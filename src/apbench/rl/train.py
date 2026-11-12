"""Train a RecurrentPPO (MlpLstmPolicy) or PPO (MlpPolicy) basal controller.

Model selection uses *separate*, fixed validation episodes (seeds
``VALIDATION_SEED_BASE + i``), not the training environments, and all seeds,
hyperparameters and software versions are written to ``metadata.json``.

Both algorithms share the environment, observation, action scaling, reward,
hyperparameters, seeds and model selection; only the policy network differs.

Usage::

    apbench-train --algorithm rppo --timesteps 3000000 --seed 0 --output runs/rppo-seed0
    cp runs/rppo-seed0/best_model.zip src/apbench/controllers/rppo/model.zip
    cp runs/rppo-seed0/metadata.json src/apbench/controllers/rppo/metadata.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

import apbench
from apbench.observation import ObsActSpec
from apbench.rl.env import TERMINATION_LOW, TERMINATION_PENALTY, SimglucoseEnv
from apbench.rl.evaluate import VALIDATION_SEED_BASE, evaluate_random_meals, nominal_simulation, run_episode

HYPERPARAMETERS = dict(
    n_steps=128,  # per environment
    batch_size=2048,  # large minibatches: sb3-contrib unrolls the LSTM step by step
    n_epochs=10,
    gamma=0.995,  # 3-minute steps: effective horizon of a few hours
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.001,
    vf_coef=0.5,
    max_grad_norm=0.5,
    learning_rate_initial=1e-3,  # linearly decayed to 0
)
ALGORITHMS = {
    "rppo": dict(
        algorithm="sb3_contrib.RecurrentPPO",
        policy="MlpLstmPolicy",
        policy_kwargs=dict(
            lstm_hidden_size=64, n_lstm_layers=1, net_arch=[64], shared_lstm=False, enable_critic_lstm=True
        ),
    ),
    "ppo": dict(algorithm="stable_baselines3.PPO", policy="MlpPolicy", policy_kwargs=dict(net_arch=[64, 64])),
}
LOG_STD_INIT = -1.0


def software_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("simglucose", "numpy", "scipy", "pandas", "torch", "stable_baselines3", "sb3_contrib", "gymnasium"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    out["simglucose_revision"] = apbench.SIMGLUCOSE_REVISION
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="rppo")
    p.add_argument("--patient", default="adolescent#001")
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--episode-minutes", type=int, default=1440)
    p.add_argument("--max-rate-u-hour", type=float, default=ObsActSpec.max_rate_u_hour)
    p.add_argument("--eval-freq", type=int, default=100_000, help="validation period in timesteps")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=HYPERPARAMETERS["learning_rate_initial"])
    p.add_argument("--log-std-init", type=float, default=LOG_STD_INIT, help="initial exploration")
    p.add_argument("--torch-threads", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    import torch
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import LinearSchedule
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from apbench.controllers.policy import PolicyController

    torch.set_num_threads(args.torch_threads)
    args.output.mkdir(parents=True, exist_ok=True)
    spec = ObsActSpec(max_rate_u_hour=args.max_rate_u_hour)
    env = make_vec_env(
        SimglucoseEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(patient=args.patient, spec=spec, episode_minutes=args.episode_minutes),
    )
    algo = ALGORITHMS[args.algorithm]
    algorithm_cls = {"rppo": RecurrentPPO, "ppo": PPO}[args.algorithm]
    policy_kwargs = dict(algo["policy_kwargs"], log_std_init=args.log_std_init)
    hp = dict(HYPERPARAMETERS, learning_rate_initial=args.learning_rate)
    hyperparameters = dict(hp)
    lr0 = hp.pop("learning_rate_initial")
    model = algorithm_cls(
        algo["policy"],
        env,
        learning_rate=LinearSchedule(lr0, 0.0, 1.0),
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device="cpu",
        verbose=0,
        tensorboard_log=str(args.output / "tb"),
        **hp,
    )
    validation_seeds = list(range(VALIDATION_SEED_BASE, VALIDATION_SEED_BASE + args.eval_episodes))

    metadata = {
        "algorithm": algo["algorithm"],
        "policy": algo["policy"],
        "policy_kwargs": policy_kwargs,
        "hyperparameters": hyperparameters,
        "obs_act_spec": spec.to_dict(),
        "training": {
            "patient": args.patient,
            "timesteps": args.timesteps,
            "seed": args.seed,
            "n_envs": args.n_envs,
            "episode_minutes": args.episode_minutes,
            "scenario": "simglucose RandomScenario; initial glucose states scaled by U(0.6, 2.0); per-episode seeds from the env RNG",
            "reward": "-(3*LBGI + HBGI)/1000 of physiological BG per control step; "
            f"-{TERMINATION_PENALTY} and termination when BG < {TERMINATION_LOW} or BG > 600",
            "device": "cpu",
            "torch_threads": args.torch_threads,
            "command": "apbench-train " + " ".join(argv if argv is not None else __import__("sys").argv[1:]),
        },
        "validation": {"seeds": validation_seeds, "selection": "max mean reward among checkpoints without nominal hypoglycemia (<70)"},
        "software": software_versions(),
    }

    class Validation(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = args.eval_freq
            self.best_key = None
            self.log = open(args.output / "validation.jsonl", "w")

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_eval:
                self.next_eval += args.eval_freq
                self.validate()
            return True

        def validate(self):
            ctrl = PolicyController(self.model, spec)
            val = evaluate_random_meals(ctrl, args.patient, validation_seeds, args.episode_minutes)["summary"]
            nom = run_episode(ctrl, nominal_simulation(args.patient), 720)
            ok = nom["min_glucose"] >= 70.0 and val["episodes_with_glucose_below_54"] == 0
            key = (ok, val["mean_reward"])
            rec = {
                "timesteps": self.num_timesteps,
                "wall_s": time.perf_counter() - tic,
                "nominal_min_glucose": nom["min_glucose"],
                "acceptable": ok,
                **val,
            }
            if self.best_key is None or key > self.best_key:
                self.best_key = key
                self.model.save(args.output / "best_model.zip")
                metadata["validation"]["best"] = rec
                rec["best"] = True
            self.log.write(json.dumps(rec) + "\n")
            self.log.flush()
            print(
                f"[{self.num_timesteps}] reward={val['mean_reward']:.1f} TIR={val['mean_time_in_range_70_180']:.1%} "
                f"<70={val['mean_time_below_70']:.2%} min={val['min_min_glucose']:.1f} "
                f"nominal_min={nom['min_glucose']:.1f} best={rec.get('best', False)}",
                flush=True,
            )

    tic = time.perf_counter()
    callback = Validation()
    model.learn(total_timesteps=args.timesteps, callback=callback, tb_log_name=args.algorithm)
    callback.validate()
    model.save(args.output / "final_model.zip")
    metadata["training"]["wall_clock_s"] = time.perf_counter() - tic
    metadata["training"]["num_timesteps"] = int(model.num_timesteps)
    with open(args.output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=1)
    env.close()
    print(f"done in {metadata['training']['wall_clock_s']:.0f}s; best: {metadata['validation'].get('best')}")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()

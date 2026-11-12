"""Evaluate controllers on reproducible held-out episodes.

Statistics use the physiological glucose sampled every minute (not CGM).

Scenarios:

* ``nominal-no-meal``: the benchmark's default initial state, no meals, 12 hours.
* ``random-meals``: Simglucose ``RandomScenario`` days with random initial glucose,
  one episode per seed (seeds disjoint from training and validation seeds).

Usage::

    apbench-evaluate --controllers rppo ppo nominal-basal --episodes 20 \
        --output src/apbench/controllers/evaluation.json
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from apbench.controllers import Controller, make_controller
from apbench.rl.env import NoMealScenario, make_episode_simulation, reward_fn
from apbench.simulator import DEFAULT_START_TIME, Simulation, SimulationError

TEST_SEED_BASE = 20_000  # held-out test episodes
VALIDATION_SEED_BASE = 10_000  # model selection during training
HYPO_EVENT_MIN_MINUTES = 15  # an excursion is a run below the threshold lasting >= 15 minutes


def count_excursions(glucose: np.ndarray, threshold: float, min_minutes: int = HYPO_EVENT_MIN_MINUTES) -> int:
    """Number of maximal runs of consecutive minute samples below ``threshold`` lasting >= ``min_minutes``."""
    below = np.asarray(glucose) < threshold
    count = run = 0
    for b in below:
        run = run + 1 if b else 0
        if run == min_minutes:
            count += 1
    return count


def run_episode(controller: Controller, sim: Simulation, minutes: int) -> dict:
    """Run one closed-loop episode to completion (no early termination)."""
    ctx = sim.reset()
    controller.reset()
    total_reward = 0.0
    error = None
    try:
        while ctx.time_min < minutes:
            ctx = sim.control_step(controller.act(ctx))
            total_reward += reward_fn(sim.ctrl_bg[-1])
    except SimulationError as e:
        error = str(e)
    g = np.asarray(sim.env.minute_glucose)
    delivered = np.asarray(sim.env.minute_delivered)
    days = len(delivered) / 1440.0
    return {
        "minutes": int(len(delivered)),
        "error": error,
        "reward": total_reward,
        "time_in_range_70_180": float(np.mean((g >= 70) & (g <= 180))),
        "time_below_70": float(np.mean(g < 70)),
        "time_below_54": float(np.mean(g < 54)),
        "time_above_180": float(np.mean(g > 180)),
        "time_above_250": float(np.mean(g > 250)),
        "mean_glucose": float(np.mean(g)),
        "min_glucose": float(np.min(g)),
        "max_glucose": float(np.max(g)),
        "hypo_excursions_70": count_excursions(g, 70.0),
        "hypo_excursions_54": count_excursions(g, 54.0),
        "insulin_total_u": float(math.fsum(delivered)),
        "insulin_u_per_day": float(math.fsum(delivered) / days) if days else 0.0,
    }


def nominal_simulation(patient: str) -> Simulation:
    return Simulation(patient, "Dexcom", "Insulet", NoMealScenario(DEFAULT_START_TIME), sensor_seed=1)


def aggregate(episodes: list[dict]) -> dict:
    keys = [k for k in episodes[0] if isinstance(episodes[0][k], (int, float)) and k != "minutes"]
    out = {f"mean_{k}": float(np.mean([e[k] for e in episodes])) for k in keys}
    out["min_min_glucose"] = float(np.min([e["min_glucose"] for e in episodes]))
    out["episodes_with_glucose_below_54"] = int(sum(e["min_glucose"] < 54 for e in episodes))
    out["episodes_with_glucose_below_70"] = int(sum(e["min_glucose"] < 70 for e in episodes))
    out["episodes_with_errors"] = int(sum(e["error"] is not None for e in episodes))
    out["n_episodes"] = len(episodes)
    return out


def evaluate_random_meals(controller: Controller, patient: str, seeds, minutes: int = 1440) -> dict:
    episodes = []
    for s in seeds:
        e = run_episode(controller, make_episode_simulation(patient, int(s)), minutes)
        e["seed"] = int(s)
        episodes.append(e)
    return {"summary": aggregate(episodes), "episodes": episodes}


def evaluate_controller(controller: Controller, patient: str, episodes: int, seed_base: int, minutes: int) -> dict:
    return {
        "nominal-no-meal": run_episode(controller, nominal_simulation(patient), 720),
        "random-meals": evaluate_random_meals(controller, patient, range(seed_base, seed_base + episodes), minutes),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--controllers", nargs="+", default=["rppo", "ppo", "nominal-basal"])
    p.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="evaluate controller NAME (rppo or ppo) with this model.zip instead of the bundled one",
    )
    p.add_argument("--patient", default="adolescent#001")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=TEST_SEED_BASE)
    p.add_argument("--minutes", type=int, default=1440, help="length of random-meal episodes")
    p.add_argument("--output", default=None, help="write results as JSON")
    args = p.parse_args(argv)

    results = {
        "patient": args.patient,
        "episodes": args.episodes,
        "seed_base": args.seed_base,
        "minutes": args.minutes,
        "controllers": {},
    }
    checkpoints = dict(c.split("=", 1) for c in args.checkpoint)
    for name in args.controllers:
        if name in checkpoints:
            from apbench.controllers.ppo import PPOController
            from apbench.controllers.rppo import RecurrentPPOController

            ctrl = {"rppo": RecurrentPPOController, "ppo": PPOController}[name](checkpoints[name])
        else:
            ctrl = make_controller(name, args.patient)
        tic = time.perf_counter()
        r = evaluate_controller(ctrl, args.patient, args.episodes, args.seed_base, args.minutes)
        results["controllers"][name] = r
        nom, rm = r["nominal-no-meal"], r["random-meals"]["summary"]
        print(
            f"{name:15s} nominal(no meal, 12h): min={nom['min_glucose']:.1f} TIR={nom['time_in_range_70_180']:.1%} "
            f"insulin={nom['insulin_total_u']:.2f}U | random meals ({rm['n_episodes']} x {args.minutes} min): "
            f"TIR={rm['mean_time_in_range_70_180']:.1%} <70={rm['mean_time_below_70']:.2%} "
            f"<54={rm['mean_time_below_54']:.2%} >250={rm['mean_time_above_250']:.1%} "
            f"min={rm['min_min_glucose']:.1f} hypo70-episodes={rm['episodes_with_glucose_below_70']} "
            f"insulin={rm['mean_insulin_u_per_day']:.1f}U/day ({time.perf_counter() - tic:.0f}s)"
        )
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()

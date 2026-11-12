"""Gymnasium training environment for the neural basal controllers.

It uses the benchmark's own :class:`~apbench.simulator.Simulation` and
:class:`~apbench.observation.ObservationBuilder`, so training and falsification
construct identical observations and apply identical pump semantics.

Each episode draws a start hour, sensor/scenario seeds and a random initial
glucose level (see ``INIT_GLUCOSE_FACTOR``) from the environment's RNG, and uses Simglucose's ``RandomScenario`` meals
(no meal announcement, no bolus: the controller is basal-only).
"""

from __future__ import annotations

import datetime

import gymnasium
import numpy as np

from apbench import _compat  # noqa: F401
from apbench.observation import ObservationBuilder, ObsActSpec
from apbench.simulator import Simulation
from simglucose.analysis.risk import risk_index
from simglucose.simulation.scenario import Scenario
from simglucose.simulation.scenario_gen import RandomScenario


class NoMealScenario(Scenario):
    def get_action(self, t):
        from simglucose.simulation.scenario import Action

        return Action(meal=0.0)

    def reset(self):
        pass


def reward_fn(bg: float) -> float:
    """Magni risk with hypoglycemia weighted 3x.

    Scaled so that discounted returns are O(1): with larger returns the value loss
    dominates PPO's jointly clipped gradient (``max_grad_norm``) and the policy barely
    learns.

    Uses physiological glucose: the reward is a training signal only and is not
    available to the controller.
    """
    lbgi, hbgi, _ = risk_index([bg], 1)
    return -(3.0 * lbgi + hbgi) / 1000.0


TERMINATION_LOW = 40.0  # mg/dL; ends a training episode (severe hypoglycemia)
TERMINATION_HIGH = 600.0
TERMINATION_PENALTY = 1.0


#: Range of the factor applied to the glucose compartments of the patient's default initial
#: state (plasma, tissue and subcutaneous glucose; x4, x5, x13). adolescent#001 starts at
#: ~149 mg/dL, so this covers roughly 90-300 mg/dL. Simglucose's own ``random_init_bg`` varies
#: glucose by only a few mg/dL, too little to make a constant basal rate a poor policy.
INIT_GLUCOSE_FACTOR = (0.6, 2.0)
GLUCOSE_STATES = (3, 4, 12)


def episode_seeds(seed: int) -> dict:
    """Deterministically derive the per-episode configuration from one integer seed."""
    rng = np.random.default_rng(seed)
    return dict(
        start_hour=int(rng.integers(0, 24)),
        sensor_seed=int(rng.integers(0, 2**31 - 1)),
        scenario_seed=int(rng.integers(0, 2**31 - 1)),
        init_glucose_factor=float(rng.uniform(*INIT_GLUCOSE_FACTOR)),
    )


def initial_state(patient: str, glucose_factor: float) -> np.ndarray:
    from simglucose.patient.t1dpatient import T1DPatient

    x0 = np.array(T1DPatient.withName(patient)._params.iloc[2:15].values, dtype=float)
    x0[list(GLUCOSE_STATES)] *= glucose_factor
    return x0


def make_episode_simulation(patient: str, seed: int, meals: bool = True) -> Simulation:
    cfg = episode_seeds(seed)
    start = datetime.datetime(2018, 1, 1, cfg["start_hour"], 0, 0)
    scenario = RandomScenario(start_time=start, seed=cfg["scenario_seed"]) if meals else NoMealScenario(start)
    return Simulation(
        patient,
        "Dexcom",
        "Insulet",
        scenario,
        sensor_seed=cfg["sensor_seed"],
        init_state=initial_state(patient, cfg["init_glucose_factor"]),
    )


class SimglucoseEnv(gymnasium.Env):
    """Basal-only control of one Simglucose patient; one step = one control period (3 min)."""

    metadata = {"render_modes": []}

    def __init__(self, patient: str = "adolescent#001", spec: ObsActSpec | None = None, episode_minutes: int = 1440):
        self.patient = patient
        self.spec = spec or ObsActSpec()
        self.episode_minutes = episode_minutes
        self.builder = ObservationBuilder(self.spec)
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(self.spec.obs_dim,), dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.sim: Simulation | None = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.sim = make_episode_simulation(self.patient, episode_seed)
        ctx = self.sim.reset()
        self.builder.reset()
        return self.builder.build(ctx), {"episode_seed": episode_seed}

    def step(self, action):
        assert self.sim is not None
        ctx = self.sim.control_step(self.spec.action_to_rate_u_min(float(np.asarray(action).reshape(-1)[0])))
        obs = self.builder.build(ctx)
        bg = self.sim.ctrl_bg[-1]
        reward = reward_fn(bg)
        terminated = bool(bg < TERMINATION_LOW or bg > TERMINATION_HIGH)
        if terminated:
            reward -= TERMINATION_PENALTY
        truncated = ctx.time_min >= self.episode_minutes
        return obs, float(reward), terminated, bool(truncated), {"bg": bg}

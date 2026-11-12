"""Deterministic closed-loop Simglucose simulation for falsification.

Time structure (all times in minutes since the start of the simulation):

* The patient ODE is integrated in 1-minute steps (Simglucose ``T1DPatient``).
  Within a minute, insulin delivery and carbohydrate intake are constant.
* The controller acts every ``sensor.sample_time`` minutes (3 minutes for Dexcom).
  This is one call of Simglucose's ``T1DSimEnv.step``.
* The falsification input (one meal) is chosen every ``interval_minutes``
  (30 minutes by default). This is one call of :meth:`Benchmark.step`.

Inputs: each meal is a finite real-valued carbohydrate amount in
[0, :data:`MAX_MEAL_GRAMS`] g. The simulator imposes no finite alphabet and no
cumulative-intake limit; admissibility of whole sequences is checked separately
(:mod:`apbench.admissibility`).

Units: the controller requests a basal *rate* in U/min (Simglucose's controller
convention). The pump (Simglucose with PR #99) quantizes and clips it using its
U/hour parameters. Delivered insulin *amounts* are in U (rate x duration).
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from apbench import _compat  # noqa: F401  (must precede simglucose imports)
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.base import Action as ControllerAction
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import Action as ScenarioAction
from simglucose.simulation.scenario import Scenario

#: Names of the six per-interval outputs, in the fixed output order.
OUTPUT_NAMES = (
    "glucose_end",
    "glucose_min",
    "glucose_max",
    "insulin_dose",
    "insulin_max_rate",
    "cgm_end",
)
#: Units of the six per-interval outputs.
OUTPUT_UNITS = ("mg/dL", "mg/dL", "mg/dL", "U", "U/hour", "mg/dL")

#: Largest accepted meal input (g of carbohydrate per interval).
MAX_MEAL_GRAMS = 80.0

#: Fixed, documented start time. The patient model does not depend on the time of day.
DEFAULT_START_TIME = datetime.datetime(2018, 1, 1, 0, 0, 0)

#: Simglucose's physiological termination thresholds on (3-minute averaged) BG, mg/dL.
DONE_LOW, DONE_HIGH = 10.0, 600.0


class SimulationError(RuntimeError):
    """The patient ODE failed or produced a non-finite state."""


class _PatientParams:
    """Patient parameters with plain attribute access.

    Simglucose's ODE right-hand side reads parameters as attributes of a pandas
    Series, which dominates the simulation time. This holds the *same* Python
    objects as plain attributes (so the arithmetic is unchanged) and keeps the
    Series for positional access (``iloc``) used in ``T1DPatient.reset``.
    """

    def __init__(self, series):
        self.__dict__.update(series.to_dict())
        self._series = series

    @property
    def iloc(self):
        return self._series.iloc


class FastT1DPatient(T1DPatient):
    """``T1DPatient`` with faster parameter access; numerically identical results."""

    def __init__(self, params, **kwargs):
        super().__init__(_PatientParams(params), **kwargs)


class MealScenario(Scenario):
    """Scenario that injects externally chosen meals, each exactly once.

    :meth:`schedule` arms a meal. Simglucose queries the scenario once per
    simulated minute, and the armed meal is returned (in grams) on the first query
    only, then cleared. Simglucose's patient model then eats it at 5 g/min.
    """

    def __init__(self, start_time: datetime.datetime):
        super().__init__(start_time=start_time)
        self._pending = 0.0

    def schedule(self, grams: float) -> None:
        if not math.isfinite(grams) or grams < 0:
            raise ValueError(f"meal must be a finite non-negative amount, got {grams!r}")
        self._pending = float(grams)

    def get_action(self, t: datetime.datetime) -> ScenarioAction:
        meal, self._pending = self._pending, 0.0
        return ScenarioAction(meal=meal)

    def reset(self) -> None:
        self._pending = 0.0


class RecordingT1DSimEnv(T1DSimEnv):
    """Simglucose ``T1DSimEnv`` that records every 1-minute sample.

    Row ``t`` of the minute log holds the physiological glucose and the raw CGM
    sensor value at minute ``t``, and the inputs applied during ``[t, t+1)``.
    """

    def _reset(self):
        super()._reset()
        self.minute_t: list[float] = [float(self.patient.t)]
        self.minute_glucose: list[float] = [float(self.patient.observation.Gsub)]
        self.minute_cgm: list[float] = [float(self.sensor._last_CGM)]
        self.minute_meal: list[float] = []
        self.minute_intake: list[float] = []
        self.minute_requested: list[float] = []
        self.minute_delivered: list[float] = []

    def mini_step(self, action):
        t0 = float(self.patient.t)
        try:
            CHO, insulin, BG, CGM = super().mini_step(action)
        except RuntimeError as e:  # T1DPatient.step re-raises with no active exception
            raise SimulationError(f"patient ODE failed at t={t0} min") from e
        state = np.asarray(self.patient.state, dtype=float)
        if not self.patient._odesolver.successful() or not np.all(np.isfinite(state)) or not math.isfinite(BG):
            raise SimulationError(f"patient ODE failed or became non-finite in minute [{t0}, {t0 + 1})")
        self.minute_meal.append(float(CHO))
        self.minute_intake.append(float(self.patient._last_action.CHO))
        self.minute_requested.append(float(action.basal))
        self.minute_delivered.append(float(insulin))
        self.minute_t.append(float(self.patient.t))
        self.minute_glucose.append(float(BG))
        self.minute_cgm.append(float(self.sensor._last_CGM))
        return CHO, insulin, BG, CGM

    # Initial sample after reset: _reset() already measured once; reset() measures
    # again for the returned observation. Keep the minute log in sync with it.
    def reset(self):
        step = super().reset()
        self.minute_cgm[0] = float(self.sensor._last_CGM)
        return step


@dataclass(frozen=True)
class ControlContext:
    """Information available to a controller at a control instant.

    ``cgm`` is the controller-visible CGM observation: Simglucose's average of
    the three 1-minute sensor readings of the preceding control step (each
    reading is held between 3-minute sensor samples).
    """

    time_min: float
    cgm: float  # mg/dL
    prev_delivered_u_min: float  # mean delivered rate over the previous control step, U/min
    max_basal_u_min: float  # pump maximum basal rate, U/min
    sample_time_min: float  # control period, minutes


class Simulation:
    """Low-level closed loop without a controller: patient, CGM, pump and scenario.

    Shared by :class:`Benchmark` and the RL training environment, so that both
    produce exactly the same :class:`ControlContext` (and hence observations).
    """

    def __init__(
        self,
        patient: str,
        sensor: str,
        pump: str,
        scenario: Scenario,
        sensor_seed: int | None,
        patient_seed: int | None = None,
        random_init_bg: bool = False,
        init_state=None,
    ):
        if sensor_seed is None:
            raise ValueError("sensor_seed must be set for reproducible CGM noise")
        self.patient_name = patient
        self.env = RecordingT1DSimEnv(
            FastT1DPatient.withName(
                patient, random_init_bg=random_init_bg, seed=patient_seed, init_state=init_state
            ),
            CGMSensor.withName(sensor, seed=sensor_seed),
            InsulinPump.withName(pump),
            scenario,
        )
        self.scenario = scenario
        self.sample_time = float(self.env.sample_time)
        if self.sample_time != int(self.sample_time):
            raise ValueError("the sensor sample time must be an integer number of minutes")

    @property
    def pump(self) -> InsulinPump:
        return self.env.pump

    def reset(self) -> ControlContext:
        step = self.env.reset()
        self.ctrl_time = [0.0]
        self.ctrl_cgm = [float(step.observation.CGM)]
        self.ctrl_bg = [float(step.info["bg"])]
        self.ctrl_done = [False]
        self.first_done_time: float | None = None
        return self._context(0.0)

    def _context(self, prev_rate: float) -> ControlContext:
        return ControlContext(
            time_min=self.ctrl_time[-1],
            cgm=self.ctrl_cgm[-1],
            prev_delivered_u_min=prev_rate,
            max_basal_u_min=float(self.pump.max_basal),
            sample_time_min=self.sample_time,
        )

    def control_step(self, basal_u_min: float) -> ControlContext:
        """Apply a basal request (U/min, bolus 0) for one control period.

        Simglucose's ``done`` flag is recorded but does not stop the simulation.
        """
        if not math.isfinite(basal_u_min):
            raise SimulationError(f"controller returned a non-finite basal rate {basal_u_min!r}")
        step = self.env.step(ControllerAction(basal=float(basal_u_min), bolus=0.0))
        self.ctrl_time.append(float(self.env.patient.t))
        self.ctrl_cgm.append(float(step.observation.CGM))
        self.ctrl_bg.append(float(step.info["bg"]))
        self.ctrl_done.append(bool(step.done))
        if step.done and self.first_done_time is None:
            self.first_done_time = self.ctrl_time[-1]
        n = int(self.sample_time)
        prev_rate = float(np.mean(self.env.minute_delivered[-n:]))
        return self._context(prev_rate)


@dataclass(frozen=True)
class IntervalSummary:
    """The six outputs of one falsification interval (see :data:`OUTPUT_NAMES`)."""

    glucose_end: float
    glucose_min: float
    glucose_max: float
    insulin_dose: float
    insulin_max_rate: float
    cgm_end: float

    def as_list(self) -> list[float]:
        return [float(getattr(self, name)) for name in OUTPUT_NAMES]


@dataclass
class Trace:
    """Full simulation trace.

    Minute grid (``minute_*``): ``minute_time[t] == t``; glucose/CGM samples have
    ``N + 1`` entries (minutes ``0..N``); inputs applied during ``[t, t+1)`` have
    ``N`` entries. Control grid (``ctrl_*``): one entry per control instant.
    """

    interval_minutes: int
    meals: list[float]
    minute_time: np.ndarray
    glucose: np.ndarray  # physiological (subcutaneous) glucose Gsub, mg/dL
    cgm_sample: np.ndarray  # raw CGM sensor value (held between sensor samples), mg/dL
    meal: np.ndarray  # meal announced at minute t (g)
    cho_intake: np.ndarray  # carbohydrate eaten during [t, t+1) (g)
    requested_basal: np.ndarray  # controller request, U/min
    delivered_rate: np.ndarray  # actually delivered insulin rate during [t, t+1), U/min
    ctrl_time: np.ndarray
    ctrl_cgm: np.ndarray  # controller-visible CGM (3-minute average), mg/dL
    ctrl_bg: np.ndarray  # Simglucose's 3-minute average BG, mg/dL
    ctrl_done: np.ndarray  # Simglucose "done" flag (BG < 10 or BG > 600)
    summaries: list[IntervalSummary] = field(default_factory=list)

    @property
    def threshold_crossed(self) -> bool:
        return bool(np.any(self.ctrl_done))

    @property
    def duration_minutes(self) -> float:
        return float(self.minute_time[-1])

    def outputs(self) -> np.ndarray:
        return np.array([s.as_list() for s in self.summaries], dtype=float).reshape(-1, len(OUTPUT_NAMES))

    def minute_frame(self):
        import pandas as pd

        n = len(self.delivered_rate)
        pad = lambda a: np.append(np.asarray(a, dtype=float), np.nan)  # noqa: E731
        cgm_obs = np.full(n + 1, np.nan)
        cgm_obs[np.asarray(self.ctrl_time, dtype=int)] = self.ctrl_cgm
        return pd.DataFrame(
            {
                "time_min": self.minute_time,
                "glucose_mg_dl": self.glucose,
                "cgm_sample_mg_dl": self.cgm_sample,
                "cgm_observation_mg_dl": cgm_obs,
                "meal_g": pad(self.meal),
                "cho_intake_g": pad(self.cho_intake),
                "requested_basal_u_min": pad(self.requested_basal),
                "delivered_rate_u_min": pad(self.delivered_rate),
                "delivered_rate_u_hour": pad(self.delivered_rate) * 60.0,
            }
        )

    def output_frame(self):
        import pandas as pd

        df = pd.DataFrame(self.outputs(), columns=list(OUTPUT_NAMES))
        df.insert(0, "meal_g", self.meals[: len(df)])
        df.insert(0, "end_min", [(k + 1) * self.interval_minutes for k in range(len(df))])
        df.insert(0, "interval", range(len(df)))
        return df

    def save(self, prefix) -> None:
        """Write ``<prefix>_minutes.csv`` (full trace) and ``<prefix>_outputs.csv`` (six outputs)."""
        self.minute_frame().to_csv(f"{prefix}_minutes.csv", index=False)
        self.output_frame().to_csv(f"{prefix}_outputs.csv", index=False)


def summarize_interval(trace_env: RecordingT1DSimEnv, start: int, end: int, cgm_end: float) -> IntervalSummary:
    """Compute the six outputs of the interval from minute ``start`` to minute ``end``.

    Glucose extrema use the right-closed interval ``(start, end]``: the samples at
    minutes ``start + 1, ..., end``. The sample at ``start`` belongs to the previous
    interval (or, for ``start == 0``, is the initial sample, which no interval
    covers). Insulin uses the rates applied during ``[start, end)``.
    """
    glucose = trace_env.minute_glucose[start + 1 : end + 1]
    rates = trace_env.minute_delivered[start:end]
    return IntervalSummary(
        glucose_end=float(glucose[-1]),
        glucose_min=float(min(glucose)),
        glucose_max=float(max(glucose)),
        insulin_dose=float(math.fsum(rates) * 1.0),  # U/min x 1 min
        insulin_max_rate=float(max(rates) * 60.0),  # U/min -> U/hour
        cgm_end=float(cgm_end),
    )


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration of the system under test. Plain Python; no external format."""

    patient: str = "adolescent#001"
    sensor: str = "Dexcom"
    pump: str = "Insulet"
    controller: str | Callable = "rppo"  # registry name or factory ``f(patient_name) -> Controller``
    start_time: datetime.datetime = DEFAULT_START_TIME
    sensor_seed: int = 1
    patient_seed: int | None = None
    random_init_bg: bool = False
    interval_minutes: int = 30
    horizon_steps: int = 24  # 24 x 30 min = 12 hours

    @property
    def horizon_minutes(self) -> int:
        return self.interval_minutes * self.horizon_steps

    def describe(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["start_time"] = self.start_time.isoformat()
        if not isinstance(self.controller, str):
            d["controller"] = getattr(self.controller, "__name__", repr(self.controller))
        return d


class Benchmark:
    """Closed-loop artificial pancreas with one meal input per interval.

    >>> bench = Benchmark(BenchmarkConfig(controller="nominal-basal"))
    >>> bench.reset()
    >>> out = bench.step(40.0)          # IntervalSummary of the first 30 minutes
    >>> trace = bench.simulate([0.0] * 24)  # resets, then runs the whole sequence
    """

    def __init__(self, config: BenchmarkConfig | None = None, controller=None):
        from apbench.controllers import make_controller

        self.config = config or BenchmarkConfig()
        self.scenario = MealScenario(self.config.start_time)
        self.sim = Simulation(
            self.config.patient,
            self.config.sensor,
            self.config.pump,
            self.scenario,
            sensor_seed=self.config.sensor_seed,
            patient_seed=self.config.patient_seed,
            random_init_bg=self.config.random_init_bg,
        )
        steps = self.config.interval_minutes / self.sim.sample_time
        if steps != int(steps) or steps < 1:
            raise ValueError("interval_minutes must be a positive multiple of the sensor sample time")
        self.control_steps_per_interval = int(steps)
        self.controller = controller if controller is not None else make_controller(self.config.controller, self.config.patient)
        self._ctx: ControlContext | None = None

    def reset(self) -> None:
        """Restore patient, scenario, sensor noise, pump and controller state."""
        self._ctx = self.sim.reset()
        self.controller.reset()
        self.meals: list[float] = []
        self.summaries: list[IntervalSummary] = []

    def step(self, meal_grams: float) -> IntervalSummary:
        """Inject one meal at the start of the next interval and simulate the interval.

        ``meal_grams`` must be finite and in [0, :data:`MAX_MEAL_GRAMS`] g. Cumulative
        intake is not restricted here (see :mod:`apbench.admissibility`).
        """
        if self._ctx is None:
            raise RuntimeError("call reset() before step()")
        meal = float(meal_grams)
        if not (math.isfinite(meal) and 0.0 <= meal <= MAX_MEAL_GRAMS):
            raise ValueError(f"meal must be a finite amount in [0, {MAX_MEAL_GRAMS:g}] g, got {meal_grams!r}")
        start = len(self.sim.env.minute_delivered)
        self.scenario.schedule(meal)
        ctx = self._ctx
        for _ in range(self.control_steps_per_interval):
            ctx = self.sim.control_step(self.controller.act(ctx))
        self._ctx = ctx
        summary = summarize_interval(self.sim.env, start, start + self.config.interval_minutes, ctx.cgm)
        self.meals.append(meal)
        self.summaries.append(summary)
        return summary

    def simulate(self, meals: Sequence[float]) -> Trace:
        """Reset, run the given meal sequence and return the full trace."""
        self.reset()
        for meal in meals:
            self.step(meal)
        return self.trace()

    def trace(self) -> Trace:
        env, sim = self.sim.env, self.sim
        return Trace(
            interval_minutes=self.config.interval_minutes,
            meals=list(self.meals),
            minute_time=np.asarray(env.minute_t, dtype=float),
            glucose=np.asarray(env.minute_glucose, dtype=float),
            cgm_sample=np.asarray(env.minute_cgm, dtype=float),
            meal=np.asarray(env.minute_meal, dtype=float),
            cho_intake=np.asarray(env.minute_intake, dtype=float),
            requested_basal=np.asarray(env.minute_requested, dtype=float),
            delivered_rate=np.asarray(env.minute_delivered, dtype=float),
            ctrl_time=np.asarray(sim.ctrl_time, dtype=float),
            ctrl_cgm=np.asarray(sim.ctrl_cgm, dtype=float),
            ctrl_bg=np.asarray(sim.ctrl_bg, dtype=float),
            ctrl_done=np.asarray(sim.ctrl_done, dtype=bool),
            summaries=list(self.summaries),
        )

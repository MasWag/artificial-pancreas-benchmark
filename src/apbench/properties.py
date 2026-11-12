"""Benchmark properties, evaluated with RTAMT on the 1-minute trace.

All properties are discrete-time STL formulas over signals sampled every minute
at ``t = 0, 1, ..., H`` (``H`` = horizon in minutes). RTAMT's default time unit is
used with a unit sampling period, so **one time unit is one minute** and time
bounds are in minutes. Robustness is RTAMT's quantitative semantics at ``t = 0``:
a property is *violated iff its robustness is negative*. A robustness of exactly 0
(a boundary case) counts as satisfied.

RTAMT evaluates bounded temporal operators over truncated windows near the end
of a finite trace. Properties with future-time windows of length ``T`` therefore
quantify over ``[0, H - T]`` only, so that every window lies inside the trace.

Signals (all from :class:`apbench.simulator.Trace`):

=================== ======== ============================================================
``glucose``         mg/dL    physiological (subcutaneous) glucose ``Gsub`` at minute t
``cgm``             mg/dL    raw CGM sensor value at minute t (held between sensor samples)
``meal``            g        meal announced at minute t (0 elsewhere)
``insulin_rate``    U/hour   insulin rate delivered during [t, t+1); 0 at t = H
``insulin_dose_60`` U        insulin delivered during [t-60, t) (interval observer)
=================== ======== ============================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from apbench.simulator import Trace

#: Scale of the meal-free assumption ``SCALE * (1 - meal) >= 0`` (meal in g). It makes the
#: assumption's robustness (+SCALE without meals, <= -SCALE * 39 with a meal) dominate, so the
#: robustness of an assumed recovery property is the glucose margin, not a mixed-unit meal margin.
MEAL_GUARD_SCALE = 1e6

SIGNALS = ("glucose", "cgm", "meal", "insulin_rate", "insulin_dose_60")


def trace_signals(trace: Trace) -> dict[str, np.ndarray]:
    """Build the 1-minute signals of the property suite from a trace."""
    n = len(trace.delivered_rate)
    rate_u_min = np.append(trace.delivered_rate, 0.0)  # nothing is delivered after the horizon
    cum = np.concatenate([[0.0], np.cumsum(trace.delivered_rate)])  # cum[t] = dose during [0, t)
    t = np.arange(n + 1)
    dose_60 = cum[t] - cum[np.maximum(t - 60, 0)]
    return {
        "time": t.astype(float),
        "glucose": np.asarray(trace.glucose, dtype=float),
        "cgm": np.asarray(trace.cgm_sample, dtype=float),
        "meal": np.append(trace.meal, 0.0),
        "insulin_rate": rate_u_min * 60.0,
        "insulin_dose_60": dose_60,
    }


@dataclass(frozen=True)
class Property:
    """A named STL property with its physical-unit documentation."""

    name: str
    formula: str  # RTAMT discrete-time STL
    description: str
    _spec: list = field(default_factory=list, repr=False, compare=False)  # lazily parsed RTAMT spec

    def _parsed(self):
        if not self._spec:
            import rtamt

            spec = rtamt.StlDiscreteTimeSpecification()
            spec.name = self.name
            for var in SIGNALS:
                spec.declare_var(var, "float")
            spec.spec = self.formula
            spec.parse()
            self._spec.append(spec)
        return self._spec[0]

    def robustness(self, signals: dict[str, np.ndarray]) -> float:
        """Robustness at time 0 (negative iff violated)."""
        dataset = {k: [float(x) for x in v] for k, v in signals.items()}
        rob = self._parsed().evaluate(dataset)
        value = float(rob[0][1])
        if math.isnan(value):
            raise ValueError(f"robustness of {self.name} is NaN")
        return value


def hypoglycemia(threshold: float) -> Property:
    return Property(
        f"hypo_{threshold:g}",
        f"always (glucose >= {threshold:g})",
        f"Physiological glucose never falls below {threshold:g} mg/dL (sampled every minute).",
    )


def hyperglycemia(threshold: float) -> Property:
    return Property(
        f"hyper_{threshold:g}",
        f"always (glucose <= {threshold:g})",
        f"Physiological glucose never exceeds {threshold:g} mg/dL (sampled every minute).",
    )


def prolonged_high(horizon: int, threshold: float = 250.0, duration: int = 120) -> Property:
    return Property(
        f"prolonged_{threshold:g}_{duration}min",
        f"always[0:{horizon - duration}] (eventually[0:{duration}] (glucose <= {threshold:g}))",
        f"Glucose never stays above {threshold:g} mg/dL for {duration} minutes: no {duration + 1} consecutive "
        f"1-minute samples (a closed {duration}-minute window) all exceed {threshold:g} mg/dL.",
    )


def recovery(horizon: int, trigger: float = 250.0, target: float = 180.0, deadline: int = 180) -> Property:
    return Property(
        f"recovery_{trigger:g}_to_{target:g}_{deadline}min",
        f"always[0:{horizon - deadline}] ((glucose > {trigger:g}) -> "
        f"((always[0:{deadline}] ({MEAL_GUARD_SCALE:g} * (1 - meal) >= 0)) -> "
        f"(eventually[0:{deadline}] (glucose <= {target:g}))))",
        f"Whenever glucose exceeds {trigger:g} mg/dL and no meal is announced in the next {deadline} minutes, "
        f"glucose returns to {target:g} mg/dL or below within {deadline} minutes (engineering deadline, "
        "not a clinical standard). Only trigger times with a complete window are checked. Meals below 1 g "
        "count as no meal; the meal guard is scaled so that its robustness never masks the glucose margin.",
    )


def insulin_dose_window(max_dose: float) -> Property:
    """Hourly delivered-dose limit, for custom suites; not part of :func:`default_properties`."""
    return Property(
        f"insulin_dose_60min_{max_dose:g}U",
        f"always (insulin_dose_60 <= {max_dose:g})",
        f"At most {max_dose:g} U of insulin is delivered in any 60-minute window (windows ending at each minute).",
    )


def insulin_suspension(glucose_threshold: float, max_rate: float) -> Property:
    return Property(
        f"suspend_below_{glucose_threshold:g}",
        f"always ((glucose < {glucose_threshold:g}) -> (insulin_rate <= {max_rate:g}))",
        f"Whenever glucose is below {glucose_threshold:g} mg/dL, the delivered insulin rate is at most "
        f"{max_rate:g} U/hour.",
    )


def default_properties(horizon_minutes: int = 720) -> list[Property]:
    """The benchmark property suite (six properties) for a given horizon (minutes)."""
    return [
        hypoglycemia(54.0),
        hypoglycemia(70.0),
        hyperglycemia(300.0),
        prolonged_high(horizon_minutes, 250.0, 120),
        recovery(horizon_minutes, 250.0, 180.0, 180),
        insulin_suspension(70.0, 1.0),
    ]


def evaluate(properties: list[Property], trace: Trace) -> dict[str, float]:
    """Robustness of every property on one trace (the trace is simulated once)."""
    signals = trace_signals(trace)
    return {p.name: p.robustness(signals) for p in properties}

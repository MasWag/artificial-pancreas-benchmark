"""Admissible meal-input sequences of the reference workload.

The simulator accepts any finite meal in [0, 80] g per interval. The constrained
falsification workload additionally restricts *sequences* (README, "Admissible
input sequences"). For the standard 12-hour workload (24 intervals of 30 minutes):

* every meal is in [0, 80] g;
* the intake over any four successive intervals (two hours) is at most 120 g,
  using the available prefix for the first intervals;
* the total intake over the 24 intervals is at most 300 g.

These are constraints on test inputs, not STL properties, and they are checked
here, separately from the simulator. Sums are computed exactly (floats converted
to :class:`fractions.Fraction`), so equality cases are admissible and no
floating-point tolerance is involved.

Other interval durations or longer horizons are workload variants: the window
length and the limits must then be given explicitly as an :class:`InputConstraints`;
:func:`reference_constraints` refuses to reuse the four-interval window for them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Sequence


@dataclass(frozen=True)
class InputConstraints:
    """Per-interval range plus rolling-window and total intake limits (grams)."""

    interval_minutes: int = 30
    horizon_steps: int = 24
    max_meal: float = 80.0
    window_intervals: int = 4  # 4 x 30 min = 2 hours
    window_limit: float = 120.0
    total_limit: float = 300.0

    def describe(self) -> dict:
        return asdict(self)


#: The reference workload: 24 x 30 minutes, <= 80 g per interval, <= 120 g per 2 hours, <= 300 g in total.
REFERENCE_CONSTRAINTS = InputConstraints()


def reference_constraints(config) -> InputConstraints:
    """The reference constraints for a benchmark configuration, if they apply to it.

    They apply to 30-minute intervals and at most 24 of them (a shorter horizon is a
    prefix of the reference workload). Anything else is a workload variant whose
    constraints must be specified explicitly.
    """
    ref = REFERENCE_CONSTRAINTS
    if config.interval_minutes != ref.interval_minutes or not 1 <= config.horizon_steps <= ref.horizon_steps:
        raise ValueError(
            f"the reference input constraints are defined for up to {ref.horizon_steps} intervals of "
            f"{ref.interval_minutes} minutes; got {config.horizon_steps} x {config.interval_minutes} minutes. "
            "Specify InputConstraints for this workload variant explicitly."
        )
    return ref


def _exact(meals: Sequence[float]) -> list[Fraction]:
    return [Fraction(float(m)) for m in meals]


def violations(meals: Sequence[float], constraints: InputConstraints = REFERENCE_CONSTRAINTS, prefix: bool = False) -> list[str]:
    """Reasons why ``meals`` is not admissible (empty if it is).

    With ``prefix=True``, ``meals`` may be shorter than the horizon: it is then
    checked as the beginning of a sequence. Because meals are non-negative, an
    inadmissible prefix cannot become admissible by appending more meals.
    """
    c = constraints
    reasons = []
    n = len(meals)
    if n > c.horizon_steps or (not prefix and n != c.horizon_steps):
        reasons.append(f"length {n} does not match the horizon of {c.horizon_steps} intervals")
    values = [float(m) for m in meals]
    bad = [k for k, m in enumerate(values) if not (math.isfinite(m) and 0.0 <= m <= c.max_meal)]
    for k in bad:
        reasons.append(f"meal {k} = {values[k]!r} g is not a finite amount in [0, {c.max_meal:g}] g")
    if bad:
        return reasons
    exact = _exact(values)
    window, total = Fraction(c.window_limit), Fraction(c.total_limit)
    for k in range(n):
        s = sum(exact[max(0, k - c.window_intervals + 1) : k + 1], Fraction(0))
        if s > window:
            reasons.append(
                f"intake over intervals {max(0, k - c.window_intervals + 1)}..{k} is {float(s):.17g} g > {c.window_limit:g} g"
            )
    s = sum(exact, Fraction(0))
    if s > total:
        reasons.append(f"total intake {float(s):.17g} g > {c.total_limit:g} g")
    return reasons


def is_admissible(meals: Sequence[float], constraints: InputConstraints = REFERENCE_CONSTRAINTS, prefix: bool = False) -> bool:
    return not violations(meals, constraints, prefix)


def remaining_allowance(meals_so_far: Sequence[float], constraints: InputConstraints = REFERENCE_CONSTRAINTS) -> float:
    """Largest admissible next meal after the admissible prefix ``meals_so_far``.

    This is the README's ``u_k = min(80, 120 - sum(m[max(0,k-3):k]), 300 - sum(m[:k]))``
    (for the reference constraints), computed exactly and rounded *down* to a
    float, so that appending any meal in [0, u_k] keeps the sequence admissible.
    """
    c = constraints
    exact = _exact(meals_so_far)
    k = len(exact)
    w = c.window_intervals - 1
    recent = sum(exact[max(0, k - w) : k], Fraction(0)) if w > 0 else Fraction(0)
    u = min(Fraction(c.max_meal), Fraction(c.window_limit) - recent, Fraction(c.total_limit) - sum(exact, Fraction(0)))
    if u <= 0:
        return 0.0
    f = float(u)
    if Fraction(f) > u:  # float() rounds to nearest; round down instead
        f = math.nextafter(f, 0.0)
    return f

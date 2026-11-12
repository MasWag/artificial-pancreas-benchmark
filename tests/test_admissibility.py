"""Admissible input sequences of the reference workload (README)."""

import math

import numpy as np
import pytest

from apbench import BenchmarkConfig
from apbench.admissibility import (
    REFERENCE_CONSTRAINTS,
    InputConstraints,
    is_admissible,
    reference_constraints,
    remaining_allowance,
    violations,
)


def pad(prefix, n=24):
    return list(prefix) + [0.0] * (n - len(prefix))


def test_reference_constraints():
    c = REFERENCE_CONSTRAINTS
    assert (c.interval_minutes, c.horizon_steps, c.max_meal) == (30, 24, 80.0)
    assert (c.window_intervals, c.window_limit, c.total_limit) == (4, 120.0, 300.0)


def test_readme_examples():
    assert is_admissible(pad([80, 20, 20, 0]))
    assert not is_admissible(pad([80, 80, 0, 0]))
    sixteen = pad([20.0] * 16)
    assert is_admissible(sixteen[:15], prefix=True)  # 300 g: equality is admissible
    v = violations(sixteen)
    assert len(v) == 1 and "total" in v[0]


def test_rolling_window_equality_and_excess():
    assert is_admissible(pad([30.0, 30.0, 30.0, 30.0]))  # exactly 120 g
    assert is_admissible(pad([0, 80, 0, 0, 40, 0, 80]))  # windows 1..4 and 4..7: 120 g each
    over = math.nextafter(30.0, 31.0)
    assert not is_admissible(pad([30.0, 30.0, 30.0, over]))
    assert not is_admissible(pad([0, 0, 0, 0, 0, 80, 0, 0, 41]))  # a window in the middle
    assert is_admissible(pad([80, 0, 0, 0, 80]))  # the first meal left the window


def test_initial_prefix_window_and_exact_sums():
    assert is_admissible([80.0, 40.0], prefix=True)
    assert not is_admissible([80.0, 40.0, 1e-9], prefix=True)
    # Sums are exact over the binary float values: no tolerance, no accumulation order effects.
    assert is_admissible([40.5, 40.25, 39.25], prefix=True)  # exactly 120 g
    assert not is_admissible([40.1, 40.2, 39.7], prefix=True)  # float sum is 120.0, exact sum exceeds it


def test_total_equality_and_excess():
    seq = [80, 40, 0, 0] * 2 + [80, 20, 0, 0]  # 340 g
    assert not is_admissible(pad(seq))
    seq = [60.0, 0.0, 0.0, 0.0] * 5  # 300 g
    assert is_admissible(pad(seq))
    seq[-4] = math.nextafter(60.0, 61.0)
    assert not is_admissible(pad(seq))


@pytest.mark.parametrize("bad", [-1e-9, 80.0001, float("nan"), float("inf")])
def test_range_and_finiteness(bad):
    v = violations(pad([bad]))
    assert v and "finite amount" in v[0]


def test_length_and_prefixes():
    assert not is_admissible([0.0] * 23)
    assert is_admissible([0.0] * 23, prefix=True)
    assert is_admissible([], prefix=True)
    assert not is_admissible([0.0] * 25, prefix=True)
    # an inadmissible prefix stays inadmissible when extended
    assert not is_admissible([80.0, 80.0], prefix=True) and not is_admissible(pad([80.0, 80.0]))


def test_reference_constraints_only_for_reference_workload():
    assert reference_constraints(BenchmarkConfig()) is REFERENCE_CONSTRAINTS
    assert reference_constraints(BenchmarkConfig(horizon_steps=6)) is REFERENCE_CONSTRAINTS  # prefix
    with pytest.raises(ValueError):
        reference_constraints(BenchmarkConfig(interval_minutes=15, horizon_steps=48))
    with pytest.raises(ValueError):
        reference_constraints(BenchmarkConfig(horizon_steps=30))


def test_explicit_variant_constraints():
    hourly = InputConstraints(interval_minutes=60, horizon_steps=12, window_intervals=2)
    assert is_admissible([80, 40] + [0] * 10, hourly)
    assert not is_admissible([80, 0, 0, 0] + [80, 0] * 4, hourly)  # 400 g total


def readme_allowance(m):
    k = len(m)
    return min(80.0, 120.0 - sum(m[max(0, k - 3) : k]), 300.0 - sum(m))


def test_remaining_allowance_matches_readme_formula():
    rng = np.random.default_rng(3)
    for _ in range(200):
        m = []
        for _ in range(24):
            u = remaining_allowance(m)
            assert u == pytest.approx(max(0.0, readme_allowance(m)), abs=1e-9)
            assert is_admissible(m + [u], prefix=True)  # the allowance itself is admissible
            m.append(float(rng.choice([0.0, u, u * rng.random()])))
    assert remaining_allowance([]) == 80.0
    assert remaining_allowance([80.0, 40.0]) == 0.0
    assert remaining_allowance([60.0, 0, 0, 0] * 5) == 0.0
    assert remaining_allowance([80.0, 20.0, 0.0]) == 20.0
    assert remaining_allowance([80.0, 20.0, 0.0, 0.0]) == 80.0  # 80 g left the window


def test_remaining_allowance_rounds_down():
    m = [0.1, 0.2]  # exact sum is slightly above 0.3 in binary
    u = remaining_allowance(m)
    assert is_admissible(m + [u], prefix=True)
    assert not is_admissible(m + [math.nextafter(u, 200.0)], prefix=True)

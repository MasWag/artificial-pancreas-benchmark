"""Tests of the trace-reuse orchestration (trace_reuse.py) with small synthetic models (no simulator)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("staliro")

from staliro import Result, SignalInput, Trace, staliro  # noqa: E402
from staliro import TestOptions as Options  # noqa: E402  (renamed so that pytest does not collect it)
from staliro.models import Model  # noqa: E402
from staliro.optimizers import DualAnnealing  # noqa: E402
from staliro.signals import piecewise_constant  # noqa: E402
from staliro.specifications import rtamt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_reuse import (  # noqa: E402
    ALL_FALSIFIED,
    BUDGET_EXHAUSTED,
    TIMEOUT,
    run_multi_property,
)

TIMES = [0.5 * i for i in range(7)]


def _trace(peak: float) -> Trace[dict[str, float]]:
    """A trace of x whose maximum is peak (reached at the middle)."""

    xs = [0.0, 1.0, 2.0, peak, 2.0, 1.0, 0.0]
    return Trace(times=TIMES, states=[{"x": float(x)} for x in xs])


class ScriptedModel(Model[dict[str, float], None]):
    """Returns the scripted peaks in order (then the last one forever), ignoring the inputs."""

    def __init__(self, peaks: list[float], clock: "FakeClock | None" = None):
        self.peaks = peaks
        self.calls = 0
        self.clock = clock

    def simulate(self, inputs):
        peak = self.peaks[min(self.calls, len(self.peaks) - 1)]
        self.calls += 1
        if self.clock is not None:
            self.clock.now += 1.0
        return Result(_trace(peak), None)


class InputModel(Model[dict[str, float], None]):
    """x follows the input signal u; records the samples it is called with."""

    def __init__(self):
        self.samples: list[list[float]] = []

    def simulate(self, inputs):
        self.samples.append(inputs.sample.values.tolist())
        u = inputs.signals["u"]
        return Result(
            Trace(times=TIMES, states=[{"x": float(u.at_time(t))} for t in TIMES]), None
        )


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _options(iterations: int, seed: int = 12345) -> Options:
    return Options(
        tspan=(0.0, 3.0),
        iterations=iterations,
        seed=seed,
        signals={"u": SignalInput(control_points=[(0.0, 100.0)] * 3, factory=piecewise_constant)},
    )


def _specs(*formulas: str):
    return [
        (f"phi{idx}", formula, rtamt.parse_discrete(formula))
        for idx, formula in enumerate(formulas, start=1)
    ]


def _run(model, formulas, iterations, **kwargs):
    return run_multi_property(
        model, _specs(*formulas), lambda: _options(iterations), log=lambda _: None, **kwargs
    )


def test_one_trace_falsifies_multiple_specifications():
    model = ScriptedModel([80.0, 97.0, 10.0])
    summary = _run(model, ["always (x < 90)", "always (x < 95)", "always (x < 1000)"], 20)
    phi1, phi2, phi3 = summary.records

    assert phi1.falsified_at_sim == phi2.falsified_at_sim == 2
    assert (phi1.via, phi2.via) == ("target", "reuse")
    assert phi1.target_status == "falsified" and phi1.target_sims == 2
    # phi2 is never targeted; phi3 exhausts exactly its budget
    assert phi2.target_status == "skipped" and phi2.target_sims == 0
    assert phi3.target_status == "budget" and phi3.target_sims == 20
    # The trace falsifying two specifications is counted once
    assert summary.simulations == model.calls == 22
    assert summary.num_falsified == 2
    assert summary.status == BUDGET_EXHAUSTED


def test_incidentally_falsified_specification_is_skipped():
    model = ScriptedModel([10.0, 10.0, 97.0, 10.0])
    summary = _run(model, ["always (x < 1000)", "always (x < 95)"], 10)
    phi1, phi2 = summary.records

    assert phi1.target_status == "budget" and phi1.target_sims == 10
    assert phi2.falsified_at_sim == 3 and phi2.via == "reuse"
    assert phi2.target_status == "skipped" and phi2.target_sims == 0
    assert summary.simulations == model.calls == 10


def test_non_violating_traces_change_nothing():
    model = ScriptedModel([10.0])
    summary = _run(model, ["always (x < 90)", "always (x < 95)"], 5)

    assert summary.num_falsified == 0
    assert all(not r.falsified and r.via is None for r in summary.records)
    assert [r.target_status for r in summary.records] == ["budget", "budget"]
    assert summary.status == BUDGET_EXHAUSTED
    assert summary.simulations == model.calls == 10


def test_all_falsified_status_and_no_extra_simulation():
    model = ScriptedModel([10.0, 97.0])
    summary = _run(model, ["always (x < 90)", "always (x < 95)"], 50)

    assert summary.status == ALL_FALSIFIED
    # No simulation is spent on phi2 after both are falsified by the second trace
    assert summary.simulations == model.calls == 2


def test_simulations_equal_cost_evaluations():
    model = InputModel()
    summary = _run(model, ["always (x < 99)", "always (x < 99.5)", "always (x < 1000)"], 30)

    targeted = sum(r.target_sims for r in summary.records)
    assert summary.simulations == len(model.samples) == targeted


@pytest.mark.parametrize("formula", ["always (x < 99)", "always (x < 1000)"])
def test_single_property_matches_native_psy_taliro(formula):
    native_model = InputModel()
    runs = staliro(native_model, rtamt.parse_discrete(formula), DualAnnealing(min_cost=0.0), _options(200))
    native_costs = [e.cost for e in runs[0].evaluations]

    model = InputModel()
    summary = _run(model, [formula], 200)
    record = summary.records[0]

    assert model.samples == native_model.samples
    assert summary.simulations == len(native_costs)
    assert record.falsified == (min(native_costs) < 0.0)
    assert record.best_cost == min(native_costs)


def test_timeout_stops_search_and_drops_late_falsification():
    clock = FakeClock()
    # Simulations finish at t = 1, 2, ...; the sixth one would falsify phi2 but ends after 5.5 s
    model = ScriptedModel([10.0] * 5 + [97.0], clock=clock)
    summary = _run(model, ["always (x < 1000)", "always (x < 95)"], 100, timeout=5.5, clock=clock)
    phi1, phi2 = summary.records

    assert summary.status == TIMEOUT
    assert summary.simulations == model.calls == 6
    assert not phi2.falsified
    assert phi1.target_status == "timeout" and phi2.target_status == "timeout"


def test_zero_robustness_is_not_a_falsification():
    model = ScriptedModel([50.0])
    summary = _run(model, ["always (x < 50)"], 15)
    record = summary.records[0]

    assert record.best_cost == 0.0
    assert not record.falsified
    # DualAnnealing(min_cost=0.0) does not stop on zero robustness either
    assert record.target_sims == 15 and record.target_status == "budget"


def test_observer_sees_each_monitored_simulation_once():
    model = ScriptedModel([80.0, 97.0, 10.0])
    calls = []
    summary = _run(
        model,
        ["always (x < 90)", "always (x < 95)", "always (x < 1000)"],
        4,
        observer=lambda sim, target, trace, costs, new: calls.append((sim, target, sorted(costs), [r.name for r in new])),
    )
    assert [c[0] for c in calls] == list(range(1, summary.simulations + 1))
    # simulation 2 falsifies phi1 (target) and phi2 (reuse); afterwards only phi3 is monitored
    assert calls[0] == (1, "phi1", ["phi1", "phi2", "phi3"], [])
    assert calls[1] == (2, "phi1", ["phi1", "phi2", "phi3"], ["phi1", "phi2"])
    assert all(c[1] == "phi3" and c[2] == ["phi3"] and c[3] == [] for c in calls[2:])


def test_observer_not_called_for_late_simulation_after_timeout():
    clock = FakeClock()
    model = ScriptedModel([10.0] * 5 + [97.0], clock=clock)
    calls = []
    summary = _run(
        model,
        ["always (x < 1000)", "always (x < 95)"],
        100,
        timeout=5.5,
        clock=clock,
        observer=lambda sim, *_: calls.append(sim),
    )
    assert summary.simulations == 6 and calls == [1, 2, 3, 4, 5]

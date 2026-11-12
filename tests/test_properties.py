"""RTAMT properties on synthetic traces with known satisfaction and robustness."""

import numpy as np
import pytest

from apbench import Benchmark, BenchmarkConfig
from apbench.properties import (
    default_properties,
    evaluate,
    hyperglycemia,
    hypoglycemia,
    insulin_dose_window,
    insulin_suspension,
    prolonged_high,
    recovery,
    trace_signals,
)
from apbench.simulator import Trace


def synthetic(glucose, meal=None, rate_u_hour=None) -> Trace:
    g = np.asarray(glucose, dtype=float)
    n = len(g) - 1
    meal = np.zeros(n) if meal is None else np.asarray(meal, dtype=float)
    rate = np.zeros(n) if rate_u_hour is None else np.asarray(rate_u_hour, dtype=float) / 60.0
    return Trace(
        interval_minutes=30,
        meals=[],
        minute_time=np.arange(n + 1, dtype=float),
        glucose=g,
        cgm_sample=g.copy(),
        meal=meal,
        cho_intake=np.zeros(n),
        requested_basal=rate,
        delivered_rate=rate,
        ctrl_time=np.arange(0, n + 1, 3, dtype=float),
        ctrl_cgm=g[::3].copy(),
        ctrl_bg=g[::3].copy(),
        ctrl_done=np.zeros(len(g[::3]), dtype=bool),
    )


def rob(prop, trace):
    return prop.robustness(trace_signals(trace))


def test_hypoglycemia_boundary_and_sign():
    g = np.full(241, 120.0)
    g[100] = 54.0
    assert rob(hypoglycemia(54), synthetic(g)) == pytest.approx(0.0)  # boundary: satisfied
    g[100] = 53.5
    assert rob(hypoglycemia(54), synthetic(g)) == pytest.approx(-0.5)
    assert rob(hypoglycemia(70), synthetic(g)) == pytest.approx(-16.5)
    g[100] = 80.0
    assert rob(hypoglycemia(54), synthetic(g)) == pytest.approx(26.0)


def test_hypoglycemia_checks_last_sample():
    g = np.full(241, 120.0)
    g[-1] = 50.0
    assert rob(hypoglycemia(54), synthetic(g)) == pytest.approx(-4.0)


def test_hyperglycemia():
    g = np.full(241, 120.0)
    g[0] = 310.0
    assert rob(hyperglycemia(300), synthetic(g)) == pytest.approx(-10.0)


@pytest.mark.parametrize("start", [0, 50, 120])  # 120 = H - 120: the run ends at the last sample
def test_prolonged_high_window(start):
    H = 240
    p = prolonged_high(H, 250.0, 120)
    g = np.full(H + 1, 150.0)
    g[start : start + 120] = 260.0  # 120 samples: spans 119 minutes -> satisfied
    assert rob(p, synthetic(g)) > 0
    g[start : start + 121] = 260.0  # 121 samples: spans 120 minutes -> violated
    assert rob(p, synthetic(g)) == pytest.approx(-10.0)


def test_prolonged_high_robustness_is_best_dip_within_window():
    H = 240
    g = np.full(H + 1, 300.0)
    g[60] = 240.0
    g[180] = 245.0  # every 121-sample window contains a dip, the worst one is +5
    assert rob(prolonged_high(H), synthetic(g)) == pytest.approx(5.0)


def test_recovery():
    H, T = 300, 120
    p = recovery(H, 250.0, 180.0, T)
    g = np.full(H + 1, 150.0)
    g[10:40] = 260.0
    g[40:110] = 200.0  # back to <= 180 at minute 110, i.e. 100 minutes after the trigger at 39
    assert rob(p, synthetic(g)) > 0
    g[40:200] = 200.0  # not recovered within 120 minutes of the triggers
    assert rob(p, synthetic(g)) == pytest.approx(-10.0)
    meal = np.zeros(H)
    meal[100] = 40.0  # a meal in every trigger's window: the assumption fails, vacuously satisfied
    assert rob(p, synthetic(g, meal=meal)) >= 0
    g2 = np.full(H + 1, 150.0)
    g2[H - T + 1 :] = 300.0  # triggers only after H - T: windows incomplete, not checked
    assert rob(p, synthetic(g2)) >= 0
    g2[H - T] = 300.0  # trigger exactly at H - T with a complete window: checked
    assert rob(p, synthetic(g2)) < 0


def test_insulin_dose_observer_and_property():
    H = 180
    s = trace_signals(synthetic(np.full(H + 1, 120.0), rate_u_hour=np.full(H, 1.0)))
    assert s["insulin_dose_60"][0] == 0.0
    assert s["insulin_dose_60"][30] == pytest.approx(0.5)
    assert np.allclose(s["insulin_dose_60"][60:], 1.0)
    assert s["insulin_rate"][-1] == 0.0 and s["insulin_rate"][0] == pytest.approx(1.0)
    assert insulin_dose_window(0.9).robustness(s) == pytest.approx(-0.1)
    assert insulin_dose_window(2.0).robustness(s) == pytest.approx(1.0)


def test_insulin_suspension():
    H = 60
    g = np.full(H + 1, 120.0)
    rate = np.full(H, 1.0)
    p = insulin_suspension(70.0, 0.5)
    assert rob(p, synthetic(g, rate_u_hour=rate)) > 0  # glucose high: no constraint
    g[20] = 65.0
    assert rob(p, synthetic(g, rate_u_hour=rate)) == pytest.approx(-0.5)
    rate[20] = 0.0
    assert rob(p, synthetic(g, rate_u_hour=rate)) > 0


def test_suite_on_simulated_trace_matches_direct_computation():
    trace = Benchmark(BenchmarkConfig(controller="nominal-basal", horizon_steps=8)).simulate([80.0] + [0.0] * 7)
    props = default_properties(240)
    assert len({p.name for p in props}) == len(props)
    r = evaluate(props, trace)
    assert r["hypo_54"] == pytest.approx(trace.glucose.min() - 54.0)
    assert r["hyper_300"] == pytest.approx(300.0 - trace.glucose.max())
    dose = insulin_dose_window(2.5)  # not in the default suite, but still evaluable on a trace
    assert evaluate([dose], trace)[dose.name] == pytest.approx(2.5 - trace_signals(trace)["insulin_dose_60"].max())


def test_default_suite_matches_specification():
    """Exactly the six README default properties (H = 720), in the documented order."""
    expected = {
        "hypo_54": "always (glucose >= 54)",
        "hypo_70": "always (glucose >= 70)",
        "hyper_300": "always (glucose <= 300)",
        "prolonged_250_120min": "always[0:600] (eventually[0:120] (glucose <= 250))",
        "recovery_250_to_180_180min": "always[0:540] ((glucose > 250) -> ((always[0:180] (1e+06 * (1 - meal) >= 0)) -> "
        "(eventually[0:180] (glucose <= 180))))",
        "suspend_below_70": "always ((glucose < 70) -> (insulin_rate <= 1))",
    }
    props = default_properties(720)
    assert [p.name for p in props] == list(expected)
    for p in props:
        assert p.formula == expected[p.name]  # README writes the guard as 1e6*(1-meal)

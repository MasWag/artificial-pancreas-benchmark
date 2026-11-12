import math

import numpy as np
import pytest

from types import SimpleNamespace

from apbench import MAX_MEAL_GRAMS, OUTPUT_NAMES, Benchmark, BenchmarkConfig, SimulationError
from apbench.simulator import ControlContext, FastT1DPatient, summarize_interval
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient

MEALS = [80.0, 0.0, 40.0, 0.0, 80.0, 80.0] + [0.0] * 6


class ConstantController:
    def __init__(self, rate_u_min):
        self.rate = rate_u_min
        self.contexts = []

    def reset(self):
        self.contexts = []

    def act(self, ctx: ControlContext) -> float:
        self.contexts.append(ctx)
        return self.rate


def make(rate=0.014, **kw):
    ctrl = ConstantController(rate)
    return Benchmark(BenchmarkConfig(horizon_steps=len(MEALS), **kw), controller=ctrl), ctrl


# --- Pump units (Simglucose PR #99) -------------------------------------------------


def test_insulet_basal_limits_in_u_per_min():
    pump = InsulinPump.withName("Insulet")
    assert pump._params["max_basal"] == 30.0  # U/hour in the parameter file
    assert pump.max_basal == pytest.approx(0.5)  # U/min at the controller interface
    assert pump.min_basal == 0.0


@pytest.mark.parametrize(
    "request_u_hour, delivered_u_hour",
    [(1.02, 1.0), (1.03, 1.05), (0.8361, 0.85), (29.99, 30.0), (100.0, 30.0), (-1.0, 0.0), (0.024, 0.0), (0.026, 0.05)],
)
def test_basal_quantized_in_0_05_u_per_hour_steps_and_clipped(request_u_hour, delivered_u_hour):
    pump = InsulinPump.withName("Insulet")
    assert pump.basal(request_u_hour / 60.0) * 60.0 == pytest.approx(delivered_u_hour)


def test_oversized_request_saturates_at_30_u_per_hour_in_trace():
    bench, _ = make(rate=10.0)  # far above the 0.5 U/min pump limit
    trace = bench.simulate(MEALS[:2])
    assert np.all(trace.requested_basal == 10.0)
    assert np.allclose(trace.delivered_rate, 0.5)
    assert all(s.insulin_max_rate == pytest.approx(30.0) for s in trace.summaries)
    assert all(s.insulin_dose == pytest.approx(15.0) for s in trace.summaries)  # 0.5 U/min x 30 min


# --- Fast patient parameters ---------------------------------------------------------


def test_fast_patient_is_bit_identical_to_simglucose():
    from simglucose.patient.t1dpatient import Action

    a, b = FastT1DPatient.withName("adolescent#001"), T1DPatient.withName("adolescent#001")
    for minute in range(120):
        act = Action(CHO=80.0 if minute == 3 else 0.0, insulin=0.03)
        a.step(act)
        b.step(act)
        assert np.array_equal(a.state, b.state)


# --- Timing, meals and step accounting ------------------------------------------------


def test_step_accounting():
    bench, ctrl = make()
    bench.reset()
    bench.step(0.0)
    assert len(ctrl.contexts) == 10  # 30 min / 3 min control period
    assert [c.time_min for c in ctrl.contexts] == [3.0 * i for i in range(10)]
    trace = bench.trace()
    assert len(trace.delivered_rate) == 30  # one record per simulated minute
    assert len(trace.glucose) == 31  # samples at minutes 0..30
    assert trace.minute_time.tolist() == list(range(31))
    assert trace.ctrl_time.tolist() == [3.0 * i for i in range(11)]


def test_full_horizon_length():
    bench = Benchmark(BenchmarkConfig(controller="nominal-basal"))
    trace = bench.simulate([0.0] * 24)
    assert len(trace.summaries) == 24
    assert trace.duration_minutes == 720.0


def test_meal_injected_once_at_interval_start():
    bench, _ = make()
    trace = bench.simulate(MEALS)
    for k, meal in enumerate(MEALS):
        announced = trace.meal[30 * k : 30 * (k + 1)]
        assert announced[0] == meal
        assert np.all(announced[1:] == 0.0)
    # Simglucose eats at 5 g/min: an 80 g meal lasts exactly 16 minutes.
    intake = trace.cho_intake
    assert np.all(intake[0:16] == 5.0) and intake[16] == 0.0
    assert math.fsum(intake) == pytest.approx(sum(MEALS))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # Simglucose's meal model overflows harmlessly for 5e-324 g
@pytest.mark.parametrize("meal", [0.0, 5e-324, 1e-9, 37.25, 79.99999999999999, 80.0, np.float64(12.5), 40])
def test_continuous_meal_inputs_accepted(meal):
    bench, _ = make()
    bench.reset()
    bench.step(meal)
    assert bench.meals == [float(meal)]


@pytest.mark.parametrize("meal", [-1e-9, -0.0 - 5e-324, 80.0001, np.nextafter(80.0, 100.0), float("nan"), float("inf"), -float("inf")])
def test_meal_inputs_outside_range_rejected(meal):
    bench, _ = make()
    bench.reset()
    with pytest.raises(ValueError):
        bench.step(meal)
    assert MAX_MEAL_GRAMS == 80.0


def test_no_built_in_alphabet_or_cumulative_limit():
    assert "meal_alphabet" not in BenchmarkConfig().describe()
    bench, _ = make()
    trace = bench.simulate([80.0] * 6)  # 480 g in 3 hours: inadmissible, but simulated
    assert math.fsum(trace.cho_intake) == pytest.approx(480.0)


def test_continuous_meal_is_eaten_completely():
    bench, _ = make()
    trace = bench.simulate([37.25, 0.0, 12.5] + [0.0] * 3)
    assert trace.meal[0] == 37.25 and trace.meal[60] == 12.5
    assert np.all(trace.cho_intake[0:7] == 5.0) and trace.cho_intake[7] == pytest.approx(2.25)
    assert math.fsum(trace.cho_intake) == pytest.approx(49.75)


# --- Determinism ------------------------------------------------------------------------


def test_reset_replays_identically():
    bench = Benchmark(BenchmarkConfig(controller="nominal-basal", horizon_steps=12))
    t1 = bench.simulate(MEALS)
    bench.reset()
    bench.step(80.0)  # perturb the state
    t2 = bench.simulate(MEALS)
    t3 = Benchmark(BenchmarkConfig(controller="nominal-basal", horizon_steps=12)).simulate(MEALS)
    for t in (t2, t3):
        assert np.array_equal(t1.glucose, t.glucose)
        assert np.array_equal(t1.cgm_sample, t.cgm_sample)
        assert np.array_equal(t1.ctrl_cgm, t.ctrl_cgm)
        assert np.array_equal(t1.outputs(), t.outputs())


def test_sensor_seed_changes_noise_only():
    a = Benchmark(BenchmarkConfig(controller="nominal-basal", sensor_seed=1)).simulate(MEALS)
    b = Benchmark(BenchmarkConfig(controller="nominal-basal", sensor_seed=2)).simulate(MEALS)
    assert np.array_equal(a.glucose, b.glucose)  # constant controller: CGM does not feed back
    assert not np.array_equal(a.cgm_sample, b.cgm_sample)


# --- Outputs ------------------------------------------------------------------------------


@pytest.mark.parametrize("interval", [30, 15, 6])
def test_six_outputs_consistent_with_full_trace(interval):
    ctrl = ConstantController(0.03)
    bench = Benchmark(BenchmarkConfig(horizon_steps=len(MEALS), interval_minutes=interval), controller=ctrl)
    trace = bench.simulate(MEALS)
    out = trace.outputs()
    assert out.shape == (len(MEALS), 6)
    steps = interval // 3
    for k in range(len(MEALS)):
        a, b = interval * k, interval * (k + 1)
        g = trace.glucose[a + 1 : b + 1]  # right-closed interval (a, b]: samples a+1, ..., b
        assert len(g) == interval
        row = dict(zip(OUTPUT_NAMES, out[k]))
        assert row["glucose_end"] == trace.glucose[b]
        assert row["glucose_min"] == g.min()
        assert row["glucose_max"] == g.max()
        assert row["insulin_dose"] == pytest.approx(np.sum(trace.delivered_rate[a:b]) * 1.0)
        assert row["insulin_max_rate"] == pytest.approx(trace.delivered_rate[a:b].max() * 60)
        assert row["cgm_end"] == trace.ctrl_cgm[steps * (k + 1)]


def _summary(glucose, start, end):
    log = SimpleNamespace(minute_glucose=list(glucose), minute_delivered=[0.01] * (len(glucose) - 1))
    return summarize_interval(log, start, end, cgm_end=0.0)


@pytest.mark.parametrize("delta", [30, 15])
def test_interval_extrema_exclude_start_and_include_end(delta):
    g = np.full(3 * delta + 1, 120.0)
    g[delta] = 40.0  # minimum only at the boundary between intervals 0 and 1
    g[2 * delta] = 400.0  # maximum only at the boundary between intervals 1 and 2
    s0, s1, s2 = (_summary(g, k * delta, (k + 1) * delta) for k in range(3))
    # a boundary sample belongs to the interval it ends ...
    assert s0.glucose_min == 40.0 and s0.glucose_end == 40.0
    assert s1.glucose_max == 400.0 and s1.glucose_end == 400.0
    # ... and not to the interval it starts
    assert s1.glucose_min == 120.0
    assert s2.glucose_max == 120.0 and s2.glucose_min == 120.0


def test_initial_sample_not_in_any_interval_extremum():
    g = np.full(61, 100.0)
    g[0] = 30.0  # extremum only at t = 0
    g[1] = 101.0
    s0 = _summary(g, 0, 30)
    assert s0.glucose_min == 100.0 and s0.glucose_max == 101.0


def test_full_minute_trace_keeps_initial_sample():
    bench, _ = make()
    bench.reset()
    g0 = bench.sim.env.patient.observation.Gsub
    trace = bench.simulate(MEALS)
    assert len(trace.glucose) == 30 * len(MEALS) + 1 and trace.minute_time[0] == 0.0
    assert trace.glucose[0] == g0


def test_insulin_dose_is_rate_integral():
    bench, _ = make(rate=1.0 / 60.0)  # 1 U/hour
    trace = bench.simulate([0.0, 0.0])
    for s in trace.summaries:
        assert s.insulin_dose == pytest.approx(0.5)  # 1 U/hour for 30 minutes
        assert s.insulin_max_rate == pytest.approx(1.0)


def test_cgm_is_three_minute_average_of_held_sensor_samples():
    bench, ctrl = make()
    trace = bench.simulate(MEALS[:2])
    s = trace.cgm_sample
    for j in range(1, len(trace.ctrl_time)):
        t = int(trace.ctrl_time[j])
        # minutes t-2 and t-1 hold the sample of minute t-3; minute t draws a new sample
        assert s[t - 2] == s[t - 3] and s[t - 1] == s[t - 3]
        assert trace.ctrl_cgm[j] == pytest.approx((s[t - 2] + s[t - 1] + s[t]) / 3.0)
    # the controller sees the CGM observation, not physiological glucose
    assert [c.cgm for c in ctrl.contexts] == trace.ctrl_cgm[:-1].tolist()
    assert not np.allclose(trace.ctrl_cgm, trace.ctrl_bg)


def test_prev_delivered_rate_in_context():
    bench, ctrl = make(rate=0.02)
    bench.simulate([0.0])
    assert ctrl.contexts[0].prev_delivered_u_min == 0.0
    expected = InsulinPump.withName("Insulet").basal(0.02)
    assert all(c.prev_delivered_u_min == pytest.approx(expected) for c in ctrl.contexts[1:])


# --- Physiological "done" and numerical failures ------------------------------------------


def test_continues_after_done():
    bench = Benchmark(BenchmarkConfig(controller="nominal-basal"))
    trace = bench.simulate([80.0] * 24)
    assert trace.threshold_crossed  # BG > 600 mg/dL
    assert bench.sim.first_done_time is not None and bench.sim.first_done_time < 720
    assert trace.duration_minutes == 720 and np.all(np.isfinite(trace.glucose))


def test_nonfinite_controller_output_raises():
    bench, _ = make(rate=float("nan"))
    bench.reset()
    with pytest.raises(SimulationError):
        bench.step(0.0)


def test_nonfinite_state_raises(monkeypatch):
    bench, _ = make()
    bench.reset()
    bench.sim.env.patient._odesolver._y[3] = np.nan
    with pytest.raises(SimulationError):
        bench.step(0.0)

"""The bundled, trained RecurrentPPO and PPO checkpoints."""

import json

import numpy as np
import pytest

from apbench import Benchmark, BenchmarkConfig
from apbench.controllers import ppo, rppo

MEALS = [80.0, 0.0, 0.0, 40.0, 0.0, 0.0, 80.0, 80.0] + [0.0] * 16
CHECKPOINTS = {
    "rppo": (rppo.MODEL_PATH, rppo.RecurrentPPOController, "sb3_contrib.RecurrentPPO", "MlpLstmPolicy"),
    "ppo": (ppo.MODEL_PATH, ppo.PPOController, "stable_baselines3.PPO", "MlpPolicy"),
}


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_checkpoint_and_metadata_present(name):
    path, cls, algorithm, policy = CHECKPOINTS[name]
    assert path.exists()
    meta = json.loads(path.with_name("metadata.json").read_text())
    assert meta["algorithm"] == algorithm and meta["policy"] == policy
    assert "simglucose_revision" in meta["software"]
    assert meta["training"]["num_timesteps"] >= meta["training"]["timesteps"]
    assert cls(path).spec.max_rate_u_hour == meta["obs_act_spec"]["max_rate_u_hour"]


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_deterministic_rollout(name):
    t1 = Benchmark(BenchmarkConfig(controller=name)).simulate(MEALS)
    bench = Benchmark(BenchmarkConfig(controller=name))
    t2 = bench.simulate(MEALS)
    t3 = bench.simulate(MEALS)
    for t in (t2, t3):
        assert np.array_equal(t1.requested_basal, t.requested_basal)
        assert np.array_equal(t1.outputs(), t.outputs())


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_trained_controller_reacts_to_glucose(name):
    """Not a constant (or untrained) policy: more insulin after meals than when fasting."""
    bench = Benchmark(BenchmarkConfig(controller=name))
    fasting = bench.simulate([0.0] * 24).delivered_rate
    fed = bench.simulate([80.0, 80.0] + [0.0] * 22).delivered_rate
    assert fed[60:240].mean() > 1.2 * fasting[60:240].mean()


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_no_hypoglycemia_without_meals(name):
    trace = Benchmark(BenchmarkConfig(controller=name)).simulate([0.0] * 24)
    assert trace.glucose.min() >= 70.0


def test_rppo_hard_properties_have_known_counterexamples():
    """Unrestricted stress inputs falsify hypoglycemia and suspension.

    These meal sequences (4-8 x 80 g in a row) are *inadmissible* for the constrained
    reference workload; they check controller behavior and that the simulator still
    accepts sequences violating the cumulative intake limits.
    """
    from apbench.admissibility import is_admissible

    assert not is_admissible([80.0] * 5 + [0.0] * 19)
    from apbench.properties import default_properties, evaluate

    props = default_properties(720)
    bench = Benchmark(BenchmarkConfig(controller="rppo"))
    r5 = evaluate(props, bench.simulate([80.0] * 5 + [0.0] * 19))
    assert r5["hypo_70"] < 0 and r5["suspend_below_70"] < 0 and r5["hypo_54"] > 0
    r8 = evaluate(props, bench.simulate([80.0] * 8 + [0.0] * 16))
    assert r8["hypo_54"] < 0


def test_ppo_hard_properties_have_known_counterexamples():
    """Unrestricted (inadmissible) stress inputs, as for rppo."""
    from apbench.properties import default_properties, evaluate

    props = default_properties(720)
    bench = Benchmark(BenchmarkConfig(controller="ppo"))
    r4 = evaluate(props, bench.simulate([80.0] * 4 + [0.0] * 20))
    assert r4["hypo_70"] < 0 and r4["suspend_below_70"] < 0 and r4["hypo_54"] > 0
    r7 = evaluate(props, bench.simulate([80.0] * 7 + [0.0] * 17))
    assert r7["hypo_54"] < 0

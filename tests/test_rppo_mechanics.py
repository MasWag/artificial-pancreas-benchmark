"""Observation construction and recurrent-state handling (untrained in-memory model)."""

import numpy as np
import pytest

from apbench import Benchmark, BenchmarkConfig
from apbench.controllers.rppo import RecurrentPPOController
from apbench.observation import ObservationBuilder, ObsActSpec
from apbench.rl.env import SimglucoseEnv, make_episode_simulation
from apbench.simulator import ControlContext


def ctx(cgm, prev=0.0, t=0.0):
    return ControlContext(time_min=t, cgm=cgm, prev_delivered_u_min=prev, max_basal_u_min=0.5, sample_time_min=3.0)


def test_observation_builder():
    spec = ObsActSpec(trend_steps=2, max_rate_u_hour=3.0)
    b = ObservationBuilder(spec)
    assert b.build(ctx(140.0)).tolist() == [0.0, 0.0, 0.0]
    b.build(ctx(150.0))
    obs = b.build(ctx(170.0, prev=1.5 / 60))
    assert obs == pytest.approx([0.3, 1.0, 0.5])  # trend over 2 steps: (170 - 140) / 30
    b.reset()
    assert b.build(ctx(200.0))[1] == 0.0


def test_action_mapping():
    spec = ObsActSpec(max_rate_u_hour=3.0)
    assert spec.action_to_rate_u_min(-1.0) == 0.0
    assert spec.action_to_rate_u_min(0.0) * 60 == pytest.approx(1.5)
    assert spec.action_to_rate_u_min(5.0) * 60 == pytest.approx(3.0)  # clipped


def test_training_env_observations_equal_benchmark_construction():
    env = SimglucoseEnv(spec=ObsActSpec())
    obs, info = env.reset(seed=3)
    sim = make_episode_simulation("adolescent#001", info["episode_seed"])
    builder = ObservationBuilder(env.spec)
    assert np.array_equal(obs, builder.build(sim.reset()))
    for a in [0.3, -0.5, 1.0, -1.0, 0.0, 0.2, 0.7]:
        obs, *_ = env.step(np.array([a], dtype=np.float32))
        expected = builder.build(sim.control_step(env.spec.action_to_rate_u_min(a)))
        assert np.array_equal(obs, expected)


@pytest.fixture(scope="module")
def untrained_controller():
    import torch
    from sb3_contrib import RecurrentPPO

    torch.manual_seed(0)
    model = RecurrentPPO("MlpLstmPolicy", SimglucoseEnv(), seed=0, device="cpu", policy_kwargs=dict(lstm_hidden_size=8))
    with torch.no_grad():  # make the (otherwise near-zero) action head sensitive to the LSTM state
        model.policy.action_net.weight.mul_(100.0)
    return RecurrentPPOController(model=model, spec=ObsActSpec())


def test_hidden_state_persists_within_query_and_resets_between(untrained_controller):
    c = untrained_controller
    bench = Benchmark(BenchmarkConfig(horizon_steps=4), controller=c)
    meals = [80.0, 0.0, 40.0, 0.0]
    t1 = bench.simulate(meals)
    hidden = c.lstm_state
    assert hidden is not None and np.any(hidden[0] != 0)
    t2 = bench.simulate(meals)  # reset() clears the state: identical replay
    assert np.array_equal(t1.requested_basal, t2.requested_basal)
    assert np.array_equal(t1.glucose, t2.glucose)
    c.reset()
    assert c.lstm_state is None
    # Without reset, the controller would continue from a stale state and act differently.
    bench.reset()
    c.reset()
    first = c.act(bench._ctx)
    c.lstm_state, c._episode_start = hidden, False
    stale = c.act(bench._ctx)
    assert first != stale

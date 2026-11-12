"""Constrained random falsifier, trace replay and FalCAuN adapter."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from apbench import Benchmark, BenchmarkConfig
from apbench.admissibility import REFERENCE_CONSTRAINTS, InputConstraints, is_admissible, remaining_allowance
from apbench.falcaun import SUL, config_from_env
from apbench.random_falsify import P_ZERO, main, run, sample_sequence
from apbench.properties import default_properties, evaluate, hyperglycemia, hypoglycemia

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
CONFIG = BenchmarkConfig(controller="nominal-basal", horizon_steps=6)


def read_runs(path):
    with open(path / "runs.csv") as f:
        return list(csv.DictReader(f))


def row_meals(row):
    return [float(m) for m in row["meals"].split()]


# --- Constrained sampler ----------------------------------------------------------------


def test_sampled_sequences_are_admissible_and_continuous():
    rng = np.random.default_rng(0)
    seqs = [sample_sequence(rng, REFERENCE_CONSTRAINTS, 24) for _ in range(3000)]
    meals = np.array(seqs)
    assert all(is_admissible(s) for s in seqs)
    assert meals.min() >= 0.0 and meals.max() <= 80.0
    positive = meals[meals > 0]
    assert len(np.unique(positive)) == len(positive)  # continuous sizes, not an alphabet
    assert not np.isin(positive, [40.0, 80.0]).any()
    assert 0.45 < np.mean(meals == 0.0) < 0.75  # zero w.p. 1/2, plus forced zeros
    assert P_ZERO == 0.5


def test_sampler_stays_within_allowance():
    rng = np.random.default_rng(1)
    for _ in range(500):
        seq = sample_sequence(rng, REFERENCE_CONSTRAINTS, 24)
        for k, m in enumerate(seq):
            assert 0.0 <= m <= remaining_allowance(seq[:k])


class ScriptedRNG:
    """Always chooses a positive meal (0.75 >= 1/2) of the full allowance (1 - 0.0)."""

    def __init__(self):
        self.values = iter([0.75, 0.0] * 100)
        self.calls = 0

    def random(self):
        self.calls += 1
        return next(self.values)


def test_sampler_selects_zero_when_allowance_is_exhausted():
    rng = ScriptedRNG()
    seq = sample_sequence(rng, REFERENCE_CONSTRAINTS, 24)
    # 80 + 40 fills the 2-hour window; zeros are forced (without consuming randomness) until
    # the 80 g meal leaves the window; then 80 + 40 + 60 reaches the 300 g total.
    assert seq[:8] == [80.0, 40.0, 0.0, 0.0, 80.0, 40.0, 0.0, 0.0]
    assert seq[8] == 60.0 and sum(seq) == 300.0 and seq[9:] == [0.0] * 15
    assert rng.calls == 2 * 5 and is_admissible(seq)


def test_sampler_tight_constraints_hit_equality_without_exceeding():
    tight = InputConstraints(horizon_steps=24, max_meal=80.0, window_limit=1.0, total_limit=3.0)
    rng = np.random.default_rng(2)
    for _ in range(2000):
        assert is_admissible(sample_sequence(rng, tight, 24), tight)


def test_sampler_reproducible_and_seed_dependent():
    a = [sample_sequence(np.random.default_rng(5), REFERENCE_CONSTRAINTS, 24) for _ in range(2)]
    b = sample_sequence(np.random.default_rng(6), REFERENCE_CONSTRAINTS, 24)
    assert a[0] == a[1] and a[0] != b


def test_falsifier_rejects_workload_variants_without_constraints(tmp_path):
    with pytest.raises(ValueError):
        run(BenchmarkConfig(controller="nominal-basal", interval_minutes=15, horizon_steps=4), [], 1, 0, tmp_path)


def test_random_falsifier_reproducible_and_complete(tmp_path):
    props = default_properties(CONFIG.horizon_minutes)
    s1 = run(CONFIG, props, 4, seed=7, output_dir=tmp_path / "a")
    s2 = run(CONFIG, props, 4, seed=7, output_dir=tmp_path / "b")
    s3 = run(CONFIG, props, 4, seed=8, output_dir=tmp_path / "c")
    assert read_runs(tmp_path / "a") == read_runs(tmp_path / "b")
    assert [r["meals"] for r in read_runs(tmp_path / "a")] != [r["meals"] for r in read_runs(tmp_path / "c")]
    assert all(is_admissible(row_meals(r), prefix=True) for r in read_runs(tmp_path / "a"))
    for key in ("properties", "total_simulations", "total_simulated_minutes", "seed", "wall_clock_s", "config"):
        assert key in s1
    assert "meal_alphabet" not in s1["config"] and "constrained continuous" in s1["method"]
    assert s1["input_constraints"] == REFERENCE_CONSTRAINTS.describe()
    assert s1["sampler"]["p_zero"] == 0.5 and s1["prefix_of_reference_horizon"] is True
    assert s1["total_simulations"] == 4 and s1["total_simulated_minutes"] == 4 * 180
    assert [p["best_robustness"] for p in s1["properties"]] == [p["best_robustness"] for p in s2["properties"]]
    rows = read_runs(tmp_path / "a")
    for prop in s1["properties"]:
        robs = [float(r[prop["name"]]) for r in rows]
        assert prop["best_robustness"] == min(robs)
        assert prop["falsified"] == (min(robs) < 0)
        if prop["falsified"]:
            first = next(i + 1 for i, r in enumerate(robs) if r < 0)
            assert prop["first_falsification_simulation"] == first
            assert (tmp_path / "a" / "counterexamples" / f"{prop['name']}_minutes.csv").exists()
    assert json.loads((tmp_path / "a" / "summary.json").read_text())["seed"] == 7


def test_input_sampling_independent_of_simulator_seeds(tmp_path):
    props = [hypoglycemia(54)]
    run(CONFIG, props, 3, seed=1, output_dir=tmp_path / "a")
    other = BenchmarkConfig(controller="nominal-basal", horizon_steps=6, sensor_seed=99)
    run(other, props, 3, seed=1, output_dir=tmp_path / "b")
    assert [r["meals"] for r in read_runs(tmp_path / "a")] == [r["meals"] for r in read_runs(tmp_path / "b")]


def test_counterexamples_and_runs_replay_exactly(tmp_path):
    props = default_properties(CONFIG.horizon_minutes) + [hyperglycemia(160.0)]  # easy within 3 hours
    summary = run(CONFIG, props, 6, seed=0, output_dir=tmp_path)
    falsified = [p for p in summary["properties"] if p["falsified"]]
    assert falsified
    bench = Benchmark(CONFIG)
    for p in falsified:
        cex = json.loads((tmp_path / "counterexamples" / f"{p['name']}_meals.json").read_text())
        assert cex["admissible"] is True and is_admissible(cex["meals"], prefix=True)
        assert evaluate(props, bench.simulate(cex["meals"]))[cex["property"]] == cex["robustness"]
    for row in read_runs(tmp_path):  # runs.csv keeps exact meal values
        rob = evaluate(props, bench.simulate(row_meals(row)))
        assert all(rob[n] == float(row[n]) for n in rob)


DEFAULT_NAMES = [p.name for p in default_properties(720)]


def test_random_falsify_cli_evaluates_default_suite(tmp_path, capsys):
    main(["--trials", "2", "--seed", "3", "--controller", "nominal-basal", "--horizon", "6", "--output-dir", str(tmp_path)])
    assert "2 simulations" in capsys.readouterr().out
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["trials"] == 2
    assert [p["name"] for p in summary["properties"]] == DEFAULT_NAMES
    with open(tmp_path / "runs.csv") as f:
        assert next(csv.reader(f)) == ["trial", "meals", *DEFAULT_NAMES]


@pytest.mark.parametrize("controller", ["rppo", "ppo"])
def test_saved_baseline_consistent_with_default_suite(controller):
    d = EXAMPLES / f"random_constrained_{controller}_seed0"
    # These CSVs were measured with the original seven-property suite. Its summary.json
    # was removed because its wall-clock timings are not for the current six-property suite.
    assert not (d / "summary.json").exists()

    props = default_properties(720)
    with open(d / "runs.csv") as f:
        assert next(csv.reader(f)) == ["trial", "meals", *DEFAULT_NAMES]
    rows = read_runs(d)
    assert len(rows) == 1000 and all(len(r) == 2 + len(DEFAULT_NAMES) for r in rows)
    assert [int(r["trial"]) for r in rows] == list(range(1, 1001))
    assert all(is_admissible(row_meals(r)) for r in rows)

    expected_files = set()
    for prop in props:
        robs = [float(r[prop.name]) for r in rows]
        assert all(np.isfinite(rob) for rob in robs)
        first = next((i + 1 for i, rob in enumerate(robs) if rob < 0), None)
        if first is not None:
            expected_files |= {f"{prop.name}_{suffix}" for suffix in ("meals.json", "minutes.csv", "outputs.csv")}
            cex = json.loads((d / "counterexamples" / f"{prop.name}_meals.json").read_text())
            assert cex["property"] == prop.name and cex["trial"] == first
            assert cex["meals"] == row_meals(rows[first - 1]) and cex["admissible"]
            assert cex["robustness"] == robs[first - 1]
    assert {f.name for f in (d / "counterexamples").iterdir()} == expected_files


def test_saved_baseline_first_rows_replay():
    d = EXAMPLES / "random_constrained_rppo_seed0"
    bench = Benchmark(BenchmarkConfig(controller="rppo"))
    for row in read_runs(d)[:2]:
        rob = evaluate(default_properties(720), bench.simulate(row_meals(row)))
        assert {n: float(row[n]) for n in DEFAULT_NAMES} == rob


def test_falcaun_adapter_matches_direct_api():
    meals = [12.5, 0.0, 37.75, 0.0, 80.0, 1e-3]  # continuous-valued inputs
    sul = SUL(CONFIG)
    for _ in range(2):  # repeated queries are independent
        sul.pre()
        outs = [sul.step([m]) for m in meals]
        sul.post()
    sul.close()
    direct = Benchmark(CONFIG).simulate(meals).outputs()
    assert np.array_equal(np.array(outs), direct)
    assert all(type(x) is float for o in outs for x in o) and all(len(o) == 6 for o in outs)


def test_falcaun_adapter_env_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("APBENCH_CONTROLLER", "nominal-basal")
    monkeypatch.setenv("APBENCH_SENSOR_SEED", "5")
    monkeypatch.setenv("APBENCH_TRACE_DIR", str(tmp_path))
    assert config_from_env().controller == "nominal-basal" and config_from_env().sensor_seed == 5
    sul = SUL()
    sul.pre()
    sul.step([40.0])
    sul.post()
    assert (tmp_path / "trace000001_outputs.csv").exists()


@pytest.mark.parametrize("meal", [-0.5, 80.5, float("nan"), float("inf")])
def test_falcaun_adapter_rejects_inputs_outside_domain(meal):
    sul = SUL(CONFIG)
    sul.pre()
    with pytest.raises(ValueError):
        sul.step([meal])

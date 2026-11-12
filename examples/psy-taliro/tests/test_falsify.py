"""falsify.py with small budgets: parametrization, costs, campaigns, result files and command line."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("staliro")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apbench import Benchmark, BenchmarkConfig  # noqa: E402
from apbench.admissibility import is_admissible, remaining_allowance  # noqa: E402
from apbench.properties import default_properties, evaluate, hyperglycemia, hypoglycemia  # noqa: E402
from falsify import (  # noqa: E402
    VARIABLES,
    ApbenchModel,
    PropertySpecification,
    make_options,
    main,
    meals_from_unit,
    run_campaign,
)
from trace_reuse import ALL_FALSIFIED, BUDGET_EXHAUSTED, TIMEOUT  # noqa: E402

DEFAULTS = default_properties(720)
EASY = hyperglycemia(160.0)  # glucose starts near 149 mg/dL: falsified by most meal sequences
QUIET = lambda _: None  # noqa: E731


def inputs(x):
    return SimpleNamespace(static=dict(zip(VARIABLES, x)))


def read_runs(d):
    with open(d / "runs.csv") as f:
        return list(csv.DictReader(f))


def row_meals(row):
    return [float(m) for m in row["meals"].split()]


# --- Input parametrization ------------------------------------------------------------------


def test_parametrization_is_deterministic_admissible_and_continuous():
    rng = np.random.default_rng(0)
    for _ in range(10_000):
        x = rng.random(24)
        meals = meals_from_unit(x)
        assert meals == meals_from_unit(x.tolist())
        assert is_admissible(meals)
        for k, m in enumerate(meals):
            assert m == min(1.0, max(0.0, x[k])) * remaining_allowance(meals[:k])
    assert len({m for m in meals if m > 0}) > 10  # continuous sizes, no alphabet


def test_parametrization_extremes_reach_limits_exactly():
    full = meals_from_unit([1.0] * 24)
    assert full[:9] == [80.0, 40.0, 0.0, 0.0, 80.0, 40.0, 0.0, 0.0, 60.0] and sum(full) == 300.0
    assert is_admissible(full)
    assert meals_from_unit([0.0] * 24) == [0.0] * 24
    assert meals_from_unit([1.7] + [-0.3] * 23) == [80.0] + [0.0] * 23  # clipped to [0, 1]


@pytest.mark.parametrize("x", [[0.5] * 23, [0.5] * 25, [float("nan")] + [0.5] * 23, [float("inf")] * 24])
def test_parametrization_rejects_invalid_variables(x):
    with pytest.raises(ValueError):
        meals_from_unit(x)


def test_options_use_24_unit_static_inputs():
    opts = make_options(7, 3)
    assert list(opts.static_inputs) == list(VARIABLES) == sorted(VARIABLES)
    assert opts.iterations == 7 and opts.seed == 3 and (opts.tspan.start, opts.tspan.end) == (0.0, 720.0)


# --- Model and costs ----------------------------------------------------------------------------


def test_costs_equal_benchmark_rtamt_robustness():
    model = ApbenchModel("nominal-basal")
    rng = np.random.default_rng(1)
    for _ in range(3):
        x = rng.random(24)
        sim = model.simulate(inputs(x)).value
        direct = evaluate(DEFAULTS, Benchmark(BenchmarkConfig(controller="nominal-basal")).simulate(meals_from_unit(x)))
        for p in DEFAULTS:
            assert PropertySpecification(p).evaluate(sim).value == direct[p.name]
    assert model.simulations == 3 and model.simulated_minutes == 3 * 720


# --- Campaigns ----------------------------------------------------------------------------------


def test_single_property_falsification_and_replay(tmp_path):
    c = run_campaign([EASY], tmp_path, controller="nominal-basal", iterations=10, seed=1, log=QUIET)
    (rec,) = c.summary.records
    assert rec.falsified and rec.via == "target" and rec.target_status == "falsified"
    assert rec.falsifying_cost < 0 and rec.best_cost <= rec.falsifying_cost
    assert c.summary.status == ALL_FALSIFIED
    rows = read_runs(tmp_path)
    # one row per SUT simulation, no extra simulations for the artifacts. SciPy's dual annealing
    # checks the stopping condition in its callback, so a few samples may follow the falsifying one.
    assert len(rows) == c.summary.simulations == c.model.simulations == rec.target_sims >= rec.falsified_at_sim
    assert all(r["target"] == EASY.name and is_admissible(row_meals(r)) for r in rows)
    assert min(float(r[EASY.name]) for r in rows) == rec.best_cost
    first = next(i for i, r in enumerate(rows) if float(r[EASY.name]) < 0)
    assert first + 1 == rec.falsified_at_sim and float(rows[first][EASY.name]) == rec.falsifying_cost
    cex = json.loads((tmp_path / "counterexamples" / f"{EASY.name}_meals.json").read_text())
    assert cex["admissible"] and cex["simulation"] == rec.falsified_at_sim and cex["via"] == "target"
    assert cex["meals"] == row_meals(rows[first])
    replay = Benchmark(BenchmarkConfig(controller="nominal-basal")).simulate(cex["meals"])
    assert evaluate([EASY], replay)[EASY.name] == cex["robustness"] == rec.falsifying_cost
    assert (tmp_path / "counterexamples" / f"{EASY.name}_minutes.csv").exists()


def test_multi_property_trace_reuse_and_skipping(tmp_path):
    hard = hypoglycemia(54.0)
    c = run_campaign([hard, EASY], tmp_path, controller="nominal-basal", iterations=4, seed=1, log=QUIET)
    rec_hard, rec_easy = c.summary.records
    assert not rec_hard.falsified and rec_hard.target_status == "budget" and rec_hard.target_sims == 4
    assert rec_easy.falsified and rec_easy.via == "reuse" and rec_easy.target_status == "skipped"
    assert rec_easy.target_sims == 0
    assert c.summary.simulations == c.model.simulations == 4 and c.summary.status == BUDGET_EXHAUSTED
    rows = read_runs(tmp_path)
    assert [int(r["simulation"]) for r in rows] == [1, 2, 3, 4] and {r["target"] for r in rows} == {hard.name}
    k = rec_easy.falsified_at_sim
    # the easy property is monitored up to (and including) its falsifying trace, then skipped
    assert all(r[EASY.name] != "" for r in rows[:k]) and all(r[EASY.name] == "" for r in rows[k:])
    assert float(rows[k - 1][EASY.name]) == rec_easy.falsifying_cost < 0
    assert all(r[hard.name] != "" for r in rows)


def test_timeout_before_first_simulation(tmp_path):
    c = run_campaign([EASY], tmp_path, controller="nominal-basal", iterations=10, seed=1, timeout=0.0, log=QUIET)
    assert c.summary.status == TIMEOUT and c.summary.simulations == 0
    assert c.summary.records[0].target_status == "timeout" and not c.summary.records[0].falsified


def test_campaign_reproducible_with_fixed_seed(tmp_path):
    props = [hypoglycemia(70.0)]
    for name in ("a", "b"):
        run_campaign(props, tmp_path / name, controller="nominal-basal", iterations=3, seed=5, log=QUIET)
    run_campaign(props, tmp_path / "c", controller="nominal-basal", iterations=3, seed=6, log=QUIET)
    a, b, c = ((tmp_path / n / "runs.csv").read_bytes() for n in "abc")
    assert a == b and a != c


@pytest.mark.parametrize("controller", ["rppo", "ppo"])
def test_neural_controllers(tmp_path, controller):
    c = run_campaign([EASY], tmp_path, controller=controller, iterations=2, seed=1, log=QUIET)
    assert c.model.config.controller == controller
    rows = read_runs(tmp_path)
    assert len(rows) == c.summary.simulations <= 2
    trace = Benchmark(BenchmarkConfig(controller=controller)).simulate(row_meals(rows[0]))
    assert evaluate([EASY], trace)[EASY.name] == float(rows[0][EASY.name])


# --- CLI ----------------------------------------------------------------------------------------


def check_summary_against_runs(summary, d):
    rows = read_runs(d)
    assert len(rows) == summary["total_simulations"] and summary["total_simulated_minutes"] == 720 * len(rows)
    assert all(is_admissible(row_meals(r)) for r in rows)
    for p in summary["properties"]:
        robs = [float(r[p["name"]]) for r in rows if r[p["name"]] != ""]
        assert p["best_robustness"] == (min(robs) if robs else None)
        first = next((int(r["simulation"]) for r in rows if r[p["name"]] != "" and float(r[p["name"]]) < 0), None)
        assert p["first_falsification_simulation"] == first and p["falsified"] == (first is not None)
        assert p["target_simulations"] == sum(r["target"] == p["name"] for r in rows)
        assert (p["counterexample"] is not None) == p["falsified"]
    return rows


def test_cli_single_property(tmp_path, capsys):
    s = main(["--properties", "hypo_54", "--controller", "nominal-basal",
              "--iterations", "3", "--seed", "2", "--output-dir", str(tmp_path), "--quiet"])
    assert "simulations" in capsys.readouterr().out
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved == json.loads(json.dumps(s))
    assert "mode" not in s and s["selected_properties"] == ["hypo_54"]
    assert s["optimizer"] == {"name": "DualAnnealing", "min_cost": 0.0, "iterations_per_target": 3, "seed": 2}
    assert s["config"]["controller"] == "nominal-basal" and s["psy_taliro_version"] == "3.2.0"
    assert [p["formula"] for p in s["properties"]] == [p.formula for p in DEFAULTS if p.name == "hypo_54"]
    rows = check_summary_against_runs(s, tmp_path)
    assert len(rows) == 3 and s["status"] == BUDGET_EXHAUSTED


def test_cli_property_subset(tmp_path):
    s = main(["--properties", "hyper_300", "hypo_70", "--controller", "nominal-basal",
              "--iterations", "2", "--seed", "1", "--output-dir", str(tmp_path), "--quiet"])
    assert s["selected_properties"] == ["hyper_300", "hypo_70"]
    rows = check_summary_against_runs(s, tmp_path)
    assert len(rows) == s["total_simulations"] == 4  # 2 evaluations per targeted property
    assert list(rows[0])[3:] == ["hyper_300", "hypo_70"]


def test_cli_default_selection_is_the_default_suite(tmp_path, monkeypatch):
    import falsify

    seen = {}
    monkeypatch.setattr(falsify, "run_campaign", lambda props, *a, **k: seen.setdefault("p", props) and 1 / 0)
    with pytest.raises(ZeroDivisionError):
        main(["--output-dir", str(tmp_path)])
    assert [p.name for p in seen["p"]] == [p.name for p in DEFAULTS]


@pytest.mark.parametrize(
    "argv",
    [
        ["--properties"],
        ["--properties", "insulin_dose_60min_2.5U"],
        ["--properties", "hypo_70", "hypo_70"],
        ["--properties", "hypo_70", "--seed", "0"],
        ["--mode", "one", "--properties", "hypo_70"],  # removed option
    ],
)
def test_cli_rejects_invalid_selection(tmp_path, argv):
    with pytest.raises(SystemExit):
        main([*argv, "--output-dir", str(tmp_path)])

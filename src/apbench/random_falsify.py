"""Constrained continuous random-sampling falsification baseline with RTAMT monitoring.

Each trial independently samples a complete *admissible* meal sequence
(:mod:`apbench.admissibility`), resets the system under test, simulates the whole
sequence once, and evaluates *all* properties on that trace with RTAMT. There is
no optimization or feedback between trials.

Sampler (README, "Random-sampling baseline"): at interval k, let ``u_k`` be the
remaining allowance ``min(80, 120 - sum(m[max(0,k-3):k]), 300 - sum(m[:k]))``.
If ``u_k == 0`` the meal is 0; otherwise it is 0 with probability 1/2 and uniform
on ``(0, u_k]`` with probability 1/2. ``u_k`` is computed exactly and rounded down,
so every sampled sequence is admissible. The distribution is not uniform over
admissible sequences. The input sampling seed (``--seed``) is independent of
Simglucose's patient and sensor seeds.

Usage::

    apbench-random-falsify --trials 1000 --seed 0 --output-dir results/random-seed0

Outputs (in ``--output-dir``):

* ``runs.csv``: one row per trial: meal sequence (shortest exact float repr, so rows
  replay exactly) and robustness of every property.
* ``summary.json``: configuration, sampler and input constraints, versions, budget, and per property the best
  (minimum) robustness and its input, whether it was falsified and at which
  simulation it was first falsified.
* ``counterexamples/<property>_{meals.json,minutes.csv,outputs.csv}``: the first
  falsifying input of each property and its full trace (re-simulated
  deterministically by ``apbench-simulate --meals ...``).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from apbench import SIMGLUCOSE_REVISION, Benchmark, BenchmarkConfig
from apbench.admissibility import InputConstraints, is_admissible, reference_constraints, remaining_allowance
from apbench.properties import Property, default_properties, evaluate

#: Probability of choosing no meal when a positive meal is allowed.
P_ZERO = 0.5


def sample_sequence(rng: np.random.Generator, constraints: InputConstraints, length: int) -> list[float]:
    """Sample one admissible meal sequence of ``length`` intervals."""
    meals: list[float] = []
    for _ in range(length):
        upper = remaining_allowance(meals, constraints)
        if upper > 0.0 and rng.random() >= P_ZERO:
            meal = upper * (1.0 - rng.random())  # uniform on (0, upper]; never exceeds the float upper
        else:
            meal = 0.0
        meals.append(float(meal))
    if not is_admissible(meals, constraints, prefix=True):  # guaranteed by construction
        raise AssertionError(f"sampled an inadmissible sequence {meals}")
    return meals


def run(
    config: BenchmarkConfig,
    properties: list[Property],
    trials: int,
    seed: int,
    output_dir: Path,
    log_every: int = 0,
    constraints: InputConstraints | None = None,
) -> dict:
    """Run ``trials`` independent simulations; ``constraints`` defaults to the reference workload."""
    if constraints is None:
        constraints = reference_constraints(config)
    if constraints.interval_minutes != config.interval_minutes or config.horizon_steps > constraints.horizon_steps:
        raise ValueError("the input constraints do not match the benchmark's interval duration and horizon")
    output_dir = Path(output_dir)
    (output_dir / "counterexamples").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    bench = Benchmark(config)
    names = [p.name for p in properties]
    best = {n: {"robustness": float("inf"), "trial": None, "meals": None} for n in names}
    first = {n: None for n in names}

    tic = time.perf_counter()
    sim_seconds = 0.0
    simulated_minutes = 0.0
    with open(output_dir / "runs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "meals", *names])
        for trial in range(1, trials + 1):
            meals = sample_sequence(rng, constraints, config.horizon_steps)
            t0 = time.perf_counter()
            trace = bench.simulate(meals)
            sim_seconds += time.perf_counter() - t0
            simulated_minutes += trace.duration_minutes
            rob = evaluate(properties, trace)
            writer.writerow([trial, " ".join(repr(m) for m in meals), *(repr(rob[n]) for n in names)])
            for n in names:
                if rob[n] < best[n]["robustness"]:
                    best[n] = {"robustness": rob[n], "trial": trial, "meals": meals}
                if rob[n] < 0 and first[n] is None:
                    first[n] = trial
                    prefix = output_dir / "counterexamples" / n
                    trace.save(prefix)
                    with open(f"{prefix}_meals.json", "w") as g:
                        json.dump(
                            {
                                "property": n,
                                "trial": trial,
                                "robustness": rob[n],
                                "admissible": is_admissible(meals, constraints, prefix=True),
                                "meals": meals,
                            },
                            g,
                            indent=1,
                        )
            if log_every and trial % log_every == 0:
                falsified = sum(v is not None for v in first.values())
                print(f"trial {trial}/{trials}: {falsified}/{len(names)} falsified", flush=True)

    wall = time.perf_counter() - tic
    summary = {
        "method": "constrained continuous random sampling of complete admissible meal sequences",
        "sampler": {
            "p_zero": P_ZERO,
            "positive_meal": "uniform on (0, u_k], u_k = remaining admissible allowance (exact, rounded down)",
            "zero_when_allowance_is_zero": True,
        },
        "input_constraints": constraints.describe(),
        "prefix_of_reference_horizon": config.horizon_steps < constraints.horizon_steps,
        "seed": seed,
        "trials": trials,
        "total_simulations": trials,
        "total_simulated_minutes": simulated_minutes,
        "wall_clock_s": wall,
        "simulation_wall_clock_s": sim_seconds,
        "config": config.describe(),
        "simglucose_revision": SIMGLUCOSE_REVISION,
        "properties": [
            {
                "name": p.name,
                "formula": p.formula,
                "description": p.description,
                "best_robustness": best[p.name]["robustness"],
                "best_trial": best[p.name]["trial"],
                "best_meals": best[p.name]["meals"],
                "falsified": first[p.name] is not None,
                "first_falsification_simulation": first[p.name],
            }
            for p in properties
        ],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--trials", type=int, default=100, help="simulation budget (number of input sequences)")
    p.add_argument("--seed", type=int, default=0, help="seed of the input sampler (independent of patient/sensor seeds)")
    p.add_argument("--patient", default=BenchmarkConfig.patient)
    p.add_argument("--controller", default=BenchmarkConfig.controller, help="rppo, ppo, nominal-basal")
    p.add_argument("--sensor-seed", type=int, default=BenchmarkConfig.sensor_seed)
    p.add_argument("--horizon", type=int, default=BenchmarkConfig.horizon_steps, help="number of 30-minute inputs (at most 24: the reference workload or a prefix of it)")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)

    config = BenchmarkConfig(
        patient=args.patient, controller=args.controller, sensor_seed=args.sensor_seed, horizon_steps=args.horizon
    )
    summary = run(
        config,
        default_properties(config.horizon_minutes),
        args.trials,
        args.seed,
        args.output_dir,
        log_every=max(1, args.trials // 10),
    )
    print(f"{'property':36s} {'best robustness':>16s} falsified first-sim")
    for prop in summary["properties"]:
        print(
            f"{prop['name']:36s} {prop['best_robustness']:16.3f} {str(prop['falsified']):9s} "
            f"{prop['first_falsification_simulation'] or '-'}"
        )
    print(
        f"{summary['total_simulations']} simulations, {summary['total_simulated_minutes'] / 60:.0f} simulated hours, "
        f"{summary['wall_clock_s']:.1f} s wall clock"
    )


if __name__ == "__main__":
    main()

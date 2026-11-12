"""Simulate one meal sequence and write its trace.

Usage::

    apbench-simulate --meals 80 0 40 0 --output results/example

writes ``results/example_minutes.csv`` (1-minute trace), ``results/example_outputs.csv``
(six outputs per 30-minute interval) and prints property robustness values and
whether the sequence is admissible for the reference workload (inadmissible
sequences are simulated too). A ``*_meals.json`` file written by
``apbench-random-falsify`` can be replayed with ``--meals-json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apbench import Benchmark, BenchmarkConfig
from apbench.admissibility import reference_constraints, violations
from apbench.properties import default_properties, evaluate


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--meals", type=float, nargs="+", help="meal per 30-minute interval, each a real amount in [0, 80] g")
    src.add_argument("--meals-json", type=Path, help="JSON file with a 'meals' list")
    p.add_argument("--patient", default=BenchmarkConfig.patient)
    p.add_argument("--controller", default=BenchmarkConfig.controller)
    p.add_argument("--sensor-seed", type=int, default=BenchmarkConfig.sensor_seed)
    p.add_argument("--output", type=Path, default=None, help="output path prefix")
    args = p.parse_args(argv)

    meals = args.meals if args.meals is not None else json.loads(args.meals_json.read_text())["meals"]
    config = BenchmarkConfig(
        patient=args.patient, controller=args.controller, sensor_seed=args.sensor_seed, horizon_steps=len(meals)
    )
    trace = Benchmark(config).simulate(meals)
    print(trace.output_frame().to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if trace.threshold_crossed:
        print("note: Simglucose's done threshold (BG < 10 or > 600 mg/dL) was crossed; simulation continued")
    try:
        reasons = violations(meals, reference_constraints(config), prefix=True)
        print("admissible (reference workload): " + ("yes" if not reasons else "no; " + "; ".join(reasons)))
    except ValueError:
        print("admissible (reference workload): n/a (workload variant)")
    for name, rob in evaluate(default_properties(config.horizon_minutes), trace).items():
        print(f"{name:36s} robustness {rob:12.3f} {'VIOLATED' if rob < 0 else 'satisfied'}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        trace.save(args.output)


if __name__ == "__main__":
    main()

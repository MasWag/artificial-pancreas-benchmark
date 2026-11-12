"""Falsify the artificial pancreas benchmark with Ψ-TaLiRo (DualAnnealing).

The file is organized in reading order:

* Ψ-TaLiRo components: 24 optimization variables x_k in [0, 1] are mapped
  sequentially to admissible meals m_k = x_k * u_k, where u_k is the remaining
  allowance of the reference workload (``apbench.admissibility.remaining_allowance``);
  the model runs one deterministic ``apbench.Benchmark`` simulation per sample; the
  cost of a property is exactly its ``Property.robustness`` (RTAMT) on that trace.
* Campaign: ``trace_reuse.run_multi_property`` targets the selected properties in
  turn and reuses every trace for all of them; ``run_campaign`` adds the result
  files ``runs.csv`` and the counterexamples.
* Command line: property selection, ``summary.json`` and the printed table.

All default properties::

    python examples/psy-taliro/falsify.py --controller rppo \\
        --iterations 1500 --seed 1 --output-dir results/psy-taliro-all

An ordered subset with ``--properties``; a single property is a one-property campaign::

    python examples/psy-taliro/falsify.py --properties hyper_300 \\
        --controller rppo --iterations 1500 --seed 1 --output-dir results/psy-taliro-hyper

``--iterations`` is the budget of *each* targeted search, so a campaign with
several properties may use more simulations in total; the actual number of unique
SUT simulations is reported as ``total_simulations``. See README.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from staliro import Result, TestOptions  # noqa: E402
from staliro.models import Model  # noqa: E402
from staliro.specifications import Specification  # noqa: E402

from apbench import SIMGLUCOSE_REVISION, Benchmark, BenchmarkConfig, Trace  # noqa: E402
from apbench.admissibility import REFERENCE_CONSTRAINTS, InputConstraints, is_admissible, remaining_allowance  # noqa: E402
from apbench.properties import Property, default_properties, trace_signals  # noqa: E402
from trace_reuse import SpecRecord, Summary, run_multi_property  # noqa: E402


# --- Ψ-TaLiRo components ---------------------------------------------------------------------------

#: Names of the static optimization variables, in input order (Ψ-TaLiRo keeps this order).
VARIABLES = tuple(f"x{k:02d}" for k in range(REFERENCE_CONSTRAINTS.horizon_steps))

PARAMETRIZATION = (
    "24 static variables x_k in [0, 1] (clipped to [0, 1]); meals are built sequentially as "
    "m_k = x_k * u_k with u_k = min(80, 120 - sum(m[max(0,k-3):k]), 300 - sum(m[:k])) computed "
    "exactly and rounded down (apbench.admissibility.remaining_allowance); every sequence is "
    "admissible. This changes the search geometry and is not the random baseline's input distribution."
)


def meals_from_unit(x: Sequence[float], constraints: InputConstraints = REFERENCE_CONSTRAINTS) -> list[float]:
    """Map optimization variables in [0, 1] to an admissible meal sequence (grams)."""
    values = [float(v) for v in x]
    if len(values) != constraints.horizon_steps:
        raise ValueError(f"expected {constraints.horizon_steps} variables, got {len(values)}")
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"non-finite optimization variable in {values}")
    meals: list[float] = []
    for v in values:
        meals.append(min(1.0, max(0.0, v)) * remaining_allowance(meals, constraints))
    if not is_admissible(meals, constraints):
        raise AssertionError(f"parametrization produced an inadmissible sequence {meals}")
    return meals


def make_options(iterations: int, seed: int) -> TestOptions:
    """Ψ-TaLiRo options: ``iterations`` evaluations per targeted property; ``seed`` must be > 0."""
    c = REFERENCE_CONSTRAINTS
    return TestOptions(
        tspan=(0.0, float(c.interval_minutes * c.horizon_steps)),
        iterations=iterations,
        seed=seed,
        static_inputs={name: (0.0, 1.0) for name in VARIABLES},
    )


@dataclass(frozen=True)
class SimulatedTrace:
    """One SUT simulation: its admissible meals, the full trace and its RTAMT signals."""

    meals: list[float]
    trace: Trace
    signals: dict[str, Any]


class ApbenchModel(Model[Any, None]):
    """Ψ-TaLiRo model running the benchmark for the reference workload."""

    def __init__(self, controller: str = "rppo"):
        self.config = BenchmarkConfig(controller=controller)
        self.benchmark = Benchmark(self.config)
        self.simulations = 0
        self.simulated_minutes = 0.0

    def simulate(self, inputs) -> Result[SimulatedTrace, None]:
        meals = meals_from_unit([inputs.static[name] for name in VARIABLES])
        trace = self.benchmark.simulate(meals)
        self.simulations += 1
        self.simulated_minutes += trace.duration_minutes
        return Result(SimulatedTrace(meals, trace, trace_signals(trace)), None)


class PropertySpecification(Specification[SimulatedTrace, float, None]):
    """A benchmark property as a Ψ-TaLiRo specification: cost = the benchmark's RTAMT robustness."""

    def __init__(self, prop: Property):
        self.prop = prop

    def evaluate(self, sim: SimulatedTrace) -> Result[float, None]:
        return Result(float(self.prop.robustness(sim.signals)), None)


# --- Campaign and result files ---------------------------------------------------------------------

class ResultWriter:
    """Observer writing ``runs.csv`` and the first counterexample of each property."""

    def __init__(self, output_dir: Path, names: list[str]):
        self.dir = Path(output_dir)
        (self.dir / "counterexamples").mkdir(parents=True, exist_ok=True)
        self.names = names
        self.counterexamples: dict[str, str] = {}
        self.file = open(self.dir / "runs.csv", "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["simulation", "target", "meals", *names])

    def __call__(
        self, simulation: int, target: str, sim: SimulatedTrace, costs: dict[str, float], new: list[SpecRecord]
    ) -> None:
        self.writer.writerow(
            [
                simulation,
                target,
                " ".join(repr(m) for m in sim.meals),
                *(repr(costs[n]) if n in costs else "" for n in self.names),
            ]
        )
        self.file.flush()
        for record in new:
            prefix = self.dir / "counterexamples" / record.name
            sim.trace.save(prefix)  # the already simulated trace: no extra SUT simulation
            with open(f"{prefix}_meals.json", "w") as f:
                json.dump(
                    {
                        "property": record.name,
                        "simulation": simulation,
                        "robustness": record.falsifying_cost,
                        "via": record.via,
                        "admissible": is_admissible(sim.meals),
                        "meals": sim.meals,
                    },
                    f,
                    indent=1,
                )
            self.counterexamples[record.name] = f"counterexamples/{record.name}_meals.json"

    def close(self) -> None:
        self.file.close()


@dataclass
class Campaign:
    summary: Summary
    model: ApbenchModel
    counterexamples: dict[str, str]


def run_campaign(
    properties: Sequence[Property],
    output_dir: Path,
    *,
    controller: str = "rppo",
    iterations: int = 1500,
    seed: int = 1,
    timeout: float | None = None,
    log: Callable[[str], None] = print,
) -> Campaign:
    """Falsify ``properties`` in order with DualAnnealing, reusing every trace for all pending ones.

    ``iterations`` is the evaluation budget of *each* targeted search; the campaign may therefore use
    more simulations in total. A single property is a campaign with one property.
    """
    model = ApbenchModel(controller)
    names = [p.name for p in properties]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate property names in {names}")
    writer = ResultWriter(output_dir, names)
    try:
        summary = run_multi_property(
            model,
            [(p.name, p.formula, PropertySpecification(p)) for p in properties],
            lambda: make_options(iterations, seed),
            timeout=timeout,
            log=log,
            observer=writer,
        )
    finally:
        writer.close()
    return Campaign(summary, model, writer.counterexamples)


# --- Command line ----------------------------------------------------------------------------------

def select_properties(props: list[str] | None) -> list[Property]:
    """The default properties named in ``props`` (in that order), or all of them if ``props`` is None."""
    defaults = {p.name: p for p in default_properties(REFERENCE_CONSTRAINTS.interval_minutes * REFERENCE_CONSTRAINTS.horizon_steps)}
    names = list(defaults) if props is None else props
    unknown = [n for n in names if n not in defaults]
    if unknown or len(set(names)) != len(names) or not names:
        raise ValueError(f"select distinct default properties {list(defaults)}; got {names}")
    return [defaults[n] for n in names]


def summarize(campaign: Campaign, args, properties: list[Property], wall: float) -> dict:
    s, model = campaign.summary, campaign.model
    return {
        "method": "Psy-TaLiRo DualAnnealing falsification with trace reuse across selected properties",
        "selected_properties": [p.name for p in properties],
        "config": model.config.describe(),
        "simglucose_revision": SIMGLUCOSE_REVISION,
        "psy_taliro_version": version("psy-taliro"),
        "optimizer": {"name": "DualAnnealing", "min_cost": 0.0, "iterations_per_target": args.iterations, "seed": args.seed},
        "timeout_s": args.timeout,
        "input_constraints": REFERENCE_CONSTRAINTS.describe(),
        "parametrization": PARAMETRIZATION,
        "total_simulations": s.simulations,
        "total_simulated_minutes": model.simulated_minutes,
        "campaign_elapsed_s": s.elapsed,
        "wall_clock_s": wall,
        "status": s.status,
        "properties": [
            {
                "name": r.name,
                "formula": r.formula,
                "falsified": r.falsified,
                "best_robustness": r.best_cost if math.isfinite(r.best_cost) else None,  # None: never monitored
                "first_falsification_simulation": r.falsified_at_sim,
                "first_falsification_time_s": r.falsified_at_time,
                "falsifying_robustness": r.falsifying_cost,
                "via": r.via,
                "target_simulations": r.target_sims,
                "target_elapsed_s": r.target_elapsed,
                "target_status": r.target_status,
                "counterexample": campaign.counterexamples.get(r.name),
            }
            for r in s.records
        ],
    }


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--properties", nargs="+", help="default properties to falsify, in order (default: all)")
    p.add_argument("--controller", default="rppo", choices=["rppo", "ppo", "nominal-basal"])
    p.add_argument("--iterations", type=int, default=1500, help="optimizer evaluations per targeted property (default 1500)")
    p.add_argument("--seed", type=int, default=1, help="Psy-TaLiRo seed (> 0), used for every targeted search")
    p.add_argument("--timeout", type=float, default=None, help="optional campaign-wide timeout in seconds")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    if args.seed <= 0:
        p.error("Psy-TaLiRo requires --seed > 0")
    if args.iterations <= 0:
        p.error("--iterations must be positive")
    try:
        properties = select_properties(args.properties)
    except ValueError as e:
        p.error(str(e))

    log = (lambda _: None) if args.quiet else (lambda m: print(m, flush=True))
    tic = time.perf_counter()
    campaign = run_campaign(
        properties,
        args.output_dir,
        controller=args.controller,
        iterations=args.iterations,
        seed=args.seed,
        timeout=args.timeout,
        log=log,
    )
    summary = summarize(campaign, args, properties, time.perf_counter() - tic)
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)

    print(f"{'property':30s} {'best robustness':>16s} falsified  first-sim  via     target-sims status")
    for r in summary["properties"]:
        print(
            f"{r['name']:30s} {'-' if r['best_robustness'] is None else format(r['best_robustness'], '.3f'):>16s} "
            f"{str(r['falsified']):10s} "
            f"{str(r['first_falsification_simulation'] or '-'):10s} {str(r['via'] or '-'):7s} "
            f"{r['target_simulations']:11d} {r['target_status']}"
        )
    print(
        f"status {summary['status']}: {summary['total_simulations']} simulations, "
        f"{summary['total_simulated_minutes'] / 60:.0f} simulated hours, {summary['wall_clock_s']:.1f} s wall clock"
    )
    return summary


if __name__ == "__main__":
    main()

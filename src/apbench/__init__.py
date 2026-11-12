"""Artificial pancreas falsification benchmark built on Simglucose."""

from apbench.simulator import (
    MAX_MEAL_GRAMS,
    OUTPUT_NAMES,
    OUTPUT_UNITS,
    Benchmark,
    BenchmarkConfig,
    IntervalSummary,
    SimulationError,
    Trace,
)

__all__ = [
    "MAX_MEAL_GRAMS",
    "OUTPUT_NAMES",
    "OUTPUT_UNITS",
    "Benchmark",
    "BenchmarkConfig",
    "IntervalSummary",
    "SimulationError",
    "Trace",
]

#: Simglucose source revision defining the benchmark semantics (PR #99, basal units).
SIMGLUCOSE_REVISION = "LI-Jialu/simglucose@fb7f1674db3337790abefbdc70a9fa86d1235f14"

"""FalCAuN adapter following the ``PythonNumericSUL`` convention.

FalCAuN runs a script with Jep and evaluates ``sul = SUL()``; a script containing
``from apbench.falcaun import SUL`` is enough (see ``examples/falcaun``). One
``step([meal])`` consumes one 30-minute meal input, any finite real amount in
[0, 80] g (the finite alphabet, if any, belongs to FalCAuN's input mapper), and
returns the six outputs (:data:`apbench.OUTPUT_NAMES`) as Python floats. The
adapter does not enforce the reference workload's cumulative intake limits
(:mod:`apbench.admissibility`). The adapter wraps the same
:class:`~apbench.Benchmark` and controller checkpoint as the standalone tools.

``SUL()`` takes no arguments; the configuration can be changed with environment
variables: ``APBENCH_CONTROLLER`` (default ``rppo``), ``APBENCH_PATIENT``
(default ``adolescent#001``), ``APBENCH_SENSOR_SEED`` (default ``1``), and
``APBENCH_TRACE_DIR`` (if set, ``post()`` saves each trace there).
"""

from __future__ import annotations

import os
from pathlib import Path

from apbench.simulator import Benchmark, BenchmarkConfig


def config_from_env() -> BenchmarkConfig:
    defaults = BenchmarkConfig()
    return BenchmarkConfig(
        controller=os.environ.get("APBENCH_CONTROLLER", defaults.controller),
        patient=os.environ.get("APBENCH_PATIENT", defaults.patient),
        sensor_seed=int(os.environ.get("APBENCH_SENSOR_SEED", defaults.sensor_seed)),
    )


class SUL:
    def __init__(self, config: BenchmarkConfig | None = None):
        self.benchmark = Benchmark(config or config_from_env())
        trace_dir = os.environ.get("APBENCH_TRACE_DIR")
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.count = 0

    def pre(self) -> None:
        self.benchmark.reset()

    def step(self, inputSignal) -> list[float]:
        (meal,) = list(inputSignal)
        return [float(x) for x in self.benchmark.step(float(meal)).as_list()]

    def post(self) -> None:
        if self.trace_dir is not None:
            self.count += 1
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self.benchmark.trace().save(self.trace_dir / f"trace{self.count:06d}")

    def close(self) -> None:
        pass

# FalCAuN integration example

This directory demonstrates how to use
[FalCAuN](https://github.com/MasWag/FalCAuN) with the
[artificial-pancreas benchmark](../../README.md). It contains the Jep
initialization script `apbench_sul.py` and the Clojure experiment
`case_study.clj`.

**Scope.** This is an integration example, not yet a complete run of the
reference constrained workload. In particular, the current Clojure
example does not enforce cumulative meal-intake constraints and
checks only four of the six default STL properties.

## Python SUL interface

`apbench.falcaun.SUL` follows FalCAuN's `PythonNumericSUL` convention:
`pre()` resets the benchmark, `step([meal])` executes one interval and
returns the six interval outputs as Python floats, `post()` optionally
saves the trace, and `close()` releases the adapter. It wraps the same
`apbench.Benchmark` and controller checkpoints as the standalone
simulation.

The parameterless `SUL()` supports the environment variables
`APBENCH_CONTROLLER`, `APBENCH_PATIENT`, and `APBENCH_SENSOR_SEED`;
`APBENCH_TRACE_DIR` enables trace saving. Each `step([meal])` forwards
a finite real-valued amount in [0, 80] g to the simulator; the SUL
does not impose a finite meal alphabet. The reference workload's
admissibility constraints are defined in the
[main README](../../README.md#meal-inputs-and-admissibility).

## Example setup

Install the benchmark in the Python environment used by Jep. Jep
requires `JAVA_HOME` when building from source.

```sh
pip install -e '.[falcaun]'
export JEP_JAVA_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/jep"
export PYTHONEXECUTABLE="$PWD/.venv/bin/python"
```

From the `example/clojure` directory of a FalCAuN checkout (where
`common.clj` and `project.clj` are located), run:

```sh
export APBENCH_EXAMPLE_DIR=/path/to/artificial-pancreas-benchmark/examples/falcaun
lein exec -p "$APBENCH_EXAMPLE_DIR/case_study.clj"
```

The example uses 24 logical steps (30 minutes each by default), and
the following finite input mapping and output thresholds:

```clojure
(def input-mapper (InputMapperReader/make [[0.0 40.0 80.0]]))
(def output-mapper-reader
  (OutputMapperReader. [[180.0 250.0] [54.0 70.0] [250.0 300.0] [1.0] [nil] [nil]]))
```

The finite alphabet is a FalCAuN configuration choice, not part of
the benchmark's continuous reference input specification. A finite
input mapper also does not impose the rolling 120 g or total 300 g
intake constraints. These must be represented separately (for
example by a finite-state input-constraint monitor or appropriate
assumptions). Membership queries may explore inadmissible sequences,
but these cannot count as counterexamples for the constrained
benchmark.

## Property and interval semantics

The four formulas in `case_study.clj` concern glucose below 54 mg/dL,
below 70 mg/dL, above 300 mg/dL, and prolonged glucose above
250 mg/dL. They use interval minima or maxima derived from 1-minute
physiological glucose samples; the example does not yet encode the
other two default properties, `recovery_250_to_180_180min` and
`suspend_below_70`.

The benchmark's `glucose_min` and `glucose_max` use the same
right-closed interval convention as FalCAuN's `previous_min` and
`previous_max`: (kΔ, (k+1)Δ]. The initial glucose sample at t = 0 is
retained in the complete trace but is not included in any interval
extremum. For full-trace invariants, the initial sample must therefore
be considered separately; the current four-property example does not
establish equivalence for arbitrary initial states.

Furthermore, time-bounded formulas over 30-minute summaries do not
automatically reproduce formulas on the canonical 1-minute RTAMT
trace. Duration-sensitive requirements may require additional
interval observers and independent equivalence checks.

See the main README for the [full property suite](../../README.md#properties)
and the [constrained input space](../../README.md#admissible-input-sequences-global-constraints).

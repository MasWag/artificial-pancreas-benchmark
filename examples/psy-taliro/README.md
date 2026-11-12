# Ψ-TaLiRo integration example

This directory falsifies the [artificial-pancreas benchmark](../../README.md)
with [Ψ-TaLiRo](https://github.com/cpslab-asu/psy-taliro) 3.2.0 and its native
`DualAnnealing` optimizer. It runs the reference constrained workload
(24 × 30-minute admissible meal inputs) on the benchmark's own simulator,
controller checkpoints and RTAMT properties.

| file | content |
|---|---|
| `falsify.py` | the benchmark-specific part: admissible input parametrization, Ψ-TaLiRo model and property specifications, campaign, result files and command line |
| `trace_reuse.py` | generic falsification loop (independent of the benchmark): targets the selected properties in turn and checks every simulated trace against all of them |
| `tests/` | synthetic orchestration tests and small-budget benchmark tests |

## Installation

Ψ-TaLiRo is an optional dependency of the benchmark:

```sh
pip install -e '.[test,psy-taliro]'
```

For the exact reference environment (Python 3.12), install the four packages that
Ψ-TaLiRo adds on top of the benchmark's lock file:

```sh
pip install -r requirements/py312-lock.txt -r examples/psy-taliro/requirements.txt
pip install --no-deps -e .
```

## Usage

A campaign targets every default property
(`apbench.properties.default_properties()`), or the subset given with
`--properties`, in order. Falsifying a single property is a campaign with one
property.

```sh
python examples/psy-taliro/falsify.py --controller rppo \
    --iterations 1500 --seed 1 --output-dir results/psy-taliro-all
python examples/psy-taliro/falsify.py --properties hyper_300 prolonged_250_120min hypo_70 \
    --iterations 1500 --seed 1 --output-dir results/psy-taliro-selected
python examples/psy-taliro/falsify.py --properties hyper_300 \
    --iterations 1500 --seed 1 --output-dir results/psy-taliro-hyper
```

| argument | meaning |
|---|---|
| `--properties NAME ...` | default properties to falsify, in order (default: all default properties) |
| `--controller` | `rppo` (default), `ppo` or `nominal-basal` |
| `--iterations N` | optimizer evaluations (= simulations) **per targeted property** (default 1500, the per-property budget customary in ARCH-COMP falsification) |
| `--seed S` | Ψ-TaLiRo seed; must be > 0 (default 1). Every targeted search uses this seed |
| `--timeout T` | optional campaign-wide timeout in seconds |
| `--output-dir DIR` | result directory |

## Search and stopping behavior

The selected properties are targeted one at a time, in the given order:

1. The first property that is not yet falsified is targeted by a native
   Ψ-TaLiRo run of `DualAnnealing(min_cost=0.0)`, whose objective is the
   robustness of that property only.
2. Every simulated trace is monitored against **all** selected properties that
   are not yet falsified. A property is falsified by the first trace with
   strictly negative robustness; zero robustness is not a falsification.
   Falsifications of other properties are recorded immediately (`via: reuse`).
3. A property falsified this way is never targeted (`target_status: skipped`).
4. When a search ends, the next unfalsified property is targeted. No optimizer
   state is shared between searches.
5. The campaign ends when all selected properties are falsified, all searches
   have used their budgets, or the timeout expires.

The robustness passed to Ψ-TaLiRo is exactly `Property.robustness` of the
benchmark (RTAMT, 1-minute samples, finite-trace semantics), computed on the one
simulated trace. Each trace counts as one SUT simulation, however many properties
are monitored on it.

**Budgets.** `--iterations` bounds each targeted search, so a campaign with
several properties may use up to (number of targeted properties) × `--iterations`
simulations. The actual number of unique simulations is reported as
`total_simulations`. SciPy's dual annealing checks the stopping condition in a
callback, so a search may evaluate a few samples after its first falsifying one;
these are counted. The timeout is checked before and after each simulation, and
a simulation that ends after the timeout is counted but cannot falsify anything.

## Admissible input parametrization

Ψ-TaLiRo optimizes 24 static variables x_k ∈ [0, 1] (values outside are
clipped), which are mapped sequentially to meals

```math
u_k=\min\left(80,\;120-\sum_{i=\max(0,k-3)}^{k-1}m_i,\;300-\sum_{i=0}^{k-1}m_i\right),
\qquad m_k = x_k u_k .
```

`u_k` is `apbench.admissibility.remaining_allowance`, computed exactly and
rounded down, so every candidate is admissible; each sequence is also checked with
`is_admissible` before simulation. Meal sizes are continuous. This parametrization
shapes the optimizer's search space (for example, x_k = 1 exhausts the
allowance) and does not reproduce the input distribution of the random baseline
(`apbench-random-falsify`).

## Results

`--output-dir` contains:

* `runs.csv`: one row per SUT simulation: simulation number, targeted property,
  meal sequence (exact float values) and the robustness of each selected
  property. A cell is empty when the property was not monitored because it was
  already falsified.
* `summary.json`: selected properties, controller and simulator
  configuration, Ψ-TaLiRo version, optimizer settings and seed, timeout, input
  constraints and parametrization, total simulations, total simulated minutes,
  wall-clock time and campaign status (`ALL_FALSIFIED`, `BUDGET_EXHAUSTED` or
  `TIMEOUT`). For each property: name, formula, `falsified`, best robustness
  observed, first falsifying simulation and its elapsed time, `via`
  (`target` or `reuse`), simulations and time spent targeting it, and
  `target_status` (`falsified`, `budget`, `skipped`, `timeout`, `not_reached`).
* `counterexamples/<property>_{meals.json,minutes.csv,outputs.csv}`: the first
  counterexample of each falsified property and its full trace, written from the
  simulated trace (no extra simulation).

Replay a counterexample with the benchmark's simulator:

```sh
apbench-simulate --meals-json results/psy-taliro-all/counterexamples/hyper_300_meals.json --controller rppo
```

## Falsification results

The table lists which default properties this script falsifies with a budget
of 5000 evaluations, for both neural controllers and seeds 1–3. Each cell is
one campaign with a single property (`--properties`), so no trace is reused
from another property's search:

```sh
for c in rppo ppo; do for s in 1 2 3; do
  for p in hypo_54 hypo_70 hyper_300 prolonged_250_120min \
           recovery_250_to_180_180min suspend_below_70; do
    python examples/psy-taliro/falsify.py --properties $p --controller $c \
        --iterations 5000 --seed $s --output-dir results/table/${c}_${p}_${s}
  done
done; done
```

The 36 campaigns are independent and can run in parallel (e.g. as a batch
array job). A falsified cell gives the number of falsified seeds and, for seeds
1, 2, 3, the simulation at which the first counterexample was found. Otherwise
it gives the lowest robustness observed over the three seeds (5000 simulations
each).

| property | `rppo` | `ppo` |
|---|---|---|
| `hypo_54` | 0/3 (best 23.94) | 0/3 (best 16.05) |
| `hypo_70` | 0/3 (best 7.94) | 0/3 (best 0.05) |
| `hyper_300` | **3/3** (5, 1, 3) | **3/3** (5, 16, 5) |
| `prolonged_250_120min` | **3/3** (2, 1, 1) | **3/3** (6, 1, 3) |
| `recovery_250_to_180_180min` | **3/3** (251, 1, 247) | **3/3** (780, 13, 412) |
| `suspend_below_70` | 0/3 (best 7.94) | 0/3 (best 0.15) |

`hypo_70` and `suspend_below_70` are nevertheless falsifiable for `ppo`.
Searches outside this example found admissible meal sequences whose replay with
the benchmark's simulator gives robustness −1.80 for `hypo_70` and −0.90 for
`suspend_below_70`, both on the same trace. These searches were an evolution
strategy minimizing the lowest glucose value, and other Ψ-TaLiRo-compatible
optimizers. Here, DualAnnealing comes within 0.05 and 0.15 but does not cross
zero, so these two properties are hard targets for it. For `rppo`, no
counterexample to either property is known; the closest margin found is
about 7.6.

A simulation takes about 0.6–0.7 s on one CPU core, so a campaign that uses
its whole budget runs for about an hour. The ppo results for
`recovery_250_to_180_180min` show that a few hundred evaluations are sometimes
needed, which is why the default budget is 1500.

## Tests

```sh
pytest examples/psy-taliro/tests
```

The tests cover the orchestration on synthetic models:
- trace reuse, incidental falsification and skipping;
- simulation accounting;
- zero robustness;
- budget and timeout;
- equivalence with a native Ψ-TaLiRo run.

They also cover the benchmark integration:
- the deterministic, admissible parametrization;
- cost equality with the benchmark's RTAMT robustness;
- counterexample replay;
- seed reproducibility;
- both neural controllers;
- the command line.

All of these run with small budgets.

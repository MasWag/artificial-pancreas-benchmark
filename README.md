# Artificial Pancreas Falsification Benchmark

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](./LICENSE)

A reusable benchmark for **black-box falsification** of a neural-network-controlled
artificial pancreas: a [Simglucose](https://github.com/jxx123/simglucose) virtual
patient, a CGM sensor, an insulin pump, a neural basal-insulin controller, and
adversarial meal inputs chosen by a falsification tool. It provides:

* a deterministic closed-loop simulator with physically defined outputs (`apbench.Benchmark`);
* trained RecurrentPPO (LSTM) and PPO (feed-forward) controllers;
* STL properties monitored with [RTAMT](https://github.com/nickovic/rtamt);
* a continuous-valued, constraint-aware random-sampling baseline (`apbench-random-falsify`);
* a [FalCAuN](https://github.com/MasWag/FalCAuN) integration example ([documentation](examples/falcaun/README.md));
* a [Ψ-TaLiRo](https://github.com/cpslab-asu/psy-taliro) integration example ([documentation](examples/psy-taliro/README.md)).

The benchmark is intended for **evaluating testing tools**, not for clinical
validation. The controllers are not medical devices. The reference falsification
workload accepts continuous-valued meal inputs and limits cumulative intake.
Its intake limits are benchmark assumptions, not medical dietary recommendations.

## Installation

Python 3.12 is the reference environment (other versions ≥ 3.10 may work).

```sh
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e '.[test]'           # add 'train' for training, 'falcaun' for Jep, 'psy-taliro' for Ψ-TaLiRo
```

`pyproject.toml` lists general, unpinned requirements. The exact environment used for
the bundled controller checkpoints and reference evaluations is `requirements/py312-lock.txt` (CPU PyTorch):

```sh
pip install -r requirements/py312-lock.txt && pip install --no-deps -e .
```

**Simglucose revision.** The benchmark semantics depend on Simglucose's pump units.
Upstream Simglucose quantizes basal requests (U/min) with `inc_basal` and clips them
with `max_basal`, both of which are in U/hour, so the Insulet pump effectively
accepted up to 30 U/min. [PR #99](https://github.com/jxx123/simglucose/pull/99) fixes
this but is not merged at the time of writing, so the dependency is pinned to the
contributor's commit
`LI-Jialu/simglucose@fb7f1674db3337790abefbdc70a9fa86d1235f14`
(also recorded as `apbench.SIMGLUCOSE_REVISION`). No other Simglucose fork is used,
and apbench adds no unit conversion of its own.

**Compatibility workaround.** Simglucose and its dependency `gym==0.9.4` import
`pkg_resources`, which Setuptools removed in version 81. Hence the requirement
`setuptools<81`. `apbench._compat` only silences the resulting deprecation warnings.

## Simulator: inputs, outputs, units and timing

```python
from apbench import Benchmark, BenchmarkConfig

bench = Benchmark(BenchmarkConfig())         # adolescent#001, Dexcom, Insulet, controller "rppo"
bench.reset()
out = bench.step(80.0)                       # first interval (30 minutes by default)
print(out.as_list())                         # six floats, see the table below
trace = bench.simulate([80, 0, 40] + [0] * 21)   # reset + 24 default intervals = 12 hours
trace.save("example")                        # example_minutes.csv, example_outputs.csv
```

`BenchmarkConfig` is a plain dataclass. Its fields are `patient`, `sensor`, `pump`,
`controller` (a registry name or a factory), `start_time` (default: 2018-01-01 00:00),
`sensor_seed` (1), `patient_seed`, `random_init_bg`, `interval_minutes` (30),
and `horizon_steps` (24). Each simulator input is a finite real-valued
carbohydrate amount in [0, 80] g. The simulator accepts continuous meal sizes,
independently of any input discretization chosen by a testing tool. Cumulative
intake constraints belong to the test workload, not the simulator interface.
There is no separate benchmark specification file format.

**Timing.**
* The patient ODE advances in 1-minute steps.
* With the default Dexcom sensor, the controller acts every 3 minutes, which
  corresponds to one Simglucose `T1DSimEnv.step`.
* Each `Benchmark.step(meal)` advances by `interval_minutes` (30 minutes by
  default), corresponding to 10 controller steps with the default sensor.
  The interval must be a positive multiple of the sensor's sampling period.

The meal is announced **once**, at the first minute of the interval. Simglucose then
eats it at 5 g/min, so 80 g takes 16 minutes.

**Units.**
* The controller requests a basal insulin **rate in U/min** (Simglucose's controller
  convention).
* The pump's limits and increments are given in **U/hour**. With PR #99 it quantizes
  the request to 0.05 U/hour and clips it to [0, 30] U/hour.
* The pump exposes `pump.max_basal == 0.5` U/min. `pump._params["max_basal"]` (30) is
  in U/hour.
* Delivered insulin **amounts** are in U (rate × duration).
* Bolus semantics are *not* corrected by PR #99. All controllers here are basal-only
  (bolus 0).

**Outputs.** Let Δ = `interval_minutes` (30 minutes by default). Each
`Benchmark.step(meal)` returns six values for interval k, in the fixed order
`apbench.OUTPUT_NAMES`. Sampled physiological-glucose extrema use the
right-closed interval (kΔ, (k+1)Δ]:

| # | name | unit | definition |
|---|---|---|---|
| 0 | `glucose_end` | mg/dL | physiological glucose (Simglucose `Gsub`) at the end of the interval |
| 1 | `glucose_min` | mg/dL | minimum of the Δ one-minute samples at kΔ+1, …, (k+1)Δ, including the current endpoint but excluding the previous one |
| 2 | `glucose_max` | mg/dL | maximum of the same samples |
| 3 | `insulin_dose` | U | insulin delivered during the interval: Σ delivered rate × 1 min (exact; the rate is constant within each minute) |
| 4 | `insulin_max_rate` | U/hour | maximum delivered (pump-quantized) rate in the interval |
| 5 | `cgm_end` | mg/dL | the controller-visible CGM observation at the end of the interval |

**Interval-boundary convention.** Interval minima and maxima are computed over
(kΔ, (k+1)Δ], excluding the start and including the end of each interval.
Consecutive intervals therefore do not double-count boundary samples. The
initial glucose sample at t = 0 remains available in the full `Trace`, but
is not included in any interval extremum. Full-trace properties evaluated
using interval summaries must account for this initial sample separately.

`cgm_end` is Simglucose's controller observation: the mean of the three 1-minute
sensor readings of the last control step. The Dexcom sensor draws a new noisy sample
only every 3 minutes and holds it in between, so this is `(2·previous + new)/3`,
clipped per sample to [39, 600] mg/dL. Physiological glucose and CGM are never mixed.

The full `Trace` keeps:
* the 1-minute physiological glucose, raw CGM sample, announced meal, carbohydrate
  intake, requested rate and delivered rate;
* the 3-minute controller observations, Simglucose's averaged BG and its `done` flag.

Simglucose's `done` (3-minute average BG < 10 or > 600 mg/dL) is recorded in
`trace.ctrl_done` / `threshold_crossed`, but the simulation **continues** through
the horizon. ODE failures and non-finite states raise `SimulationError`.

`reset()` restores the patient, scenario, sensor-noise generator, pump, and
controller's hidden state. The same configuration and input sequence produce
identical traces.

To simulate a sequence from the command line:

```sh
apbench-simulate --meals 80 0 40 0 0 0 --controller rppo --output results/example
```

## Meal inputs and admissibility

The reference workload accepts one carbohydrate amount at the start of each
interval. By default, intervals last 30 minutes and the 24-interval horizon
covers 12 hours. Zero denotes no meal; Simglucose consumes a positive amount
according to its meal model.

### Continuous-valued input domain

For the default 12-hour workload, the $k$-th interval's input is any finite
real-valued carbohydrate amount $m_k$ with

```math
0 \le m_k \le 80\ \mathrm{g}, \qquad k=0,\ldots,23.
```

Input amounts are continuous-valued but arrive at discrete interval
boundaries. The simulator checks the per-interval [0, 80] g range, without
restricting meals to any finite categories. It does not reject a sequence
solely because cumulative intake exceeds the reference workload limits:
these constraints are checked separately by the testing workload.

### Admissible input sequences (global constraints)

The constrained workload defines one admissible input set shared by
**all** STL properties. These are constraints on test sequences, not additional
STL safety properties. **Users must configure their falsification tool's input
signal generator to produce sequences satisfying these constraints.** The
simulator does not enforce them. For the standard 12-hour workload
(Δ = 30 minutes and 24 intervals), admissible inputs must satisfy both conditions:

1. **Rolling two-hour intake:** no more than 120 g across any four
   successive 30-minute intervals (using the available prefix initially):

   ```math
   \sum_{i=\max(0,k-3)}^{k}m_i \le 120\ \mathrm{g},
   \qquad k=0,\ldots,23.
   ```

2. **Total intake:** no more than 300 g across the 12-hour sequence:

   ```math
   \sum_{i=0}^{23}m_i \le 300\ \mathrm{g}.
   ```

For example, `[80, 20, 20, 0]` meets the rolling limit if there is
no preceding intake, whereas `[80, 80, 0, 0]` does not. Sixteen consecutive
20 g inputs meet the rolling limit but violate the total limit.

The limits are **engineering choices** for a plausible stress-test
workload, not clinical dietary recommendations. A sequence satisfying
these limits is *admissible*; this does not mean it represents a typical diet.
The simulator also supports bounded sequences that violate the cumulative
constraints for unrestricted stress experiments, but they cannot serve as
counterexamples to the constrained workload. Since all inputs are nonnegative,
an inadmissible prefix cannot become admissible after appending more inputs.
Other interval durations or horizons constitute workload variants; the
two-hour window and total-intake limit must then be specified accordingly.

## Controllers

| name | description |
|---|---|
| `rppo` (default) | RecurrentPPO (`MlpLstmPolicy`, LSTM), trained for this benchmark |
| `ppo` | PPO (`MlpPolicy`, feed-forward), trained for this benchmark with the same setup |
| `nominal-basal` | constant steady-state basal of the patient (reference, not neural) |

A controller implements `reset()` and `act(ctx) -> basal U/min`. `ctx` is a
`ControlContext` holding the CGM observation, the time, the rate delivered in the
previous control step, and the pump limits.

### Neural controllers (`src/apbench/controllers/{rppo,ppo}/`)

Both controllers are trained by `apbench-train` with the same environment,
observation, action scaling, reward, hyperparameters, seed and model selection. Only
the policy differs:
* `rppo`: a 64-unit LSTM followed by a 64-unit layer, for both the actor and the
  critic.
* `ppo`: two 64-unit layers, with no memory.

The shared setup:
* **Observation** (`apbench.observation.ObservationBuilder`, shared by training and
  deployment): `[(CGM − 140)/100, (CGM − CGM 15 min ago)/30, previous delivered
  rate / max]`. No internal patient states are used.
* **Action** in [−1, 1], mapped linearly to [0, 3] U/hour basal.
* **Training** on adolescent#001 with Simglucose's `RandomScenario` meals (no meal
  announcement, no bolus) and episode seeds drawn from the environment RNG.
  * Each episode lasts 24 hours.
  * The initial glucose is spread over ~90–300 mg/dL by scaling the glucose states of
    the default initial state. Simglucose's own `random_init_bg` varies glucose by
    only a few mg/dL, which lets a constant basal rate look optimal.
  * Reward: −(3·LBGI + HBGI)/1000 per step on physiological glucose, plus termination
    with a penalty at BG < 40 mg/dL. The scale keeps returns O(1), so that the value
    loss does not dominate PPO's clipped gradient.
  * Hyperparameters: 16 environments, n_steps 128, minibatch 2048, 10 epochs,
    γ = 0.995, learning rate 1e-3 decayed linearly, initial log-std −1, 3M steps,
    seed 0.
  * Model selection: fixed validation episodes (seeds 10000+), separate from the
    training environments. The best checkpoint is bundled.
  * Held-out evaluation: seeds 20000+.
* **Inference** is deterministic, with SB3's native checkpoint loaded on CPU. The
  LSTM state of `rppo` persists across all control steps of a query and is cleared
  by `reset()`.
* Each checkpoint directory holds `model.zip`, `metadata.json` (hyperparameters,
  seeds, training command, software versions, Simglucose revision) and
  `validation.jsonl` (the training validation log).

Bundled checkpoints:
* `rppo`: best validation checkpoint at 1.8M steps (50 min of training on CPU).
* `ppo`: best validation checkpoint at 1.2M steps (37 min of training on CPU).

The held-out evaluation (`src/apbench/controllers/evaluation.json`) used 20 unseen
24-hour `RandomScenario` days (seeds 20000–20019) and a 12-hour fasting run from the
benchmark's initial state. Statistics use physiological glucose sampled every minute:

| controller | fasting 12 h: min glucose | random days: TIR 70–180 | time < 70 | time > 250 | min glucose | insulin |
|---|---|---|---|---|---|---|
| `rppo` | 102.5 mg/dL | 96.3% | 0.00% | 0.6% | 82.7 mg/dL | 36.5 U/day |
| `ppo` | 105.1 mg/dL | 96.5% | 0.00% | 0.6% | 80.3 mg/dL | 36.6 U/day |
| `nominal-basal` | 147.5 mg/dL | 52.5% | 0.00% | 2.1% | 92.9 mg/dL | 20.4 U/day |

RecurrentPPO and feed-forward PPO perform similarly on these held-out
`RandomScenario` episodes. These are not the constrained 12-hour
falsification sequences, so the results characterize nominal controller
behavior rather than relative falsification difficulty.

To retrain and re-evaluate:

```sh
pip install -e '.[train]'
apbench-train --algorithm rppo --timesteps 3000000 --seed 0 --max-rate-u-hour 3 --learning-rate 1e-3 --output runs/rppo
apbench-train --algorithm ppo  --timesteps 3000000 --seed 0 --max-rate-u-hour 3 --learning-rate 1e-3 --output runs/ppo
for c in rppo ppo; do
    cp runs/$c/best_model.zip src/apbench/controllers/$c/model.zip
    cp runs/$c/metadata.json runs/$c/validation.jsonl src/apbench/controllers/$c/
done
apbench-evaluate --controllers rppo ppo nominal-basal --episodes 20 \
    --output src/apbench/controllers/evaluation.json
```

Training is reproducible only up to floating-point nondeterminism across machines
and thread counts. The bundled checkpoints, not retraining, define the benchmark.

## Properties

`apbench.properties.default_properties(H)` returns RTAMT discrete-time STL properties
over 1-minute signals sampled at t = 0, …, H minutes.
* One RTAMT time unit is one minute.
* Robustness is evaluated at t = 0, and a property is **violated iff its robustness
  is negative**.
* RTAMT truncates bounded windows at the end of a finite trace, so properties with
  a future window of length T quantify only over [0, H − T]. Every window then lies
  inside the trace.

Signals:

| signal | unit | meaning |
|---|---|---|
| `glucose` | mg/dL | physiological glucose |
| `meal` | g | meal announced at that minute |
| `insulin_rate` | U/hour | delivered rate during [t, t+1) |
| `insulin_dose_60` | U | insulin delivered during [t−60, t), an interval observer computed in Python (available for custom properties) |

| name | STL (RTAMT) | meaning |
|---|---|---|
| `hypo_54` | `always (glucose >= 54)` | no severe hypoglycemia |
| `hypo_70` | `always (glucose >= 70)` | no hypoglycemia |
| `hyper_300` | `always (glucose <= 300)` | upper glucose safety |
| `prolonged_250_120min` | `always[0:H-120] (eventually[0:120] (glucose <= 250))` | glucose does not stay above 250 mg/dL for 120 minutes, i.e. no 121 consecutive 1-minute samples above 250 |
| `recovery_250_to_180_180min` | `always[0:H-180] ((glucose > 250) -> ((always[0:180] (1e6*(1-meal) >= 0)) -> (eventually[0:180] (glucose <= 180))))` | after glucose exceeds 250 mg/dL, and *assuming no meal (≥ 1 g) in the next 180 minutes*, glucose is back to ≤ 180 mg/dL within 180 minutes |
| `suspend_below_70` | `always ((glucose < 70) -> (insulin_rate <= 1.0))` | while glucose is below 70 mg/dL, at most 1 U/hour is delivered |

(H = 720 for the default 12-hour horizon.)

### Known falsifiability

Status of each property under the reference constrained workload:

| name | `rppo` | `ppo` |
|---|---|---|
| `hypo_54` | likely not falsifiable | likely not falsifiable |
| `hypo_70` | likely not falsifiable | falsifiable |
| `hyper_300` | falsifiable | falsifiable |
| `prolonged_250_120min` | falsifiable | falsifiable |
| `recovery_250_to_180_180min` | falsifiable | falsifiable |
| `suspend_below_70` | likely not falsifiable | falsifiable |

"Falsifiable" means that an admissible counterexample is known and reproduces
with `apbench-simulate`. "Likely not falsifiable" means that no counterexample
was found by any search we ran: random sampling, Ψ-TaLiRo's DualAnnealing,
Part-X, ConBO and an evolution strategy minimizing the lowest glucose value.
This is empirical evidence, not a proof.

The default suite contains no simple hourly insulin-dose limit: such a limit
is readily violated by ordinary high-rate controller behavior (the neural
controllers may deliver up to 3 U/hour). `apbench.properties.insulin_dose_window`
remains available for custom property suites.

The meal guard in `recovery` is scaled by 10⁶ to prevent its robustness
from dominating the glucose margin when the no-meal assumption holds.
The STL requirements are evaluated on the one-minute physiological trace;
input admissibility is checked independently on the 30-minute meal sequence.

## Random-sampling baseline

The reference baseline `apbench-random-falsify` samples **continuous-valued,
admissible** meal sequences. At step $k$, it computes the remaining permitted
carbohydrate amount:

```math
u_k = \min\left(
80,\;120-\sum_{i=\max(0,k-3)}^{k-1}m_i,\;
300-\sum_{i=0}^{k-1}m_i
\right),
```

with empty sums defined as zero. If $u_k=0$, the sampler sets $m_k=0$.
Otherwise, it chooses $m_k=0$ with probability 1/2, or draws a positive
meal size uniformly from $(0,u_k]$ with probability 1/2. Thus the
baseline samples both fasting intervals and **continuous positive meal
sizes** while respecting the global intake constraints. The resulting
distribution is not uniform over the set of admissible sequences.

```sh
apbench-random-falsify --trials 1000 --seed 0 --controller rppo --horizon 24 --output-dir results/random-seed0
```

Every trial resets the system, simulates one full sequence, and evaluates
**all** properties on the same 1-minute physiological trace using RTAMT.
Sampling does not adapt to previous counterexamples or robustness values.
`--seed` controls the input sampler independently of the patient and CGM
seeds.

The output directory contains:

* `runs.csv`: sampled sequences and their quantitative robustness values;
* `summary.json`: configuration, seeds, simulation budget, best robustness,
  first falsification for each property, total simulated time, and wall time;
* `counterexamples/`: the meal sequences and full traces of first
  counterexamples. Replay with `apbench-simulate --meals-json PATH`.

## Implementation notes

The simulator uses a local `T1DPatient` subclass that exposes the same
patient parameters as ordinary Python attributes for the ODE right-hand
side, avoiding repeated pandas Series lookups. In the reference
environment this gives bit-identical results and approximately an
eightfold speedup (about 0.5 seconds for a default 12-hour simulation).
This is a local optimization, not an additional modification of the
pinned Simglucose dependency described under Installation.

## Tests

Run the project's test suite with:

```sh
pytest
```

The suite (`tests/`) covers:

* `test_simulator.py`: pump units, the [0, 80] g continuous input domain
  (endpoints accepted; negative, larger, NaN and infinite values rejected),
  meal timing, step accounting, right-closed interval extrema (including
  boundary-only extrema, the excluded t = 0 sample and non-default interval
  durations), the six outputs against the full 1-minute trace, CGM labeling,
  deterministic replay, and continuation after Simglucose's `done`;
* `test_admissibility.py`: the rolling 120 g and total 300 g limits, including
  equality cases, prefixes, exact sums, and the remaining-allowance formula;
* `test_tools.py`: admissibility, reproducibility and seed independence of the
  constrained random sampler, the `apbench-random-falsify` outputs, exact replay
  of counterexamples and `runs.csv` rows, and agreement of the FalCAuN Python
  SUL with `Benchmark` on continuous inputs;
* `test_properties.py`: the six default STL definitions, the insulin-dose
  observer and window property, and RTAMT robustness on
  synthetic traces with known values;
* `test_checkpoints.py`: the bundled PPO and RecurrentPPO checkpoints
  (metadata, deterministic rollouts, glucose response, no hypoglycemia when
  fasting, and known counterexamples under unrestricted stress inputs);
* `test_rppo_mechanics.py`: the observation and action scaling shared by
  training and deployment, and the LSTM state's persistence within a query
  and reset between queries.

## License

Apache License 2.0 (see `LICENSE`), Copyright 2026 Masaki Waga.

The benchmark depends on [Simglucose](https://github.com/jxx123/simglucose) (MIT
license, Copyright (c) 2018 Jinyu Xie) at the revision
`LI-Jialu/simglucose@fb7f1674db3337790abefbdc70a9fa86d1235f14`, which contains the
basal-unit correction proposed in [PR #99](https://github.com/jxx123/simglucose/pull/99).
Simglucose is installed as a dependency, not redistributed. apbench subclasses it
without modifying it.

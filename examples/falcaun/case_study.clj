;; Falsify the artificial pancreas benchmark with FalCAuN (Clojure, Jep-based PythonNumericSUL).
;;
;; Run from FalCAuN's example/clojure directory (for common.clj and project.clj), with
;; the benchmark installed in the Python environment used by Jep:
;;
;;   export JEP_JAVA_LIBRARY_PATH=/path/to/venv/lib/python3.12/site-packages/jep
;;   export PYTHONEXECUTABLE=/path/to/venv/bin/python
;;   export APBENCH_EXAMPLE_DIR=/path/to/artificial-pancreas-benchmark/examples/falcaun
;;   lein exec -p "$APBENCH_EXAMPLE_DIR/case_study.clj"
;;
;; One FalCAuN step = one 30-minute interval. The six outputs are, in order:
;;   signal(0) glucose_end [mg/dL]     signal(3) insulin_dose [U]
;;   signal(1) glucose_min [mg/dL]     signal(4) insulin_max_rate [U/hour]
;;   signal(2) glucose_max [mg/dL]     signal(5) cgm_end [mg/dL]
;; FalCAuN's STL here is over the 30-minute outputs. The per-interval min/max are taken over
;; the 1-minute samples of (kΔ, (k+1)Δ], so the initial sample at t = 0 is in no interval, and
;; these formulas are approximations of, not equivalents to, the 1-minute RTAMT properties.
;;
;; The finite input mapper [0 40 80] is a FalCAuN configuration choice: the SUL accepts any
;; real meal in [0, 80] g. This example does not enforce the reference workload's 120 g / 2 h
;; and 300 g / 12 h intake limits, so its counterexamples must be checked for admissibility
;; (apbench.admissibility.is_admissible) before they count for the constrained benchmark.

(import '(net.maswag.falcaun AdaptiveSTLList OutputMapperReader))
(import '(net.maswag.falcaun.python PythonNumericSUL))

(load-file "common.clj")
(suppress-logs)

(def example-dir (or (System/getenv "APBENCH_EXAMPLE_DIR") "."))
(def signal-length 24) ; 24 x 30 minutes = 12 hours
(def signal-step 1.0)

(def input-mapper (InputMapperReader/make [[0.0 40.0 80.0]]))
(def output-mapper-reader
  (let [ignore [nil]]
    ;; Abstraction thresholds per output (glucose_end, glucose_min, glucose_max, dose, max rate, cgm)
    (OutputMapperReader. [[180.0 250.0] [54.0 70.0] [250.0 300.0] [1.0] ignore ignore])))
(def mapper (make-mapper input-mapper output-mapper-reader))

(def stl-list
  (parse-stl-list
   ["[] (signal(1) >= 54.0)"                     ; no severe hypoglycemia (samples after t = 0)
    "[] (signal(1) >= 70.0)"                     ; no hypoglycemia (samples after t = 0)
    "[] (signal(2) <= 300.0)"                    ; no glucose above 300 mg/dL (samples after t = 0)
    "! <> ([]_[0,3] (signal(1) > 250.0))"]       ; not 4 intervals (120 one-minute samples) above 250;
                                                 ; the RTAMT property needs 121 samples
   input-mapper
   output-mapper-reader))
(def properties (AdaptiveSTLList. stl-list signal-length))

(def sul (PythonNumericSUL. (str example-dir "/apbench_sul.py")))
(def verifier (make-verifier sul signal-step properties mapper signal-length
                             50000 200 0.5 0.01 30))
(def result (.run verifier))
(print-results verifier result)
(print-stats verifier)

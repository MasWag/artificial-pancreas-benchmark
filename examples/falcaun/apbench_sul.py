# Initialization script for FalCAuN's PythonNumericSUL (run by Jep).
# Requires the artificial-pancreas-benchmark package in the Python environment used by Jep.
# Optional configuration: APBENCH_CONTROLLER, APBENCH_PATIENT, APBENCH_SENSOR_SEED, APBENCH_TRACE_DIR.
from apbench.falcaun import SUL  # noqa: F401

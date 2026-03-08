# Mean HbA1c Sensitivity Analysis (SageMaker)

The notebook `mean_a1c_sensitivity.ipynb` computes a sensitivity analysis using the mean of all HbA1c values within each 6-month window (rather than the most recent single value). This requires database access to query raw laboratory results.

## Requirements

- AWS SageMaker environment with access to the Waymark core database
- `waymark` Python library (provides `get_waymark_core_db_engine()`)

## Pre-Computed Results

Results from this analysis are stored in `../output/mean_a1c_sensitivity_results.json` and do not need to be recomputed to reproduce the other analyses.

Key result: DML ATE = -0.56 percentage points (95% CI: -1.01 to -0.11; P = 0.014; Cohen's d = 0.41) on N = 261 patients with multiple A1c values in both windows.

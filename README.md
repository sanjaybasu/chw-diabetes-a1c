# CHW Diabetes A1c Analysis

Replication code for: "Association of Multidisciplinary Care Team Engagement With Hemoglobin A1c Among Medicaid Enrollees With Diabetes: A Doubly-Robust Machine Learning Analysis"

Authors: Basu S, Baum A, Goldhirsh M, Morgan J, Batniji R

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

Install:
```
pip install -r requirements.txt
```

## Data

Parquet files must be placed in `data/` before running. These files contain de-identified patient-level data and are not included in this repository. See `DATA_NOTE.md` for details.

Required files:
- `data/analytic_cohort.parquet` — Full cohort (N=372, 118 treated, 254 control)
- `data/matched_treated.parquet` — PS-matched treated group (N=114)
- `data/matched_control.parquet` — PS-matched control group (N=114)

## Reproduction

Run all analyses sequentially:
```
make all
```

Or individually:
```
make reanalysis        # Primary DML + GRF heterogeneity
make sensitivity       # Time-adjusted, subgroup, binary outcome
make cost_effectiveness  # RECODe-based cost projections
make figures           # Figure 1 + eFigure 2
```

The mean-A1c primary analysis (`sagemaker/`) requires database access and cannot be run locally. Pre-computed results are in `output/mean_a1c_sensitivity_results.json`.

## Directory Structure

```
code/                  Analysis scripts
  reanalysis.py        Primary DML, matched DML, GRF heterogeneity, E-value, negative control
  sensitivity_analyses.py  Time-adjusted DML, subgroup DML, binary outcome
  cost_effectiveness.py    RECODe-based cost-effectiveness projection, Monte Carlo PSA
  generate_figures.py      Figure 1 (two-panel) and eFigure 2 (CE plane)
data/                  Input parquet files (not tracked)
output/                JSON results
tables/                CSV tables
figures/               Generated figures (PNG + PDF)
manuscript/            Manuscript, appendix, cover letter
sagemaker/             Mean-A1c sensitivity notebook (requires DB access)
```

## Claim-to-Code Traceability

Mapping of key quantitative claims in the manuscript to the script and output that produces each number.

| Manuscript Claim | Value | Script | Output File | JSON Key |
|---|---|---|---|---|
| DML ATE, mean HbA1c (primary) | −0.56 (95% CI: −1.01 to −0.11; P=0.014) | `sagemaker/` | `output/mean_a1c_sensitivity_results.json` | `mean_a1c_full_cohort.ate`, `.ci_lo`, `.ci_hi`, `.p_value` |
| Cohen's d, mean HbA1c (primary) | 0.41 | `sagemaker/` | `output/mean_a1c_sensitivity_results.json` | `mean_a1c_full_cohort.cohens_d` |
| DML ATE, last HbA1c (sensitivity) | −0.289 (95% CI: −0.57 to −0.01) | `code/reanalysis.py` | `output/reanalysis_results.json` | `dml_full_cohort.ate`, `.ci_lower`, `.ci_upper` |
| Cohen's d, last HbA1c (sensitivity) | 0.23 | `code/reanalysis.py` | `output/reanalysis_results.json` | `dml_full_cohort.cohens_d` |
| N after PS trimming (last value) | 369 | `code/reanalysis.py` | `output/reanalysis_results.json` | `propensity_score.n_after_trimming` |
| N primary cohort (mean A1c) | 261 | `sagemaker/` | `output/mean_a1c_sensitivity_results.json` | `mean_a1c_full_cohort.n` |
| GRF feature importance (baseline A1c) | 0.560 | `code/reanalysis.py` | `output/reanalysis_results.json` | `grf_full_cohort.feature_importance.baseline_a1c` |
| GATE Q1 (highest benefit) | −0.579 (95% CI: −1.34 to 0.18) | `code/reanalysis.py` | `output/reanalysis_results.json` | `grf_full_cohort.gate_quartiles["Q1 (highest benefit)"]` |
| Treatment x baseline A1c interaction P | 0.019 | `code/reanalysis.py` | `output/reanalysis_results.json` | `rtm_sensitivity.interaction_pval` |
| Interaction coefficient | −0.31 (95% CI: 0.05 to 0.57) | `code/reanalysis.py` | `output/reanalysis_results.json` | `rtm_sensitivity.interaction_coef`, `.interaction_ci` |
| E-value (point, primary) | 2.26 | computed from Cohen's d | — | RR = exp(0.91 × 0.41) |
| Net savings per person (primary) | $2,246 | `code/cost_effectiveness.py` | `output/cost_effectiveness_results.json` | `overall.deterministic.net_cost` (absolute value) |
| Prob cost-saving (primary) | 99.8% | `code/cost_effectiveness.py` | `output/cost_effectiveness_results.json` | `overall.probabilistic.prob_cost_saving` |
| QALY gain (primary) | 0.044 | `code/cost_effectiveness.py` | `output/cost_effectiveness_results.json` | `overall.deterministic.total_qaly_gain` |
| Intervention cost per course | $262 | `code/cost_effectiveness.py` | `output/cost_effectiveness_results.json` | `overall.deterministic.intervention_cost_per_course` |
| Total encounters dose-response (joint model) | coef = −0.073, P = 0.011 | `code/activity_outcome_analysis.py` | `output/activity_outcome_results.json` | `adjusted_ols_joint.n_total_encounters` |

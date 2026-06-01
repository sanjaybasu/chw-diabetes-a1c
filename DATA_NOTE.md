# Data Note

The `data/` directory contains de-identified patient-level Medicaid claims and laboratory data. These files are excluded from version control (`.gitignore`) because they contain protected health information (PHI)-adjacent data derived from Medicaid administrative claims and electronic health records.

## Required Files

| File | Description |
|---|---|
| `data/analytic_cohort.parquet` | Base diabetes cohort with treatment assignment, covariates, and index date |
| `data/raw/*.parquet` | HbA1c labs, eligible universe (IPCW denominator), and status-event dates produced by `code/extract_cohort.py` |

`code/extract_cohort.py` reads `data/analytic_cohort.parquet` plus a read-only coredb
connection and writes the `data/raw/` inputs consumed by `code/analysis.py`. Both
directories are git-ignored.

## Key Columns in analytic_cohort.parquet

| Column | Type | Description |
|---|---|---|
| `way_id` | str | De-identified patient identifier |
| `treated` | int | 1 = care team engaged, 0 = control |
| `baseline_a1c` | float | HbA1c within 6 months before index date (analysis recomputes mean and last value from `data/raw/` labs) |
| `followup_a1c` | float | HbA1c after index date (analysis recomputes the mean of values 90-365 days after index) |
| `a1c_change` | float | followup_a1c - baseline_a1c |
| `age` | float | Age at index date |
| `risk_percentile` | float | Composite risk score percentile |
| `comorbidity_count` | int | Count of chronic conditions |
| `pre_ed` | int | ED visits in 6 months before index |
| `pre_ip` | int | Inpatient admissions in 6 months before index |
| `pre_pcp` | int | PCP visits in 6 months before index |
| `has_bh` | int | Behavioral health diagnosis indicator |
| `has_htn` | int | Hypertension indicator |
| `has_chf` | int | Heart failure indicator |
| `has_pulm` | int | Pulmonary disease indicator |
| `polypharmacy` | int | Polypharmacy indicator |
| `high_ed_ip` | int | High ED/inpatient utilizer indicator |
| `index_date` | date | Care team activation (treated) or first targeting (control) |
| `followup_a1c_date` | date | Date of follow-up HbA1c measurement |

## Provenance

Data were extracted from a Medicaid managed care organization's analytics warehouse (Tuva Health data model) linking administrative claims, EHR laboratory results, and care management platform records. HbA1c labs are identified by `source_description ILIKE '%a1c%'`, restricted to 3-20%, and de-duplicated to one value per person-day. `code/extract_cohort.py` documents the queries against `dbt_tuva_core.lab_result`, `dbt.stg_patient_spine_zero_date`, `dbt.im__chronic_conditions_dashboard`, and `dbt.stg__status_events`.

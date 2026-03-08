# Data Note

The `data/` directory contains de-identified patient-level Medicaid claims and laboratory data. These files are excluded from version control (`.gitignore`) because they contain protected health information (PHI)-adjacent data derived from Medicaid administrative claims and electronic health records.

## Required Files

| File | Rows | Description |
|---|---|---|
| `analytic_cohort.parquet` | 372 | Full diabetes cohort with baseline/follow-up A1c, treatment assignment, covariates |
| `matched_treated.parquet` | 114 | Propensity score-matched treated group |
| `matched_control.parquet` | 114 | Propensity score-matched control group |

## Key Columns in analytic_cohort.parquet

| Column | Type | Description |
|---|---|---|
| `way_id` | str | De-identified patient identifier |
| `treated` | int | 1 = care team engaged, 0 = control |
| `baseline_a1c` | float | Most recent HbA1c within 6 months before index date |
| `followup_a1c` | float | Most recent HbA1c >=90 days after index date, within 6 months |
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

Data were extracted from a Medicaid managed care organization's analytics warehouse (Tuva Health data model) linking administrative claims, EHR laboratory results, and care management platform records. The extraction was performed on AWS SageMaker with database access via SSM-managed credentials.

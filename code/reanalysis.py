"""
Reanalysis: CHW Diabetes A1c Study.

  1. DML on full cohort (primary) + matched cohort (sensitivity)
  2. Propensity score overlap/positivity assessment
  3. Formal heterogeneity test (BLP + GATE with CIs)
  4. Regression-to-the-mean sensitivity analysis
  5. Covariate balance assessment (SMD < 0.10)
  6. Negative control outcome (pre-period placebo test)
"""

import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from econml.dml import LinearDML, CausalForestDML
from lightgbm import LGBMRegressor, LGBMClassifier
from scipy import stats
import statsmodels.api as sm

# ============================================================
# 0. Load data
# ============================================================
BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
OUTPUT_DIR = BASE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

cohort = pd.read_parquet(DATA_DIR / 'analytic_cohort.parquet')
matched_t = pd.read_parquet(DATA_DIR / 'matched_treated.parquet')
matched_c = pd.read_parquet(DATA_DIR / 'matched_control.parquet')

COVARIATES = [
    'age', 'baseline_a1c', 'risk_percentile', 'comorbidity_count',
    'pre_ed', 'pre_ip', 'pre_pcp', 'has_bh', 'has_htn',
    'has_chf', 'has_pulm', 'polypharmacy', 'high_ed_ip'
]

Y_col = 'a1c_change'
T_col = 'treated'

# Prepare full cohort
full = cohort.dropna(subset=COVARIATES + [Y_col]).copy()
print(f"Full cohort after dropping NAs: {len(full)} (treated={full[T_col].sum()}, control={(1-full[T_col]).sum()})")

X_full = full[COVARIATES].values.astype(float)
Y_full = full[Y_col].values.astype(float)
T_full = full[T_col].values.astype(float)

# Prepare matched cohort
matched = pd.concat([matched_t, matched_c], ignore_index=True)
matched = matched.dropna(subset=COVARIATES + [Y_col])
X_matched = matched[COVARIATES].values.astype(float)
Y_matched = matched[Y_col].values.astype(float)
T_matched = matched[T_col].values.astype(float)
print(f"Matched cohort: {len(matched)} (treated={matched[T_col].sum()}, control={(1-matched[T_col]).sum()})")

results = {}

# ============================================================
# 1. PROPENSITY SCORE OVERLAP ASSESSMENT
# ============================================================
print("\n=== 1. PROPENSITY SCORE OVERLAP ===")
ps_model = LogisticRegression(max_iter=1000, C=1.0)
ps_model.fit(X_full, T_full)
ps = ps_model.predict_proba(X_full)[:, 1]
full['ps'] = ps

ps_treated = ps[T_full == 1]
ps_control = ps[T_full == 0]

print(f"PS treated: min={ps_treated.min():.3f}, max={ps_treated.max():.3f}, mean={ps_treated.mean():.3f}")
print(f"PS control: min={ps_control.min():.3f}, max={ps_control.max():.3f}, mean={ps_control.mean():.3f}")

# Common support
cs_min = max(ps_treated.min(), ps_control.min())
cs_max = min(ps_treated.max(), ps_control.max())
in_cs = (ps >= cs_min) & (ps <= cs_max)
print(f"Common support: [{cs_min:.3f}, {cs_max:.3f}], {in_cs.sum()}/{len(full)} in support ({100*in_cs.mean():.1f}%)")

# Trimmed cohort (remove extreme PS < 0.05 or > 0.95)
trimmed = (ps >= 0.05) & (ps <= 0.95)
n_trimmed = (~trimmed).sum()
print(f"Trimming PS <0.05 or >0.95: {n_trimmed} removed, {trimmed.sum()} remain")

results['propensity_score'] = {
    'ps_treated_min': round(float(ps_treated.min()), 3),
    'ps_treated_max': round(float(ps_treated.max()), 3),
    'ps_treated_mean': round(float(ps_treated.mean()), 3),
    'ps_control_min': round(float(ps_control.min()), 3),
    'ps_control_max': round(float(ps_control.max()), 3),
    'ps_control_mean': round(float(ps_control.mean()), 3),
    'common_support_min': round(float(cs_min), 3),
    'common_support_max': round(float(cs_max), 3),
    'pct_in_common_support': round(100 * float(in_cs.mean()), 1),
    'n_trimmed': int(n_trimmed),
    'n_after_trimming': int(trimmed.sum())
}

# ============================================================
# 2. COVARIATE BALANCE (strict SMD < 0.10 threshold)
# ============================================================
print("\n=== 2. COVARIATE BALANCE ===")
def compute_smd(df, covs, t_col):
    t = df[df[t_col] == 1]
    c = df[df[t_col] == 0]
    smds = {}
    for cov in covs:
        mean_t = t[cov].mean()
        mean_c = c[cov].mean()
        sd_t = t[cov].std()
        sd_c = c[cov].std()
        pooled_sd = np.sqrt((sd_t**2 + sd_c**2) / 2)
        smd = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0
        smds[cov] = round(abs(smd), 3)
    return smds

smd_unmatched = compute_smd(full, COVARIATES, T_col)
smd_matched = compute_smd(matched, COVARIATES, T_col)

print("Covariate | Unmatched SMD | Matched SMD | Pass (0.10)")
for cov in COVARIATES:
    u = smd_unmatched[cov]
    m = smd_matched[cov]
    flag = "PASS" if m < 0.10 else f"FAIL ({m:.3f})"
    print(f"  {cov:25s} | {u:.3f} | {m:.3f} | {flag}")

# Count how many fail at 0.10 vs 0.20
fail_010 = sum(1 for v in smd_matched.values() if v >= 0.10)
fail_020 = sum(1 for v in smd_matched.values() if v >= 0.20)
print(f"\nSMDs >= 0.10: {fail_010}/{len(COVARIATES)}")
print(f"SMDs >= 0.20: {fail_020}/{len(COVARIATES)}")

results['covariate_balance'] = {
    'unmatched': smd_unmatched,
    'matched': smd_matched,
    'n_fail_010': fail_010,
    'n_fail_020': fail_020
}

# ============================================================
# 3. PRIMARY ANALYSIS: DML ON FULL COHORT
# ============================================================
print("\n=== 3. DML ON FULL COHORT (PRIMARY) ===")

# Use trimmed cohort for DML
X_trim = X_full[trimmed]
Y_trim = Y_full[trimmed]
T_trim = T_full[trimmed]

dml_full = LinearDML(
    model_y=LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, verbose=-1),
    model_t=LogisticRegression(max_iter=1000),
    discrete_treatment=True,
    cv=5,
    random_state=42
)
dml_full.fit(Y_trim, T_trim, X=None, W=X_trim)

ate_full = dml_full.ate()
ate_full_ci = dml_full.ate_interval(alpha=0.05)
print(f"DML Full Cohort ATE: {ate_full:.3f} (95% CI: {ate_full_ci[0]:.3f} to {ate_full_ci[1]:.3f})")

# Pooled SD for Cohen's d
pooled_sd_full = np.sqrt(
    (np.var(Y_trim[T_trim == 1]) + np.var(Y_trim[T_trim == 0])) / 2
)
d_full = abs(ate_full) / pooled_sd_full
d_ci_lo = abs(ate_full_ci[1]) / pooled_sd_full  # CI bound closest to 0
d_ci_hi = abs(ate_full_ci[0]) / pooled_sd_full
print(f"Cohen's d: {d_full:.2f} (95% CI: {d_ci_lo:.2f} to {d_ci_hi:.2f})")

results['dml_full_cohort'] = {
    'n': int(trimmed.sum()),
    'n_treated': int(T_trim.sum()),
    'n_control': int((1-T_trim).sum()),
    'ate': round(float(ate_full), 3),
    'ci_lower': round(float(ate_full_ci[0]), 3),
    'ci_upper': round(float(ate_full_ci[1]), 3),
    'cohens_d': round(float(d_full), 2),
    'cohens_d_ci_lower': round(float(d_ci_lo), 2),
    'cohens_d_ci_upper': round(float(d_ci_hi), 2),
    'pooled_sd': round(float(pooled_sd_full), 3)
}

# ============================================================
# 4. SENSITIVITY: DML ON MATCHED COHORT
# ============================================================
print("\n=== 4. DML ON MATCHED COHORT (SENSITIVITY) ===")
dml_matched = LinearDML(
    model_y=LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, verbose=-1),
    model_t=LogisticRegression(max_iter=1000),
    discrete_treatment=True,
    cv=5,
    random_state=42
)
dml_matched.fit(Y_matched, T_matched, X=None, W=X_matched)
ate_matched = dml_matched.ate()
ate_matched_ci = dml_matched.ate_interval(alpha=0.05)
print(f"DML Matched ATE: {ate_matched:.3f} (95% CI: {ate_matched_ci[0]:.3f} to {ate_matched_ci[1]:.3f})")

pooled_sd_matched = np.sqrt(
    (np.var(Y_matched[T_matched == 1]) + np.var(Y_matched[T_matched == 0])) / 2
)
d_matched = abs(ate_matched) / pooled_sd_matched

results['dml_matched'] = {
    'n': int(len(matched)),
    'ate': round(float(ate_matched), 3),
    'ci_lower': round(float(ate_matched_ci[0]), 3),
    'ci_upper': round(float(ate_matched_ci[1]), 3),
    'cohens_d': round(float(d_matched), 2),
    'pooled_sd': round(float(pooled_sd_matched), 3)
}

# ============================================================
# 5. GRF ON FULL COHORT WITH FORMAL HETEROGENEITY TEST
# ============================================================
print("\n=== 5. GRF ON FULL COHORT WITH HETEROGENEITY TESTS ===")

# Use full trimmed cohort for GRF
cf = CausalForestDML(
    model_y=LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, verbose=-1),
    model_t=LogisticRegression(max_iter=1000),
    discrete_treatment=True,
    cv=5,
    n_estimators=1000,
    min_samples_leaf=10,
    random_state=42
)

# For CATE estimation, we need X (heterogeneity variables) and W (confounders)
# X = covariates that may drive heterogeneity
# W = all covariates for confounding adjustment
cf.fit(Y_trim, T_trim, X=X_trim, W=X_trim)

# ATE from GRF
grf_ate = cf.ate(X_trim)
grf_ate_ci = cf.ate_interval(X_trim, alpha=0.05)
print(f"GRF Full Cohort ATE: {grf_ate:.3f} (95% CI: {grf_ate_ci[0]:.3f} to {grf_ate_ci[1]:.3f})")

# CATEs
cates = cf.effect(X_trim)

# Feature importance
feat_imp = dict(zip(COVARIATES, [round(float(x), 3) for x in cf.feature_importances_]))
print(f"Feature importance: {feat_imp}")

# CATE quartiles with proper CIs (GATE - Group Average Treatment Effects)
cate_q = pd.qcut(cates, 4, labels=['Q1 (highest benefit)', 'Q2', 'Q3', 'Q4 (lowest benefit)'])
gate_results = {}
full_trim = full[trimmed].copy()
full_trim['cate'] = cates
full_trim['cate_quartile'] = np.array(cate_q)

for q in ['Q1 (highest benefit)', 'Q2', 'Q3', 'Q4 (lowest benefit)']:
    mask = cate_q == q
    q_cates = cates[mask]
    q_data = full_trim[full_trim['cate_quartile'] == q]

    # Observed change for calibration
    obs_change_treated = q_data[q_data[T_col] == 1][Y_col].mean() if (q_data[T_col] == 1).any() else np.nan
    obs_change_control = q_data[q_data[T_col] == 0][Y_col].mean() if (q_data[T_col] == 0).any() else np.nan

    # GATE: get CI from the forest for this subgroup
    gate_point = cf.ate(X_trim[mask])
    gate_ci = cf.ate_interval(X_trim[mask], alpha=0.05)

    gate_results[q] = {
        'n': int(mask.sum()),
        'n_treated': int((q_data[T_col] == 1).sum()),
        'n_control': int((q_data[T_col] == 0).sum()),
        'mean_predicted_cate': round(float(q_cates.mean()), 3),
        'gate': round(float(gate_point), 3),
        'gate_ci_lower': round(float(gate_ci[0]), 3),
        'gate_ci_upper': round(float(gate_ci[1]), 3),
        'mean_baseline_a1c': round(float(q_data['baseline_a1c'].mean()), 1),
        'observed_change_treated': round(float(obs_change_treated), 3) if not np.isnan(obs_change_treated) else None,
        'observed_change_control': round(float(obs_change_control), 3) if not np.isnan(obs_change_control) else None,
        'mean_age': round(float(q_data['age'].mean()), 1)
    }
    print(f"  {q}: N={mask.sum()}, GATE={gate_point:.3f} ({gate_ci[0]:.3f}, {gate_ci[1]:.3f}), "
          f"baseline A1c={q_data['baseline_a1c'].mean():.1f}")

# Formal heterogeneity test: Best Linear Projection (BLP)
# Regress the outcome residuals on treatment residuals * predicted CATE
# H0: coefficient on treatment_resid * centered_CATE = 0
# This tests whether predicted heterogeneity predicts actual heterogeneity
print("\n--- Formal Heterogeneity Test (BLP) ---")
# Use the forest's predicted CATEs
cate_centered = cates - cates.mean()

# First stage residuals
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(
    LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, verbose=-1),
    X_trim, Y_trim, cv=5
)
t_pred = cross_val_predict(
    LogisticRegression(max_iter=1000),
    X_trim, T_trim, cv=5, method='predict_proba'
)[:, 1]

y_resid = Y_trim - y_pred
t_resid = T_trim - t_pred

# BLP regression: y_resid ~ t_resid + t_resid * cate_centered
blp_X = np.column_stack([t_resid, t_resid * cate_centered])
blp_model = sm.OLS(y_resid, sm.add_constant(blp_X)).fit(cov_type='HC1')
blp_coef = blp_model.params[2]  # coefficient on t_resid * cate_centered
blp_pval = blp_model.pvalues[2]
blp_ci = blp_model.conf_int(alpha=0.05)[2]
print(f"BLP heterogeneity coefficient: {blp_coef:.3f} (95% CI: {blp_ci[0]:.3f} to {blp_ci[1]:.3f}), p={blp_pval:.4f}")
print(f"Interpretation: coefficient > 0 and p < 0.05 => significant treatment effect heterogeneity")

# Also report the ATE coefficient from BLP
blp_ate_coef = blp_model.params[1]
blp_ate_pval = blp_model.pvalues[1]
print(f"BLP ATE coefficient: {blp_ate_coef:.3f}, p={blp_ate_pval:.4f}")

results['grf_full_cohort'] = {
    'n': int(trimmed.sum()),
    'ate': round(float(grf_ate), 3),
    'ci_lower': round(float(grf_ate_ci[0]), 3),
    'ci_upper': round(float(grf_ate_ci[1]), 3),
    'feature_importance': feat_imp,
    'gate_quartiles': gate_results,
    'blp_heterogeneity_coef': round(float(blp_coef), 3),
    'blp_heterogeneity_pval': round(float(blp_pval), 4),
    'blp_heterogeneity_ci': [round(float(blp_ci[0]), 3), round(float(blp_ci[1]), 3)],
    'blp_ate_coef': round(float(blp_ate_coef), 3),
    'blp_ate_pval': round(float(blp_ate_pval), 4)
}

# ============================================================
# 6. REGRESSION TO THE MEAN SENSITIVITY
# ============================================================
print("\n=== 6. REGRESSION TO THE MEAN SENSITIVITY ===")
# If regression to the mean drives the heterogeneity, then:
# (a) The ATE should attenuate substantially when baseline A1c is included
# (b) The treatment × baseline_a1c interaction should fully explain heterogeneity

# Test: parametric interaction model
# y = b0 + b1*T + b2*baseline_a1c + b3*T*baseline_a1c + other covariates
trim_df = full[trimmed].copy()
X_rtm = trim_df[COVARIATES].copy()
X_rtm['treatment'] = trim_df[T_col].values
X_rtm['treatment_x_baseline_a1c'] = trim_df[T_col].values * trim_df['baseline_a1c'].values
X_rtm = sm.add_constant(X_rtm)

rtm_model = sm.OLS(trim_df[Y_col].values, X_rtm.astype(float)).fit(cov_type='HC1')
treatment_coef = rtm_model.params['treatment']
treatment_pval = rtm_model.pvalues['treatment']
interaction_coef = rtm_model.params['treatment_x_baseline_a1c']
interaction_pval = rtm_model.pvalues['treatment_x_baseline_a1c']
treatment_ci = rtm_model.conf_int(alpha=0.05).loc['treatment']
interaction_ci = rtm_model.conf_int(alpha=0.05).loc['treatment_x_baseline_a1c']

print(f"Treatment main effect: {treatment_coef:.3f} (95% CI: {treatment_ci[0]:.3f} to {treatment_ci[1]:.3f}), p={treatment_pval:.4f}")
print(f"Treatment × baseline A1c: {interaction_coef:.3f} (95% CI: {interaction_ci[0]:.3f} to {interaction_ci[1]:.3f}), p={interaction_pval:.4f}")
print(f"Interpretation: Significant negative interaction means treatment benefit increases with baseline A1c")

results['rtm_sensitivity'] = {
    'treatment_coef': round(float(treatment_coef), 3),
    'treatment_ci': [round(float(treatment_ci[0]), 3), round(float(treatment_ci[1]), 3)],
    'treatment_pval': round(float(treatment_pval), 4),
    'interaction_coef': round(float(interaction_coef), 3),
    'interaction_ci': [round(float(interaction_ci[0]), 3), round(float(interaction_ci[1]), 3)],
    'interaction_pval': round(float(interaction_pval), 4)
}

# ============================================================
# 7. NEGATIVE CONTROL: PRE-PERIOD PLACEBO TEST
# ============================================================
print("\n=== 7. NEGATIVE CONTROL: PRE-PERIOD PCP VISITS ===")
# If the treatment effect is real, there should be no "effect" on pre-period outcomes
# Use pre_pcp as a negative control outcome (should not differ)
dml_placebo = LinearDML(
    model_y=LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, verbose=-1),
    model_t=LogisticRegression(max_iter=1000),
    discrete_treatment=True,
    cv=5,
    random_state=42
)
# Covariates for placebo: exclude pre_pcp from W since it's the outcome
placebo_covs = [c for c in COVARIATES if c != 'pre_pcp']
W_placebo = full[trimmed][placebo_covs].values.astype(float)
Y_placebo = full[trimmed]['pre_pcp'].values.astype(float)

dml_placebo.fit(Y_placebo, T_trim, X=None, W=W_placebo)
placebo_ate = dml_placebo.ate()
placebo_ci = dml_placebo.ate_interval(alpha=0.05)
print(f"Placebo (pre-period PCP visits) ATE: {placebo_ate:.3f} (95% CI: {placebo_ci[0]:.3f} to {placebo_ci[1]:.3f})")
print(f"Interpretation: CI including 0 supports no pre-treatment differences")

results['negative_control'] = {
    'outcome': 'pre_period_pcp_visits',
    'ate': round(float(placebo_ate), 3),
    'ci_lower': round(float(placebo_ci[0]), 3),
    'ci_upper': round(float(placebo_ci[1]), 3),
    'null_included': bool(placebo_ci[0] <= 0 <= placebo_ci[1])
}

# ============================================================
# 8. E-VALUE ON FULL COHORT PRIMARY ESTIMATE
# ============================================================
print("\n=== 8. E-VALUE ===")
d_for_evalue = abs(ate_full) / pooled_sd_full
rr = np.exp(0.91 * d_for_evalue)
e_value_point = rr + np.sqrt(rr * (rr - 1))

# For CI bound (use the bound closest to null)
ci_bound = max(ate_full_ci[0], ate_full_ci[1], key=lambda x: abs(x))
ci_bound_closest_null = min(abs(ate_full_ci[0]), abs(ate_full_ci[1]))
d_ci = ci_bound_closest_null / pooled_sd_full
rr_ci = np.exp(0.91 * d_ci)
e_value_ci = rr_ci + np.sqrt(rr_ci * (rr_ci - 1)) if rr_ci > 1 else 1.0

print(f"E-value (point): {e_value_point:.2f}")
print(f"E-value (CI bound): {e_value_ci:.2f}")

results['e_value'] = {
    'point': round(float(e_value_point), 2),
    'ci_bound': round(float(e_value_ci), 2),
    'rr_point': round(float(rr), 2),
    'cohens_d': round(float(d_for_evalue), 3)
}

# ============================================================
# 9. DESCRIPTIVE STATS FOR REVISED MANUSCRIPT
# ============================================================
print("\n=== 9. DESCRIPTIVE STATS ===")
# Full cohort descriptives
for group, label in [(1, 'Treated'), (0, 'Control')]:
    g = full[full[T_col] == group]
    print(f"\n{label} (N={len(g)}):")
    print(f"  Age: {g['age'].mean():.1f} (SD {g['age'].std():.1f})")
    print(f"  Baseline A1c: {g['baseline_a1c'].mean():.2f} (SD {g['baseline_a1c'].std():.2f})")
    print(f"  Follow-up A1c: {g['followup_a1c'].mean():.2f} (SD {g['followup_a1c'].std():.2f})")
    print(f"  A1c change: {g['a1c_change'].mean():.2f}")
    print(f"  Comorbidity count: {g['comorbidity_count'].mean():.1f} (SD {g['comorbidity_count'].std():.1f})")
    print(f"  Behavioral health: {g['has_bh'].mean()*100:.1f}%")

results['descriptives'] = {
    'full_cohort_n': int(len(full)),
    'full_treated_n': int(full[T_col].sum()),
    'full_control_n': int((1-full[T_col]).sum()),
    'trimmed_n': int(trimmed.sum()),
    'treated_baseline_a1c_mean': round(float(full[full[T_col]==1]['baseline_a1c'].mean()), 2),
    'treated_baseline_a1c_sd': round(float(full[full[T_col]==1]['baseline_a1c'].std()), 2),
    'treated_followup_a1c_mean': round(float(full[full[T_col]==1]['followup_a1c'].mean()), 2),
    'treated_a1c_change_mean': round(float(full[full[T_col]==1]['a1c_change'].mean()), 2),
    'control_baseline_a1c_mean': round(float(full[full[T_col]==0]['baseline_a1c'].mean()), 2),
    'control_baseline_a1c_sd': round(float(full[full[T_col]==0]['baseline_a1c'].std()), 2),
    'control_followup_a1c_mean': round(float(full[full[T_col]==0]['followup_a1c'].mean()), 2),
    'control_a1c_change_mean': round(float(full[full[T_col]==0]['a1c_change'].mean()), 2),
}

# ============================================================
# SAVE ALL RESULTS
# ============================================================
print("\n=== SAVING RESULTS ===")
out = OUTPUT_DIR / 'reanalysis_results.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"Results saved to {out}")

# Print summary
print("\n" + "="*60)
print("SUMMARY OF KEY RESULTS")
print("="*60)
print(f"Primary (DML, full cohort, N={results['dml_full_cohort']['n']}): "
      f"ATE = {results['dml_full_cohort']['ate']:.3f} "
      f"(95% CI: {results['dml_full_cohort']['ci_lower']:.3f} to {results['dml_full_cohort']['ci_upper']:.3f})")
print(f"Sensitivity (DML, matched, N={results['dml_matched']['n']}): "
      f"ATE = {results['dml_matched']['ate']:.3f} "
      f"(95% CI: {results['dml_matched']['ci_lower']:.3f} to {results['dml_matched']['ci_upper']:.3f})")
print(f"GRF (full cohort): ATE = {results['grf_full_cohort']['ate']:.3f} "
      f"(95% CI: {results['grf_full_cohort']['ci_lower']:.3f} to {results['grf_full_cohort']['ci_upper']:.3f})")
print(f"BLP heterogeneity test: coef = {results['grf_full_cohort']['blp_heterogeneity_coef']:.3f}, "
      f"p = {results['grf_full_cohort']['blp_heterogeneity_pval']:.4f}")
print(f"E-value: {results['e_value']['point']:.2f} (CI bound: {results['e_value']['ci_bound']:.2f})")
print(f"Negative control (pre-PCP): ATE = {results['negative_control']['ate']:.3f} "
      f"(null included: {results['negative_control']['null_included']})")
print(f"RTM interaction: coef = {results['rtm_sensitivity']['interaction_coef']:.3f}, "
      f"p = {results['rtm_sensitivity']['interaction_pval']:.4f}")

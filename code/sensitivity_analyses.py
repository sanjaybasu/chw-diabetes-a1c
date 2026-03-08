"""
Sensitivity analyses for CHW diabetes A1c manuscript.

Analyses:
  1a. Time-adjusted DML (days_to_followup as additional covariate)
  1b. Subgroup DML: baseline HbA1c >= 8.0
  1c. Binary outcome: achieved glycemic control (A1c < 8) among uncontrolled at baseline
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from econml.dml import LinearDML, CausalForestDML
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats

warnings.filterwarnings("ignore")

# --- Paths ---
BASE = Path(__file__).resolve().parents[1]
PARQUET = BASE / "data" / "analytic_cohort.parquet"
OUTPUT_DIR = BASE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Covariates (same as reanalysis.py) ---
COVARIATES = [
    "age", "baseline_a1c", "risk_percentile", "comorbidity_count",
    "pre_ed", "pre_ip", "pre_pcp", "has_bh", "has_htn",
    "has_chf", "has_pulm", "polypharmacy", "high_ed_ip",
]

PS_LO, PS_HI = 0.05, 0.95


def load_and_trim(extra_covariates=None):
    """Load parquet, apply PS trimming, return ready-to-use arrays."""
    cohort = pd.read_parquet(PARQUET)
    cohort = cohort.dropna(subset=COVARIATES + ["a1c_change"]).copy()

    # PS trimming
    from sklearn.linear_model import LogisticRegression as LR
    X_ps = cohort[COVARIATES].values.astype(float)
    T_ps = cohort["treated"].values.astype(float)
    ps = LR(max_iter=1000).fit(X_ps, T_ps).predict_proba(X_ps)[:, 1]
    mask = (ps >= PS_LO) & (ps <= PS_HI)
    cohort = cohort[mask].copy()

    covs = COVARIATES + (extra_covariates or [])
    return cohort, covs


def run_dml(Y, T, W, label=""):
    """Run LinearDML, return dict of results."""
    dml = LinearDML(
        model_y=LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, verbose=-1),
        model_t=LogisticRegression(max_iter=1000),
        discrete_treatment=True, cv=5, random_state=42,
    )
    dml.fit(Y, T, X=None, W=W)
    ate = float(dml.ate())
    ci = dml.ate_interval(alpha=0.05)
    ci_lo, ci_hi = float(ci[0]), float(ci[1])
    pooled_sd = float(np.sqrt((Y[T == 1].var() + Y[T == 0].var()) / 2))
    d = abs(ate) / pooled_sd if pooled_sd > 0 else 0
    # p-value from Wald test
    se = (ci_hi - ci_lo) / (2 * 1.96)
    z = ate / se if se > 0 else 0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    print(f"  [{label}] ATE={ate:.3f} (95% CI: {ci_lo:.3f} to {ci_hi:.3f}), d={d:.3f}, P={p:.4f}")
    return {
        "label": label, "ate": round(ate, 4), "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4), "cohens_d": round(d, 3), "p_value": round(p, 4),
        "n": int(len(Y)), "n_treated": int(T.sum()), "n_control": int((1 - T).sum()),
        "pooled_sd": round(pooled_sd, 4),
    }


def analysis_1a_time_adjusted():
    """DML with days_to_followup as additional covariate."""
    print("\n=== 1a. Time-adjusted DML ===")
    cohort, covs = load_and_trim()
    cohort["days_to_followup"] = (
        pd.to_datetime(cohort["followup_a1c_date"]) - pd.to_datetime(cohort["index_date"])
    ).dt.days.astype(float)
    covs_aug = covs + ["days_to_followup"]
    cohort = cohort.dropna(subset=covs_aug + ["a1c_change"])
    W = cohort[covs_aug].values.astype(float)
    Y = cohort["a1c_change"].values.astype(float)
    T = cohort["treated"].values.astype(float)
    result = run_dml(Y, T, W, label="DML, time-adjusted")
    result["mean_days_to_followup_treated"] = round(float(cohort.loc[cohort["treated"] == 1, "days_to_followup"].mean()), 1)
    result["mean_days_to_followup_control"] = round(float(cohort.loc[cohort["treated"] == 0, "days_to_followup"].mean()), 1)
    return result


def analysis_1b_subgroup_uncontrolled():
    """DML restricted to baseline HbA1c >= 8.0."""
    print("\n=== 1b. Subgroup DML: baseline HbA1c >= 8.0 ===")
    cohort, covs = load_and_trim()
    sub = cohort[cohort["baseline_a1c"] >= 8.0].copy()
    # Re-estimate PS and trim within subgroup
    from sklearn.linear_model import LogisticRegression as LR
    X_ps = sub[covs].values.astype(float)
    T_ps = sub["treated"].values.astype(float)
    if T_ps.sum() < 5 or (1 - T_ps).sum() < 5:
        print("  Insufficient sample for subgroup analysis")
        return {"label": "DML, baseline HbA1c >= 8.0", "error": "insufficient sample"}
    ps = LR(max_iter=1000).fit(X_ps, T_ps).predict_proba(X_ps)[:, 1]
    mask = (ps >= PS_LO) & (ps <= PS_HI)
    sub = sub[mask].copy()
    W = sub[covs].values.astype(float)
    Y = sub["a1c_change"].values.astype(float)
    T = sub["treated"].values.astype(float)
    result = run_dml(Y, T, W, label="DML, baseline A1c >= 8.0")
    result["mean_baseline_a1c"] = round(float(sub["baseline_a1c"].mean()), 2)
    return result


def analysis_1b2_subgroup_poor_control():
    """DML restricted to baseline HbA1c >= 9.0 (HEDIS poor control)."""
    print("\n=== 1b2. Subgroup DML: baseline HbA1c >= 9.0 ===")
    cohort, covs = load_and_trim()
    sub = cohort[cohort["baseline_a1c"] >= 9.0].copy()
    from sklearn.linear_model import LogisticRegression as LR
    X_ps = sub[covs].values.astype(float)
    T_ps = sub["treated"].values.astype(float)
    if T_ps.sum() < 5 or (1 - T_ps).sum() < 5:
        print("  Insufficient sample for subgroup analysis")
        return {"label": "DML, baseline HbA1c >= 9.0", "error": "insufficient sample"}
    ps = LR(max_iter=1000).fit(X_ps, T_ps).predict_proba(X_ps)[:, 1]
    mask = (ps >= PS_LO) & (ps <= PS_HI)
    sub = sub[mask].copy()
    if sub["treated"].sum() < 5:
        print("  Insufficient treated after trimming")
        return {"label": "DML, baseline HbA1c >= 9.0", "error": "insufficient after trimming"}
    W = sub[covs].values.astype(float)
    Y = sub["a1c_change"].values.astype(float)
    T = sub["treated"].values.astype(float)
    result = run_dml(Y, T, W, label="DML, baseline A1c >= 9.0")
    result["mean_baseline_a1c"] = round(float(sub["baseline_a1c"].mean()), 2)
    return result


def analysis_1c_binary_control():
    """Binary outcome: achieved glycemic control (A1c < 8) among those uncontrolled at baseline."""
    print("\n=== 1c. Binary outcome: achieved control (A1c < 8) | baseline >= 8 ===")
    cohort, covs = load_and_trim()
    sub = cohort[cohort["baseline_a1c"] >= 8.0].copy()
    sub["achieved_control"] = (sub["followup_a1c"] < 8.0).astype(float)
    from sklearn.linear_model import LogisticRegression as LR
    X_ps = sub[covs].values.astype(float)
    T_ps = sub["treated"].values.astype(float)
    if T_ps.sum() < 5 or (1 - T_ps).sum() < 5:
        return {"label": "Binary: achieved control", "error": "insufficient sample"}
    ps = LR(max_iter=1000).fit(X_ps, T_ps).predict_proba(X_ps)[:, 1]
    mask = (ps >= PS_LO) & (ps <= PS_HI)
    sub = sub[mask].copy()
    W = sub[covs].values.astype(float)
    Y = sub["achieved_control"].values.astype(float)
    T = sub["treated"].values.astype(float)

    dml = LinearDML(
        model_y=LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, verbose=-1),
        model_t=LogisticRegression(max_iter=1000),
        discrete_treatment=True, cv=5, random_state=42,
    )
    dml.fit(Y, T, X=None, W=W)
    ate = float(dml.ate())
    ci = dml.ate_interval(alpha=0.05)
    ci_lo, ci_hi = float(ci[0]), float(ci[1])
    se = (ci_hi - ci_lo) / (2 * 1.96)
    z = ate / se if se > 0 else 0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))

    # Crude rates
    rate_treated = float(sub.loc[sub["treated"] == 1, "achieved_control"].mean())
    rate_control = float(sub.loc[sub["treated"] == 0, "achieved_control"].mean())
    nnt = round(1 / abs(ate), 1) if abs(ate) > 0.001 else float("inf")

    print(f"  [Binary] ATE={ate:.3f} (95% CI: {ci_lo:.3f} to {ci_hi:.3f}), P={p:.4f}")
    print(f"  Rate treated={rate_treated:.3f}, control={rate_control:.3f}, NNT={nnt}")

    return {
        "label": "Binary: achieved control (A1c < 8) | baseline >= 8",
        "ate": round(ate, 4), "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
        "p_value": round(p, 4),
        "rate_treated": round(rate_treated, 3), "rate_control": round(rate_control, 3),
        "nnt": nnt,
        "n": int(len(Y)), "n_treated": int(T.sum()), "n_control": int((1 - T).sum()),
    }


if __name__ == "__main__":
    results = {}

    results["time_adjusted"] = analysis_1a_time_adjusted()
    results["subgroup_a1c_ge8"] = analysis_1b_subgroup_uncontrolled()
    results["subgroup_a1c_ge9"] = analysis_1b2_subgroup_poor_control()
    results["binary_control"] = analysis_1c_binary_control()

    out = OUTPUT_DIR / "sensitivity_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")

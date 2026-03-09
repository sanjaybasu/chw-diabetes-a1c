"""
Exploratory analysis: association between specific care team activities
and HbA1c change within the treated group.

Hypothesis-generating only (N~93 with encounter documentation).
Confounding by indication is a concern: sicker patients may receive
more encounters. Adjusted for baseline A1c, age, and comorbidity count.
"""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
OUTPUT_DIR = BASE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# External data paths (encounter notes and goals from care management platform)
REAL_INPUTS = BASE.parents[1] / "data" / "real_inputs"
ENCOUNTER_NOTES_CSV = REAL_INPUTS / "notes" / "encounter notes.csv"
MEMBER_GOALS_CSV = REAL_INPUTS / "member_goals.csv"
MEMBER_ATTRS = REAL_INPUTS / "member_attributes.parquet"

# ── Load analytic cohort ──────────────────────────────────────────────
cohort = pd.read_parquet(DATA_DIR / "analytic_cohort.parquet")
treated = cohort[cohort["treated"] == 1].copy()
print(f"Treated cohort: N = {len(treated)}")

# ── Load encounter notes ──────────────────────────────────────────────
if not ENCOUNTER_NOTES_CSV.exists():
    raise FileNotFoundError(f"Encounter notes CSV not found: {ENCOUNTER_NOTES_CSV}")

enc = pd.read_csv(ENCOUNTER_NOTES_CSV, low_memory=False)
enc = enc.rename(columns={"WaymarkId": "way_id"})
enc["dateOfEncounter"] = pd.to_datetime(enc["dateOfEncounter"], errors="coerce")

# Filter to treated patients with occurred encounters
enc_treated = enc[
    (enc["way_id"].isin(treated["way_id"]))
    & (enc["encounterOccurred"] == "YES")
].copy()
print(f"Occurred encounters for treated patients: {len(enc_treated)}")
print(f"Treated patients with encounters: {enc_treated['way_id'].nunique()}")

# Filter to encounters between index_date and follow-up window
# Join index_date from cohort
enc_treated = enc_treated.merge(
    treated[["way_id", "index_date", "followup_a1c_date"]].rename(
        columns={"index_date": "idx_date"}
    ),
    on="way_id",
    how="left",
)
enc_treated["idx_date"] = pd.to_datetime(enc_treated["idx_date"])
enc_treated["followup_a1c_date"] = pd.to_datetime(enc_treated["followup_a1c_date"])

# Keep encounters during the intervention period (index to follow-up)
enc_window = enc_treated[
    (enc_treated["dateOfEncounter"] >= enc_treated["idx_date"])
    & (enc_treated["dateOfEncounter"] <= enc_treated["followup_a1c_date"])
].copy()
print(f"Encounters in intervention window: {len(enc_window)}")
print(f"Patients with in-window encounters: {enc_window['way_id'].nunique()}")

# ── Role mapping ──────────────────────────────────────────────────────
ROLE_MAP = {
    "PHARMACY_TECH": "pharmacy",
    "PHARMACIST_LEAD": "pharmacy",
    "CHW": "chw",
    "CHW_LEAD": "chw",
    "NATIONAL_CHW_LEAD": "chw",
    "CARE_COORDINATOR": "care_coord",
    "CARE_COORDINATOR_LEAD": "care_coord",
    "THERAPIST": "therapist",
    "THERAPIST_LEAD": "therapist",
}
enc_window["role"] = enc_window["title"].map(ROLE_MAP).fillna("other")

# ── Diabetes-relevant keywords in note text ───────────────────────────
KEYWORDS = {
    "med": r"medication|prescription|refill|rx|pharmacy|prior auth|formulary",
    "insulin": r"insulin",
    "cgm": r"cgm|continuous glucose|monitor|dexcom|libre",
    "nutrition": r"food|nutrition|diet|meal|pantry|snap|wic",
    "diabetes": r"diabetes|diabetic|a1c|hemoglobin|hba1c|glyco|metformin",
    "appointment": r"appointment|pcp|primary care|provider visit",
}

enc_window["text_lower"] = enc_window["text"].fillna("").str.lower()
for kw_name, pattern in KEYWORDS.items():
    enc_window[f"kw_{kw_name}"] = enc_window["text_lower"].str.contains(
        pattern, flags=re.IGNORECASE, na=False
    ).astype(int)

# ── Construct per-patient activity features ───────────────────────────
# Role counts
role_counts = (
    enc_window.groupby(["way_id", "role"])
    .size()
    .unstack(fill_value=0)
    .rename(columns=lambda c: f"n_{c}")
)

# Total encounters
total_enc = enc_window.groupby("way_id").size().rename("n_total_encounters")

# Keyword fractions
kw_cols = [c for c in enc_window.columns if c.startswith("kw_")]
kw_means = enc_window.groupby("way_id")[kw_cols].mean()
kw_means.columns = [c.replace("kw_", "pct_") + "_notes" for c in kw_means.columns]

# Has any diabetes-related note
has_diabetes = (
    enc_window.groupby("way_id")["kw_diabetes"].sum() > 0
).astype(int).rename("has_any_diabetes_note")

# Combine
features = (
    role_counts
    .join(total_enc, how="outer")
    .join(kw_means, how="outer")
    .join(has_diabetes, how="outer")
    .fillna(0)
)

# ── Goals features ────────────────────────────────────────────────────
if MEMBER_GOALS_CSV.exists() and MEMBER_ATTRS.exists():
    goals = pd.read_csv(MEMBER_GOALS_CSV, low_memory=False)
    attrs = pd.read_parquet(MEMBER_ATTRS)

    # Crosswalk: member_id → waymark_patient_number (= way_id)
    xwalk = attrs[["member_id", "waymark_patient_number"]].drop_duplicates()
    goals = goals.merge(xwalk, on="member_id", how="left")
    goals = goals.rename(columns={"waymark_patient_number": "way_id"})
    goals = goals[goals["way_id"].isin(treated["way_id"]) & (goals["deleted"] != True)]

    goal_features = goals.groupby("way_id").agg(
        n_total_goals=("goal_id", "count"),
        n_completed_goals=("status", lambda x: (x == "COMPLETED").sum()),
        n_med_adherence_goals=("category", lambda x: (x == "MEDICATION_ADHERENCE").sum()),
        n_diabetes_goals=("category", lambda x: (x == "DIABETES").sum()),
    )
    goal_features["goal_completion_rate"] = (
        goal_features["n_completed_goals"] / goal_features["n_total_goals"]
    ).fillna(0)

    features = features.join(goal_features, how="outer").fillna(0)
    print(f"Patients with goal data: {(goal_features.index.isin(treated['way_id'])).sum()}")
else:
    print("Warning: goals or member_attributes not found; skipping goal features")

# ── Merge with treated cohort ─────────────────────────────────────────
analysis = treated.set_index("way_id").join(features, how="inner")
print(f"\nFinal analysis cohort: N = {len(analysis)}")
print(f"  (treated patients with in-window encounters)")

# ── Descriptive: Spearman correlations ────────────────────────────────
activity_cols = [c for c in features.columns if c.startswith(("n_", "pct_", "has_", "goal_"))]
# Ensure columns exist in analysis
activity_cols = [c for c in activity_cols if c in analysis.columns]

correlations = {}
for col in activity_cols:
    if analysis[col].std() > 0:
        rho, p = spearmanr(analysis[col], analysis["a1c_change"])
        correlations[col] = {"spearman_rho": round(rho, 3), "p_value": round(p, 4)}

print("\n=== Spearman correlations: activity feature vs. A1c change ===")
for feat, vals in sorted(correlations.items(), key=lambda x: abs(x[1]["spearman_rho"]), reverse=True):
    print(f"  {feat:30s}  rho={vals['spearman_rho']:+.3f}  P={vals['p_value']:.4f}")

# ── Adjusted OLS: pre-specified features ──────────────────────────────
# Pre-specified: n_pharmacy, n_chw, pct_med_notes, n_total_encounters
# Adjusted for: baseline_a1c, age, comorbidity_count
PRESPECIFIED = ["n_pharmacy", "n_chw", "pct_med_notes", "n_total_encounters"]
CONFOUNDERS = ["baseline_a1c", "age", "comorbidity_count"]

# Ensure all columns present
prespec_available = [c for c in PRESPECIFIED if c in analysis.columns]
missing = [c for c in PRESPECIFIED if c not in analysis.columns]
if missing:
    print(f"\nWarning: missing pre-specified features: {missing}")

ols_results = {}
for feat in prespec_available:
    X = analysis[[feat] + CONFOUNDERS].copy()
    X = sm.add_constant(X)
    y = analysis["a1c_change"]

    model = sm.OLS(y, X).fit(cov_type="HC1")
    coef = model.params[feat]
    ci = model.conf_int().loc[feat]
    p = model.pvalues[feat]

    ols_results[feat] = {
        "coefficient": round(coef, 4),
        "ci_lower": round(ci[0], 4),
        "ci_upper": round(ci[1], 4),
        "p_value": round(p, 4),
        "n": int(model.nobs),
    }

print("\n=== Adjusted OLS: individual activity features ===")
print(f"    (adjusted for {', '.join(CONFOUNDERS)}; HC1 SEs)")
for feat, vals in ols_results.items():
    print(
        f"  {feat:30s}  coef={vals['coefficient']:+.4f}  "
        f"95% CI [{vals['ci_lower']:.4f}, {vals['ci_upper']:.4f}]  "
        f"P={vals['p_value']:.4f}"
    )

# ── Joint model: all pre-specified features together ──────────────────
if len(prespec_available) >= 2:
    X_joint = analysis[prespec_available + CONFOUNDERS].copy()
    X_joint = sm.add_constant(X_joint)
    y = analysis["a1c_change"]
    joint_model = sm.OLS(y, X_joint).fit(cov_type="HC1")

    joint_results = {}
    for feat in prespec_available:
        ci = joint_model.conf_int().loc[feat]
        joint_results[feat] = {
            "coefficient": round(joint_model.params[feat], 4),
            "ci_lower": round(ci[0], 4),
            "ci_upper": round(ci[1], 4),
            "p_value": round(joint_model.pvalues[feat], 4),
        }
    joint_results["r_squared"] = round(joint_model.rsquared, 4)
    joint_results["n"] = int(joint_model.nobs)

    print(f"\n=== Joint model (all pre-specified features) ===")
    print(f"    R² = {joint_results['r_squared']:.4f}, N = {joint_results['n']}")
    for feat in prespec_available:
        v = joint_results[feat]
        print(
            f"  {feat:30s}  coef={v['coefficient']:+.4f}  "
            f"95% CI [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]  "
            f"P={v['p_value']:.4f}"
        )
else:
    joint_results = {}

# ── Descriptive summary of activity features ──────────────────────────
desc = {}
for col in activity_cols:
    if col in analysis.columns:
        s = analysis[col]
        desc[col] = {
            "mean": round(s.mean(), 2),
            "sd": round(s.std(), 2),
            "median": round(s.median(), 2),
            "min": round(s.min(), 2),
            "max": round(s.max(), 2),
            "pct_nonzero": round((s > 0).mean() * 100, 1),
        }

# ── Save results ──────────────────────────────────────────────────────
output = {
    "n_treated_total": int(len(treated)),
    "n_with_encounters": int(enc_treated["way_id"].nunique()),
    "n_with_in_window_encounters": int(len(analysis)),
    "total_in_window_encounters": int(len(enc_window)),
    "spearman_correlations": correlations,
    "adjusted_ols_individual": ols_results,
    "adjusted_ols_joint": joint_results,
    "descriptive_stats": desc,
    "confounders": CONFOUNDERS,
    "prespecified_features": PRESPECIFIED,
}

def _convert(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} not serializable")

out_path = OUTPUT_DIR / "activity_outcome_results.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=_convert)
print(f"\nResults saved to {out_path}")

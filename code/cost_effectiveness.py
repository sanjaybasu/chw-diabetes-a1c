"""
Cost-effectiveness projection for CHW diabetes A1c manuscript.
Translates A1c reductions into projected morbidity/mortality and cost impacts.

Uses:
- RECODe risk equations (Basu et al. Lancet Diabetes Endocrinol 2017; Basu et al. Diabetes Care 2018)
  Cox PH beta coefficients for HbA1c per 1 percentage point, derived from ACCORD,
  validated in DPPOS, Look AHEAD, MESA, and JHS
- Intervention costs from Baum et al. NEJM Catalyst 2024 spreadsheet
- Medicaid acute care costs from T-MSIS
- Standard diabetes utility weights (Beaudet et al. 2014; Sullivan PW 2011)

CHEERS 2022-compliant reporting structure.
"""

import json
import numpy as np
from pathlib import Path

np.random.seed(42)
N_SIM = 10_000

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# INPUT PARAMETERS (all from published sources or study data)
# ============================================================

# --- A1c treatment effects from our study ---
# Primary: mean-A1c DML (averaging all values in each window)
# Sensitivity: last-value DML (most recent single value)
EFFECTS = {
    "overall": {"ate": -0.558, "ci_lo": -1.006, "ci_hi": -0.111, "label": "Full cohort (mean A1c, primary)"},
    "overall_last_value": {"ate": -0.289, "ci_lo": -0.570, "ci_hi": -0.007, "label": "Full cohort (last value, sensitivity)"},
}

# --- RECODe risk equations (Basu et al. Lancet Diabetes Endocrinol 2017) ---
# Cox PH beta coefficients for HbA1c (per 1 percentage point)
# HR per 1% HbA1c increase = exp(beta)
RECODE_BETA_PER_1PCT = {
    "myocardial_infarction": 0.21350,   # HR = 1.238
    "stroke": 0.33650,                   # HR = 1.400
    "renal_failure": 0.13690,            # HR = 1.147 (ESRD)
    "severe_vision_loss": 0.14490,       # HR = 1.156
    "congestive_heart_failure": 0.20920, # HR = 1.233
    "all_cause_mortality": 0.16590,      # HR = 1.180
}

# Derived HRs per 1% increase (for readability/reporting)
RECODE_HR_PER_1PCT = {k: np.exp(v) for k, v in RECODE_BETA_PER_1PCT.items()}

# --- 10-year baseline complication rates for Medicaid diabetes population ---
# (Gregg et al. NEJM 2014; CDC Diabetes Report 2022; Medicaid-specific estimates)
BASELINE_10YR_RATES = {
    "myocardial_infarction": 0.082,     # ~8.2% 10-year MI rate
    "stroke": 0.055,                     # ~5.5% 10-year stroke
    "renal_failure": 0.045,              # ~4.5% 10-year ESRD/renal failure
    "severe_vision_loss": 0.075,         # ~7.5% 10-year severe vision loss
    "congestive_heart_failure": 0.065,   # ~6.5% 10-year CHF
    "all_cause_mortality": 0.065,        # ~6.5% 10-year all-cause mortality
}

# --- Complication costs (Medicaid, 2024 USD) ---
# (T-MSIS, Medical Expenditure Panel Survey, Zhuo et al. Diabetes Care 2014)
COMPLICATION_COSTS = {
    "myocardial_infarction": 45_000,    # Acute MI hospitalization + 1yr follow-up
    "stroke": 55_000,                    # Acute stroke + rehab
    "renal_failure": 85_000,             # Dialysis initiation + first year
    "severe_vision_loss": 15_000,        # Laser photocoagulation + low vision services
    "congestive_heart_failure": 32_000,  # CHF hospitalization + follow-up
    "all_cause_mortality": 0,            # No cost (death); captured in QALYs
}

# --- QALY decrements per complication (Beaudet et al. 2014; Sullivan PW 2011) ---
QALY_DECREMENTS = {
    "myocardial_infarction": 0.055,     # Per year, lasting
    "stroke": 0.164,                     # Major disability
    "renal_failure": 0.078,              # ESRD/dialysis (Beaudet)
    "severe_vision_loss": 0.050,         # Visual impairment (Sullivan)
    "congestive_heart_failure": 0.075,   # CHF (Beaudet)
    "all_cause_mortality": 1.0,          # Full QALY loss per year of life lost
}

# Assume 15 years remaining life expectancy for Medicaid diabetes cohort (mean age ~48)
REMAINING_LIFE_YEARS = 15.0
DISCOUNT_RATE = 0.03  # 3% annual discount

# --- Intervention costs (from Baum et al. NEJM Catalyst 2024 spreadsheet) ---
INTERVENTION_COST_PER_MONTH = {"mean": 67.74, "lo": 31.81, "hi": 147.33}
ENGAGEMENT_DAYS = {"mean": 118, "lo": 84, "hi": 163}

# --- Acute care cost offsets (from spreadsheet) ---
MONTHLY_SAVINGS = {"mean": 252.70, "lo": 49.69, "hi": 519.67}


def discount_factor(year, rate=DISCOUNT_RATE):
    return 1.0 / (1.0 + rate) ** year


def compute_risk_reduction(a1c_change_pp, complication):
    """Compute absolute risk reduction over 10 years from A1c change using RECODe."""
    beta = RECODE_BETA_PER_1PCT.get(complication, 0)
    # HR for the change: exp(beta * a1c_change). Since change is negative, HR < 1.
    hr_change = np.exp(beta * a1c_change_pp)
    baseline_rate = BASELINE_10YR_RATES.get(complication, 0)
    new_rate = baseline_rate * hr_change
    arr = baseline_rate - new_rate  # absolute risk reduction (positive = fewer events)
    return arr, new_rate


def run_deterministic(a1c_change):
    """Deterministic cost-effectiveness for a given A1c change."""
    total_cost_savings = 0
    total_qaly_gain = 0

    complications_detail = {}
    for comp in BASELINE_10YR_RATES:
        arr, new_rate = compute_risk_reduction(a1c_change, comp)
        events_averted_per_1000 = arr * 1000
        cost_averted = arr * COMPLICATION_COSTS.get(comp, 0)

        # QALY gain: events averted * QALY decrement * discounted remaining years
        if comp == "all_cause_mortality":
            # Deaths averted: each death = REMAINING_LIFE_YEARS of life lost
            discounted_years = sum(discount_factor(y) for y in range(10, int(REMAINING_LIFE_YEARS)))
            qaly_gain = arr * discounted_years
        else:
            # Morbidity: QALY decrement per year * remaining years after event (assume event at year 5)
            years_after = REMAINING_LIFE_YEARS - 5
            discounted_years = sum(discount_factor(y) for y in range(5, int(years_after + 5)))
            qaly_gain = arr * QALY_DECREMENTS[comp] * discounted_years

        complications_detail[comp] = {
            "baseline_10yr_rate": round(BASELINE_10YR_RATES[comp], 4),
            "projected_10yr_rate": round(new_rate, 4),
            "absolute_risk_reduction": round(arr, 5),
            "events_averted_per_1000": round(events_averted_per_1000, 2),
            "cost_averted_per_person": round(cost_averted, 2),
            "qaly_gain_per_person": round(qaly_gain, 5),
            "recode_hr_per_1pct": round(RECODE_HR_PER_1PCT[comp], 3),
        }
        total_cost_savings += cost_averted
        total_qaly_gain += qaly_gain

    # Intervention costs (per course)
    months = ENGAGEMENT_DAYS["mean"] / 30.5
    intervention_cost = INTERVENTION_COST_PER_MONTH["mean"] * months

    # Acute care savings during engagement (from Baum et al.)
    acute_savings = MONTHLY_SAVINGS["mean"] * months

    net_cost = intervention_cost - acute_savings - total_cost_savings
    icer = net_cost / total_qaly_gain if total_qaly_gain > 0 else float("inf")

    return {
        "a1c_change": round(a1c_change, 3),
        "intervention_cost_per_course": round(intervention_cost, 2),
        "acute_care_savings_during_engagement": round(acute_savings, 2),
        "projected_10yr_complication_savings": round(total_cost_savings, 2),
        "total_savings": round(acute_savings + total_cost_savings, 2),
        "net_cost": round(net_cost, 2),
        "total_qaly_gain": round(total_qaly_gain, 5),
        "icer": round(icer, 2) if abs(icer) < 1e8 else "dominant" if net_cost < 0 else "dominated",
        "cost_saving": net_cost < 0,
        "complications": complications_detail,
    }


def run_probabilistic(a1c_change_mean, a1c_change_se):
    """Monte Carlo probabilistic sensitivity analysis."""
    icers = []
    net_costs = []
    qaly_gains = []
    cost_saving_count = 0

    for _ in range(N_SIM):
        # Draw A1c change from normal
        a1c = np.random.normal(a1c_change_mean, a1c_change_se)

        # Draw intervention cost
        cost_per_month = np.random.triangular(
            INTERVENTION_COST_PER_MONTH["lo"],
            INTERVENTION_COST_PER_MONTH["mean"],
            INTERVENTION_COST_PER_MONTH["hi"],
        )
        days = np.random.triangular(
            ENGAGEMENT_DAYS["lo"], ENGAGEMENT_DAYS["mean"], ENGAGEMENT_DAYS["hi"]
        )
        months = days / 30.5
        intervention_cost = cost_per_month * months

        # Draw acute savings
        acute_savings = np.random.triangular(
            MONTHLY_SAVINGS["lo"], MONTHLY_SAVINGS["mean"], MONTHLY_SAVINGS["hi"]
        ) * months

        # Compute complication savings + QALYs
        total_savings = 0
        total_qaly = 0
        for comp in BASELINE_10YR_RATES:
            arr, _ = compute_risk_reduction(a1c, comp)
            total_savings += arr * COMPLICATION_COSTS.get(comp, 0)
            if comp == "all_cause_mortality":
                discounted_years = sum(discount_factor(y) for y in range(10, int(REMAINING_LIFE_YEARS)))
                total_qaly += arr * discounted_years
            else:
                years_after = REMAINING_LIFE_YEARS - 5
                discounted_years = sum(discount_factor(y) for y in range(5, int(years_after + 5)))
                total_qaly += arr * QALY_DECREMENTS[comp] * discounted_years

        net = intervention_cost - acute_savings - total_savings
        net_costs.append(net)
        qaly_gains.append(total_qaly)
        if net < 0:
            cost_saving_count += 1
        if total_qaly > 0:
            icers.append(net / total_qaly)

    return {
        "prob_cost_saving": round(cost_saving_count / N_SIM, 3),
        "prob_cost_effective_50k": round(sum(1 for i in icers if i < 50_000) / len(icers), 3) if icers else 0,
        "prob_cost_effective_100k": round(sum(1 for i in icers if i < 100_000) / len(icers), 3) if icers else 0,
        "net_cost_mean": round(float(np.mean(net_costs)), 2),
        "net_cost_95ci": [round(float(np.percentile(net_costs, 2.5)), 2),
                          round(float(np.percentile(net_costs, 97.5)), 2)],
        "qaly_gain_mean": round(float(np.mean(qaly_gains)), 5),
        "qaly_gain_95ci": [round(float(np.percentile(qaly_gains, 2.5)), 5),
                           round(float(np.percentile(qaly_gains, 97.5)), 5)],
        "n_simulations": N_SIM,
    }


if __name__ == "__main__":
    results = {}

    for key, eff in EFFECTS.items():
        print(f"\n=== {eff['label']} (ATE = {eff['ate']}) ===")
        det = run_deterministic(eff["ate"])
        se = (eff["ci_hi"] - eff["ci_lo"]) / (2 * 1.96)
        prob = run_probabilistic(eff["ate"], se)

        print(f"  Intervention cost/course: ${det['intervention_cost_per_course']:.0f}")
        print(f"  Acute care savings (engagement): ${det['acute_care_savings_during_engagement']:.0f}")
        print(f"  10-yr complication savings: ${det['projected_10yr_complication_savings']:.0f}")
        print(f"  Net cost: ${det['net_cost']:.0f}")
        print(f"  QALY gain: {det['total_qaly_gain']:.4f}")
        print(f"  ICER: {det['icer']}")
        print(f"  Prob cost-saving: {prob['prob_cost_saving']:.1%}")
        print(f"  Prob CE at $50k/QALY: {prob['prob_cost_effective_50k']:.1%}")
        print(f"  Prob CE at $100k/QALY: {prob['prob_cost_effective_100k']:.1%}")

        for comp, detail in det["complications"].items():
            print(f"    {comp}: HR/1%={detail['recode_hr_per_1pct']:.3f}, "
                  f"ARR={detail['absolute_risk_reduction']:.4f}, "
                  f"events averted/1000={detail['events_averted_per_1000']:.1f}, "
                  f"savings=${detail['cost_averted_per_person']:.0f}")

        results[key] = {"deterministic": det, "probabilistic": prob}

    # CHEERS metadata
    results["cheers_metadata"] = {
        "perspective": "Medicaid payer",
        "time_horizon": "10-year projection for complications; engagement period for acute care",
        "discount_rate": "3% annual",
        "currency_year": "2024 USD",
        "outcome_measure": "QALYs",
        "sensitivity_analysis": "Probabilistic (Monte Carlo, 10,000 draws)",
        "risk_equations": "RECODe (Basu et al. Lancet Diabetes Endocrinol 2017; Basu et al. Diabetes Care 2018)",
        "data_sources": [
            "RECODe equations (Basu et al. Lancet Diabetes Endocrinol 2017) for 10-year complication risk",
            "RECODe validation (Basu et al. Diabetes Care 2018) in MESA and JHS cohorts",
            "Baum et al. NEJM Catalyst 2024 for intervention costs",
            "T-MSIS for Medicaid complication costs",
            "Beaudet et al. 2014; Sullivan PW 2011 for utility weights",
        ],
    }

    out = OUTPUT_DIR / "cost_effectiveness_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")

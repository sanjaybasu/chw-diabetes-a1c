"""
Extract analysis inputs from the Tuva-Health-based analytics warehouse (coredb).

Produces, under data/raw/ (all protected health information; git-ignored):
  - cohort372_a1c_labs.parquet      HbA1c labs for the analytic cohort
  - eligible_universe.parquet       targeted VA adults with diabetes (for IPCW denominator)
  - eligible_universe_a1c.parquet   HbA1c labs for the eligible universe
  - treated_status_dates.parquet    targeted/activated dates for treated people
  - atrisk_cohort.parquet           eligible-with-baseline-HbA1c cohort + observed indicator

Requires the base analytic cohort at data/analytic_cohort.parquet (PHI, not shared;
built by the care-management-platform extraction described in DATA_NOTE.md) and a
read-only connection to coredb. The connection helper (wm_conn) performs the
Vault-backed credential handshake; substitute your own SQLAlchemy engine if needed.

A1c labs are identified by source_description ILIKE '%a1c%' (the normalized LOINC
fields are null in this mart); results are restricted to the plausible range 3-20%
and de-duplicated to one value per person-day (minimum).
"""
from __future__ import annotations
import sys, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; RAW = DATA / "raw"; RAW.mkdir(parents=True, exist_ok=True)

ANALYSIS_START, ANALYSIS_END = "2023-07-01", "2025-12-31"
A1C_TEXT = "source_description ILIKE '%%a1c%%'"
A1C_NUMERIC = "translate(result,'<>','') ~ '^[0-9]+(\\.[0-9]+)?$'"

def get_engine():
    """Read-only SQLAlchemy engine for coredb (prod). Uses the waymark-data-access helper."""
    sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
    from wm_conn import coredb
    return coredb("prod")

def clean_a1c(df):
    df = df.copy()
    df["a1c"] = pd.to_numeric(
        df["result"].astype(str).str.replace("<", "", regex=False).str.replace(">", "", regex=False),
        errors="coerce")
    df["collection_date"] = pd.to_datetime(df["collection_date"], errors="coerce")
    df = df[df["a1c"].between(3.0, 20.0)]
    return df.groupby(["person_id", "collection_date"], as_index=False)["a1c"].min()

def main():
    from sqlalchemy import text  # noqa
    eng = get_engine()
    import pandas as pd
    read = lambda q, **kw: pd.read_sql(q, eng, params=kw or None)

    cohort = pd.read_parquet(DATA / "analytic_cohort.parquet")
    cohort["index_date"] = pd.to_datetime(cohort["index_date"])
    ids = tuple(cohort["way_id"].dropna().unique().tolist())

    # 1) A1c labs for the analytic cohort (way_id == Tuva person_id)
    labs = read(f"""SELECT person_id, collection_date, result FROM dbt_tuva_core.lab_result
                    WHERE {A1C_TEXT} AND {A1C_NUMERIC} AND person_id IN %(ids)s""", ids=ids)
    clean_a1c(labs).to_parquet(RAW / "cohort372_a1c_labs.parquet", index=False)

    # 2) Eligible universe: VA adults ever-targeted in window (first targeting), diabetes via
    #    chronic-conditions dashboard OR any A1c lab (stable definition robust to spine drift)
    sp = read(f"""SELECT person_id, zero_date, ever_targeted, ever_activated, rr_activated, risk_percentile,
                         any_bh, htn, chf, copd, asthma, polypharmacy, high_ed_ip, age, diabetes
                  FROM dbt.stg_patient_spine_zero_date
                  WHERE state='VIRGINIA' AND age>=18 AND ever_targeted=1
                    AND zero_date BETWEEN %(s)s AND %(e)s""", s=ANALYSIS_START, e=ANALYSIS_END)
    sp["zero_date"] = pd.to_datetime(sp["zero_date"])
    sp = sp.sort_values("zero_date").drop_duplicates("person_id", keep="first")
    uids = tuple(sp["person_id"].tolist())
    cc = read("""SELECT DISTINCT person_id FROM dbt.im__chronic_conditions_dashboard
                 WHERE dashboard_condition ILIKE '%%diabet%%' AND person_id IN %(ids)s""", ids=uids)
    ulab = read(f"""SELECT person_id, collection_date, result FROM dbt_tuva_core.lab_result
                    WHERE {A1C_TEXT} AND {A1C_NUMERIC} AND person_id IN %(ids)s""", ids=uids)
    dm = set(cc["person_id"]) | set(ulab["person_id"])
    elig = sp[sp["person_id"].isin(dm)].copy()
    elig.to_parquet(RAW / "eligible_universe.parquet", index=False)
    ulab.to_parquet(RAW / "eligible_universe_a1c.parquet", index=False)

    # 3) Status-event dates for treated people (first targeted / first activated-or-deeper)
    tids = tuple(cohort.loc[cohort["treated"] == 1, "way_id"].tolist())
    se = read("""SELECT person_id, status, date_time, activated_or_deeper_status
                 FROM dbt.stg__status_events WHERE person_id IN %(ids)s""", ids=tids)
    se["date_time"] = pd.to_datetime(se["date_time"]).dt.normalize()
    tgt = se[se["status"].str.upper().isin(["TARGETED", "OUTREACH", "ASSIGNED"])].groupby("person_id")["date_time"].min().rename("targeted_at")
    act = se[se["activated_or_deeper_status"] == 1].groupby("person_id")["date_time"].min().rename("activated_at")
    tsd = pd.DataFrame(index=pd.Index(cohort.loc[cohort["treated"] == 1, "way_id"], name="person_id")).join(tgt).join(act).reset_index()
    tsd = tsd.merge(cohort[["way_id", "index_date"]].rename(columns={"way_id": "person_id"}), on="person_id", how="left")
    tsd.to_parquet(RAW / "treated_status_dates.parquet", index=False)

    # 4) At-risk cohort for IPCW: eligible-with-baseline-HbA1c; observed = has a follow-up HbA1c
    #    index = activation (activated) else first targeting (zero_date)
    act_ids = tuple(elig.loc[elig["ever_activated"] == 1, "person_id"].tolist())
    ea = read("""SELECT person_id, min(date_time) act_at FROM dbt.stg__status_events
                 WHERE activated_or_deeper_status=1 AND person_id IN %(ids)s GROUP BY person_id""", ids=act_ids)
    ea["act_at"] = pd.to_datetime(ea["act_at"]).dt.normalize()
    ar = elig.merge(ea, on="person_id", how="left")
    ar["treated"] = ar["ever_activated"].astype(int)
    ar["index_date"] = pd.to_datetime(np.where(ar["treated"] == 1, ar["act_at"], ar["zero_date"]))
    ar = ar[ar["index_date"].notna()].copy()
    lab = clean_a1c(ulab).merge(ar[["person_id", "index_date"]], on="person_id", how="inner").sort_values("collection_date")
    bl = lab[(lab.collection_date >= lab.index_date - pd.DateOffset(months=6)) & (lab.collection_date < lab.index_date)].groupby("person_id")["a1c"].last().rename("baseline_a1c")
    fu = lab[(lab.collection_date >= lab.index_date + pd.Timedelta(days=90)) & (lab.collection_date <= lab.index_date + pd.Timedelta(days=365))].groupby("person_id")["a1c"].last().rename("followup_a1c")
    ar = ar.merge(bl, on="person_id", how="left").merge(fu, on="person_id", how="left")
    ar["observed"] = ar["followup_a1c"].notna()
    ar[ar["baseline_a1c"].notna()].to_parquet(RAW / "atrisk_cohort.parquet", index=False)
    print("Wrote data/raw/: cohort372_a1c_labs, eligible_universe(+a1c), treated_status_dates, atrisk_cohort")

if __name__ == "__main__":
    main()

"""
Generate publication-quality tables for the CHW diabetes A1c manuscript.
Diabetes Care: max 4 tables+figures combined. We use 2 tables + 2 figures.
Table 1: Baseline characteristics. Table 2: GRF CATE quartiles.
"""
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "diabetes")


def table1_baseline():
    """Table 1: Baseline characteristics of matched cohort."""
    df = pd.read_csv(os.path.join(DATA_DIR, "table1_baseline_characteristics.csv"))
    print("Table 1. Baseline Characteristics (N = 114 pairs)")
    print("=" * 80)
    print(f"{'Variable':<40} {'Treated (N=114)':<20} {'Control (N=114)':<20} {'SMD':<10}")
    print("-" * 80)
    for _, row in df.iterrows():
        smd = f"{row['SMD']:.3f}" if pd.notna(row["SMD"]) and row["SMD"] != "" else ""
        print(f"{row['Variable']:<40} {row['Treated (N=114)']:<20} {row['Control (N=114)']:<20} {smd:<10}")
    print()
    return df


def table2_cate_quartiles():
    """Table 2: GRF CATE quartiles."""
    df = pd.read_csv(os.path.join(DATA_DIR, "table_cf_cate_quartiles.csv"))
    print("Table 2. Conditional Average Treatment Effects by GRF Quartile")
    print("=" * 90)
    print(f"{'Quartile':<20} {'N':<6} {'N tx':<6} {'N ctrl':<8} {'Mean CATE':<12} {'Obs \u0394':<10} {'Baseline A1c':<12}")
    print("-" * 90)
    for _, row in df.iterrows():
        print(
            f"{row['CATE quartile']:<20} {row['N']:<6} {row['N treated']:<6} "
            f"{row['N control']:<8} {row['Mean CATE']:<12.3f} {row['Observed DiD']:<10.3f} "
            f"{row['Mean baseline A1c']:<12.1f}"
        )
    print()
    return df


if __name__ == "__main__":
    table1_baseline()
    table2_cate_quartiles()
    print("All tables generated.")

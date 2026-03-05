"""
Generate publication-quality figures for the CHW diabetes A1c manuscript.
Focused on DML (primary) + GRF (heterogeneity).
Diabetes Care: max 4 tables+figures combined. We use 2 tables + 2 figures.
"""
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "diabetes")
OUT_DIR = os.path.dirname(__file__)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})


def figure1_a1c_trajectories():
    """
    Figure 1: Mean HbA1c trajectories for matched treated and control patients.
    """
    with open(os.path.join(DATA_DIR, "summary_stats.json")) as f:
        stats = json.load(f)

    treated_pre = stats["baseline_a1c_treated"]
    treated_post = stats["followup_a1c_treated"]
    control_pre = stats["baseline_a1c_control"]
    control_post = stats["followup_a1c_control"]

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot([0, 1], [treated_pre, treated_post], "o-", color="#1f4e79",
            linewidth=2.5, markersize=10, label="Care team (treated)", zorder=5)
    ax.plot([0, 1], [control_pre, control_post], "s--", color="#9dc3e6",
            linewidth=2.5, markersize=10, label="Usual care (control)", zorder=5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Follow-up"], fontsize=10)
    ax.set_ylabel("Mean HbA1c (%)", fontsize=10)
    ax.set_ylim(6.2, 7.4)
    ax.set_xlim(-0.15, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    ax.annotate(f"\u0394 = {treated_post - treated_pre:+.2f}",
                xy=(0.5, (treated_pre + treated_post) / 2 - 0.05),
                fontsize=9, color="#1f4e79", ha="center")
    ax.annotate(f"\u0394 = {control_post - control_pre:+.2f}",
                xy=(0.5, (control_pre + control_post) / 2 + 0.05),
                fontsize=9, color="#5a8fb8", ha="center")

    ax.set_title("Figure 1. HbA1c Trajectories",
                 fontsize=11, fontweight="bold", loc="left")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUT_DIR, f"figure1_a1c_trajectory.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print("Figure 1 saved.")


def figure2_forest_plot():
    """
    Figure 2: Forest plot — DML (primary) and GRF ATE.
    """
    with open(os.path.join(DATA_DIR, "summary_stats.json")) as f:
        stats = json.load(f)

    estimators = [
        ("Doubly-Robust DML\n(primary)", stats["dml_ate"],
         float(stats["dml_ci"].strip("()").split(",")[0]),
         float(stats["dml_ci"].strip("()").split(",")[1])),
        ("Generalized Random\nForest ATE", stats["cf_ate"],
         float(stats["cf_ate_ci"].strip("()").split(",")[0]),
         float(stats["cf_ate_ci"].strip("()").split(",")[1])),
    ]

    fig, ax = plt.subplots(figsize=(7, 2.5))

    colors = ["#1f4e79", "#2e75b6"]
    y_positions = [1, 0]

    for i, (label, est, lo, hi) in enumerate(estimators):
        y = y_positions[i]
        color = colors[i]

        ax.plot([lo, hi], [y, y], color=color, linewidth=2.5, solid_capstyle="round")

        if i == 0:
            diamond = plt.Polygon(
                [[est, y + 0.12], [est + 0.03, y], [est, y - 0.12], [est - 0.03, y]],
                closed=True, facecolor=color, edgecolor=color
            )
            ax.add_patch(diamond)
        else:
            ax.plot(est, y, "o", color=color, markersize=9, zorder=5)

        ci_text = f"{est:.2f} ({lo:.2f}, {hi:.2f})"
        ax.text(0.20, y, ci_text, va="center", fontsize=9, color="#333333")

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([e[0] for e in estimators], fontsize=10)
    ax.set_xlabel("Estimated association with HbA1c change (percentage points)",
                  fontsize=9)
    ax.set_xlim(-1.0, 0.35)
    ax.set_ylim(-0.5, 1.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    primary_patch = mpatches.Patch(color="#1f4e79", label="Primary (DML)")
    grf_patch = mpatches.Patch(color="#2e75b6", label="GRF")
    ax.legend(handles=[primary_patch, grf_patch],
              loc="lower right", fontsize=8, frameon=False)

    ax.set_title("Figure 2. Estimated Treatment Associations With HbA1c",
                 fontsize=11, fontweight="bold", loc="left")
    ax.annotate("\u2190 Favors treatment", xy=(-0.9, -0.4), fontsize=8, color="gray")
    ax.annotate("Favors control \u2192", xy=(0.05, -0.4), fontsize=8, color="gray")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUT_DIR, f"figure2_forest_plot.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print("Figure 2 saved.")


if __name__ == "__main__":
    figure1_a1c_trajectories()
    figure2_forest_plot()
    print("All figures generated.")

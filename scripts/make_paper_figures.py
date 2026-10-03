#!/usr/bin/env python3
"""
make_paper_figures.py
---------------------
Regenerates every result figure used in the paper from the evaluation
artifacts, so no figure can drift from the numbers it claims to show.

Figures written to paper/figures/:
  rho.png              held-out engagement correlation vs baselines
  coverage.png         assignment coverage, split by grouping semantics
  anonymity.png        within-group hashtag Jaccard vs baseline
  partition.png        coverage decomposition (the key fairness figure)
  purity.png           size-matched visual purity with permutation p-values
  groupsize.png        group size profiles (are baselines emitting trends?)

Run:  venv/bin/python scripts/make_paper_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

FIG_DIR = ROOT / "paper" / "figures"
EVAL_CSV = ROOT / "evaluation_real_results.csv"
FAIR_JSON = ROOT / "baseline_fairness_summary.json"
FAIR_CSV = ROOT / "baseline_fairness_results.csv"

OURS = "Proposed solution"
OURS_SHORT = "Proposed"
HASHTAG = "Hashtag"
KEYWORD = "Keyword"

# Colourblind-safe, print-tolerant palette.
C_OURS = "#1b6ca8"
C_BASE = "#b0b0b0"
C_ALT = "#e8a33d"
GRID = {"color": "#d9d9d9", "linewidth": 0.7}


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, **GRID)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def load_eval() -> pd.DataFrame:
    return pd.read_csv(EVAL_CSV)


def get(df: pd.DataFrame, detector: str, metric: str) -> float:
    m = df[(df["detector"] == detector) & (df["metric"] == metric)]["value"]
    return float(m.iloc[0]) if len(m) else float("nan")


def map_det(name: str) -> str:
    """Normalise detector labels between the two artifacts."""
    if name.startswith("TrendLens") or name == OURS:
        return OURS_SHORT
    if name.startswith("Hashtag"):
        return HASHTAG
    if name.startswith("Keyword"):
        return name
    return name


OUR_EVAL_LABELS = ("Proposed solution", "Proposed", "TrendLens (visual clusters)")


def resolve(df: pd.DataFrame, short: str) -> str:
    """Find the row label this detector actually uses in the eval CSV."""
    for cand in df["detector"].unique():
        if cand == short:
            return cand
        if cand.startswith("TrendLens") and short == OURS_SHORT:
            return cand
    return short


# ── figures ────────────────────────────────────────────────────────────────
def fig_rho(df: pd.DataFrame) -> None:
    """Held-out engagement correlation. The paper's headline quantitative win."""
    order = [resolve(df, OURS_SHORT), "Hashtag-frequency detection",
             "Keyword/Google-Trends-style"]
    labels = ["Proposed\nsolution", "Hashtag\nbaseline", "Keyword\nbaseline"]
    vals = [get(df, d, "spearman_pred_vs_actual") for d in order]
    vals = [v if np.isfinite(v) else 0.0 for v in vals]
    colors = [C_OURS, C_BASE, C_BASE]

    fig, ax = plt.subplots(figsize=(3.6, 3.0), layout="constrained")
    bars = ax.bar(labels, vals, color=colors, width=0.62, zorder=3)
    ax.axhline(0, color="#555", linewidth=0.8, zorder=4)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=9, fontweight="bold",
                    xytext=(0, 3 if v >= 0 else -3), textcoords="offset points")
    ax.set_ylabel(r"Spearman $\rho$", fontsize=9)
    # Headroom above the tallest bar so the value label never reaches the title.
    span = max(vals) - min(min(vals), 0.0)
    ax.set_ylim(min(min(vals), 0.0) - 0.18 * span,
                max(vals) + 0.34 * span)
    ax.set_title("Engagement prediction on unseen accounts", fontsize=9.5, pad=8)
    _style(ax)
    fig.savefig(FIG_DIR / "rho.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_coverage(df: pd.DataFrame) -> None:
    """Raw assignment coverage, with the fragility of that number made visible."""
    order = [resolve(df, OURS_SHORT), "Hashtag-frequency detection",
             "Keyword/Google-Trends-style"]
    labels = ["Proposed\nsolution", "Hashtag\nbaseline", "Keyword\nbaseline"]
    vals = [get(df, d, "assignment_coverage") for d in order]
    vals = [v if np.isfinite(v) else 0.0 for v in vals]

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    bars = ax.bar(labels, vals, color=[C_OURS, C_BASE, C_ALT], width=0.62, zorder=3)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    xytext=(0, 3), textcoords="offset points")
    ax.set_ylabel("Fraction of corpus assignable", fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_title("Assignment coverage (any group)", fontsize=9.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "coverage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_anonymity(df: pd.DataFrame) -> None:
    """Within-group hashtag overlap against the hashtag detector."""
    # NOTE: evaluation_real_results.csv still labels our detector with the old
    # project name, so resolve it explicitly rather than by the short alias.
    order = ["TrendLens (visual clusters)", "Hashtag-frequency detection"]
    labels = ["Proposed\nsolution", "Hashtag\nbaseline"]
    vals = [get(df, d, "mean_hashtag_jaccard") for d in order]
    vals = [v if np.isfinite(v) else 0.0 for v in vals]

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    bars = ax.bar(labels, vals, color=[C_OURS, C_BASE], width=0.5, zorder=3)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    xytext=(0, 3), textcoords="offset points")
    ax.set_ylabel("Mean within-group hashtag Jaccard", fontsize=9)
    ax.set_title("Lower = groups not defined by shared tags", fontsize=9.5)
    ax.set_ylim(0, 1.16 * max(vals))
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "anonymity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_partition(fair: dict) -> None:
    """
    The fairness figure: coverage split into 'in at least one group' versus
    'assigned to exactly one trend'. Keyword's high raw coverage comes from
    overlapping bags, not from resolved assignment.
    """
    det = fair["detectors"]
    names = [OURS_SHORT, HASHTAG, "Keyword", "Keyword-distinctive"]
    pretty = ["Proposed\nsolution", "Hashtag\nbaseline",
              "Keyword\nbaseline", "Keyword\n(distinctive)"]
    any_ = [det[n]["coverage_any_group"] for n in names]
    one = [det[n]["coverage_exactly_one_group"] for n in names]
    x = np.arange(len(names))
    w = 0.36

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    b1 = ax.bar(x - w / 2, any_, w, label="in at least one group",
                color=C_ALT, zorder=3)
    b2 = ax.bar(x + w / 2, one, w, label="assigned to exactly one trend",
                color=C_OURS, zorder=3)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7.6,
                        xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels(pretty, fontsize=8.6)
    ax.set_ylabel("Fraction of corpus", fontsize=9)
    ax.set_ylim(0, 1.22)
    ax.set_title("Coverage depends on grouping semantics, not only on grouping quality",
                 fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.0))
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "partition.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_purity(fair: dict) -> None:
    """
    Size-matched visual purity: the metric that directly tests the paper's
    central claim. Measured in one embedding space for every detector, with
    baselines subsampled to the proposed solution's group sizes.
    """
    det = fair["detectors"]
    names = [OURS_SHORT, HASHTAG, "Keyword", "Keyword-distinctive"]
    pretty = ["Proposed\nsolution", "Hashtag\nbaseline",
              "Keyword\nbaseline", "Keyword\n(distinctive)"]
    vals = [det[n]["visual_purity_size_matched"] for n in names]
    ps = [det[n]["permutation_p_vs_proposed"] for n in names]
    colors = [C_OURS] + [C_BASE] * 3

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    bars = ax.bar(pretty, vals, color=colors, width=0.6, zorder=3)
    for i, (b, v, p) in enumerate(zip(bars, vals, ps)):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    xytext=(0, 3), textcoords="offset points")
        if p is not None:
            txt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            if p >= 0.05:
                txt = "n.s."
            ax.annotate(txt, (b.get_x() + b.get_width() / 2, 0.012),
                        ha="center", va="bottom", fontsize=7.8, rotation=90,
                        color="white", fontweight="bold")
    ax.set_ylabel("Size-matched intra-group CLIP cosine", fontsize=9)
    ax.set_ylim(0, 1.16 * max(vals))
    ax.set_title("Visual coherence of discovered groups (permutation test vs proposed)",
                 fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "purity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_groupsize(fair: dict) -> None:
    """Median group size: baselines' 'trends' are not sized like trends."""
    det = fair["detectors"]
    names = [OURS_SHORT, HASHTAG, "Keyword", "Keyword-distinctive"]
    pretty = ["Proposed\nsolution", "Hashtag\nbaseline",
              "Keyword\nbaseline", "Keyword\n(distinctive)"]
    med = [det[n]["median_group_size"] for n in names]
    ngrp = [det[n]["n_groups"] for n in names]
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    colors = [C_OURS] + [C_BASE] * 3
    bars = ax.bar(pretty, med, color=colors, width=0.6, zorder=3)
    for b, v, g in zip(bars, med, ngrp):
        ax.annotate(f"{v:.0f} posts\n({g} groups)", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=7.8,
                    xytext=(0, 3), textcoords="offset points")
    ax.axhline(10, color=C_ALT, linestyle="--", linewidth=1.1, zorder=4)
    ax.annotate("minimum size for a trend claim", (len(names) - 0.45, 10),
                ha="right", va="bottom", fontsize=7.6, color="#8a5a10")
    ax.set_ylabel("Median posts per group", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.8, max(med) * 3.4)
    ax.set_title("Group size profile of each detector (log scale)", fontsize=9.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "groupsize.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_eval()
    print("Loaded", EVAL_CSV.name, f"({len(df)} rows)")

    if FAIR_JSON.exists():
        fair = json.loads(FAIR_JSON.read_text())
        print("Loaded", FAIR_JSON.name)
        fig_partition(fair)
        fig_purity(fair)
        fig_groupsize(fair)
        print("  wrote partition.png, purity.png, groupsize.png")
    else:
        print(f"WARNING: {FAIR_JSON.name} missing; run "
              "scripts/analyze_baseline_fairness.py first. "
              "Skipping partition/purity/groupsize figures.")

    fig_rho(df)
    fig_coverage(df)
    fig_anonymity(df)
    print("  wrote rho.png, coverage.png, anonymity.png")
    print("Done ->", FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
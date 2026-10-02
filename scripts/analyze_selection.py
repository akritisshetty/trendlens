#!/usr/bin/env python3
"""
analyze_selection.py
--------------------
Paired comparison of candidate encoders on the selection metric.

Why a paired test and not a ranking of means
--------------------------------------------
Held-out engagement rho varies a lot with WHICH accounts are held out: the
per-split standard deviation observed on this corpus is of the same order as
the differences between encoders (~0.10 vs ~0.02-0.04). Comparing two means
drawn from noisy per-split values therefore says very little on its own.

Because every encoder is scored on the IDENTICAL set of account splits, the
comparison can be paired: for each split, take the difference between the
challenger and the incumbent. Pairing removes the split-to-split variance,
which is the dominant noise source, and tests the thing actually of interest —
whether the encoder changes the score on a given corpus partition at all.

Reported per challenger:
  mean paired delta   average of (challenger rho - incumbent rho) over splits
  paired t            t-statistic over those per-split differences
  wins/losses/ties    how many splits the challenger was better on
  holdout rho         challenger rho on splits never used during selection

A paired delta whose spread across splits is as large as its mean is reported
as indistinguishable rather than as an improvement.

Run:  venv/bin/python scripts/analyze_selection.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
from scripts.evaluate_real import (  # noqa: E402
    assignment_coverage,
    detect_trendlens,
    heldout_engagement_prediction,
    trend_precision,
)
from src.clustering import run_hdbscan  # noqa: E402
from src.data_quality import clean_engagement  # noqa: E402

OUT = ROOT / "encoder_paired_comparison.csv"
SEL_CSV = ROOT / "model_selection_results.csv"

SELECTION_SEEDS = (42, 7, 13, 21, 99)
HOLDOUT_SEEDS = (1234, 5678, 4321)


def load_meta() -> pd.DataFrame:
    meta = clean_engagement(
        pd.read_parquet(config.INSTAGRAM_DIR / "embed_meta.parquet")
    )
    meta["timestamp"] = pd.to_datetime(meta["timestamp"], utc=True, errors="coerce")
    return meta


def labels_for(emb: np.ndarray, row) -> np.ndarray:
    cache = config.ARTIFACTS_DIR / "model_selection" / "reduce" / (
        row["encoder"].replace("/", "__")
    )
    red = np.load(cache / f"umap_{int(row['umap_components'])}d_s42.npy")
    labels, _, _ = run_hdbscan(
        red,
        min_cluster_size=int(row["min_cluster_size"]),
        min_samples=int(row["min_samples"]),
    )
    assert len(labels) == len(emb)
    return labels


def rho_series(meta: pd.DataFrame, labels: np.ndarray, seeds) -> list[float]:
    m = meta.copy()
    m["_vis_label"] = labels
    groups = detect_trendlens(m, labels)
    out = []
    for s in seeds:
        r = heldout_engagement_prediction(m, groups, seed=s)
        out.append(r.get("spearman", np.nan))
    return out


def main() -> int:
    sel = pd.read_csv(SEL_CSV)
    meta = load_meta()

    # Best eligible config per encoder, as the selection rule chose it.
    elig = sel[sel["eligible"]]
    if elig.empty:
        print("No eligible configurations in the selection results.")
        return 1
    best = elig.sort_values("rho", ascending=False).groupby("encoder", as_index=False).first()

    out_dir = config.ARTIFACTS_DIR / "model_selection"
    embs = {
        name: np.load(out_dir / f"emb_{name.replace('/', '__')}.npy")
        for name in best["encoder"]
    }

    # Incumbent = the encoder/hyperparameters production runs today.
    incumbent_cfg = sel[
        (sel["encoder"] == config.CLIP_MODEL)
        & (sel["umap_components"] == config.UMAP_COMPONENTS)
        & (sel["min_cluster_size"] == config.HDBSCAN_MIN_CLUSTER_SIZE)
        & (sel["min_samples"] == config.HDBSCAN_MIN_SAMPLES)
    ]
    if incumbent_cfg.empty:
        inc_row = best[best["encoder"] == config.CLIP_MODEL].iloc[0].to_dict()
    else:
        inc_row = incumbent_cfg.iloc[0].to_dict()

    inc_labels = labels_for(embs[inc_row["encoder"]], inc_row)
    inc_sel = rho_series(meta, inc_labels, SELECTION_SEEDS)
    inc_hold = rho_series(meta, inc_labels, HOLDOUT_SEEDS)

    print("=" * 104)
    print(f"INCUMBENT  {inc_row['encoder']}  umap={int(inc_row['umap_components'])} "
          f"mcs={int(inc_row['min_cluster_size'])} ms={int(inc_row['min_samples'])}")
    print(f"  selection rho per split: "
          f"{['%+.3f' % v for v in inc_sel]}  mean={np.mean(inc_sel):+.3f}")
    print(f"  holdout   rho per split: "
          f"{['%+.3f' % v for v in inc_hold]}  mean={np.mean(inc_hold):+.3f}")
    print("=" * 104)

    rows = []
    for _, row in best.iterrows():
        name = row["encoder"]
        labels = labels_for(embs[name], row)
        m = meta.copy()
        m["_vis_label"] = labels
        groups = detect_trendlens(m, labels)

        sel_rho = rho_series(meta, labels, SELECTION_SEEDS)
        hold_rho = rho_series(meta, labels, HOLDOUT_SEEDS)
        diff = np.array(sel_rho) - np.array(inc_sel)
        d = diff[~np.isnan(diff)]
        if len(d) > 1 and np.std(d) > 1e-12:
            t = float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))))
        else:
            t = float("nan")
        tp = trend_precision(m, groups)

        rows.append(
            {
                "encoder": name,
                "is_incumbent": name == inc_row["encoder"],
                "umap_components": int(row["umap_components"]),
                "min_cluster_size": int(row["min_cluster_size"]),
                "min_samples": int(row["min_samples"]),
                "selection_rho_mean": float(np.nanmean(sel_rho)),
                "holdout_rho_mean": float(np.nanmean(hold_rho)),
                "paired_delta_mean": float(np.nanmean(diff)) if len(d) else float("nan"),
                "paired_delta_std": float(np.std(d, ddof=1)) if len(d) > 1 else float("nan"),
                "paired_t": t,
                "wins": int((d > 1e-6).sum()),
                "losses": int((d < -1e-6).sum()),
                "ties": int((np.abs(d) <= 1e-6).sum()),
                "coverage": assignment_coverage(groups, len(meta)),
                "trend_precision_at_5": tp["precision"],
                "n_clusters": int(len(np.unique(labels[labels >= 0]))),
            }
        )

    df = pd.DataFrame(rows).sort_values("holdout_rho_mean", ascending=False)
    df.to_csv(OUT, index=False)

    print(f"\n{'encoder':<42}{'umap':>5}{'mcs':>4}{'sel rho':>9}{'hold rho':>10}"
          f"{'Δ(paired)':>11}{'t':>7}{'W/L':>7}{'cov':>7}{'prc':>5}")
    print("-" * 104)
    for _, r in df.iterrows():
        print(
            f"{r['encoder']:<42}{r['umap_components']:>5}{r['min_cluster_size']:>4}"
            f"{r['selection_rho_mean']:>+9.3f}{r['holdout_rho_mean']:>+10.3f}"
            f"{r['paired_delta_mean']:>+11.3f}{r['paired_t']:>7.2f}"
            f"{str(r['wins']) + '/' + str(r['losses']):>7}"
            f"{r['coverage']:>7.3f}{r['trend_precision_at_5']:>5.2f}"
        )

    best_row = df.iloc[0]
    print("\n" + "=" * 104)
    print(f"HIGHEST HELD-OUT ENGAGEMENT CORRELATION: {best_row['encoder']}")
    print(f"  holdout rho {best_row['holdout_rho_mean']:+.3f} vs incumbent "
          f"{df[df['is_incumbent']].iloc[0]['holdout_rho_mean']:+.3f}")
    print(
        f"  paired delta {best_row['paired_delta_mean']:+.3f} "
        f"(sd {best_row['paired_delta_std']:.3f} across splits, "
        f"{best_row['wins']}W/{best_row['losses']}L)"
    )
    verdict = (
        "IMPROVEMENT — paired delta is large relative to its spread"
        if abs(best_row["paired_delta_mean"]) > 2 * best_row["paired_delta_std"]
        else "NOT DISTINGUISHABLE from the incumbent — paired spread is as "
             "large as the mean delta"
    )
    print(f"  verdict: {verdict}")
    print(f"\nSaved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
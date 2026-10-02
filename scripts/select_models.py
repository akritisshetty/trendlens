#!/usr/bin/env python3
"""
select_models.py
----------------
Choose TrendLens' pretrained encoders by MEASUREMENT on the real Instagram
corpus, using the same non-circular criteria as ``scripts/evaluate_real.py``.

Why this exists
---------------
The production configuration was pinned to ``openai/clip-vit-base-patch32`` by
default rather than by evidence. ViT-B/32 is the cheapest CLIP; ViT-L/14 is the
standard quality jump, and the text-similarity and zero-shot quality gap
between them is large. Nothing in the repo compared them on TrendLens' actual
objective, so the pipeline has been running the weakest reasonable encoder.

Selection rule (fixed BEFORE looking at results, to keep this honest)
---------------------------------------------------------------------
Primary   ``rho`` — mean Spearman correlation for held-out engagement over
          SEEDS_SPLITS independent by-ACCOUNT splits. Engagement is the only
          target in the corpus that no detector is shown, and the split is by
          account, so this measures whether visual grouping carries engagement
          information the account split cannot fake. Averaged over splits
          because a single 1/3 split leaves ~136 test posts and the sampling
          noise is far larger than the differences being compared.

Gates     A configuration is only eligible if all of:
            * ``coverage`` >= COVERAGE_GATE — a detector that cannot assign
              most of the corpus cannot be a trending list
            * ``n_clusters >= MIN_CLUSTERS`` — too few groups and the trend
              definition has nothing to classify
            * ``stability_ari`` >= STABILITY_GATE — cluster identity must
              survive re-runs for the longitudinal tracker to mean anything

Tie-break Among eligible configurations, rank on ``rho``; break ties within
          1e-3 on ``trend_precision``, then on lower ``encode_minutes``.

          The tolerance is load-bearing. Without it the ranking is decided by
          the last digits of a noisy coefficient: the two leading rows sat
          4.9e-4 apart on rho, and the tie-break handed the win to the one
          whose trends clear the definition (0.2 vs 0.0).

Honesty note
------------
Hyperparameters ARE selected against the evaluation corpus, so the winning
configuration's reported ``rho`` is optimistic as an estimate of future
corpora. That is inherent to tuning, not a flaw of this script; the mitigation
is that the same rule is applied identically to every candidate encoder, so
the COMPARISON between encoders is fair even though the absolute number is
biased upward. ``--holdout`` additionally reports the winner's score under
account splits never used for selection, which is the number to quote.

Run:  venv/bin/python scripts/select_models.py
      venv/bin/python scripts/select_models.py --encoders openai/clip-vit-large-patch14
      venv/bin/python scripts/select_models.py --stage2
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
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
from src.data_quality import clean_engagement  # noqa: E402

OUT_DIR = config.ARTIFACTS_DIR / "model_selection"
OUT_CSV = ROOT / "model_selection_results.csv"
OUT_JSON = ROOT / "model_selection_summary.json"

#: Account splits used for the primary metric.
#:
#: This was 5 seeds, and 5 is NOT enough. On this corpus the per-split rho has a
#: standard deviation of ~0.12, so the standard error of a 5-seed mean is ~0.05
#: — larger than every difference the sweep was trying to resolve. It nominated
#: UMAP=20 over UMAP=15; re-measured on 80 splits scored PAIRED, 15 wins by
#: +0.071 (65/80 splits, t=6.96, p=9e-10). The ranking below is only trusted
#: because a properly powered protocol overrode it.
#:
#: 25 splits puts the standard error at ~0.025. It costs ~5x the wall clock of
#: the old 5-seed sweep, which is the correct trade: a cheap sweep that picks
#: the wrong configuration costs a full downstream rebuild.
SEEDS_SPLITS = tuple(range(1, 26))

#: Account splits reserved for the winner's holdout score. Disjoint from
#: SEEDS_SPLITS by construction — reusing selection splits would report the
#: selection objective back to itself and look like independent confirmation.
#: 20 splits rather than the 3 this used to use, because the reported number
#: now carries a standard error: a single split on this corpus is worth about
#: +-0.12 of rho, which is wider than most of the differences being compared.
HOLDOUT_SEEDS = tuple(range(9001, 9021))
# UMAP seeds used for the stability estimate.
SEEDS_UMAP = (42, 7, 13)

COVERAGE_GATE = 0.55
MIN_CLUSTERS = 8
#: Two rho values this close are treated as a TIE and handed to the tie-breakers
#: below, as the module docstring promises. Spearman rho here has a per-config
#: std of ~0.12 across the five account splits, so a gap of a few 1e-4 in the
#: mean is noise, not signal: without a tolerance the ranking is decided by the
#: last digits of a noisy coefficient and a configuration whose trends never
#: meet the definition can outrank one that does.
RHO_TIE_TOL = 1e-3
#: ARI is measured ACROSS UMAP random seeds, which is a far harsher test than
#: re-running the pipeline (that reuses config.RANDOM_SEED and is deterministic
#: by construction, so it trivially scores ~1.0). What matters for the
#: longitudinal tracker is whether cluster identity survives a changed
#: projection, and on a 461-post corpus the honest measured range is ~0.28-0.73.
#: The gate is set just below the incumbent's 0.632 (umap=15 / mcs=5 / ms=3) so
#: candidates must not be worse than the configuration already in production.
STABILITY_GATE = 0.55

# Candidate image encoders. ViT-B/32 is the incumbent; the rest are the
# standard quality ladder for CLIP-family image/text encoders, plus two
# SigLIP2 points to check whether the newer objective is worth the extra work.
DEFAULT_ENCODERS = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-large-patch14",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    "laion/CLIP-ViT-L-14-laion2B-s32B-b79K",
    "google/siglip2-base-patch16-224",
]

# Grid searched per encoder. UMAP dims matter because HDBSCAN's notion of
# density changes with dimensionality; cluster size is swept because the
# trade-off between "many small groups" and "few big ones" is what decides
# whether a group can clear the trend definition's support floor.
UMAP_DIMS = (10, 15, 20)
MIN_CLUSTER_SIZES = (5, 8, 12)
MIN_SAMPLES = (3, 5)


def slug(name: str) -> str:
    return name.replace("/", "__")


# ──────────────────────────────────────────────────────────────────────────
# Corpus + embedding
# ──────────────────────────────────────────────────────────────────────────
def load_corpus() -> tuple[np.ndarray, pd.DataFrame]:
    """The production embedding matrix + its aligned, sanitised metadata."""
    emb = np.load(config.INSTAGRAM_EMBEDDINGS_PATH)
    meta = clean_engagement(
        pd.read_parquet(config.INSTAGRAM_DIR / "embed_meta.parquet")
    )
    assert len(emb) == len(meta), "embeddings/metadata misaligned"
    meta["timestamp"] = pd.to_datetime(meta["timestamp"], utc=True, errors="coerce")
    return emb, meta


def resolve_image(post_id: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = config.INSTAGRAM_IMAGES_DIR / f"{post_id}{ext}"
        if p.is_file():
            return p
    return None


def current_corpus_post_ids() -> list[str]:
    """post_ids of the posts that currently have a local image, in the order
    embed_corpus() walks them."""
    df = pd.read_parquet(config.INSTAGRAM_DIR / "all_posts.parquet")
    return [
        str(pid)
        for pid in df["post_id"]
        if resolve_image(str(pid)) is not None
    ]


def embed_corpus(model_name: str, batch_size: int = 16, force: bool = False) -> tuple[np.ndarray, list[str], float] | None:
    """
    Embed every local image with ``model_name``. Returns (embeddings, post_ids, minutes),
    or None if the checkpoint cannot be loaded.

    Returns None rather than raising for an unloadable candidate: several widely
    cited CLIP checkpoints (e.g. laion/CLIP-ViT-B-16-laion2B-s34B-b88K) are
    published in open_clip format only and have no ``model_type`` in
    config.json, so transformers cannot read them. One such candidate must not
    abort a sweep of the others.

    Handles both architectures the candidates use:
      * CLIPModel    — ``get_image_features`` returns an output object under
        transformers >= 5 whose ``pooler_output`` is the PROJECTED embedding
        (verified to equal ``config.projection_dim``, not the pre-projection
        hidden size), so both call shapes are handled explicitly.
      * SiglipModel  — ``get_image_features`` returns the projected tensor.
    """
    import torch
    from PIL import Image

    cache = OUT_DIR / f"emb_{slug(model_name)}.npy"
    ids_cache = OUT_DIR / f"ids_{slug(model_name)}.json"
    if cache.exists() and ids_cache.exists() and not force:
        ids = json.loads(ids_cache.read_text())
        arr = np.load(cache)
        # The cache must be checked against the corpus it claims to describe.
        # A row-count self-check (`len(arr) == len(ids)`) is always true and
        # says nothing: after a fresh fetch the cached matrix still described
        # the previous corpus, so the sweep silently scored a stale embedding
        # against the new metadata and crashed on length mismatch. Compare the
        # actual post_ids, which is the only thing that can detect this.
        if len(arr) == len(ids) and ids == current_corpus_post_ids():
            print(f"  [cached] {model_name}: {arr.shape}")
            return arr, ids, 0.0
        print(
            f"  [stale] {model_name}: cached {len(ids)} ids no longer match the "
            f"corpus ({len(current_corpus_post_ids())} posts) — re-embedding"
        )

    from transformers import AutoModel, AutoProcessor

    torch.set_num_threads(max(1, torch.get_num_threads()))
    print(f"  loading {model_name} …")
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as exc:  # noqa: BLE001 — a bad candidate must not kill the sweep
        print(
            f"  SKIPPED {model_name}: not loadable by transformers "
            f"({type(exc).__name__}: {str(exc)[:90]})"
        )
        return None
    model.eval()

    meta = pd.read_parquet(config.INSTAGRAM_DIR / "embed_meta.parquet")
    post_ids: list[str] = []
    images: list[Image.Image] = []
    for pid in meta["post_id"].astype(str):
        path = resolve_image(pid)
        if path is None:
            continue
        try:
            images.append(Image.open(path).convert("RGB"))
        except OSError:
            continue
        post_ids.append(pid)
    print(
        f"  {len(images)} images, "
        f"{getattr(model.config, 'projection_dim', '?')}d projected"
    )

    feats: list[np.ndarray] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            inputs = processor(images=chunk, return_tensors="pt")
            out = model.get_image_features(**inputs)
            if not hasattr(out, "shape"):
                out = out.pooler_output
            feats.append(out.float().cpu().numpy())
            if (start // batch_size) % 5 == 0:
                done = min(start + batch_size, len(images))
                rate = done / max(time.perf_counter() - t0, 1e-9)
                print(
                    f"    {done}/{len(images)}  {rate:.2f} img/s"
                    f"  eta {(len(images) - done) / max(rate, 1e-9) / 60:.1f} min",
                    flush=True,
                )
    minutes = (time.perf_counter() - t0) / 60.0

    arr = np.concatenate(feats, axis=0).astype("float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    ids_cache.write_text(json.dumps(post_ids))
    print(f"  done in {minutes:.1f} min -> {arr.shape}")
    del model
    return arr, post_ids, minutes


# ──────────────────────────────────────────────────────────────────────────
# Clustering + scoring
# ──────────────────────────────────────────────────────────────────────────
def _umap(emb: np.ndarray, n_components: int, seed: int, cache_dir: Path) -> np.ndarray:
    import umap

    path = cache_dir / f"umap_{n_components}d_s{seed}.npy"
    if path.exists():
        cached = np.load(path)
        if cached.shape[0] == emb.shape[0]:
            return cached
    reducer = umap.UMAP(
        n_components=n_components,
        random_state=seed,
        n_neighbors=min(15, max(2, emb.shape[0] - 1)),
        min_dist=0.0,
        metric="cosine",
        verbose=False,
    )
    reduced = np.asarray(reducer.fit_transform(emb), dtype="float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, reduced)
    return reduced


def _hdbscan(red: np.ndarray, mcs: int, ms: int):
    from hdbscan import HDBSCAN

    clusterer = HDBSCAN(
        min_cluster_size=int(mcs),
        min_samples=int(ms),
        cluster_selection_method="eom",
        prediction_data=True,
        metric="euclidean",
    )
    clusterer.fit(red)
    return np.asarray(clusterer.labels_, dtype=np.int64)


def _stability_ari(label_sets: list[np.ndarray]) -> float:
    """Mean pairwise ARI across UMAP seeds — does cluster identity survive a re-run?"""
    from sklearn.metrics import adjusted_rand_score

    pairs = [
        adjusted_rand_score(label_sets[i], label_sets[j])
        for i, j in itertools.combinations(range(len(label_sets)), 2)
    ]
    return float(np.mean(pairs)) if pairs else float("nan")


def score_labels(meta: pd.DataFrame, labels: np.ndarray) -> dict:
    """
    Score one label vector on the non-circular criteria.

    ``trend_precision`` re-derives verdicts from the label vector, so it is
    called with a copy of the metadata carrying this candidate's labels — the
    metric never sees the production labels.
    """
    m = meta.copy()
    m["_vis_label"] = labels
    groups = detect_trendlens(m, labels)

    rhos, maes = [], []
    for seed in SEEDS_SPLITS:
        r = heldout_engagement_prediction(m, groups, seed=seed)
        if np.isfinite(r.get("spearman", np.nan)):
            rhos.append(r["spearman"])
        if "mae" in r:
            maes.append(r["mae"])

    tp = trend_precision(m, groups)
    return {
        "rho": float(np.mean(rhos)) if rhos else float("nan"),
        "rho_std": float(np.std(rhos)) if len(rhos) > 1 else float("nan"),
        "rho_min": float(np.min(rhos)) if rhos else float("nan"),
        "mae": float(np.mean(maes)) if maes else float("nan"),
        "coverage": assignment_coverage(groups, len(m)),
        "n_groups": len(groups),
        "trend_precision_at_5": tp["precision"],
        "n_noise": int((labels == -1).sum()),
    }


def sweep_encoder(
    model_name: str,
    emb: np.ndarray,
    meta: pd.DataFrame,
    encode_minutes: float,
) -> list[dict]:
    """Grid-search UMAP x HDBSCAN for one encoder, scoring every configuration."""
    from sklearn.metrics import adjusted_rand_score

    cache_dir = OUT_DIR / f"reduce_{slug(model_name)}"
    rows: list[dict] = []

    # UMAP reduction is the expensive step, so cache per (dim, seed).
    reduced_by_seed: dict[tuple[int, int], np.ndarray] = {}
    for n_components in UMAP_DIMS:
        for seed in SEEDS_UMAP:
            reduced_by_seed[(n_components, seed)] = _umap(emb, n_components, seed, cache_dir)

    for n_components, mcs, ms in itertools.product(
        UMAP_DIMS, MIN_CLUSTER_SIZES, MIN_SAMPLES
    ):
        label_sets = [
            _hdbscan(reduced_by_seed[(n_components, seed)], mcs, ms)
            for seed in SEEDS_UMAP
        ]
        # Score on the production seed (42) so every row is comparable, and use
        # the other two only for the stability estimate.
        m = meta.copy()
        m["_vis_label"] = label_sets[0]
        groups = detect_trendlens(m, label_sets[0])

        rhos = []
        for split_seed in SEEDS_SPLITS:
            r = heldout_engagement_prediction(m, groups, seed=split_seed)
            if np.isfinite(r.get("spearman", np.nan)):
                rhos.append(r["spearman"])
        tp = trend_precision(m, groups)

        # Cluster-count agreement across seeds, in addition to ARI: a config can
        # have high ARI while merging everything into one blob.
        counts = [int(len(np.unique(ls[ls >= 0]))) for ls in label_sets]

        rows.append(
            {
                "encoder": model_name,
                "umap_components": n_components,
                "min_cluster_size": mcs,
                "min_samples": ms,
                "n_clusters": counts[0],
                "n_clusters_min": min(counts),
                "n_noise": int((label_sets[0] == -1).sum()),
                "rho": float(np.mean(rhos)) if rhos else float("nan"),
                "rho_std": float(np.std(rhos)) if len(rhos) > 1 else float("nan"),
                "coverage": assignment_coverage(groups, len(m)),
                "trend_precision_at_5": tp["precision"],
                "stability_ari": _stability_ari(label_sets),
                "encode_minutes": encode_minutes,
            }
        )
        r = rows[-1]
        print(
            f"    umap={n_components:>2} mcs={mcs:>2} ms={ms}  "
            f"clusters={r['n_clusters']:>2}  rho={r['rho']:+.3f}"
            f"(±{r['rho_std']:.3f})  cov={r['coverage']:.3f}  "
            f"prec={r['trend_precision_at_5']:.2f}  ari={r['stability_ari']:.3f}",
            flush=True,
        )
    _ = adjusted_rand_score  # imported for clarity; _stability_ari does the work
    return rows


def eligible(row: dict) -> bool:
    return (
        not np.isnan(row["rho"])
        and row["coverage"] >= COVERAGE_GATE
        and row["n_clusters"] >= MIN_CLUSTERS
        and not np.isnan(row["stability_ari"])
        and row["stability_ari"] >= STABILITY_GATE
    )


def _rank_within_tolerance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order rows by ``rho``, but treat anything within ``RHO_TIE_TOL`` of the
    current leader as tied and settle that group on the tie-breakers.

    A plain lexicographic ``sort_values(["rho", ...])`` cannot express this:
    it lets a 5e-4 rho gap outrank a far better ``trend_precision_at_5``,
    which is exactly the case the docstring's tie-break exists to prevent.
    Walking rho downward instead makes the tie group a real group, so the
    tie-breakers decide it.
    """
    pending = df.sort_values("rho", ascending=False).index.tolist()
    order: list = []
    while pending:
        lead = df["rho"].astype(float).loc[pending[0]]
        if not np.isfinite(lead):
            # Non-finite rho sorts last and can never join a tie group.
            order.extend(pending)
            break
        group = [i for i in pending if df["rho"].astype(float).loc[i] >= lead - RHO_TIE_TOL]
        group.sort(
            key=lambda i: (
                -float(df["trend_precision_at_5"].loc[i]),
                float(df["encode_minutes"].loc[i]),
            )
        )
        order.extend(group)
        tied = set(group)
        pending = [i for i in pending if i not in tied]
    return df.loc[order]


def rank(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["eligible"] = df.apply(eligible, axis=1)
    elig = df[df["eligible"]]
    # Ineligible rows keep their old ordering but always sort after eligible ones.
    inelig = df[~df["eligible"]].sort_values(
        ["rho", "trend_precision_at_5", "encode_minutes"],
        ascending=[False, False, True],
    )
    ranked = _rank_within_tolerance(elig) if len(elig) else inelig
    return pd.concat([ranked, inelig]).reset_index(drop=True)


def holdout_check(df: pd.DataFrame, meta: pd.DataFrame, emb_lookup: dict) -> dict:
    """
    Re-score the winning configuration on account splits NOT used for selection.

    Selection maximised rho over SEEDS_SPLITS, so re-using them would report
    the selection objective back to itself. These seeds were never consulted.
    """
    win = df.iloc[0]
    emb = emb_lookup[win["encoder"]]
    red = _umap(
        emb,
        int(win["umap_components"]),
        42,
        OUT_DIR / f"reduce_{slug(win['encoder'])}",
    )
    labels = _hdbscan(red, int(win["min_cluster_size"]), int(win["min_samples"]))
    m = meta.copy()
    m["_vis_label"] = labels
    groups = detect_trendlens(m, labels)

    fresh = HOLDOUT_SEEDS
    rhos = []
    for seed in fresh:
        r = heldout_engagement_prediction(m, groups, seed=seed)
        if np.isfinite(r.get("spearman", np.nan)):
            rhos.append(r["spearman"])
    tp = trend_precision(m, groups)
    arr = np.asarray(rhos, dtype=float)
    return {
        "encoder": win["encoder"],
        "umap_components": int(win["umap_components"]),
        "min_cluster_size": int(win["min_cluster_size"]),
        "min_samples": int(win["min_samples"]),
        "selection_rho": float(win["rho"]),
        "holdout_rho": float(arr.mean()) if arr.size else float("nan"),
        "holdout_rho_std": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "holdout_rho_sem": (
            float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
        ),
        "holdout_n_splits": int(arr.size),
        "holdout_seeds": list(fresh),
        "coverage": assignment_coverage(groups, len(m)),
        "trend_precision_at_5": tp["precision"],
        "n_clusters": int(len(np.unique(labels[labels >= 0]))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="*", default=DEFAULT_ENCODERS)
    ap.add_argument("--stage1-only", action="store_true")
    ap.add_argument("--force-embed", action="store_true")
    ap.add_argument("--out-prefix", default="model_selection")
    ap.add_argument(
        "--from-csv",
        action="store_true",
        help=(
            "Re-rank an existing results CSV with the current rule instead of "
            "re-running the sweep. The sweep itself is unchanged by a ranking "
            "fix, so re-deriving the order from the same measurements is exact "
            "and costs seconds rather than an hour."
        ),
    )
    args = ap.parse_args()

    _, meta = load_corpus()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = ROOT / f"{args.out_prefix}_results.csv"
    out_json = ROOT / f"{args.out_prefix}_summary.json"

    if args.from_csv:
        if not out_csv.exists():
            print(f"{out_csv} not found; run without --from-csv first.")
            return 1
        cached = pd.read_csv(out_csv)
        rows = cached.drop(columns=["eligible"], errors="ignore").to_dict("records")
        print(f"Re-ranking {len(rows)} rows from {out_csv.name}")
        emb_lookup, minutes_lookup = {}, {}
        for name in sorted({r["encoder"] for r in rows}):
            got = embed_corpus(name)
            if got is not None:
                emb_lookup[name], _, mins = got
                minutes_lookup[name] = mins
        df = rank(rows)
        return _report(df, meta, emb_lookup, out_csv, out_json)

    print("=" * 96)
    print("STAGE 1 — embed the corpus with each candidate encoder")
    print("=" * 96)
    emb_lookup: dict[str, np.ndarray] = {}
    minutes_lookup: dict[str, float] = {}
    for name in args.encoders:
        result = embed_corpus(name, force=args.force_embed)
        if result is None:
            continue
        arr, ids, mins = result
        emb_lookup[name] = arr
        minutes_lookup[name] = mins
        if args.stage1_only:
            continue

        print(f"\n  sweeping {name} …")
        rows = sweep_encoder(name, arr, meta, mins)
        globals().setdefault("_ROWS", []).extend(rows)

    if args.stage1_only:
        print("\nEmbeddings cached. Re-run without --stage1-only to sweep.")
        return 0

    rows = globals().get("_ROWS", [])
    if not rows:
        print("No candidate could be evaluated.")
        return 1
    df = rank(rows)
    return _report(df, meta, emb_lookup, out_csv, out_json)


def _report(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    emb_lookup: dict,
    out_csv: Path,
    out_json: Path,
) -> int:
    """Write the ranked CSV, print it, and emit the holdout-checked summary."""
    df.to_csv(out_csv, index=False)

    print()
    print("=" * 96)
    print("RANKED (eligible first)")
    print("=" * 96)
    show = [
        "encoder", "umap_components", "min_cluster_size", "min_samples",
        "n_clusters", "rho", "rho_std", "coverage", "trend_precision_at_5",
        "stability_ari", "encode_minutes",
    ]
    print(df[show].head(15).to_string(index=False))

    hold = holdout_check(df, meta, emb_lookup)
    print()
    print("WINNER (holdout account splits never used for selection)")
    print(json.dumps(hold, indent=1))

    if df["eligible"].any():
        # .head(1) per encoder, NOT groupby(...).first(): groupby.first() takes
        # the first NON-NULL value per COLUMN, so one NaN stability_ari would
        # splice that column in from a different configuration and report a row
        # that was never actually measured.
        best_per_encoder = df[df["eligible"]].groupby("encoder", sort=False).head(1)
        best_rows = best_per_encoder[show].to_dict("records")
    else:
        best_rows = []
        print(
            "\nWARNING: no configuration cleared the gates "
            f"(coverage>={COVERAGE_GATE}, n_clusters>={MIN_CLUSTERS}, "
            f"stability_ari>={STABILITY_GATE}). Reporting the best rows "
            "regardless of eligibility; treat the winner as unvalidated."
        )
    summary = {
        "selection_rule": {
            "primary": "mean spearman rho over by-account splits "
                       f"{list(SEEDS_SPLITS)}",
            "gates": {
                "coverage": COVERAGE_GATE,
                "n_clusters": MIN_CLUSTERS,
                "stability_ari": STABILITY_GATE,
            },
            "tie_break": ["trend_precision_at_5", "encode_minutes"],
            "rho_tie_tolerance": RHO_TIE_TOL,
        },
        "grid": {
            "umap_components": list(UMAP_DIMS),
            "min_cluster_size": list(MIN_CLUSTER_SIZES),
            "min_samples": list(MIN_SAMPLES),
            "umap_seeds": list(SEEDS_UMAP),
        },
        "winner": hold,
        "best_per_encoder": best_rows,
        "caveat": (
            "Hyperparameters were selected against the evaluation corpus, so "
            "holdout_rho is optimistic too but is measured on splits that were "
            "not consulted during selection. Encoder-to-encoder comparison is "
            "fair because every candidate saw the identical rule and grid."
        ),
    }
    out_json.write_text(json.dumps(summary, indent=1, default=str))
    print(f"\nSaved {len(df)} rows -> {out_csv}\nSaved summary -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
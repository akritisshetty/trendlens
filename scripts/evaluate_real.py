#!/usr/bin/env python3
"""
evaluate_real.py
----------------
Evaluation of TrendLens on the REAL Instagram corpus, against realistic
baselines, using only non-circular criteria.

This replaces ``evaluate_system.py``, whose reported numbers were artifacts of
how it was built rather than properties of the system:

  * CLUSTERING was measured on ``synthetic 512-dim`` data with 500 samples and
    5 planted Gaussian clusters. HDBSCAN and KMeans both recovered them
    perfectly, so both scored ARI 1.0000 / NMI 1.0000 / silhouette 0.8255 —
    identical numbers, presented in the README as an "HDBSCAN vs KMeans"
    comparison. Recovering 5 separated blobs is a tautology, not evidence.

  * LIFECYCLE scored 1.00 precision / recall / F1 on all three classes. The
    code built a hardcoded 20-row DataFrame, derived the expected labels by
    applying the same ``>= 0.25`` rule that ``classify_lifecycle`` implements,
    and then compared the two. It always scores 1.0 and had no connection to
    any real data.

  * RAG retrieval used a 15-document corpus and 8 queries. Real, but far too
    small to support the conclusions drawn from it.

This script measures four things, each chosen because it cannot be gamed by the
implementation being measured.

── 1. ASSIGNMENT COVERAGE ────────────────────────────────────────────────
What fraction of the corpus can each detector put into a group at all?
Text-based detectors can only group posts that carry hashtags or keywords, so
posts whose posters wrote nothing are invisible to them. This is a capability
difference, not a quality judgement, and it is measured identically for every
detector.

── 2. HELD-OUT ENGAGEMENT PREDICTION (the load-bearing metric) ────────────
Each detector groups posts. Group labels are then built from a TRAIN split and
used to predict engagement on a TEST split the group statistics never saw.

Two things make this non-circular:
  * the split is BY ACCOUNT, so an account's posts cannot appear in both
    train and test. Without this, a detector that groups by anything an
    account does consistently would score near-perfectly on memorisation of
    that account's audience size rather than on anything visual.
  * the target is engagement, but no detector is given engagement as input.
    Detectors see only images (TrendLens) or only text (baselines).

Reported as Spearman correlation between predicted and actual engagement on
held-out posts, plus the mean-absolute-error of a "predict the train global
median" control. If a detector cannot beat that control, its grouping carries
no engagement information and the number is reported as a negative result.

── 3. TREND-DEFINITION PRECISION ─────────────────────────────────────────
How many of each detector's top-ranked groups are confirmed Rising under
``src.trend_definition``? A detector that flags many things will always get
some hits by luck, so precision is paired with the count of claims made.

── 4. TEXTUAL ANONYMY (the core product claim) ───────────────────────────
Mean pairwise hashtag/keyword Jaccard inside each detector's groups. Low
overlap means the grouped posts share no name — the situation where
Google-Trends-style keyword tools have nothing to search for. This is a
property of the data, not a quality score, and is reported as such.

Nothing here is tuned to favour TrendLens. The engagement-prediction control
in particular is a floor that a strong-looking but uninformative grouping will
fail to clear.

Run:  venv/bin/python scripts/evaluate_real.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.data_quality import clean_engagement  # noqa: E402
from src.trend_definition import DEFAULT_DEFINITION, classify_corpus  # noqa: E402

OUT = ROOT / "evaluation_real_results.csv"
SEED = 42

_STOP = set(
    """the a an and or but for with from this that these those are was were is not
    on at by to of in it its you your we our have has had be been being do does did
    as if so then just like what when where how why which who whom via very really
    such more most than about only into over out up down now get got also new one
    two first last day week month year good great best top photo photos video
    instagram repost original content new watch love insta""".split()
)


def hashtags(caption: str) -> set[str]:
    return set(re.findall(r"#([A-Za-z0-9_]+)", str(caption or "")))


def words(caption: str) -> set[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", str(caption or "").lower())
    return {t for t in toks if t not in _STOP}


def load_data():
    emb = np.load(config.INSTAGRAM_EMBEDDINGS_PATH)
    meta = clean_engagement(pd.read_parquet(config.INSTAGRAM_DIR / "embed_meta.parquet"))
    assert len(emb) == len(meta), "embeddings/metadata misaligned"
    meta["timestamp"] = pd.to_datetime(meta["timestamp"], utc=True, errors="coerce")
    return emb, meta


def trendlens_labels(emb: np.ndarray) -> np.ndarray:
    """
    Production cluster labels for the given embedding matrix, cached.

    The cache is keyed on the embedding matrix itself, not just its row count:
    every candidate encoder yields the same number of rows, so a row-count key
    returns the previous checkpoint's labels and reports them as if they were
    the current model's.

    It is ALSO keyed on the clustering parameters. Keying only on the encoder
    and the embeddings meant that changing UMAP components or min_cluster_size
    silently returned the previous configuration's labels — so a sweep could
    change the hyperparameters, be told the evaluation was cached, and report
    the OLD configuration's metrics as the new one's. Any input that changes
    the labels has to be part of the key.
    """
    from src.clustering import embedding_fingerprint, reduce_dimensions, run_hdbscan

    params = (
        f"u{config.UMAP_COMPONENTS}"
        f"_mcs{config.HDBSCAN_MIN_CLUSTER_SIZE}"
        f"_ms{config.HDBSCAN_MIN_SAMPLES}"
        f"_{config.HDBSCAN_SELECTION_METHOD}"
    )
    fp = embedding_fingerprint(emb)
    stem = f"labels_instagram_{config.CLIP_MODEL.replace('/', '__')}_{params}_{fp}"
    path = config.CLUSTER_MODELS_DIR / f"{stem}.npy"
    if path.exists():
        cached = np.load(path)
        if len(cached) == len(emb):
            return cached

    red = reduce_dimensions(
        emb, method="umap", n_components=config.UMAP_COMPONENTS, seed=config.RANDOM_SEED
    )
    labels, _, _ = run_hdbscan(
        red,
        min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=config.HDBSCAN_MIN_SAMPLES,
        cluster_selection_method=config.HDBSCAN_SELECTION_METHOD,
    )
    np.save(path, labels)
    return labels


# ──────────────────────────────────────────────────────────────────────────
# Detectors. Each returns a dict group_key -> list of row positions.
# ──────────────────────────────────────────────────────────────────────────
def detect_trendlens(meta: pd.DataFrame, labels: np.ndarray) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, lb in enumerate(labels):
        if lb >= 0:
            groups[f"visual_{lb}"] .append(i)
    return dict(groups)


def detect_hashtag(meta: pd.DataFrame, top_k: int = 40) -> dict[str, list[int]]:
    tags = meta["caption"].map(hashtags)
    counts = tags.explode().dropna().value_counts()
    groups: dict[str, list[int]] = {}
    for tag in counts.head(top_k).index:
        members = [i for i, s in enumerate(tags) if tag in s]
        if len(members) > 1:
            groups[f"#{tag}"] = members
    return groups


def detect_keyword(meta: pd.DataFrame, top_k: int = 40) -> dict[str, list[int]]:
    ws = meta["caption"].map(words)
    counts = ws.explode().dropna().value_counts()
    groups: dict[str, list[int]] = {}
    for w in counts.head(top_k).index:
        members = [i for i, s in enumerate(ws) if w in s]
        if len(members) > 1:
            groups[w] = members
    return groups


def detect_engagement(meta: pd.DataFrame, n_groups: int = 12) -> dict[str, list[int]]:
    """
    An engagement-ranked 'trending list', expressed as groups so it can be
    compared on equal terms: the corpus is cut into n_groups bands of equal
    size by engagement rank.

    This is a strong baseline on prediction by construction (each band has a
    nearly constant engagement), which is exactly why the control in the
    prediction metric is needed — the question is whether VISUAL grouping
    recovers engagement structure the account split hides.
    """
    order = meta["likes"].fillna(0).add(meta["comments"].fillna(0)).fillna(0).argsort()
    groups: dict[str, list[int]] = {}
    per = max(2, len(order) // n_groups)
    for g, start in enumerate(range(0, len(order), per)):
        chunk = order[start : start + per]
        if len(chunk) > 1:
            groups[f"eng_band_{g}"] = [int(i) for i in chunk]
    return groups


# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────
def assignment_coverage(groups: dict[str, list[int]], n_posts: int) -> float:
    if not groups:
        return 0.0
    assigned = set()
    for members in groups.values():
        assigned.update(members)
    return len(assigned) / n_posts if n_posts else 0.0


def heldout_engagement_prediction(
    meta: pd.DataFrame,
    groups: dict[str, list[int]],
    min_group: int = 5,
    seed: int = SEED,
) -> dict:
    """
    Predict held-out engagement from group membership learned on a train split.

    The train/test split is BY ACCOUNT. A random post-level split would leak:
    an account's posts cluster together in both halves, so a detector could
    score well by learning "posts by big accounts get many likes" instead of
    anything about the group. Splitting by account removes that path.

    Prediction for a test post is the median engagement of its group's TRAIN
    posts; posts in groups absent from train get the global train median.

    ``seed`` selects which accounts are held out. Model selection calls this
    with several seeds and averages, because a single 1/3 account split leaves
    the correlation too noisy to choose between encoders on: with ~136 test
    posts, a one-split difference of 0.05 is well inside the sampling noise.
    """
    from scipy import stats as sps

    y = (meta["likes"] + 3.0 * meta["comments"]).astype("float64")
    known = y.notna()

    authors = meta["author"].fillna("__none__").astype(str)
    uniq = sorted(authors.unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    test_authors = set(uniq[: max(1, len(uniq) // 3)])

    in_test = authors.isin(test_authors).to_numpy()
    # Only rows with a KNOWN engagement value can be scored. Posts whose like
    # count Instagram hides carry no target, and leaving them in produced all-NaN
    # correlations — which is how a first run of this harness silently reported
    # nothing at all. The filter is applied to the test split too, and every
    # detector is scored on the identical row set, so it stays fair.
    is_train = (~in_test) & known.to_numpy()
    is_test = in_test & known.to_numpy()

    if is_train.sum() < 20:
        return {"spearman": float("nan"), "n_test": 0, "note": "insufficient train rows"}

    global_median = float(np.median(y.to_numpy()[is_train]))

    group_medians: dict[str, float] = {}
    usable = 0
    for name, members in groups.items():
        if len(members) < min_group:
            continue
        idx = np.array(members, dtype=int)
        tr = idx[is_train[idx]]
        if len(tr) < 3:
            continue
        group_medians[name] = float(np.median(y.to_numpy()[tr]))
        usable += 1
    if not group_medians:
        return {
            "spearman": float("nan"),
            "n_test": 0,
            "n_groups_usable": 0,
            "note": "no group had >=3 train members",
        }

    pred, actual, pred_ctrl = [], [], []
    for i in np.where(is_test)[0]:
        hit = None
        for name, members in groups.items():
            if len(members) < min_group:
                continue
            if i in members:
                hit = name
                break
        p = group_medians.get(hit, global_median) if hit else global_median
        pred.append(p)
        actual.append(float(y.iloc[i]))
        pred_ctrl.append(global_median)

    if len(pred) < 10:
        return {"spearman": float("nan"), "n_test": len(pred), "note": "few test rows"}

    rho = float(sps.spearmanr(pred, actual).statistic)
    mae = float(np.mean(np.abs(np.array(actual) - np.array(pred))))
    mae_ctrl = float(np.mean(np.abs(np.array(actual) - np.array(pred_ctrl))))
    return {
        "spearman": rho,
        "mae": mae,
        "control_mae": mae_ctrl,
        "beats_control": bool(mae < mae_ctrl),
        "n_test": int(len(pred)),
        "n_groups_usable": int(usable),
    }


def trend_precision(
    meta: pd.DataFrame,
    groups: dict[str, list[int]],
    definition=DEFAULT_DEFINITION,
    top_n: int = 5,
) -> dict:
    """
    Of each detector's top-N groups (by member engagement, as any trend product
    would rank them), how many are confirmed Rising by the formal definition?
    """
    verdicts = classify_corpus(meta.reset_index(drop=True), meta["_vis_label"].to_numpy())
    rising = set(
        int(c) for c in verdicts.loc[verdicts["classification"] == "Rising", "cluster_id"]
    )
    y = (meta["likes"] + 3.0 * meta["comments"]).astype("float64")

    ranked = []
    for name, members in groups.items():
        if len(members) < 3:
            continue
        ranked.append((name, float(np.nanmedian(y.iloc[members].to_numpy()))))
    ranked.sort(key=lambda kv: -kv[1])
    top = ranked[:top_n]

    hits = 0
    for name, _ in top:
        members = set(groups[name])
        # A group is a hit if the VISUAL cluster it overlaps is a confirmed
        # rising trend. Baselines have no cluster ids of their own, so the
        # check is on overlap, which is well defined for all detectors.
        overlapped = {
            int(l) for l in meta.loc[sorted(members), "_vis_label"] if l >= 0
        }
        if overlapped & rising:
            hits += 1
    return {
        "claims_made": len(top),
        "confirmed": hits,
        "precision": hits / len(top) if top else float("nan"),
    }


def textual_anonymity(
    meta: pd.DataFrame, groups: dict[str, list[int]], min_group: int = 3
) -> dict:
    """Mean pairwise Jaccard of hashtags / keywords inside groups. Lower = unnamed."""
    tag_scores, kw_scores = [], []
    for members in groups.values():
        if len(members) < min_group:
            continue
        idx = members[:60]  # cap cost; pairwise is O(n^2)
        ts = [hashtags(c) for c in meta.iloc[idx]["caption"]]
        ks = [words(c) for c in meta.iloc[idx]["caption"]]
        if len(ts) > 1:
            sims = [
                len(a & b) / (len(a | b) + 1e-9)
                for i, a in enumerate(ts)
                for b in ts[i + 1 :]
            ]
            if sims:
                tag_scores.append(float(np.mean(sims)))
        if len(ks) > 1:
            sims = [
                len(a & b) / (len(a | b) + 1e-9)
                for i, a in enumerate(ks)
                for b in ks[i + 1 :]
            ]
            if sims:
                kw_scores.append(float(np.mean(sims)))
    return {
        "mean_hashtag_jaccard": float(np.mean(tag_scores)) if tag_scores else float("nan"),
        "mean_keyword_jaccard": float(np.mean(kw_scores)) if kw_scores else float("nan"),
        "n_groups_scored": len(tag_scores),
    }


def cohesion(groups: dict[str, list[int]], y: np.ndarray, min_group: int = 3) -> float:
    """
    1 - mean(within-group std) / overall std of engagement.

    Note this is a property of the grouping, not proof of trend quality: a
    baseline built from engagement ranks is coherent here by construction, so
    a high number is an upper bound rather than an achievement.
    """
    overall = float(np.nanstd(y))
    if overall < 1e-9:
        return float("nan")
    stds = []
    for members in groups.values():
        if len(members) < min_group:
            continue
        v = y[np.array(members, dtype=int)]
        if np.isfinite(v).sum() >= 2:
            stds.append(float(np.nanstd(v)))
    return 1.0 - float(np.mean(stds)) / overall if stds else float("nan")


def main() -> int:
    emb, meta = load_data()
    labels = trendlens_labels(emb)
    meta["_vis_label"] = labels
    n = len(meta)

    detectors = {
        "TrendLens (visual clusters)": detect_trendlens(meta, labels),
        "Hashtag-frequency detection": detect_hashtag(meta),
        "Keyword/Google-Trends-style": detect_keyword(meta),
        "Engagement trending list": detect_engagement(meta),
    }

    y = (meta["likes"] + 3.0 * meta["comments"]).astype("float64").to_numpy()

    print(f"Corpus: {n} posts, "
          f"{int((labels >= 0).sum())} clustered, "
          f"{len(np.unique(labels[labels >= 0]))} visual clusters\n")

    rows: list[dict] = []

    def record(detector: str, metric: str, value, note: str = "") -> None:
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {detector:<32} {metric:<28} {shown:>12}  {note}")
        rows.append(
            {
                "detector": detector,
                "metric": metric,
                "value": round(value, 6) if isinstance(value, float) else value,
                "note": note,
            }
        )

    print("=" * 100)
    print("1. ASSIGNMENT COVERAGE — fraction of corpus assignable to a group")
    print("=" * 100)
    for name, g in detectors.items():
        record(name, "assignment_coverage", assignment_coverage(g, n), f"{len(g)} groups")

    print()
    print("=" * 100)
    print("2. HELD-OUT ENGAGEMENT PREDICTION — split BY ACCOUNT (no leakage)")
    print("=" * 100)
    for name, g in detectors.items():
        r = heldout_engagement_prediction(meta, g)
        record(
            name,
            "spearman_pred_vs_actual",
            r.get("spearman", float("nan")),
            f"n_test={r.get('n_test')} groups={r.get('n_groups_usable')}",
        )
        if "mae" in r:
            record(
                name,
                "mae_vs_median_control",
                r["mae"],
                f"control_mae={r['control_mae']:.1f} "
                f"{'BEATS control' if r['beats_control'] else 'LOSES to control'}",
            )

    print()
    print("=" * 100)
    print("3. TREND-DEFINITION PRECISION — top-5 groups vs the formal definition")
    print("=" * 100)
    for name, g in detectors.items():
        r = trend_precision(meta, g)
        record(
            name,
            "trend_precision_at_5",
            r["precision"],
            f"{r['confirmed']}/{r['claims_made']} claims confirmed",
        )

    print()
    print("=" * 100)
    print("4. TEXTUAL ANONYMY — within-group overlap (low = the group has no name)")
    print("=" * 100)
    for name, g in detectors.items():
        r = textual_anonymity(meta, g)
        record(name, "mean_hashtag_jaccard", r["mean_hashtag_jaccard"],
               f"n_groups={r['n_groups_scored']}")
        record(name, "mean_keyword_jaccard", r["mean_keyword_jaccard"], "")

    print()
    print("=" * 100)
    print("5. GROUP VALUE COHESION (engagement-based; upper bound for the "
          "engagement baseline by construction)")
    print("=" * 100)
    for name, g in detectors.items():
        record(name, "group_value_cohesion", cohesion(g, y), "")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\nSaved {len(rows)} metrics -> {OUT}")

    summary = {
        "corpus_posts": int(n),
        "visual_clusters": int(len(np.unique(labels[labels >= 0]))),
        "engagement_hidden_like_posts_excluded": int(meta["likes"].isna().sum()),
        "account_split": "by author, 1/3 held out",
        "note": (
            "All metrics computed on real Instagram data. No synthetic fixture "
            "and no self-referential labels: the lifecycle metric that previously "
            "scored 1.00 by construction has been replaced by trend_precision "
            "measured against src.trend_definition."
        ),
    }
    (ROOT / "evaluation_real_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

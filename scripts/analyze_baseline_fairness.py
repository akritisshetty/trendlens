#!/usr/bin/env python3
"""
analyze_baseline_fairness.py
-----------------------------
Produces the group-structure numbers reported in the paper's comparison section.

Motivation
----------
The headline metrics in the main comparison table (assignment coverage, mean
within-group hashtag Jaccard, group value cohesion) are all *relative* to how
a detector chooses to group the corpus, and two of them can be inflated
without any group being a meaningful trend:

  * assignment coverage counts a post if it falls in AT LEAST ONE group. A
    detector that emits many overlapping bags drives coverage to 1.0 while
    assigning no coherent trend to anybody.
  * mean within-group Jaccard is averaged per group, so it shrinks toward 0 as
    groups get smaller, independent of what the groups mean.

Measured here, on the same corpus and embeddings as the main evaluation:

  1. PARTITION COVERAGE — fraction of posts assigned to exactly ONE group.
     This is the coverage a trend product can actually claim, since a post in
     three groups has not been given one trend.
  2. GROUP SIZE PROFILE — n_groups, median size, and how many groups fall below
     a minimum trend size (too few posts to establish persistence).
  3. VISUAL PURITY — mean pairwise CLIP cosine similarity inside each group,
     reported size-matched against the proposed solution's group sizes, with a
     permutation test. This is the metric that tests the central claim directly:
     are the groups visually coherent? Text-driven grouping cannot be assumed to
     achieve it, and it is measured in the same embedding space for all
     detectors, so it is comparable.
  4. A distinctive-keyword variant of the keyword baseline (document frequency
     capped) to rule out the objection that the baseline is merely picking up
     boilerplate tokens.

Run:  venv/bin/python scripts/analyze_baseline_fairness.py
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_real import (  # noqa: E402
    detect_hashtag,
    detect_keyword,
    load_data,
    trendlens_labels,
    words,
)

OUT_CSV = ROOT / "baseline_fairness_results.csv"
OUT_JSON = ROOT / "baseline_fairness_summary.json"

MIN_TREND_SIZE = 10  # below this a group cannot show persistence or a growth rate
N_PERM = 20000
SEED = 0

# Tokens that carry no topical meaning on Instagram. Used ONLY to report a
# diagnostic (what fraction of the baseline's groups are keyed on boilerplate),
# never to improve a baseline's score.
BOILERPLATE = set(
    """link bio all can see full his look their some they through every time her them
years tap back little way off the a an and or but for with from this that these those are
was were is not on at by to of in it its you your we our have has had be been being do does
did as if so then just like what when where how why which who whom will would should could
more most than about only into over out up down now get got also new one two first last day""".split()
)


# ── metrics ────────────────────────────────────────────────────────────────
def membership_counts(groups: dict[str, list[int]], n_posts: int) -> np.ndarray:
    counts: defaultdict[int, int] = defaultdict(int)
    for members in groups.values():
        for i in members:
            counts[i] += 1
    return np.array([counts.get(i, 0) for i in range(n_posts)], dtype=int)


def visual_purity(members: list[int], emb: np.ndarray) -> float:
    """Mean pairwise CLIP cosine similarity within one group."""
    idx = np.asarray(members, dtype=int)
    X = emb[idx]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms
    S = X @ X.T
    iu = np.triu_indices(len(idx), 1)
    return float(S[iu].mean())


def group_profile(groups: dict[str, list[int]], n_posts: int) -> dict:
    sizes = np.array([len(v) for v in groups.values()], dtype=int)
    counts = membership_counts(groups, n_posts)
    return {
        "n_groups": int(sizes.size),
        "median_group_size": float(np.median(sizes)) if sizes.size else None,
        "mean_group_size": round(float(sizes.mean()), 2) if sizes.size else None,
        "max_group_size": int(sizes.max()) if sizes.size else 0,
        "coverage_any_group": round(float((counts > 0).mean()), 4),
        "coverage_exactly_one_group": round(float((counts == 1).mean()), 4),
        "mean_groups_per_post": round(float(counts.mean()), 3),
        "share_posts_in_multi_groups": round(float((counts > 1).mean()), 4),
        "groups_below_min_trend_size": int((sizes < MIN_TREND_SIZE).sum()),
        "share_posts_in_small_groups": (
            round(float(sizes[sizes < MIN_TREND_SIZE].sum() / sizes.sum()), 4)
            if sizes.sum() else None
        ),
    }


def size_matched_purity(groups: dict[str, list[int]], emb: np.ndarray,
                        target_sizes: list[int], trials: int = 20,
                        seed: int = SEED) -> tuple[float, float]:
    """
    Subsample each baseline group to the proposed solution's group sizes, so
    purity is compared at equal group size and cannot be inflated by
    fragmentation.
    """
    pool = [m for m in groups.values() if len(m) >= 3]
    if not pool or not target_sizes:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(trials):
        per_trial = []
        for target in target_sizes:
            best = min(pool, key=lambda m: abs(len(m) - target))
            take = min(target, len(best))
            per_trial.append(visual_purity(rng.choice(best, size=take, replace=False), emb))
        means.append(float(np.mean(per_trial)))
    return float(np.mean(means)), float(np.std(means))


def permutation_p(pure_a: np.ndarray, pure_b: np.ndarray,
                  seed: int = SEED) -> float:
    """One-sided p: is mean(a) > mean(b)? Pooled, two-sample style."""
    rng = np.random.default_rng(seed)
    obs = float(pure_a.mean())
    pool = np.concatenate([pure_a, pure_b])
    hits = sum(
        1 for _ in range(N_PERM)
        if np.mean(rng.choice(pool, size=pure_a.size, replace=True)) >= obs
    )
    return (hits + 1) / (N_PERM + 1)


def raw_hashtag_jaccard(groups: dict[str, list[int]], meta: pd.DataFrame,
                       min_size: int = 3) -> float:
    """Mean within-group hashtag Jaccard, averaged over all groups >= min_size."""
    from scripts.evaluate_real import hashtags as _tags

    scores = []
    for members in groups.values():
        if len(members) < min_size:
            continue
        caps = [_tags(meta.iloc[i]["caption"]) for i in members]
        sims = [len(a & b) / (len(a | b) + 1e-9)
                for i, a in enumerate(caps) for b in caps[i + 1:]]
        if sims:
            scores.append(float(np.mean(sims)))
    return float(np.mean(scores)) if scores else float("nan")


def per_group_purity(groups: dict[str, list[int]], emb: np.ndarray,
                     min_size: int = 3) -> np.ndarray:
    return np.array(
        [visual_purity(m, emb) for m in groups.values() if len(m) >= min_size],
        dtype=float,
    )


# ── baselines ──────────────────────────────────────────────────────────────
def detect_distinctive_keyword(meta: pd.DataFrame, max_df_frac: float = 0.05
                               ) -> dict[str, list[int]]:
    """
    Keyword baseline restricted to reasonably distinctive tokens, so it cannot
    win by grouping on words that appear nearly everywhere. This is a STRONGER
    baseline than detect_keyword, and it is reported to show the conclusion is
    not an artefact of boilerplate tokens.
    """
    ws = meta["caption"].map(words)
    df: Counter = Counter()
    for s in ws:
        for w in s:
            df[w] += 1
    cap = max_df_frac * len(meta)
    groups: dict[str, list[int]] = {}
    for w, c in df.most_common():
        if c <= cap:
            members = [i for i, s in enumerate(ws) if w in s]
            if len(members) > 1:
                groups[w] = members
    return groups


def main() -> int:
    emb, meta = load_data()
    labels = trendlens_labels(emb)
    n = len(meta)

    detectors = {
        "Proposed": {
            f"v{lb}": [i for i, x in enumerate(labels) if x == lb]
            for lb in np.unique(labels[labels >= 0])
        },
        "Hashtag": detect_hashtag(meta),
        "Keyword": detect_keyword(meta),
        "Keyword-distinctive": detect_distinctive_keyword(meta),
    }

    target_sizes = sorted(
        (len(v) for v in detectors["Proposed"].values()), reverse=True
    )
    pure = {k: per_group_purity(v, emb) for k, v in detectors.items()}
    prop_pure = pure["Proposed"]

    rows: list[dict] = []

    def add(detector: str, metric: str, value, note: str = "") -> None:
        rows.append(dict(detector=detector, metric=metric, value=value, note=note))
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {detector:<22} {metric:<34} {shown:>12}  {note}")

    print("=" * 100)
    print(f"BASELINE FAIRNESS / GROUP-STRUCTURE DIAGNOSTIC  "
          f"(corpus = {n} posts, min trend size = {MIN_TREND_SIZE})")
    print("=" * 100)

    print("\n1. GROUP STRUCTURE")
    for name, g in detectors.items():
        print(f"\n{name}")
        for k, v in group_profile(g, n).items():
            add(name, k, v)

    print("\n2. VISUAL PURITY (mean intra-group CLIP cosine)")
    for name in detectors:
        add(name, "visual_purity_mean", round(float(pure[name].mean()), 4),
            f"k={pure[name].size} groups")
    for name in ("Hashtag", "Keyword", "Keyword-distinctive"):
        m, sd = size_matched_purity(detectors[name], emb, target_sizes)
        add(name, "visual_purity_size_matched", round(m, 4), f"sd={sd:.4f}")
        add(name, "permutation_p_vs_proposed",
            round(permutation_p(prop_pure, pure[name]), 4), "one-sided, 20k perms")

    print("\n3. WHAT THE KEYWORD BASELINE IS ACTUALLY GROUPING ON")
    kw = detectors["Keyword"]
    n_boiler = sum(1 for k in kw if k.lower() in BOILERPLATE)
    boiler_posts = sum(
        len(v) for k, v in kw.items() if k.lower() in BOILERPLATE
    )
    add("Keyword", "groups_keyed_on_boilerplate", n_boiler, f"of {len(kw)} groups")
    add("Keyword", "assignments_on_boilerplate_groups", boiler_posts,
        "posts (overlapping, so > corpus size)")
    for key in ("link", "bio", "all"):
        if key in kw:
            add("Keyword", f"largest_group_{key}", len(kw[key]),
                f"{len(kw[key])/n:.1%} of corpus in one group")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    summary = {
        "corpus_posts": n,
        "min_trend_size": MIN_TREND_SIZE,
        "permutation_draws": N_PERM,
        "detectors": {},
    }
    for name, g in detectors.items():
        summary["detectors"][name] = {
            **group_profile(g, n),
            "visual_purity_mean": round(float(pure[name].mean()), 4),
            "visual_purity_size_matched": (
                round(size_matched_purity(g, emb, target_sizes)[0], 4)
                if name != "Proposed" else round(float(pure[name].mean()), 4)
            ),
            "permutation_p_vs_proposed": (
                None if name == "Proposed"
                else round(permutation_p(prop_pure, pure[name]), 4)
            ),
            "mean_hashtag_jaccard": round(raw_hashtag_jaccard(g, meta), 4),
        }
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    hdr = (f"{'detector':<22}{'cov(any)':>10}{'cov(exactly1)':>16}"
           f"{'med_sz':>9}{'purity':>10}{'purity_sm':>11}{'p':>9}")
    print(hdr)
    for name, s in summary["detectors"].items():
        p = s["permutation_p_vs_proposed"]
        p_s = f"{p:.4f}" if p is not None else "-"
        print(f"{name:<22}{s['coverage_any_group']:>10.4f}"
              f"{s['coverage_exactly_one_group']:>16.4f}"
              f"{s['median_group_size']:>9.1f}"
              f"{s['visual_purity_mean']:>10.4f}"
              f"{s['visual_purity_size_matched']:>11.4f}{p_s:>9}")

    print(f"\nSaved -> {OUT_CSV}")
    print(f"Saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
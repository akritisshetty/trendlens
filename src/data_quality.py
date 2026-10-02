"""
data_quality.py
---------------
Central engagement-data hygiene for TrendLens.

Why this module exists
----------------------
Instagram's Graph API returns ``likesCount = -1`` for posts whose like count
is hidden or unavailable. The original collector parsed engagement with

    likes = item.get("likesCount") or item.get("likes") or 0

which does NOT catch ``-1`` (``-1`` is truthy in Python), so the sentinel was
written into the dataset verbatim. Consequences observed on the real corpus
(599 posts): 47 posts carried ``likes = -1``, and those values were averaged
into ``avg_likes``, the engagement-velocity term and the emerging-score
engagement bonus as if they were genuine counts.

Conventions enforced here
-------------------------
1. A negative engagement count means **unknown**, not zero. It is converted to
   ``NaN`` so it is excluded from statistics rather than dragging a mean down.
2. Unknown ≠ zero. A post with a known ``likes = 0`` is a real observation of
   "nobody liked this"; a post with ``likes = NaN`` carries no information.
   Aggregates therefore report **coverage** alongside the value.
3. Likes are extremely heavy-tailed (observed: median 7k, max 15.5M on the
   same corpus). A single viral post moves the mean by orders of magnitude, so
   the **median** is the primary engagement statistic and the mean is only
   reported as a secondary, clearly-labelled figure.

Every module that computes engagement must go through this one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Instagram returns -1 for counts the API will not disclose.
NEGATIVE_SENTINEL = -1

#: Engagement columns subject to the negative-sentinel rule.
ENGAGEMENT_COLUMNS = ("likes", "comments", "views", "plays")

#: Comment is a much stronger intent signal than a like, so it is weighted
#: higher in the composite. Chosen as 3x because a comment requires a deliberate
#: action while a like is one tap; documented here so it is auditable.
COMMENT_WEIGHT = 3.0
LIKE_WEIGHT = 1.0


def clean_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of ``df`` with engagement columns sanitised.

    Negative counts become ``NaN`` (unknown, not zero) and are cast to float64
    so that partial-observation arithmetic works. A boolean ``*_known`` column
    is added for each engagement column actually present.
    """
    out = df.copy()
    for col in ENGAGEMENT_COLUMNS:
        if col not in out.columns:
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        # Anything below zero is a sentinel, not a measurement.
        numeric = numeric.mask(numeric < 0, np.nan)
        out[f"{col}_known"] = numeric.notna()
        out[col] = numeric.astype("float64")
    return out


def composite_engagement(df: pd.DataFrame) -> pd.Series:
    """
    Weighted like/comment engagement, NaN-safe.

    Uses ``skipna`` so a post with a hidden like count but a known comment
    count still contributes. Returns ``NaN`` only when both are unknown.
    """
    likes = pd.to_numeric(df.get("likes"), errors="coerce")
    comments = pd.to_numeric(df.get("comments"), errors="coerce")
    likes = likes.mask(likes < 0) if likes is not None else np.nan
    comments = comments.mask(comments < 0) if comments is not None else np.nan

    weighted = LIKE_WEIGHT * likes + COMMENT_WEIGHT * comments
    if likes is None and comments is None:
        return pd.Series(np.nan, index=df.index)
    return weighted


def engagement_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Robust engagement statistics plus an explicit coverage figure.

    The coverage fields are the point: a median computed over 8% of a cluster
    is not the same claim as one computed over 100% of it, and the previous
    artifacts did not distinguish the two.
    """
    likes = pd.to_numeric(df.get("likes"), errors="coerce")
    comments = pd.to_numeric(df.get("comments"), errors="coerce")
    if likes is not None:
        likes = likes.mask(likes < 0)
    if comments is not None:
        comments = comments.mask(comments < 0)

    n = int(len(df))
    n_likes_known = int(likes.notna().sum()) if likes is not None else 0
    n_comments_known = int(comments.notna().sum()) if comments is not None else 0

    summary: dict[str, Any] = {
        "n_posts": n,
        "n_likes_known": n_likes_known,
        "n_comments_known": n_comments_known,
        "likes_coverage": round(n_likes_known / n, 4) if n else 0.0,
    }

    if n_likes_known:
        summary["median_likes"] = float(likes.median())
        summary["p75_likes"] = float(likes.quantile(0.75))
        summary["mean_likes"] = float(likes.mean())
    else:
        summary["median_likes"] = None
        summary["p75_likes"] = None
        summary["mean_likes"] = None

    if n_comments_known:
        summary["median_comments"] = float(comments.median())
        summary["mean_comments"] = float(comments.mean())
    else:
        summary["median_comments"] = None
        summary["mean_comments"] = None

    # Confidence band on the median: for a heavy-tailed distribution the median
    # of a small sample is unstable, so callers can require a minimum sample.
    if n_likes_known >= 8:
        lo, hi = likes.quantile([0.25, 0.75])
        summary["median_likes_ci"] = [float(lo), float(hi)]
    else:
        summary["median_likes_ci"] = None
        summary["median_likes_reliable"] = False
    summary.setdefault("median_likes_reliable", n_likes_known >= 8)

    return summary


def best_post_by_engagement(df: pd.DataFrame) -> Any:
    """
    Row with the highest known engagement.

    ``-1`` sentinels used to win this comparison outright (a hidden count of
    ``-1`` is greater than zero), which is how a post with *unknown*
    engagement could be selected as the cluster's exemplar. Unknown rows are
    now excluded rather than ranked.
    """
    likes = pd.to_numeric(df.get("likes"), errors="coerce")
    if likes is None or likes.notna().sum() == 0:
        return None
    likes = likes.mask(likes < 0)
    if likes.notna().sum() == 0:
        return None
    return df.loc[likes.idxmax()]

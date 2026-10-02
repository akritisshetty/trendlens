"""
trend_definition.py
-------------------
A single, falsifiable definition of what TrendLens means by a "trend", plus the
longitudinal snapshot store that makes the definition testable over time.

Why this module had to be written
--------------------------------
The project previously used "trend" to mean several different things computed
in several different places, none of them documented against a fixed denominator:

  * ``compute_temporal_trends`` divided a raw post count by ``log1p(total)``,
    producing an unbounded score where a 50-post cluster structurally dominates
    a 10-post cluster. Values ranged 0.0 to 13.5 on the real corpus and the
    number was not comparable between clusters.
  * ``src/trends.py`` computed ``(recent - prior) / (prior + 1)`` over 3-period
    windows, with no support requirement.
  * The README described growth as "+200% vs prior period".

None of these agree, and the growth figures they produced on the real corpus
were dominated by sampling noise: 87% of the 599 posts fall in a single 11-day
window, so a 3-day recent-vs-prior comparison over per-cluster daily counts of
1-5 is comparing 2 posts against 3.

The definition below is deliberately strict. Its purpose is to make TrendLens
say "I do not know" on a single snapshot instead of emitting a confident number
that cannot be supported.

The definition
--------------
A visual cluster ``c`` is a **rising trend** at time ``t`` only if ALL of:

  G1. SUPPORT      recent window has >= ``min_recent_posts`` posts
  G2. PERSISTENCE  the recent window is not a single spike: the cluster appears
                   on >= ``min_active_days`` distinct days
  G3. GROWTH       recent rate exceeds the prior rate by a relative margin
                   >= ``growth_margin``, where both rates are normalised
                   per-day so unequal window lengths cannot manufacture growth
  G4. SIGNIFICANCE the growth survives a Poisson/binomial null: the probability
                   of seeing >= the observed recent count under a
                   no-growth (equal-rate) null is <= ``alpha``
  G5. ENGAGEMENT   median engagement per post in the recent window is not
                   materially below the corpus median (optional, on by default)
  G6. COVERAGE     engagement is known for >= ``min_engagement_coverage`` of
                   the recent-window posts

Otherwise the cluster is reported as **insufficient data** rather than being
silently ranked. ``insufficient data`` is a first-class outcome: on the current
single-snapshot corpus most clusters land there, and that is the correct
answer, not a failure.

Two distinct notions of growth are reported and never conflated:

  * ``post_volume_growth``   — are there more POSTS of this look? (supply)
  * ``engagement_growth``    — do the SAME posts earn more likes? (demand)

These can and do move independently. Instagram supply can spike because one
large account posted a 12-part series; that is not a trend.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import config
from src.data_quality import engagement_summary


# ──────────────────────────────────────────────────────────────────────────
# Definition parameters (all documented, all overridable)
# ──────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TrendDefinition:
    """
    Thresholds that operationalise the definition above.

    Defaults are set conservatively for the *observed* corpus scale
    (n ~= 600 posts across ~100 accounts). On a substantially larger corpus
    ``min_recent_posts`` and ``min_active_days`` should be raised; they are
    parameters, not constants, precisely so this is a one-line change.
    """

    #: G1 — minimum posts in the recent window before growth may be claimed.
    min_recent_posts: int = 8
    #: G2 — minimum distinct days with >= 1 post, to exclude single-spike noise.
    min_active_days: int = 3
    #: G3 — required relative margin of recent over prior per-day rate.
    growth_margin: float = 0.50
    #: G4 — one-sided significance threshold for the growth test.
    alpha: float = 0.05
    #: G6 — minimum share of recent posts with a known engagement value.
    min_engagement_coverage: float = 0.60
    #: G5 — reject if recent median engagement is below this multiple of corpus median.
    engagement_floor_multiple: float = 0.75
    #: Length of the recent and prior comparison windows, in days.
    recent_days: int = 4
    prior_days: int = 4
    #: Absolute floor on prior-window posts; below this growth is unmeasurable
    #: (a 0->4 jump is 100% growth in the arithmetic and 0% in reality).
    min_prior_posts: int = 4
    #: Minimum number of distinct accounts contributing to the recent window.
    #: A "trend" driven by one account is that account's content strategy.
    min_recent_authors: int = 3

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Default definition used by the pipeline.
DEFAULT_DEFINITION = TrendDefinition()

#: Outcomes a cluster can be assigned. `INSUFFICIENT` is a real, expected state.
RISING = "Rising"
FALLING = "Falling"
FLAT = "Flat"
INSUFFICIENT = "InsufficientData"


# ──────────────────────────────────────────────────────────────────────────
# Significance testing
# ──────────────────────────────────────────────────────────────────────────
def poisson_growth_test(
    n_recent: int,
    n_prior: int,
    recent_days: float,
    prior_days: float,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    One-sided test of ``recent rate > prior rate``.

    Under the null hypothesis of *no growth* the two windows observe the same
    underlying rate. The recent window then behaves as a binomial draw with
    ``p = n_prior / (n_prior + n_recent)`` (ratio-of-counts estimate, which is
    also the MLE for the pooled rate), and we ask how surprising the observed
    split is in the upward tail.

    Returns the exact one-sided p-value plus the observed rate ratio. A plain
    percentage is deliberately NOT sufficient: ``0 -> 4`` posts is +400% and
    carries no evidence whatsoever, and this test is what prevents that from
    being reported as a trend.
    """
    n_recent = int(n_recent)
    n_prior = int(n_prior)
    total = n_recent + n_prior
    if total == 0 or n_recent == 0:
        return {
            "p_value": 1.0,
            "rate_ratio": None,
            "recent_rate_per_day": n_recent / recent_days if recent_days else 0.0,
            "prior_rate_per_day": n_prior / prior_days if prior_days else 0.0,
            "testable": False,
            "reason": "no recent posts" if n_recent == 0 else "no posts in either window",
        }

    p_null = n_prior / total
    # P(X >= n_recent) under Binomial(total, p_null)
    p_value = float(
        sum(math.comb(total, k) * p_null**k * (1 - p_null) ** (total - k)
            for k in range(n_recent, total + 1))
    )
    p_value = min(1.0, max(0.0, p_value))

    recent_rate = n_recent / recent_days if recent_days else 0.0
    prior_rate = n_prior / prior_days if prior_days else 0.0
    if prior_rate > 0:
        rate_ratio: Optional[float] = recent_rate / prior_rate
        rel_growth = recent_rate / prior_rate - 1.0
    else:
        rate_ratio = None
        rel_growth = None

    return {
        "p_value": p_value,
        "rate_ratio": rate_ratio,
        "relative_growth": rel_growth,
        "recent_rate_per_day": recent_rate,
        "prior_rate_per_day": prior_rate,
        "testable": True,
        "significant": bool(p_value <= alpha),
    }


# ──────────────────────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────────────────────
def _window_masks(
    ts: pd.Series, now: pd.Timestamp, recent_days: int, prior_days: int
) -> tuple[pd.Series, pd.Series]:
    recent_cutoff = now - timedelta(days=recent_days)
    prior_cutoff = recent_cutoff - timedelta(days=prior_days)
    ts_utc = ts.dt.tz_convert(timezone.utc) if ts.dt.tz is not None else ts
    return (ts_utc >= recent_cutoff), (ts_utc >= prior_cutoff) & (ts_utc < recent_cutoff)


def classify_cluster_trend(
    members: pd.DataFrame,
    now: pd.Timestamp,
    definition: TrendDefinition = DEFAULT_DEFINITION,
    corpus_median_likes: Optional[float] = None,
) -> dict[str, Any]:
    """
    Apply the formal definition to one cluster's posts.

    Returns a verdict record containing the classification, every component
    criterion as a named boolean, and the evidence behind it. The named
    criteria matter: when a cluster is rejected the caller can report *which*
    condition failed instead of a bare "not a trend".
    """
    ts = pd.to_datetime(members["timestamp"], utc=True, errors="coerce")
    recent_mask, prior_mask = _window_masks(
        ts, now, definition.recent_days, definition.prior_days
    )

    n_recent = int(recent_mask.sum())
    n_prior = int(prior_mask.sum())
    n_total = int(len(members))

    recent_posts = members[recent_mask]
    active_days = int(recent_posts["timestamp"].dt.date.nunique()) if n_recent else 0
    author_col = "author" if "author" in members.columns else None
    n_authors = (
        int(recent_posts[author_col].nunique())
        if n_recent and author_col
        else 0
    )

    test = poisson_growth_test(
        n_recent=n_recent,
        n_prior=n_prior,
        recent_days=definition.recent_days,
        prior_days=definition.prior_days,
        alpha=definition.alpha,
    )

    # Engagement demand signal + coverage
    recent_eng = engagement_summary(recent_posts) if n_recent else {
        "median_likes": None,
        "likes_coverage": 0.0,
        "n_likes_known": 0,
        "median_likes_reliable": False,
    }
    coverage = float(recent_eng.get("likes_coverage", 0.0))
    rel_growth = test.get("relative_growth")

    criteria = {
        "G1_support": n_recent >= definition.min_recent_posts,
        "G2_persistence": active_days >= definition.min_active_days,
        "G2b_author_breadth": n_authors >= definition.min_recent_authors,
        "G3_growth_margin": (
            rel_growth is not None and rel_growth >= definition.growth_margin
        ),
        "G4_significance": bool(test.get("significant", False)),
        "G5_engagement_not_depressed": (
            corpus_median_likes is None
            or recent_eng.get("median_likes") is None
            or recent_eng["median_likes"]
            >= definition.engagement_floor_multiple * corpus_median_likes
        ),
        "G6_engagement_coverage": coverage >= definition.min_engagement_coverage,
    }

    # A prior window too small to compare against makes growth unmeasurable.
    measurable = n_prior >= definition.min_prior_posts

    failed = [k for k, ok in criteria.items() if not ok]

    if not measurable:
        classification = INSUFFICIENT
        reason = (
            f"only {n_prior} post(s) in the prior {definition.prior_days}d window "
            f"(need >= {definition.min_prior_posts} to measure growth); "
            f"a small absolute jump here is sampling noise, not a trend"
        )
    elif all(criteria.values()):
        classification = RISING
        reason = "all definition criteria satisfied"
    else:
        classification = INSUFFICIENT
        pretty = {
            "G1_support": f"only {n_recent} recent post(s), need >= {definition.min_recent_posts}",
            "G2_persistence": f"present on {active_days} day(s), need >= {definition.min_active_days}",
            "G2b_author_breadth": f"from {n_authors} account(s), need >= {definition.min_recent_authors}",
            "G3_growth_margin": (
                f"relative growth {rel_growth:+.0%}, need >= +{definition.growth_margin:.0%}"
                if rel_growth is not None
                else "growth unmeasurable (zero prior rate)"
            ),
            "G4_significance": (
                f"p={test.get('p_value', 1.0):.3f} > alpha={definition.alpha}"
            ),
            "G5_engagement_not_depressed": (
                f"recent median {recent_eng.get('median_likes')} below corpus median {corpus_median_likes}"
            ),
            "G6_engagement_coverage": (
                f"engagement known for {coverage:.0%} of recent posts, "
                f"need >= {definition.min_engagement_coverage:.0%}"
            ),
        }
        reason = "; ".join(pretty.get(k, k) for k in failed)

    return {
        "classification": classification,
        "reason": reason,
        "failed_criteria": failed,
        "criteria": criteria,
        "n_total": n_total,
        "n_recent": n_recent,
        "n_prior": n_prior,
        "recent_active_days": active_days,
        "recent_authors": n_authors,
        "recent_rate_per_day": round(float(test.get("recent_rate_per_day", 0.0)), 4),
        "prior_rate_per_day": round(float(test.get("prior_rate_per_day", 0.0)), 4),
        "rate_ratio": (
            round(test["rate_ratio"], 4) if test.get("rate_ratio") is not None else None
        ),
        "relative_growth": (
            round(rel_growth, 4) if rel_growth is not None else None
        ),
        "p_value": round(float(test.get("p_value", 1.0)), 6),
        "significant": bool(test.get("significant", False)),
        "recent_median_likes": recent_eng.get("median_likes"),
        "recent_likes_coverage": round(coverage, 4),
        "recent_median_reliable": bool(recent_eng.get("median_likes_reliable", False)),
    }


# ──────────────────────────────────────────────────────────────────────────
# Corpus-level driver
# ──────────────────────────────────────────────────────────────────────────
def classify_corpus(
    df: pd.DataFrame,
    labels: np.ndarray,
    definition: TrendDefinition = DEFAULT_DEFINITION,
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """
    Classify every cluster in the corpus under the formal definition.
    """
    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame[cluster_col] = labels

    from src.data_quality import clean_engagement

    frame = clean_engagement(frame)

    # HDBSCAN's -1 bucket is "unclustered noise", not a trend. It is the
    # largest single group and trivially out-grows every real cluster, so
    # leaving it in would manufacture a headline trend out of residue.
    frame = frame[frame[cluster_col] >= 0]

    now = frame["timestamp"].max()
    if pd.isna(now):
        now = pd.Timestamp.now(tz=timezone.utc)
    corpus_median = (
        float(frame["likes"].median()) if frame["likes"].notna().any() else None
    )

    rows = []
    for cid, members in frame.groupby(cluster_col):
        verdict = classify_cluster_trend(
            members, now=now, definition=definition, corpus_median_likes=corpus_median
        )
        eng = engagement_summary(members)
        verdict["cluster_id"] = int(cid)
        verdict["median_likes_alltime"] = eng.get("median_likes")
        verdict["mean_likes_alltime"] = eng.get("mean_likes")
        verdict["median_comments_alltime"] = eng.get("median_comments")
        verdict["likes_coverage_alltime"] = eng.get("likes_coverage")
        rows.append(verdict)

    out = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Longitudinal snapshot store
# ──────────────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id   TEXT PRIMARY KEY,
    captured_at   TEXT NOT NULL,
    source        TEXT,
    n_posts       INTEGER,
    n_clusters    INTEGER,
    window_days   INTEGER,
    corpus_fp     TEXT,
    definition    TEXT
);

-- One row per (cluster, snapshot). Cluster identity is the stable UUID from
-- the cluster registry, so a cluster keeps its history across re-clustering.
CREATE TABLE IF NOT EXISTS cluster_observations (
    snapshot_id        TEXT NOT NULL,
    cluster_id         TEXT NOT NULL,
    theme_name         TEXT,
    n_total            INTEGER,
    n_recent           INTEGER,
    n_prior            INTEGER,
    recent_rate_per_day  REAL,
    prior_rate_per_day   REAL,
    rate_ratio         REAL,
    relative_growth    REAL,
    p_value            REAL,
    classification     TEXT,
    reason             TEXT,
    median_likes       REAL,
    likes_coverage     REAL,
    PRIMARY KEY (snapshot_id, cluster_id),
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id)
);

-- Daily post counts per cluster, so history can be re-derived at any window
-- length without re-scraping.
CREATE TABLE IF NOT EXISTS daily_counts (
    cluster_id  TEXT NOT NULL,
    day         TEXT NOT NULL,
    n_posts     INTEGER NOT NULL,
    PRIMARY KEY (cluster_id, day)
);

CREATE INDEX IF NOT EXISTS idx_obs_cluster ON cluster_observations(cluster_id);
CREATE INDEX IF NOT EXISTS idx_daily_day ON daily_counts(day);
"""


class TrendHistory:
    """
    SQLite-backed store for cross-snapshot trend history.

    Today the pipeline has exactly one snapshot, so this is initially a
    single-row history. Its purpose is that every subsequent run appends, and
    the longitudinal claims in the README become true as the collector runs
    daily — rather than being asserted on day one and never checked.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else config.DATA_DIR / "trend_history.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        # Migrations for stores created before corpus_fp existed.
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(snapshots)")}
        if "corpus_fp" not in cols:
            self.conn.execute("ALTER TABLE snapshots ADD COLUMN corpus_fp TEXT")
        self.conn.commit()

    # -- corpus identity -------------------------------------------------
    @staticmethod
    def corpus_fingerprint(post_ids: Any) -> str:
        """
        Stable fingerprint of WHICH posts an observation covers.

        This exists because of a concrete failure. Re-running the pipeline over
        an unchanged corpus — for example after editing a clustering
        hyperparameter — is a RECOMPUTATION, not a new observation in time. With
        461 posts re-clustered from 12 groups to 18, an unguarded store reported
        a 50% jump in cluster count as though the corpus had changed, and any
        longitudinal delta computed across those two snapshots would have
        described a config edit rather than the passage of time.

        The fingerprint lets the store refuse to treat a re-scrape of the same
        posts as new history.
        """
        ids = sorted(str(p) for p in pd.Series(post_ids).dropna().unique())
        return hashlib.sha256("|".join(ids).encode()).hexdigest()[:16]

    def _same_corpus(self, fingerprint: str) -> bool:
        row = self.conn.execute(
            "SELECT snapshot_id FROM snapshots WHERE corpus_fp = ? "
            "ORDER BY captured_at DESC LIMIT 1",
            (fingerprint,),
        ).fetchone()
        return row is not None

    # -- snapshots -------------------------------------------------------
    def record_snapshot(
        self,
        observations: pd.DataFrame,
        daily: pd.DataFrame,
        source: str = "instagram",
        window_days: Optional[int] = None,
        definition: TrendDefinition = DEFAULT_DEFINITION,
        n_posts: int = 0,
        post_ids: Optional[Any] = None,
        allow_duplicate_corpus: bool = False,
    ) -> Optional[str]:
        """
        Append one observation set. Re-running with the same ``snapshot_id``
        replaces that snapshot rather than duplicating it.

        Returns the new snapshot id, or ``None`` if the observation was
        rejected as a duplicate of the existing corpus (see
        ``corpus_fingerprint``). Pass ``allow_duplicate_corpus=True`` to
        record it anyway — for example when a genuinely new scrape returns
        exactly the same post set because the accounts posted nothing new, in
        which case the unchanged count is itself the informative result.
        """
        fingerprint = self.corpus_fingerprint(post_ids) if post_ids is not None else None
        if (
            fingerprint
            and not allow_duplicate_corpus
            and self._same_corpus(fingerprint)
        ):
            return None

        snapshot_id = datetime.now(timezone.utc).strftime("snap_%Y%m%dT%H%M%SZ")
        captured = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            "INSERT OR REPLACE INTO snapshots "
            "(snapshot_id, captured_at, source, n_posts, n_clusters, window_days, "
            " corpus_fp, definition)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (
                snapshot_id,
                captured,
                source,
                int(n_posts),
                int(len(observations)),
                window_days,
                fingerprint,
                json.dumps(definition.as_dict(), sort_keys=True),
            ),
        )

        rows = [
            (
                snapshot_id,
                str(r["cluster_id"]),
                r.get("theme_name"),
                r.get("n_total"),
                r.get("n_recent"),
                r.get("n_prior"),
                r.get("recent_rate_per_day"),
                r.get("prior_rate_per_day"),
                r.get("rate_ratio"),
                r.get("relative_growth"),
                r.get("p_value"),
                r.get("classification"),
                r.get("reason"),
                r.get("median_likes"),
                r.get("likes_coverage"),
            )
            for _, r in observations.iterrows()
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO cluster_observations VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )

        if not daily.empty:
            self.conn.executemany(
                "INSERT OR REPLACE INTO daily_counts VALUES (?,?,?)",
                [
                    (str(r["cluster_id"]), str(r["day"]), int(r["n_posts"]))
                    for _, r in daily.iterrows()
                ],
            )

        self.conn.commit()
        return snapshot_id

    # -- queries ---------------------------------------------------------
    def n_snapshots(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0])

    def series_for_cluster(self, cluster_id: str) -> pd.DataFrame:
        """Observation history for one cluster, oldest first."""
        return pd.read_sql_query(
            "SELECT captured_at, n_total, n_recent, n_prior, rate_ratio, "
            "relative_growth, p_value, classification, median_likes "
            "FROM cluster_observations o JOIN snapshots s USING(snapshot_id) "
            "WHERE cluster_id = ? ORDER BY s.captured_at",
            self.conn,
            params=(str(cluster_id),),
        )

    def longitudinal_summary(self) -> pd.DataFrame:
        """
        Per-cluster change between the first and most recent snapshot.

        Only snapshots over DIFFERENT post sets are compared. Two runs over the
        same 461 posts are the same observation of the world recomputed, and
        differencing them would report a config edit as a trend. An empty frame
        is returned when fewer than two distinct corpora exist — which is the
        honest behaviour, and the reason this cannot yet confirm any
        longitudinal claim.
        """
        rows = self.conn.execute(
            "SELECT snapshot_id, corpus_fp FROM snapshots ORDER BY captured_at"
        ).fetchall()
        if not rows:
            return self._empty_longitudinal()
        first_fp = rows[0][1]
        # Most recent snapshot whose corpus differs from the first.
        last = next((r for r in reversed(rows) if r[1] != first_fp), None)
        if last is None:
            return self._empty_longitudinal()
        first_id, last_id = rows[0][0], last[0]
        return pd.read_sql_query(
            """
            SELECT o.cluster_id,
                   COUNT(*)                AS n_snapshots,
                   MAX(CASE WHEN o.snapshot_id = ? THEN o.n_total END)      AS first_n_total,
                   MAX(CASE WHEN o.snapshot_id = ? THEN o.n_total END)      AS latest_n_total,
                   MAX(CASE WHEN o.snapshot_id = ? THEN o.median_likes END) AS first_median_likes,
                   MAX(CASE WHEN o.snapshot_id = ? THEN o.median_likes END) AS latest_median_likes
            FROM cluster_observations o
            WHERE o.snapshot_id IN (?, ?)
            GROUP BY o.cluster_id
            """,
            self.conn,
            params=(first_id, last_id, first_id, last_id, first_id, last_id),
        )

    @staticmethod
    def _empty_longitudinal() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "n_snapshots",
                "first_n_total",
                "latest_n_total",
                "delta_n_total",
                "first_median_likes",
                "latest_median_likes",
            ]
        )

    def history_status(self) -> dict[str, Any]:
        """
        Whether the store currently supports a longitudinal claim.

        Reported to callers so the UI and any report can say "insufficient
        history" explicitly instead of silently presenting a single snapshot's
        numbers as if they were a trend.
        """
        rows = self.conn.execute(
            "SELECT corpus_fp, captured_at FROM snapshots ORDER BY captured_at"
        ).fetchall()
        distinct = {r[0] for r in rows if r[0] is not None}
        legacy = sum(1 for r in rows if r[0] is None)
        return {
            "n_snapshots": len(rows),
            "n_distinct_corpora": len(distinct),
            "n_snapshots_without_fingerprint": legacy,
            "supports_longitudinal": len(distinct) >= 2,
            "first_captured_at": rows[0][1] if rows else None,
            "latest_captured_at": rows[-1][1] if rows else None,
            "note": (
                "Longitudinal comparison requires at least two snapshots over "
                "differing post sets. Re-running over an unchanged corpus does "
                "not count."
            ),
        }

    def close(self) -> None:
        self.conn.close()

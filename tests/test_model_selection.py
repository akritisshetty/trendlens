"""
Regression tests for the MODEL SELECTION protocol.

Two defects here were found by measurement rather than by reading code, and
both had the same shape: the selection machinery was confidently reporting a
decision that the underlying numbers did not support.

1. ``rank()`` ignored the tie tolerance its own docstring promised. Two
   configurations 4.9e-4 apart on rho were ordered by that gap instead of by
   the documented tie-breakers, so the row whose trends never clear the trend
   definition (precision@5 = 0.0) was ranked above one that does (0.2).

2. The seed count was too small to resolve the differences being compared.
   Per-split rho has a sd of ~0.12 on this corpus, so a 5-seed mean carries a
   standard error of ~0.05 — wider than the effect. That nominated UMAP=20,
   which loses to 15-d by +0.071 on a paired 80-split test (p = 9e-10).
   Production stayed at 15-d; the guard here stops the sweep quietly
   reintroducing a 5-seed protocol.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def _row(**kw):
    base = dict(
        encoder="openai/clip-vit-base-patch32",
        umap_components=15,
        min_cluster_size=5,
        min_samples=3,
        n_clusters=17,
        rho=0.13,
        rho_std=0.10,
        coverage=0.82,
        trend_precision_at_5=0.2,
        stability_ari=0.63,
        encode_minutes=0.0,
    )
    base.update(kw)
    return base


@pytest.fixture(scope="module")
def select_models():
    sys.path.insert(0, str(ROOT / "scripts"))
    import select_models as sm

    return sm


class TestTieTolerance:
    def test_tie_broken_by_trend_precision_not_by_rho_digits(self, select_models):
        """The real pair from model_selection_results.csv: 4.9e-4 apart."""
        rows = [
            _row(umap_components=20, min_cluster_size=5, rho=0.166151,
                 trend_precision_at_5=0.2, stability_ari=0.726),
            _row(umap_components=20, min_cluster_size=8, rho=0.166644,
                 trend_precision_at_5=0.0, stability_ari=0.721),
        ]
        out = select_models.rank(rows)
        assert out.iloc[0]["min_cluster_size"] == 5
        assert out.iloc[0]["trend_precision_at_5"] == 0.2

    def test_gap_beyond_tolerance_still_ranks_on_rho(self, select_models):
        """A real difference must NOT be swallowed by the tolerance band."""
        rows = [
            _row(min_cluster_size=5, rho=0.30, trend_precision_at_5=0.0),
            _row(min_cluster_size=8, rho=0.20, trend_precision_at_5=1.0),
        ]
        out = select_models.rank(rows)
        assert out.iloc[0]["rho"] == 0.30

    def test_tie_broken_by_encode_time_when_precision_equal(self, select_models):
        rows = [
            _row(min_cluster_size=5, rho=0.166151, trend_precision_at_5=0.2,
                 encode_minutes=9.0),
            _row(min_cluster_size=8, rho=0.166644, trend_precision_at_5=0.2,
                 encode_minutes=1.0),
        ]
        out = select_models.rank(rows)
        assert out.iloc[0]["encode_minutes"] == 1.0

    def test_ineligible_rows_sort_after_eligible(self, select_models):
        """Gates are the gate: a high rho does not buy an ineligible row."""
        rows = [
            _row(rho=0.90, coverage=0.10),  # fails coverage
            _row(rho=0.10, coverage=0.90),  # eligible
        ]
        out = select_models.rank(rows)
        assert bool(out.iloc[0]["eligible"]) is True
        assert bool(out.iloc[1]["eligible"]) is False

    def test_nan_rho_does_not_hang_or_jump_the_queue(self, select_models):
        rows = [
            _row(rho=np.nan, stability_ari=np.nan),
            _row(rho=0.20, min_cluster_size=8),
        ]
        out = select_models.rank(rows)
        assert np.isfinite(out.iloc[0]["rho"])
        assert np.isnan(out.iloc[1]["rho"])

    def test_best_per_encoder_does_not_splice_columns(self, select_models):
        """groupby().first() takes the first NON-NULL per column, which would
        build a row that was never measured if one metric were NaN."""
        import pandas as pd

        rows = [
            _row(encoder="a", rho=0.30, stability_ari=np.nan),
            _row(encoder="a", rho=0.20, stability_ari=0.80, min_cluster_size=8),
            _row(encoder="b", rho=0.10, min_cluster_size=12),
        ]
        df = select_models.rank(rows)
        best = df[df["eligible"]].groupby("encoder", sort=False).head(1)
        assert len(best) == 2
        for _, r in best.iterrows():
            # rho and stability_ari must come from the SAME measured row.
            if np.isnan(r["rho"]):
                assert np.isnan(r["stability_ari"])
            else:
                assert not np.isnan(r["stability_ari"])


class TestSeedCountIsAdequate:
    def test_selection_seed_count_resolves_the_metric(self, select_models):
        """Guard the reason the sweep misled us.

        Per-split rho sd is ~0.12 on this corpus, so the standard error of the
        mean is 0.12/sqrt(n). The sweep needs n large enough that its error is
        smaller than the gap it is trying to resolve (~0.03).
        """
        per_split_sd = 0.12
        n = len(select_models.SEEDS_SPLITS)
        sem = per_split_sd / np.sqrt(n)
        assert sem < 0.03, (
            f"{n} splits gives SE={sem:.3f}; that cannot resolve the ~0.03 "
            "differences this sweep compares and will pick noise again"
        )

    def test_holdout_seeds_disjoint_from_selection_seeds(self, select_models):
        """A holdout that reuses selection seeds reports the objective back
        to itself and looks like independent confirmation."""
        assert not set(select_models.SEEDS_SPLITS) & set(select_models.HOLDOUT_SEEDS)
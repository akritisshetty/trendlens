import numpy as np
import pandas as pd
from PIL import Image
import pytest

import config
from src import clustering


def make_metadata(n=60):
    return pd.DataFrame(
        {
            "post_id": [f"p{i}" for i in range(n)],
            "user_id": [f"u{i % 5}" for i in range(n)],
            "image_path": [f"train/u{i}/img{i}.jpg" for i in range(n)],
            "timestamp": pd.date_range("2016-01-01", periods=n, freq="D", tz="UTC"),
            "likes": list(range(n)),
            "comments": [0] * n,
        }
    )


def gaussian_blobs(n=90, dims=8, seed=0, n_clusters=3):
    """Well-separated normal clusters in R^dims."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dims)) * 3.0
    pts = []
    for i in range(n):
        c = centers[i % n_clusters]
        pts.append(rng.normal(loc=c, scale=0.2, size=dims))
    emb = np.stack(pts)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    return (emb / norm).astype("float32")


class TestReduceDimensions:
    def test_pca_shape_and_cache(self, tmp_path):
        emb = gaussian_blobs(n=40)
        cache = tmp_path / "pca_5d.npy"
        r1 = clustering.reduce_dimensions(emb, method="pca", n_components=5, cache_path=cache)
        r2 = clustering.reduce_dimensions(emb, method="pca", n_components=5, cache_path=cache)
        assert r1.shape == (40, 5)
        assert r1.dtype == np.float32
        assert np.array_equal(r1, r2)  # cached
        assert cache.exists()

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            clustering.reduce_dimensions(np.zeros((10, 4)), method="nope")

    def test_cache_shape_mismatch_recomputes(self, tmp_path):
        emb = gaussian_blobs(n=40)
        cache = tmp_path / "pca_5d.npy"
        clustering.reduce_dimensions(emb, method="pca", n_components=5, cache_path=cache)
        r = clustering.reduce_dimensions(emb, method="pca", n_components=4, cache_path=cache)
        assert r.shape == (40, 4)


class TestRunHDBSCAN:
    def test_recovers_blobs(self):
        emb = gaussian_blobs(n=300, seed=1)
        labels, probs, clusterer = clustering.run_hdbscan(
            emb, min_cluster_size=20, min_samples=5
        )
        # 3 real clusters + some noise allowed; exactly-3 is ideal but robust:
        n_clusters = int(labels.max() + 1) if labels.max() >= 0 else 0
        assert n_clusters >= 2
        assert labels.shape == (300,)
        assert probs.shape == (300,)
        # each of the 3 seeded groups mostly shares a label (>= 60%)
        for g in range(3):
            group = labels[g::3]
            group = group[group >= 0]
            assert len(group) > 0
            most_common = np.bincount(group).argmax()
            frac = (group == most_common).mean()
            assert frac >= 0.6

    def test_noise_label_is_minus_one(self):
        # Contract: noise points are labelled -1, every other label is a
        # valid non-negative cluster id, probabilities are in [0, 1].
        emb = gaussian_blobs(n=120, seed=2, n_clusters=1, dims=4)
        labels, probs, _ = clustering.run_hdbscan(emb, min_cluster_size=20, min_samples=5)
        n_clusters = int(labels.max() + 1) if labels.max() >= 0 else 0
        assert np.all(labels >= -1)
        assert np.all(labels < n_clusters)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_saves_model_manifest(self, tmp_path):
        emb = gaussian_blobs(n=60)
        p = tmp_path / "model.pkl"
        clustering.run_hdbscan(emb, min_cluster_size=15, min_samples=5, save_model=p)
        assert p.exists()


class TestClusterReport:
    def test_report_well_separated(self):
        emb = gaussian_blobs(n=300, seed=3)
        labels, probs, clusterer = clustering.run_hdbscan(
            emb, min_cluster_size=20, min_samples=5
        )
        stats = clustering.cluster_report(emb, labels, probs, clusterer, silhouette_sample=None)
        assert stats["n_points"] == 300
        assert stats["n_clusters"] >= 2
        assert 0 <= stats["noise_pct"] <= 1.0
        assert stats["silhouette"] is not None and stats["silhouette"] > 0.5
        assert "cluster_sizes" in stats
        assert sum(stats["cluster_sizes"].values()) == 300 - stats["n_noise"]

    def test_all_noise_handled(self):
        emb = np.random.default_rng(0).normal(size=(30, 4)).astype("float32")
        # tiny min size impossible to satisfy on 30 random pts may still form
        # clusters; use a deliberately large min_cluster_size to force noise
        labels, probs, clusterer = clustering.run_hdbscan(
            emb, min_cluster_size=100, min_samples=10
        )
        stats = clustering.cluster_report(emb, labels, probs, clusterer, silhouette_sample=None)
        assert stats["n_clusters"] == 0
        assert stats["silhouette"] is None


class TestClusterSummary:
    def test_basic(self, tmp_path):
        emb = gaussian_blobs(n=60)
        labels, probs, _ = clustering.run_hdbscan(emb, min_cluster_size=15, min_samples=5)
        df = make_metadata(60)
        summary = clustering.cluster_summary(
            df, labels, probs, out_path=tmp_path / "cluster_summary.csv"
        )
        assert list(summary.columns) == [
            "cluster_id", "size", "percentage", "mean_probability",
            "median_engagement", "mean_engagement",
        ]
        assert (summary["size"] > 0).all()
        assert (tmp_path / "cluster_summary.csv").exists()


class TestRepresentativeImages:
    def test_selects_highest_probability_first(self):
        emb = gaussian_blobs(n=90, seed=5)
        labels, probs, _ = clustering.run_hdbscan(emb, min_cluster_size=15, min_samples=5)
        df = make_metadata(90)
        reps = clustering.representative_images(df, emb, labels, probs, k=4)
        # reps keys are exactly the non-noise clusters
        assert set(reps) == set(int(c) for c in np.unique(labels[labels >= 0]))
        for c, rows in reps.items():
            assert len(rows) == min(4, int((labels == c).sum()))
            assert rows[0]["probability"] >= rows[-1]["probability"] - 1e-9
            assert all("image_path" in r and "post_id" in r for r in rows)

    def test_no_noise_clusters_returns_empty(self):
        labels = np.full(20, -1)
        reps = clustering.representative_images(make_metadata(20), np.zeros((20, 4)), labels, np.zeros(20), k=3)
        assert reps == {}

    def test_save_load_roundtrip(self, tmp_path):
        reps = {0: [{"row_index": 1, "post_id": "p1", "image_path": "x.jpg", "probability": 0.9, "medoid_distance": 0.1}]}
        p = clustering.save_representatives(reps, tmp_path / "r.json")
        assert p.exists()
        import json
        assert json.loads(p.read_text())["0"][0]["post_id"] == "p1"


class TestContactSheets:
    def test_creates_pngs(self, tmp_path):
        img_dir = tmp_path / "train" / "u0"
        img_dir.mkdir(parents=True)
        Image.new("RGB", (32, 32), (255, 0, 0)).save(img_dir / "img0.jpg")
        reps = {0: [{"image_path": "train/u0/img0.jpg"}] * 4}
        out = tmp_path / "figs"
        paths = clustering.contact_sheets(reps, image_root=tmp_path, out_dir=out, grid=(2, 2), size=(16, 16))
        assert paths[0].exists()
        assert paths[0].suffix == ".jpg"

    def test_placeholder_on_failed_load(self, tmp_path):
        reps = {0: [{"image_path": "train/u9/missing.jpg"}] * 4}
        out = tmp_path / "figs"
        paths = clustering.contact_sheets(reps, image_root=tmp_path, out_dir=out, grid=(2, 2), size=(16, 16))
        assert paths[0].exists()


class TestParameterSweep:
    def test_returns_comparison_table(self):
        emb = gaussian_blobs(n=120, seed=7)
        df = clustering.parameter_sweep(
            emb, min_cluster_sizes=[20, 40], min_samples_list=[5, 10], methods=["eom"]
        )
        assert len(df) == 4
        assert set(["min_cluster_size", "min_samples", "method", "n_clusters", "noise_pct", "silhouette"]).issubset(df.columns)

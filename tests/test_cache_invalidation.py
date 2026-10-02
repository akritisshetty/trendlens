"""
Regression tests for CACHE INVALIDATION across model/embedding changes.

Every cache in this pipeline is keyed on shapes, and shapes do not change when
the encoder does: the same corpus produces the same row count through
ViT-B/32 and ViT-L/14, and both are 512/768-d only at the very end. Each test
below pins one cache that previously reused a stale artifact silently.
"""

import json

import numpy as np
import pytest

import config
from src import clustering, style_tags as st


def _rand(n=40, d=16, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype("float32")
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class TestEmbeddingFingerprint:
    def test_same_input_same_fingerprint(self):
        a = _rand(seed=1)
        assert clustering.embedding_fingerprint(a) == clustering.embedding_fingerprint(
            a.copy()
        )

    def test_different_data_different_fingerprint(self):
        assert clustering.embedding_fingerprint(
            _rand(seed=1)
        ) != clustering.embedding_fingerprint(_rand(seed=2))

    def test_different_width_different_fingerprint(self):
        """The encoder swap that motivated this: same rows, different width."""
        a = _rand(n=40, d=512, seed=1)
        b = _rand(n=40, d=768, seed=1)
        assert a.shape[0] == b.shape[0]
        assert clustering.embedding_fingerprint(a) != clustering.embedding_fingerprint(b)

    def test_is_stable_across_dtypes_and_layouts(self):
        a = _rand(seed=3)
        same_values = np.asfortranarray(a.astype("float32"))
        assert clustering.embedding_fingerprint(a) == clustering.embedding_fingerprint(
            same_values
        )


class TestReduceDimensionsCacheKey:
    def test_cache_path_depends_on_embedding_content(self, tmp_path, monkeypatch):
        """Default cache filename must not collide across encoder swaps."""
        import umap

        # Hermetic: keep the default cache dir inside tmp_path.
        monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")

        # Stub the fit so the test asserts the CACHE KEY, not UMAP's behaviour.
        def fake_fit_transform(self, X):
            return np.zeros((len(X), self.n_components), dtype="float32")

        monkeypatch.setattr(umap.UMAP, "fit_transform", fake_fit_transform)

        cache_dir = config.ARTIFACTS_DIR / "embeddings"
        n_after_first = len(list(cache_dir.glob("umap_3d_*"))) if cache_dir.exists() else 0

        clustering.reduce_dimensions(_rand(n=30, d=16, seed=1), n_components=3)
        after_first = len(list(cache_dir.glob("umap_3d_*")))

        # Same row count, different embedding width — exactly what swapping
        # ViT-B/32 for ViT-L/14 does to the corpus.
        clustering.reduce_dimensions(_rand(n=30, d=32, seed=2), n_components=3)
        after_second = len(list(cache_dir.glob("umap_3d_*")))

        assert after_first > n_after_first
        assert after_second == after_first + 1, (
            "a second, differently-shaped embedding matrix reused the first "
            "matrix's UMAP cache entry"
        )

    def test_same_matrix_hits_the_cache(self, tmp_path, monkeypatch):
        """The key must also be stable, or nothing would ever be reused."""
        import umap

        monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
        calls = {"n": 0}

        def counting_fit(self, X):
            calls["n"] += 1
            return np.zeros((len(X), self.n_components), dtype="float32")

        monkeypatch.setattr(umap.UMAP, "fit_transform", counting_fit)

        emb = _rand(n=30, d=16, seed=7)
        clustering.reduce_dimensions(emb, n_components=3)
        clustering.reduce_dimensions(emb.copy(), n_components=3)
        assert calls["n"] == 1

    def test_explicit_cache_path_still_respected(self, tmp_path):
        out = tmp_path / "custom.npy"
        arr = clustering.reduce_dimensions(
            _rand(n=30, d=8, seed=5), n_components=3, cache_path=out
        )
        assert out.exists()
        assert arr.shape == (30, 3)


class TestStyleTagCacheModelKey:
    def _write_cache(self, emb_path, meta_path, model, dim):
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, np.zeros((len(st.ALL_PROMPTS), dim), dtype="float32"))
        meta_path.write_text(
            json.dumps(
                {
                    "version": st.STYLE_VERSION,
                    "model": model,
                    "prompts": st.ALL_PROMPTS,
                    "dim": dim,
                }
            )
        )

    def test_cache_rejected_when_model_differs(self, tmp_path, monkeypatch):
        emb = tmp_path / "emb.npy"
        meta = tmp_path / "meta.json"
        self._write_cache(emb, meta, "some/old-checkpoint", 512)
        monkeypatch.setattr(st, "STYLE_EMB_PATH", emb)
        monkeypatch.setattr(st, "STYLE_META_PATH", meta)
        monkeypatch.setattr(config, "CLIP_MODEL", "a/different-checkpoint")

        assert st._style_text_embeddings_cached() is None, (
            "a style-prompt cache built by a different encoder was reused; its "
            "vectors are not in the image embeddings' space"
        )

    def test_cache_accepted_for_matching_model(self, tmp_path, monkeypatch):
        emb = tmp_path / "emb.npy"
        meta = tmp_path / "meta.json"
        self._write_cache(emb, meta, "the/right-checkpoint", 512)
        monkeypatch.setattr(st, "STYLE_EMB_PATH", emb)
        monkeypatch.setattr(st, "STYLE_META_PATH", meta)
        monkeypatch.setattr(config, "CLIP_MODEL", "the/right-checkpoint")

        out = st._style_text_embeddings_cached()
        assert out is not None
        assert out.shape == (len(st.ALL_PROMPTS), 512)

    def test_cache_still_rejects_version_and_prompt_drift(self, tmp_path, monkeypatch):
        emb = tmp_path / "emb.npy"
        meta = tmp_path / "meta.json"
        self._write_cache(emb, meta, "m", 512)
        monkeypatch.setattr(st, "STYLE_EMB_PATH", emb)
        monkeypatch.setattr(st, "STYLE_META_PATH", meta)
        monkeypatch.setattr(config, "CLIP_MODEL", "m")

        bad = json.loads(meta.read_text())
        bad["prompts"] = ["something", "else"]
        meta.write_text(json.dumps(bad))
        assert st._style_text_embeddings_cached() is None


class TestModelNamesAreConfigurable:
    """No module may carry its own copy of a model checkpoint string."""

    def test_config_exposes_the_three_models(self):
        for attr in ("CLIP_MODEL", "RAG_EMBED_MODEL", "BLIP_MODEL"):
            val = getattr(config, attr)
            assert isinstance(val, str) and val
            assert "/" in val, f"{attr} should be a hub id, got {val!r}"

    def test_embeddings_default_follows_config(self):
        """The re-export must equal config at import time (no second copy)."""
        from src import embeddings

        assert embeddings.DEFAULT_CLIP_MODEL == config.CLIP_MODEL

    def test_load_clip_default_signature_tracks_config(self):
        import inspect

        from src import embeddings

        sig = inspect.signature(embeddings.load_clip)
        assert sig.parameters["model_name"].default == config.CLIP_MODEL

    def test_interpretation_default_follows_config(self):
        from src import interpretation

        assert interpretation.DEFAULT_BLIP_MODEL == config.BLIP_MODEL

    def test_no_hardcoded_clip_checkpoint_in_src(self):
        """Guards against a second copy creeping back in."""
        import pathlib
        import re

        pattern = re.compile(r"[\"'](?:openai|laion)/clip-[\w.\-]+[\"']")
        src = pathlib.Path(config.ROOT) / "src"
        offenders = [
            str(p.relative_to(config.ROOT))
            for p in src.rglob("*.py")
            if pattern.search(p.read_text())
        ]
        assert not offenders, (
            f"hard-coded CLIP checkpoint in {offenders}; use config.CLIP_MODEL"
        )
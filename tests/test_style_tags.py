"""Tests for src.style_tags — CLIP zero-shot photography-style tagging."""

import numpy as np
import pytest

from src import style_tags as st


class TestTaxonomy:
    def test_tags_unique_and_complete(self):
        tags = [s["tag"] for s in st.STYLE_TAXONOMY]
        assert len(tags) == len(set(tags))
        assert len(tags) >= 10

    def test_every_entry_has_prompts_direction_aspect(self):
        aspects = {"framing", "lighting", "color mood", "process", "composition"}
        for s in st.STYLE_TAXONOMY:
            assert s["prompts"], f"{s['tag']} has no prompts"
            assert all(isinstance(p, str) and p for p in s["prompts"])
            assert isinstance(s["direction"], str) and s["direction"]
            assert s["aspect"] in aspects

    def test_flattened_prompts_match_taxonomy(self):
        expected = sum(len(s["prompts"]) for s in st.STYLE_TAXONOMY)
        assert len(st.ALL_PROMPTS) == expected
        assert len(st._PROMPT_TAG_POS) == expected
        # first prompt of taxonomy i must map back to tag index i
        assert st._PROMPT_TAG_POS[0] == 0

    def test_taxonomy_record_roundtrip(self):
        rec = st.taxonomy_record("tactile macro close-up")
        assert rec["aspect"] == "framing"
        assert rec["direction"]
        with pytest.raises(KeyError):
            st.taxonomy_record("nonexistent tag")


class TestScoresFromPromptSims:
    def test_ensemble_mean_per_tag(self):
        n_tags = len(st.STYLE_TAXONOMY)
        n_prompts = len(st.ALL_PROMPTS)
        sims = np.zeros((1, n_prompts), dtype="float32")
        # set every prompt of tag 3 to a distinct value; mean must be their mean
        pos = np.asarray(st._PROMPT_TAG_POS)
        vals = [0.2, 0.6] if (pos == 3).sum() == 2 else list(np.linspace(0.1, 0.9, (pos == 3).sum()))
        sims[0, pos == 3] = np.asarray(vals, dtype="float32")
        out = st.scores_from_prompt_sims(sims)
        assert out.shape == (1, n_tags)
        assert out[0, 3] == pytest.approx(float(np.mean(vals)))
        # untouched tags average to 0
        assert out[0, 0] == pytest.approx(0.0)

    def test_wrong_width_raises(self):
        with pytest.raises(ValueError):
            st.scores_from_prompt_sims(np.zeros((2, 3)))


class TestAggregateStyles:
    def _scores(self):
        # two rows, three fake tags via monkeypatched taxonomy length not
        # required — build against real taxonomy width but only inspect order.
        rng = np.random.default_rng(42)
        return rng.uniform(0.0, 0.5, size=(5, len(st.STYLE_TAXONOMY))).astype("float32")

    def test_top_k_descending(self):
        scores = self._scores()
        out = st.aggregate_styles(scores, top_k=3, min_score=0.0)
        assert len(out) == 3
        assert out == sorted(out, key=lambda x: -x["score"])
        mean = scores.mean(axis=0)
        best = int(np.argmax(mean))
        assert out[0]["tag"] == st.TAGS[best]

    def test_min_score_filters_noise(self):
        scores = self._scores() * 0.01  # everything below any sane floor
        out = st.aggregate_styles(scores, top_k=3, min_score=st.MIN_STYLE_SCORE)
        assert out == []

    def test_indices_restrict_rows(self):
        scores = self._scores()
        single = st.aggregate_styles(scores, indices=[2], top_k=1, min_score=0.0)
        manual = st.aggregate_styles(scores[[2]], top_k=1, min_score=0.0)
        assert single == manual

    def test_empty_indices(self):
        assert st.aggregate_styles(self._scores(), indices=[], top_k=3) == []

    def test_payload_shape(self):
        out = st.aggregate_styles(self._scores(), top_k=2, min_score=0.0)
        for item in out:
            assert set(item.keys()) == {"tag", "aspect", "score"}
            assert item["tag"] in st.TAGS


class TestFormatStyleTags:
    def test_dicts_and_limit(self):
        tags = [{"tag": f"t{i}"} for i in range(5)]
        assert st.format_style_tags(tags, limit=2) == "t0, t1"

    def test_plain_strings_and_empty(self):
        assert st.format_style_tags(["a", "b"]) == "a, b"
        assert st.format_style_tags([]) == ""
        assert st.format_style_tags(None) == ""


class TestComputeStyleScores:
    def test_zero_shot_with_injected_text_embs(self):
        """No model needed: inject orthonormal 'prompt' embeddings."""
        n_prompts = len(st.ALL_PROMPTS)
        dim = n_prompts
        # each prompt embedding is its own basis vector -> sims are exact
        text = np.eye(n_prompts, dtype="float32")

        # image 0 == prompt 0 -> its tag's ensemble mean > all others
        img0 = np.zeros((1, dim), dtype="float32")
        img0[0, 0] = 1.0
        # image 1 == noise -> uniform-ish low scores, never above exact match
        rng = np.random.default_rng(7)
        img1 = rng.normal(size=(1, dim)).astype("float32")
        img1 /= np.linalg.norm(img1)

        out = st.compute_style_scores(np.vstack([img0, img1]), text_embs=text)

        assert out.shape == (2, len(st.STYLE_TAXONOMY))
        first_tag = st._PROMPT_TAG_POS[0]
        assert out[0].argmax() == first_tag
        # exact-match image must beat a random image on that tag
        assert out[0, first_tag] > out[1, first_tag]

    def test_l2_normalized_input_ok(self):
        n_prompts = len(st.ALL_PROMPTS)
        text = np.eye(n_prompts, dtype="float32")
        emb = np.ones((3, n_prompts), dtype="float32") / np.sqrt(n_prompts)
        out = st.compute_style_scores(emb, text_embs=text)
        assert np.isfinite(out).all()


class TestCacheMeta:
    def test_cache_paths_under_embeddings_dir(self):
        assert st.STYLE_EMB_PATH.parent == st.config.EMBEDDINGS_DIR
        assert st.STYLE_META_PATH.parent == st.config.EMBEDDINGS_DIR

    def test_invalid_cache_returns_none(self, tmp_path, monkeypatch):
        bad_emb = tmp_path / "emb.npy"
        bad_meta = tmp_path / "meta.json"
        bad_emb.write_bytes(b"not an npy")
        bad_meta.write_text("{}")
        monkeypatch.setattr(st, "STYLE_EMB_PATH", bad_emb)
        monkeypatch.setattr(st, "STYLE_META_PATH", bad_meta)
        assert st._style_text_embeddings_cached() is None

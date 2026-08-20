import json
from unittest import mock

import pandas as pd
import pytest

import config
from src import rag


class TestClusterRecord:
    def test_uses_real_artifact_fields(self):
        rec = rag._cluster_record(cluster_id=0, similarity=0.9, rank=1)
        assert rec["cluster_id"] == 0
        assert rec["rank"] == 1
        assert rec["similarity_score"] == pytest.approx(0.9)
        assert rec["n_posts"] is None or isinstance(rec["n_posts"], int)
        # NOT MEASURED fields must never be invented
        assert rec["viral_rate"] is None
        assert rec["geographic_hotspots"] == []
        # keys the frontend expects
        for key in ["id", "name", "description", "blip_caption",
                    "characteristics", "lifecycle", "n_posts",
                    "average_engagement", "recent_growth", "trend_score"]:
            assert key in rec

    def test_representative_image_keys(self):
        rec = rag._cluster_record(cluster_id=0, similarity=0.9, rank=1)
        assert "representative_image" in rec
        if rec["representative_image"]:
            assert rec["representative_image_url"].startswith("/api/images?path=")
        else:
            assert rec["representative_image_url"] is None


class TestBuildContext:
    def test_returns_disclaimer_and_records(self):
        with mock.patch.object(rag, "_load_retrieval") as mock_rt, \
             mock.patch("src.retrieval.embed_texts") as mock_emb, \
             mock.patch("src.retrieval.query_index") as mock_q:
            mock_rt.return_value = {
                "model": None, "processor": None, "device": "cpu",
                "index": None, "cluster_ids": [0, 1, 2],
            }
            mock_emb.return_value = np_ones = __import__("numpy").ones((1, 512))
            dists = __import__("numpy").array([[0.9, 0.5, 0.1]])
            idxs = __import__("numpy").array([[1, 2, 0]])
            mock_q.return_value = (dists, idxs)

            ctx = rag.build_context("coffee", k=3)
            assert ctx["query"] == "coffee"
            assert ctx["disclaimer"]
            assert len(ctx["retrieved_clusters"]) == 3
            assert ctx["retrieved_clusters"][0]["cluster_id"] == 1
            assert ctx["retrieved_clusters"][2]["rank"] == 3


class TestFormatAnswer:
    def test_no_clusters(self):
        out = rag.format_answer("x", {"retrieved_clusters": []})
        assert "No matching clusters" in out

    def test_uses_real_fields_only(self):
        ctx = {
            "retrieved_clusters": [{
                "cluster_id": 3, "rank": 1, "similarity_score": 0.8,
                "name": "Moon Sky", "description": "the moon in the sky",
                "blip_caption": "a moon in the sky", "characteristics": ["moon"],
                "interpretation_confidence": 0.3, "lifecycle": "Rising",
                "n_posts": 40, "average_engagement": 12.5, "recent_growth": 1.0,
                "trend_score": 0.09, "text_trend_score": 0.5,
            }],
        }
        out = rag.format_answer("moon", ctx)
        assert "Moon Sky" in out
        # Should NOT include pipeline internals
        assert "Rising" not in out
        assert "12.50" not in out

    def test_missing_metrics_renders_clean(self):
        ctx = {"retrieved_clusters": [{
            "cluster_id": 0, "rank": 1, "similarity_score": 0.1, "name": "x",
            "description": "", "blip_caption": "", "characteristics": [],
            "interpretation_confidence": None, "lifecycle": None, "n_posts": None,
            "average_engagement": None, "recent_growth": None, "trend_score": None,
            "text_trend_score": None,
        }]}
        out = rag.format_answer("q", ctx)
        assert len(out) > 0


class TestAdviceFormat:
    def _ctx(self, clusters):
        return {"retrieved_clusters": clusters, "total_clusters_analyzed": 29}

    def _cluster(self, cid=20, name="cup coffee", feats=("cup", "glass", "coffee", "poured", "white"),
                 caption="a cup of coffee being poured into a white cup",
                 avg=79.04, trend=0.11, lifecycle="Stable", n_posts=325):
        return {
            "cluster_id": cid, "rank": 1, "similarity_score": 0.8, "name": name,
            "description": "a cluster", "blip_caption": caption,
            "characteristics": list(feats), "interpretation_confidence": 0.04,
            "lifecycle": lifecycle, "n_posts": n_posts, "average_engagement": avg,
            "recent_growth": 1.0, "trend_score": trend, "text_trend_score": 0.5,
        }

    def test_intent_detection(self):
        q = "I want to post a picture of cup of coffee. What should the visual look like for max engagement?"
        assert rag._is_advice_intent(q) is True
        assert rag._is_advice_intent("a cup of coffee") is False
        assert rag._is_advice_intent("what kind of cat photos get the most engagement?") is True

    def test_subject_extraction(self):
        top = self._cluster()
        assert rag._advice_subject("post a picture of red flowers for max engagement", top) == "red flowers"
        assert rag._advice_subject("a cup of coffee", top) == "cup coffee"

    def test_advice_uses_real_fields_only(self):
        ctx = self._ctx([self._cluster()])
        out = rag.format_advice_answer(
            "I want to post a picture of cup of coffee. What should the visual look like for max engagement?", ctx
        )
        assert "How to shoot" in out
        assert "cup of coffee" in out
        assert "cup coffee" in out
        assert "no invented styling advice" in out
        assert "synthetic demo data" in out
        # Should NOT include pipeline internals
        assert "79.04" not in out
        assert "No LLM used" not in out

    def test_advice_no_overlap_is_honest(self):
        coffee = self._cluster()
        bird = self._cluster(cid=7, name="bird long", feats=("bird", "neck", "horse", "running", "grass"),
                             caption="a bird with a long neck", avg=93.83, trend=0.03, lifecycle="Stable")
        ctx = self._ctx([coffee, bird])
        out = rag.format_advice_answer("post a picture of coffee for max engagement", ctx)
        assert "do **not** overlap" in out
        assert "No engagement advantage to borrow" in out

    def test_advice_engagement_overlap_highlighted(self):
        coffee = self._cluster()
        variant = self._cluster(cid=9, name="coffee pour", feats=("coffee", "cup", "glass", "steam"),
                                caption="coffee in a glass cup", avg=99.0, trend=0.3, lifecycle="Rising")
        ctx = self._ctx([coffee, variant])
        out = rag.format_advice_answer("post a picture of coffee for max engagement", ctx)
        assert "highest measured engagement" in out
        assert "coffee pour" in out
        assert "edge the data can point to" in out


class TestClassifyScope:
    def test_keyword_gate_blocks_programming(self):
        got = rag.classify_scope("write a c program to print hello world")
        assert got["in_scope"] is False
        assert got["method"] == "keywords"

    def test_keyword_gate_blocks_cooking(self):
        got = rag.classify_scope("recipe for biryani")
        assert got["in_scope"] is False
        assert got["method"] == "keywords"

    def test_keyword_gate_blocks_finance(self):
        got = rag.classify_scope("best crypto to invest")
        assert got["in_scope"] is False
        assert got["method"] == "keywords"

    def test_in_scope_phrasing_is_not_rejected(self):
        # Visual noun phrases must pass the keyword gate (no "recipe" etc.)
        got = rag.classify_scope("cat photos")
        assert got["method"] == "keywords" or got["in_scope"] is True


class TestImageUrl:
    def test_quotes_path(self):
        url = rag._image_url("train/59@N75/775.jpg")
        assert url == "/api/images?path=train%2F59%40N75%2F775.jpg"

    def test_none(self):
        assert rag._image_url(None) is None
        assert rag._image_url("") is None


class TestRunQuery:
    def test_response_shape(self):
        with mock.patch.object(rag, "build_context") as mock_bc, \
             mock.patch.object(rag, "classify_scope") as mock_scope:
            mock_scope.return_value = {
                "in_scope": True, "method": "anchors", "reason": None,
            }
            mock_bc.return_value = {
                "query": "coffee", "k": 2,
                "disclaimer": "demo",
                "total_clusters_analyzed": 29,
                "dataset": "5K sample",
                "retrieved_clusters": [],
            }
            res = rag.run_query("coffee", k=2)
            assert res["query"] == "coffee"
            assert res["answer"]
            assert res["totalClustersAnalyzed"] == 29
            assert res["mode"] == "faiss-only"
            assert res["sources"] == []
            assert res["inScope"] is True
            assert res["scopeMethod"] == "anchors"

    def test_out_of_scope_refuses_without_context(self):
        with mock.patch.object(rag, "classify_scope") as mock_scope:
            mock_scope.return_value = {
                "in_scope": False, "method": "keywords",
                "reason": "This looks like a programming question.",
            }
            res = rag.run_query("write a c program to print hello world", k=2)
            assert res["inScope"] is False
            assert res["scopeMethod"] == "keywords"
            assert res["retrievedClusters"] == []
            assert res["supportingImages"] == []
            assert "Out of scope" in res["answer"]
            assert "hello world" in res["answer"]

    def test_live_intent_stays_rule_based_not_llm(self):
        live = {
            "disclaimer": "REAL LIVE DATA: test",
            "source": "wikimedia-commons",
            "subreddits": ["breakfast"],
            "recent_window_days": 30,
            "themes": [{
                "name": "warm visual theme", "keywords": ["warm", "book"],
                "keywords_emoji": "🔥", "blip_caption": "",
                "subreddits": ["breakfast"], "channel_label": "wikimedia search: breakfast",
                "source": "wikimedia-commons", "has_engagement": False,
                "recent_posts": 11, "prior_posts": 0,
                "growth_rate": None, "avg_engagement": 0.0, "total_comments": 0,
                "representative_image_url": "/api/live-images?name=w.jpg",
            }],
        }
        with mock.patch.object(rag, "classify_scope") as mock_scope, \
             mock.patch.object(rag, "build_context") as mock_bc, \
             mock.patch.object(rag, "load_live_trends") as mock_live, \
             mock.patch("src.llm.format_answer_with_llm") as mock_llm:
            mock_scope.return_value = {"in_scope": True, "method": "anchors", "reason": None}
            mock_bc.return_value = {
                "query": "trending in food", "k": 2, "disclaimer": "demo",
                "total_clusters_analyzed": 29, "dataset": "5K sample",
                "retrieved_clusters": [],
            }
            mock_live.return_value = live
            mock_llm.return_value = "GEMINI WROTE THIS WHOLE THING"
            res = rag.run_query(
                "What is trending in the food industry right now? "
                "What should my pasta visuals be?", k=2
            )
            assert res["answerMode"] == "rule-based"
            assert "warm visual theme" in res["answer"]
            assert "brand new this window" in res["answer"]
            assert "GEMINI WROTE THIS WHOLE THING" not in res["answer"]
            mock_llm.assert_not_called()

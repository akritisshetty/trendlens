import json

import pytest

import config


@pytest.fixture
def ctx():
    return {
        "query": "what are the biggest coffee trends",
        "total_clusters_analyzed": 29,
        "dataset": "5,000-image sample of 69,226",
        "disclaimer": "synthetic demo",
        "retrieved_clusters": [
            {"name": "latte art", "average_engagement": 320.0, "n_posts": 40}
        ],
    }


class TestEnabled:
    def test_disabled_without_provider(self, monkeypatch, ctx):
        monkeypatch.delenv("TRENDLENS_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("TRENDLENS_LLM_API_KEY", raising=False)
        monkeypatch.delenv("TRENDLENS_LLM_BASE_URL", raising=False)
        from src import llm

        assert llm.llm_enabled() is False
        assert llm.format_answer_with_llm("q", ctx) is None

    def test_enabled_gemini(self, monkeypatch, ctx):
        monkeypatch.setenv("TRENDLENS_LLM_PROVIDER", "gemini")
        monkeypatch.setenv("TRENDLENS_LLM_API_KEY", "k")
        from src import llm

        assert llm.llm_enabled() is True

    def test_ollama_no_key_ok(self, monkeypatch, ctx):
        monkeypatch.setenv("TRENDLENS_LLM_PROVIDER", "ollama")
        monkeypatch.delenv("TRENDLENS_LLM_API_KEY", raising=False)
        monkeypatch.setenv("TRENDLENS_LLM_BASE_URL", "http://localhost:11434")
        from src import llm

        assert llm.llm_enabled() is True


class TestFormatting:
    def test_gemini_parse(self, monkeypatch, ctx):
        monkeypatch.setenv("TRENDLENS_LLM_PROVIDER", "gemini")
        monkeypatch.setenv("TRENDLENS_LLM_API_KEY", "k")

        def fake_post(url, params=None, json=None, timeout=None, headers=None):
            assert "key=k" in url
            assert llm.llm_config()["model"] in url
            assert "contents" in (json or {})

            class R:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "candidates": [
                            {"content": {"parts": [{"text": "Here is a fluent answer."}]}}
                        ]
                    }

            return R()

        monkeypatch.setattr("requests.post", fake_post)
        from src import llm

        out = llm.format_answer_with_llm("q", ctx)
        assert out == "Here is a fluent answer."

    def test_openai_parse(self, monkeypatch, ctx):
        monkeypatch.setenv("TRENDLENS_LLM_PROVIDER", "openai")
        monkeypatch.setenv("TRENDLENS_LLM_API_KEY", "k")

        def fake_post(url, params=None, json=None, timeout=None, headers=None):
            class R:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"choices": [{"message": {"content": "OpenAI prose."}}]}

            return R()

        monkeypatch.setattr("requests.post", fake_post)
        from src import llm

        assert llm.format_answer_with_llm("q", ctx) == "OpenAI prose."

    def test_ollama_parse(self, monkeypatch, ctx):
        monkeypatch.setenv("TRENDLENS_LLM_PROVIDER", "ollama")
        monkeypatch.setenv("TRENDLENS_LLM_BASE_URL", "http://localhost:11434")

        def fake_post(url, params=None, json=None, timeout=None, headers=None):
            assert url == "http://localhost:11434/api/chat"

            class R:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"message": {"content": "Ollama prose."}}

            return R()

        monkeypatch.setattr("requests.post", fake_post)
        from src import llm

        assert llm.format_answer_with_llm("q", ctx) == "Ollama prose."

    def test_network_failure_falls_back_to_none(self, monkeypatch, ctx):
        monkeypatch.setenv("TRENDLENS_LLM_PROVIDER", "gemini")
        monkeypatch.setenv("TRENDLENS_LLM_API_KEY", "k")

        def boom(*a, **k):
            raise RuntimeError("down")

        monkeypatch.setattr("requests.post", boom)
        from src import llm

        assert llm.format_answer_with_llm("q", ctx) is None

    def test_malformed_response_returns_none(self, monkeypatch, ctx):
        monkeypatch.setenv("TRENDLENS_LLM_PROVIDER", "openai")
        monkeypatch.setenv("TRENDLENS_LLM_API_KEY", "k")

        def fake_post(url, params=None, json=None, timeout=None, headers=None):
            class R:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"choices": []}

            return R()

        monkeypatch.setattr("requests.post", fake_post)
        from src import llm

        assert llm.format_answer_with_llm("q", ctx) is None

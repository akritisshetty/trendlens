import base64

import pytest

import config
from src import chat_ai


class _Response:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


TEXT = """**Verdict:** Sunlit tables are leading.

## Trends to watch

**Sunlit table** - food
- What it is: hard light on simple food.
- Why it is landing now: it feels immediate and unpolished.
- How to shoot it:
  1. Use hard side light.
  2. Shoot portrait 4:5.
  3. Keep the palette warm.

IMAGE PROMPTS:
1. Photorealistic editorial food photograph of a sunlit Mediterranean table with tomatoes, olive oil, and linen, hard side light, warm cream palette, portrait framing, shallow depth of field.
2. Photorealistic beauty close-up with glossy skin, peach blush, soft window light, warm neutral palette, minimal styling, editorial portrait composition.
"""


def test_requires_gemini_key(monkeypatch):
    for var in (
        "TRENDLENS_GEMINI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "TRENDLENS_LLM_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(chat_ai, "_legacy_fallback", lambda query, reason: None)

    result = chat_ai.chat_query("What food trends are rising?")

    assert result["degraded"] is True
    assert result["supportingImages"] == []
    assert "not connected" in result["answer"]


def test_direct_answer_parses_prompts_and_generates_images(monkeypatch, tmp_path):
    monkeypatch.setenv("TRENDLENS_GEMINI_API_KEY", "k")
    monkeypatch.setenv("TRENDLENS_GEMINI_IMAGE_COUNT", "2")
    monkeypatch.setattr(config, "CHAT_IMAGES_DIR", tmp_path)
    monkeypatch.setattr(chat_ai, "_corpus_reference_images", lambda *a, **k: [])
    monkeypatch.setattr(chat_ai, "_yandex_reference_images", lambda *a, **k: [])

    image_data = base64.b64encode(b"fake-image").decode()

    def fake_post(url, json=None, timeout=None):
        if "image" in url:
            return _Response({
                "candidates": [{
                    "content": {
                        "parts": [{
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": image_data,
                            }
                        }]
                    }
                }]
            })
        return _Response({
            "candidates": [{
                "content": {"parts": [{"text": TEXT}]},
                "groundingMetadata": {
                    "groundingChunks": [{
                        "web": {"uri": "https://example.com", "title": "Source"}
                    }]
                },
            }]
        })

    monkeypatch.setattr(chat_ai.requests, "post", fake_post)
    monkeypatch.setattr(chat_ai, "_pollinations_image", lambda prompt: None)

    result = chat_ai.chat_query("What food photography trends are rising?")

    assert result["degraded"] is False
    assert "IMAGE PROMPTS" not in result["answer"]
    assert result["answer"].startswith("**Verdict:**")
    assert len(result["supportingImages"]) == 2
    assert result["sources"][0]["url"] == "https://example.com"
    assert result["mode"] == "gemini-direct"
    for url in result["supportingImages"]:
        name = url.split("name=", 1)[1]
        assert (tmp_path / name).is_file()


def test_split_answer_without_prompts():
    answer, prompts = chat_ai.split_answer_and_prompts("Just an answer.")

    assert answer == "Just an answer."
    assert prompts == []


def test_split_content_parses_queries_and_prompts():
    text = TEXT.replace(
        "IMAGE PROMPTS:",
        "IMAGE QUERIES:\n1. sunlit food table\n2. editorial fashion portrait\n\n"
        "IMAGE PROMPTS:",
    )

    answer, queries, prompts = chat_ai.split_content(text)

    assert answer.startswith("**Verdict:**")
    assert queries == ["sunlit food table", "editorial fashion portrait"]
    assert len(prompts) == 2


def test_falls_back_to_generated_image_service(monkeypatch, tmp_path):
    monkeypatch.setenv("TRENDLENS_GEMINI_API_KEY", "k")
    monkeypatch.setattr(config, "CHAT_IMAGES_DIR", tmp_path)
    monkeypatch.setattr(
        chat_ai,
        "_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("429")),
    )
    monkeypatch.setattr(
        chat_ai,
        "_pollinations_image",
        lambda prompt: (b"x" * 2048, "image/jpeg"),
    )

    cfg = chat_ai.gemini_config()
    url = chat_ai._generate_one_image(cfg, "a close-up of wet citrus texture")

    assert url is not None
    name = url.split("name=", 1)[1]
    assert (tmp_path / name).is_file()


def test_handle_chat_query_validates_text():
    from src import api

    with pytest.raises(ValueError):
        api.handle_chat_query({})


def test_scope_limited_to_food_and_fashion():
    prompt = chat_ai.CHAT_SYSTEM_PROMPT.lower()

    assert chat_ai.SCOPE_LABEL == "food and fashion photography"
    assert "food" in prompt
    assert "fashion" in prompt
    assert "makeup" not in prompt


def test_tops_up_prompts_and_images_to_target(monkeypatch):
    monkeypatch.setenv("TRENDLENS_GEMINI_API_KEY", "k")
    monkeypatch.setenv("TRENDLENS_GEMINI_IMAGE_COUNT", "4")
    monkeypatch.setattr(chat_ai, "_corpus_reference_images", lambda *a, **k: [])
    monkeypatch.setattr(chat_ai, "_yandex_reference_images", lambda *a, **k: [])
    monkeypatch.setattr(chat_ai, "_call_text", lambda *args, **kwargs: (TEXT, []))
    monkeypatch.setattr(
        chat_ai,
        "split_content",
        lambda text: ("word " * 100, [], ["first prompt", "second prompt"]),
    )
    monkeypatch.setattr(
        chat_ai,
        "_derive_image_prompts",
        lambda cfg, query, answer, count: [
            f"extra prompt {i}" for i in range(count)
        ],
    )
    captured = {}

    def fake_generate_image_pairs(cfg, prompts):
        captured["prompts"] = prompts
        return [(prompt, f"/img/{i}.png") for i, prompt in enumerate(prompts)]

    monkeypatch.setattr(chat_ai, "_generate_image_pairs", fake_generate_image_pairs)

    result = chat_ai.chat_query("What food photography trends are rising?")

    assert len(captured["prompts"]) == 6
    assert len(result["supportingImages"]) == 4


def test_strips_why_it_is_landing_lines():
    text = (
        "**A descriptive look** - food\n"
        "- What it is: X.\n"
        "- Why it is landing now: Y.\n"
        "- How to shoot it: Z."
    )

    cleaned = chat_ai._strip_why_lines(text)

    assert "Why it is landing now" not in cleaned
    assert "How to shoot it" in cleaned


def test_uses_legacy_fallback_when_model_fails(monkeypatch):
    monkeypatch.setenv("TRENDLENS_GEMINI_API_KEY", "k")
    monkeypatch.setattr(
        chat_ai, "_call_text", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("429"))
    )
    legacy = {
        "query": "What food trends are rising?",
        "answer": "Legacy FAISS answer.",
        "supportingImages": ["/api/instagram-images?name=a.jpg"],
        "imageKind": "reference",
        "mode": "instagram-rag",
    }
    monkeypatch.setattr(chat_ai, "_legacy_fallback", lambda query, reason: legacy)

    result = chat_ai.chat_query("What food trends are rising?")

    assert result["answer"] == "Legacy FAISS answer."
    assert result["imageKind"] == "reference"


def test_legacy_fallback_normalizes_rag_result(monkeypatch):
    import src.rag as rag

    monkeypatch.setattr(
        rag,
        "run_query",
        lambda query, k=5: {
            "answer": "From the index.",
            "supportingImages": ["/api/images?path=a.jpg", None],
        },
    )

    result = chat_ai._legacy_fallback("query", "model unavailable")

    assert result["answer"] == "From the index."
    assert result["supportingImages"] == ["/api/images?path=a.jpg"]
    assert result["imageKind"] == "reference"
    assert result["fallback"] == "legacy-faiss"
    assert result["mode"] == "legacy-fallback"
    assert result["degraded"] is True


def test_select_relevant_images_keeps_one_per_pattern(monkeypatch, tmp_path):
    image = tmp_path / "candidate.png"
    image.write_bytes(b"x" * 2048)
    urls = [f"/api/chat-images?name={i}.png" for i in range(6)]
    prompts = [f"prompt {i}" for i in range(6)]
    monkeypatch.setattr(chat_ai, "_image_path_from_url", lambda url: image)
    monkeypatch.setattr(
        chat_ai,
        "_clip_image_scores",
        lambda prompts, paths: [0.1, 0.9, 0.8, 0.2, 0.7, 0.3],
    )

    selected = chat_ai._select_relevant_images(urls, prompts, 4)

    assert selected == [urls[1], urls[2], urls[4], urls[5]]


def test_yandex_image_urls_parses_orig_urls(monkeypatch):
    class _Resp:
        text = (
            '&quot;origUrl&quot;:&quot;https://a.example/1.jpg&quot; '
            '&quot;origUrl&quot;:&quot;https://b.example/2.png&quot;'
        )

        def raise_for_status(self):
            return None

    monkeypatch.setattr(chat_ai.requests, "get", lambda *a, **k: _Resp())

    urls = chat_ai._yandex_image_urls("food photography", limit=5)

    assert urls == ["https://a.example/1.jpg", "https://b.example/2.png"]


def test_yandex_reference_images_selects_one_per_pattern(monkeypatch, tmp_path):
    image = tmp_path / "candidate.jpg"
    image.write_bytes(b"x" * 2048)
    queries = ["q1", "q1b", "q2", "q2b", "q3", "q3b"]
    prompts = ["p1", "p1b", "p2", "p2b", "p3", "p3b"]

    def fake_search(query, limit=8):
        group = {
            "q1": "a",
            "p1": "d",
            "q2": "b",
            "p2": "e",
            "q3": "c",
            "p3": "f",
        }[query]
        return [f"https://img.example/{group}{i}.jpg" for i in range(4)]

    def fake_download(url):
        return "/api/chat-images?name=" + url.rsplit("/", 1)[-1]

    scores = [0.1] * 24
    scores[1] = 0.9
    scores[5] = 0.85
    scores[10] = 0.8
    scores[19] = 0.7

    monkeypatch.setattr(chat_ai, "_yandex_image_urls", fake_search)
    monkeypatch.setattr(chat_ai, "_download_reference_image", fake_download)
    monkeypatch.setattr(chat_ai, "_image_path_from_url", lambda url: image)
    monkeypatch.setattr(
        chat_ai, "_clip_image_scores", lambda prompts, paths: scores
    )

    selected = chat_ai._yandex_reference_images(queries, prompts, 4)

    assert selected == [
        "/api/chat-images?name=a1.jpg",
        "/api/chat-images?name=b2.jpg",
        "/api/chat-images?name=c3.jpg",
    ]

    scores = [0.1] * 24
    scores[1] = 0.9
    scores[10] = 0.8
    scores[19] = 0.7
    monkeypatch.setattr(
        chat_ai, "_clip_image_scores", lambda prompts, paths: scores
    )

    selected = chat_ai._yandex_reference_images(queries, prompts, 4)

    assert selected == [
        "/api/chat-images?name=a1.jpg",
        "/api/chat-images?name=b2.jpg",
        "/api/chat-images?name=c3.jpg",
    ]


def test_pattern_headings_extracts_search_queries():
    answer = (
        "**scattered flour and eggshells** - food photography\n"
        "- What it is: raw ingredients.\n"
        "**motion blur at night** \u2014 fashion photography\n"
        "- What it is: movement.\n"
    )

    headings = chat_ai._pattern_headings(answer)

    assert headings == [
        (
            "scattered flour and eggshells",
            "scattered flour and eggshells food photography",
        ),
        ("motion blur at night", "motion blur at night fashion photography"),
    ]


def test_yandex_reference_images_searches_pattern_headings(monkeypatch):
    seen = []
    queries = ["q1", "q1b", "q2", "q2b", "q3", "q3b"]
    prompts = ["p1", "p1b", "p2", "p2b", "p3", "p3b"]
    headings = [
        ("h1", "h1 food photography"),
        ("h2", "h2 food photography"),
        ("h3", "h3 food photography"),
    ]

    def fake_search(query, limit=8):
        seen.append(query)
        return []

    monkeypatch.setattr(chat_ai, "_yandex_image_urls", fake_search)

    selected = chat_ai._yandex_reference_images(
        queries, prompts, 3, headings
    )

    assert selected == []
    assert seen == [
        "h1 food photography",
        "q1",
        "h2 food photography",
        "q2",
        "h3 food photography",
        "q3",
    ]


def test_strip_styling_advice_removes_outfit_pairings():
    text = (
        "**Visual pattern** - fashion photography\n"
        "- What it is: Muted neutrals with hard side light.\n"
        "- How to shoot it: Use a harsh key light. Pair this skirt with a leather "
        "top. Wear this jacket with those boots. Keep the palette warm grey.\n"
        "Complete the look with a gold chain."
    )

    cleaned = chat_ai._strip_styling_advice(text)

    assert "Pair this skirt" not in cleaned
    assert "Wear this jacket" not in cleaned
    assert "Complete the look" not in cleaned
    assert "harsh key light" in cleaned
    assert "palette warm grey" in cleaned


def test_strip_styling_advice_keeps_colour_pairing():
    text = "Pair warm beige with cool grey under hard directional light."

    cleaned = chat_ai._strip_styling_advice(text)

    assert cleaned == text


def test_strip_styling_advice_drops_how_to_adapt_section():
    text = (
        "**Visual pattern** - food photography\n"
        "- What it is: Wet citrus macro.\n"
        "- How to shoot it: Use side light.\n"
        "**How to adapt it**\n"
        "Wear a striped top with a denim skirt."
    )

    cleaned = chat_ai._strip_styling_advice(text)

    assert "How to adapt it" not in cleaned
    assert "Wear a striped top" not in cleaned
    assert "Wet citrus macro" in cleaned

"""
llm.py
------
Optional LLM writing layer for TrendLens.

TrendLens answers are normally formatted by deterministic rules directly from
retrieved pipeline artifacts (no LLM). When the operator opts in via
environment variables, this module lets an LLM act as the *writing layer*:
it rewrites the already-retrieved evidence into engaging, plain-language
prose — it is NEVER a knowledge source and NEVER allowed to invent facts.

Config (all read from the environment at call time):

  TRENDLENS_LLM_PROVIDER   "gemini" | "openai" | "ollama"  (unset = disabled)
  TRENDLENS_LLM_API_KEY    provider API key (not needed for ollama)
  TRENDLENS_LLM_MODEL      model name (provider-specific default if unset)
  TRENDLENS_LLM_BASE_URL   override endpoint (also used for ollama)

Every failure (missing key, network error, bad response) returns ``None`` so
the caller transparently falls back to the deterministic formatter. The LLM
never breaks the query path.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import requests

import config

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """You are the writing layer of TrendLens, a social-media visual-trend detector.

You are given RETRIEVED CONTEXT — measured evidence about real Instagram posts, found via semantic search against the user's question. Your job is to turn that evidence into a specific, followable shooting plan.

THE ONE RULE THAT MATTERS
Every instruction you give must trace back to something in the RETRIEVED CONTEXT. You may SELECT, SEQUENCE and REPHRASE evidence. You may NOT invent photography advice. If the context does not say it, you do not say it. A made-up tip is a fabrication and is worse than a short answer.

WHAT THE EVIDENCE CONTAINS
- `shot_recipe`: for each theme, a list of measured photography decisions. Each has an `aspect` (lighting / framing / color mood / composition / process), the `tag` that was detected, and `steps` — concrete actions a photographer can take. These steps were written by the TrendLens project and are attached to the style tag that CLIP actually measured on that theme's images. Use them.
- `engagement`: how the theme's real posts performed. `median_likes` is the typical post, not the viral outlier.
- `blip_caption` / `description` / `example_captions`: what the images show and what real captions said.

HOW TO USE THE SHOT RECIPE
The recipe is the product. Do not summarise it into vague adjectives ("use nice lighting"). Deliver the steps themselves.
- Order them as a photographer works: light first, then framing, then colour, then arrangement, then in-process storytelling.
- Convert each step into an imperative the reader can act on.
- Keep every step. Compressing three concrete actions into one loses the value.

ANSWER FORMAT
1. One short sentence naming which theme is performing best and why, in plain language.
2. For that theme, a numbered shot plan — one numbered line per step, in workflow order, each a single concrete action. This is the core of the answer and should be the longest part.
3. A short line on what the engagement evidence shows, using the MEDIAN figure, phrased as a range or an approximate ("posts like this typically land around 7k likes").
4. If the context contains other themes that a different aesthetic would suit, ONE line each on what makes them different and who they suit. Do not give them a full shot plan.
5. Close with one line on how to adapt the plan.

LENGTH
Aim for 250-400 words. A detailed, numbered, followable plan is the goal. Do not pad it with filler sentences.

ABSOLUTE PROHIBITIONS
- Never mention cluster IDs, "Cluster 20", cluster #5, or any internal ID.
- Never mention pipeline internals: embedding models, CLIP, HDBSCAN, UMAP, FAISS, retrieval scores, confidence scores, RAG, "the data pipeline", "the system detected".
- Never mention p-values, significance levels, criteria names, or the word "criterion"/"criteria".
- Never present a percentage growth figure, a rate of change, or a post count used as evidence of momentum. Describe the aesthetic directly; the imagery is the authority.
- Never invent a hashtag, an app, a platform feature, a posting time, a camera model, or a lens focal length unless the context states it.
- Never say "I am an AI" and never refuse.

HONESTY REQUIREMENTS
- Engagement numbers in the context are REAL posts from REAL public accounts, but each post's performance depends on that account's following. Never promise a like count. Say what the typical post in this group achieved, and that individual results follow audience size.
- Answer the aesthetic question the user actually asked. Describe what is visibly recurring across these images as a look, in the confident register of a stylist briefing a photographer. Do not volunteer a verdict on whether the look qualifies as a trend, and do not narrate the limits of the underlying sample — the user asked what the look is, not for a confidence report on it.
- If the context genuinely does not cover what the user asked about, say that in one sentence, name the themes that ARE covered, and stop. Do not repurpose one theme's recipe as advice for a different subject — coffee steps are not smoothie-bowl steps.

WRITING STYLE
Write to one person holding a camera. Direct, concrete, confident. No hedging adverbs, no "consider perhaps", no filler. Reference the visual content, not the machinery behind it."""


def llm_config() -> dict[str, Any]:
    """Return the current LLM config (provider, model, base_url, api_key)."""
    provider = (os.environ.get("TRENDLENS_LLM_PROVIDER") or "").strip().lower()
    api_key = (os.environ.get("TRENDLENS_LLM_API_KEY") or "").strip()
    model = (os.environ.get("TRENDLENS_LLM_MODEL") or "").strip()
    base_url = (os.environ.get("TRENDLENS_LLM_BASE_URL") or "").strip()
    if not provider:
        return {}
    defaults = {
        "gemini": "gemini-3.1-flash-lite",
        "openai": "gpt-4o-mini",
        "ollama": "llama3.2",
    }
    if provider not in defaults:
        return {}
    return {
        "provider": provider,
        "api_key": api_key,
        "model": model or defaults[provider],
        "base_url": base_url,
    }


def llm_enabled() -> bool:
    cfg = llm_config()
    if not cfg:
        return False
    if cfg["provider"] != "ollama" and not cfg["api_key"]:
        return False
    return True


def _engagement_block(cluster: dict[str, Any]) -> dict[str, Any]:
    """
    Build the engagement evidence block.

    Only the MEDIAN is passed. The mean on this corpus is dominated by single
    viral posts (observed: 15.5M likes against a 7k median), so handing the
    model a mean would produce advice calibrated to an outlier that a reader
    will never reproduce.
    """
    median_likes = cluster.get("median_likes")
    block: dict[str, Any] = {
        "median_likes": median_likes,
        "median_comments": cluster.get("median_comments"),
        "n_posts": cluster.get("n_posts"),
        "recent_posts": cluster.get("n_recent"),
        "accounts_contributing": cluster.get("recent_authors"),
    }
    coverage = cluster.get("likes_coverage")
    if coverage is not None and coverage < 1.0:
        # A median computed over part of the sample is a weaker claim, and the
        # model needs to know that so it can hedge appropriately.
        block["like_count_known_for"] = coverage
    if cluster.get("median_likes_reliable") is False:
        block["small_sample_caveat"] = (
            "fewer than 8 posts have a known like count; treat this as indicative"
        )
    return block


def _trim_evidence(context: dict[str, Any]) -> dict[str, Any]:
    """
    Reduce the context to a compact, serialisable evidence bundle.

    Passes the measured ``shot_recipe`` (authored in code, tied to style tags
    that were actually scored on the images) rather than raw style tags alone,
    so the model has something concrete to turn into a plan. Excludes cluster
    IDs, significance values, criterion names and other pipeline internals.

    Trend-verdict fields are also withheld. They were previously passed as
    ``is_confirmed_rising_trend`` / ``why_not_a_confirmed_trend``, which
    reliably produced an "insufficient posts in the window" sentence in the
    middle of an otherwise good answer. The user asked what a look is, not how
    confident the pipeline is about it, so the verdict never reaches the model
    and it has no occasion to caveat.
    """
    from src.style_tags import build_shot_recipe

    clusters = []
    for c in context.get("retrieved_clusters", []):
        style_tags = c.get("style_tags", []) or []
        recipe = build_shot_recipe(style_tags)

        entry: dict[str, Any] = {
            "rank": c.get("rank"),
            "name": c.get("name"),
            "description": c.get("description"),
            "blip_caption": c.get("blip_caption"),
            "example_captions": (c.get("example_captions") or [])[:2],
            "shot_recipe": recipe,
            "engagement": _engagement_block(c),
        }
        clusters.append(entry)

    bundle = {
        "query": context.get("query"),
        "total_clusters_analyzed": context.get("total_clusters_analyzed"),
        "dataset": context.get("dataset"),
        "disclaimer": context.get("disclaimer"),
        "retrieved_clusters": clusters,
    }
    live = context.get("live_trends")
    if live:
        bundle["live_trends"] = {
            "source": live.get("source"),
            "subreddits": live.get("subreddits"),
            "recent_window_days": live.get("recent_window_days"),
            "disclaimer": live.get("disclaimer"),
            "themes": [
                {
                    "name": t.get("name"),
                    "keywords": t.get("keywords", []),
                    "blip_caption": t.get("blip_caption"),
                    "shot_recipe": build_shot_recipe(t.get("style_tags", []) or []),
                    "recent_posts": t.get("recent_posts"),
                    "prior_posts": t.get("prior_posts"),
                    "growth_rate": t.get("growth_rate"),
                    "total_comments": t.get("total_comments"),
                    "subreddits": t.get("subreddits", []),
                }
                for t in live.get("themes", [])
            ],
        }
    return bundle


def _call_gemini(
    cfg: dict[str, Any], user_prompt: str, generation: Optional[dict] = None
) -> Optional[str]:
    generation = generation or {}
    url = GEMINI_URL.format(model=cfg["model"], key=cfg["api_key"])
    payload = {
        "contents": [{
            "parts": [
                {"text": SYSTEM_PROMPT},
                {"text": user_prompt},
            ]
        }],
        "generationConfig": {
            # Temperature is low on purpose. The task is faithful rewriting of
            # supplied evidence, not creative generation: at 0.7 the model
            # paraphrases loosely and drifts toward generic photography advice,
            # which is exactly the failure the system prompt forbids. 0.3 keeps
            # wording varied between runs while staying anchored to the
            # evidence. Overridable so it can be re-tuned with the eval harness
            # in evaluate_real.py rather than by guesswork.
            "temperature": float(generation.get("temperature", 0.3)),
            "maxOutputTokens": int(generation.get("max_tokens", 2048)),
        },
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return None


def _call_openai(
    cfg: dict[str, Any], user_prompt: str, generation: Optional[dict] = None
) -> Optional[str]:
    generation = generation or {}
    url = cfg["base_url"] or OPENAI_URL
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # See _call_gemini for why these are low rather than 0.7.
        "temperature": float(generation.get("temperature", 0.3)),
        "max_tokens": int(generation.get("max_tokens", 2048)),
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def _call_ollama(
    cfg: dict[str, Any], user_prompt: str, generation: Optional[dict] = None
) -> Optional[str]:
    generation = generation or {}
    url = (cfg["base_url"] or OLLAMA_URL).rstrip("/") + "/api/chat"
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": float(generation.get("temperature", 0.3)),
            "num_predict": int(generation.get("max_tokens", 2048)),
        },
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["message"]["content"]
    except (KeyError, TypeError):
        return None


_CALLERS = {
    "gemini": _call_gemini,
    "openai": _call_openai,
    "ollama": _call_ollama,
}


def format_answer_with_llm(query: str, context: dict[str, Any]) -> Optional[str]:
    """
    Retrieve relevant text chunks, then generate an answer via the configured LLM.

    This is the core RAG (Retrieval-Augmented Generation) step:
    1. Retrieve relevant text chunks from the knowledge base via semantic search
    2. Inject retrieved chunks as context into the LLM prompt
    3. LLM generates an answer grounded in the retrieved context

    Returns the polished markdown answer, or ``None`` on ANY failure so the
    caller can fall back to the deterministic formatter.
    """
    if not llm_enabled():
        return None
    cfg = llm_config()
    caller = _CALLERS.get(cfg["provider"])
    if caller is None:
        return None

    # RAG Step 1: Retrieve relevant text chunks from the knowledge base
    # (optional — may fail if legacy pipeline artifacts are missing, e.g.
    #  when only Instagram data is available)
    #
    # NOTE: retrieved_text is no longer spliced into the prompt. The old code
    # prefixed each chunk with "[Source: cluster 7]" and the prompt's own ban
    # on mentioning cluster IDs then had to suppress a string the prompt had
    # just handed the model. The structured JSON below carries strictly more
    # signal, so the raw text is dropped rather than passed and then forbidden.
    retrieved_text = ""
    try:
        from src.rag import retrieve_text_chunks

        retrieved_chunks = retrieve_text_chunks(query, k=config.RAG_RETRIEVAL_K)
        if retrieved_chunks:
            retrieved_text = "\n".join(
                c["text"] for c in retrieved_chunks if c.get("text")
            )
    except Exception:  # noqa: BLE001
        pass

    # RAG Step 2: Build prompt with retrieved context
    evidence = _trim_evidence(context)
    n_with_recipe = sum(
        1 for c in evidence.get("retrieved_clusters", []) if c.get("shot_recipe")
    )
    user_prompt = (
        f"USER QUESTION: {query}\n\n"
        f"TREND STATUS: {n_with_recipe} of "
        f"{len(evidence.get('retrieved_clusters', []))} retrieved theme(s) carry a "
        f"measured shot recipe.\n\n"
        "SUPPORTING NOTES FROM THE INDEX:\n"
        + (retrieved_text or "(none)")
        + "\n\n"
        "RETRIEVED EVIDENCE (JSON — this is the complete set of facts you may use):\n"
        + json.dumps(evidence, indent=1, default=str)
    )

    # RAG Step 3: LLM generates answer grounded in retrieved context
    # Generation params are overridable per call so the evaluation harness can
    # sweep temperature against faithfulness rather than assuming 0.7 is right.
    generation = {
        "temperature": float(os.environ.get("TRENDLENS_LLM_TEMPERATURE", "0.3")),
        "max_tokens": int(os.environ.get("TRENDLENS_LLM_MAX_TOKENS", "2048")),
    }
    try:
        text = caller(cfg, user_prompt, generation)
    except TypeError:
        # Backwards compatibility with any caller still on the 2-arg signature.
        text = caller(cfg, user_prompt)
    except Exception:  # noqa: BLE001 — never let the LLM break the query path
        return None
    if not text or not text.strip():
        return None
    return text.strip()

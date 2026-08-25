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

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """You are the writing layer of TrendLens, a social-media visual-trend detector.

You are given RETRIEVED CONTEXT — text chunks from a visual-trend knowledge base, found via semantic search against the user's question. Your job is to generate an answer that is DIRECTLY GROUNDED in this retrieved context. Every piece of advice must reference what the retrieved data says.

STRICT RULES — violations are unacceptable:
1. Base EVERY claim on the RETRIVED CONTEXT. Never invent subjects, colors, stats, platforms, hashtags, posting times, or engagement numbers that are not present in the retrieved data.
2. The engagement/likes/timestamps in the evidence are SYNTHETIC DEMO data unless the evidence explicitly says otherwise. Present them as demo estimates, never as real platform numbers.
3. If the retrieved context contains no real match for the subject, say so honestly and describe the closest matches from what WAS retrieved.
4. Answer the user's question directly. Do not say you are an AI/LLM. Do not refuse.
5. Follow the OUTPUT FORMAT below exactly.

OUTPUT FORMAT — scannable structure with substantive content:
1. Open with ONE short intro sentence framing what the data shows.
2. Render each relevant theme as its own markdown bullet, separated by BLANK
   LINES, ranked by relevance, max 5 themes. Format:
   - **Theme name:** What the trend is — the subject, mood, or behaviour driving
     it — plus one concrete execution cue from its style tags. Max two sentences.
3. Close with ONE short sentence synthesising the directions so the reader can
   choose between them (e.g. technical vs rustic).
4. Standalone shooting-advice bullets ONLY when the question explicitly asks HOW
   to shoot; otherwise fold the key cue into each theme's description.
5. Separate every block with a BLANK LINE so markdown renders as clean lists.
   Never use tables, section headers, or horizontal rules. Target 150–250 words.
   Every sentence must be grounded in the retrieved evidence.

CRITICAL — NEVER mention these in your answer:
- Cluster IDs (e.g. "Cluster 20", "cluster 17", "Cluster #5")
- Engagement scores, post counts, or trend scores (e.g. "79.04", "325 posts", "trend score 0.11")
- Lifecycle labels (e.g. "Rising", "Stable", "Declining")
- Trend category labels (e.g. "Trending up", "Fading", "Steady presence", "Just appeared")
- Any numeric metrics from the pipeline internals

When discussing what is trending, use natural conversational language like:
- "Currently, [X] is trending in cafe visuals."
- "Right now, [X] is seeing a lot of activity on social media."
- "[X] is a popular visual theme at the moment."
Never label trends with stiff categories. Write as if recommending to a friend.

Instead, describe the visual patterns and actionable advice in plain language. Reference the visual content (keywords, characteristics, captions) not the pipeline internals.

SUBJECT vs EXECUTION — critical distinction:
Each retrieved cluster may include "style_tags": a measured photography-execution profile (framing, lighting, color mood, process storytelling, composition) scored directly from the images, plus a BLIP caption describing the subject.
- Keywords/captions = WHAT is being shot (subjects).
- style_tags = HOW it is being shot (execution).
When the user asks about photography styles, aesthetics, or how to shoot something for engagement, lead with the execution evidence (style tags) rather than listing subjects. When giving advice, translate the style tags into concrete shooting guidance — but ONLY the tags present in the evidence, never generic styling tips.

NO ADAPTATION RULE — strictest rule of all:
The retrieved evidence defines exactly which subjects exist in the data (each chunk names its theme, e.g. "manual latte art").
- If the user asks about a subject that is NOT among the retrieved themes, your ENTIRE answer must be a short refusal: state that no data on that subject exists yet and name the subjects that ARE covered. Then stop.
- In that case add NOTHING else — no style tags, no shooting advice, no "how those subjects are being captured" section, no offer of further help. Information about other themes must not appear in the answer at all.
- You may NOT repurpose, re-label, or "adapt" one subject's evidence as advice for a different subject. Coffee tips are coffee tips — presenting them as smoothie-bowl or burger advice is fabrication and is forbidden.
- Partial keyword overlap (e.g. both are drinks) does not count as a match. The theme name itself must cover what the user asked about."""


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


def _trim_evidence(context: dict[str, Any]) -> dict[str, Any]:
    """Reduce the context to a compact, serialisable evidence bundle.

    Excludes cluster IDs, engagement scores, lifecycle labels, and other
    pipeline internals — only the visual content (name, description,
    characteristics, captions) is passed to the LLM.
    """
    clusters = []
    for c in context.get("retrieved_clusters", []):
        clusters.append(
            {
                "rank": c.get("rank"),
                "name": c.get("name"),
                "description": c.get("description"),
                "blip_caption": c.get("blip_caption"),
                "characteristics": c.get("characteristics", []),
                "style_tags": c.get("style_tags", []),
                "interpretation_confidence": c.get("interpretation_confidence"),
            }
        )
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
                    "style_tags": t.get("style_tags", []),
                    "recent_posts": t.get("recent_posts"),
                    "prior_posts": t.get("prior_posts"),
                    "growth_rate": t.get("growth_rate"),
                    "avg_engagement": t.get("avg_engagement"),
                    "total_comments": t.get("total_comments"),
                    "subreddits": t.get("subreddits", []),
                }
                for t in live.get("themes", [])
            ],
        }
    return bundle


def _call_gemini(cfg: dict[str, Any], user_prompt: str) -> Optional[str]:
    url = GEMINI_URL.format(model=cfg["model"], key=cfg["api_key"])
    payload = {
        "contents": [{
            "parts": [
                {"text": SYSTEM_PROMPT},
                {"text": user_prompt},
            ]
        }],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 4096},
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return None


def _call_openai(cfg: dict[str, Any], user_prompt: str) -> Optional[str]:
    url = cfg["base_url"] or OPENAI_URL
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def _call_ollama(cfg: dict[str, Any], user_prompt: str) -> Optional[str]:
    url = (cfg["base_url"] or OLLAMA_URL).rstrip("/") + "/api/chat"
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
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
    retrieved_text = ""
    try:
        from src.rag import retrieve_text_chunks

        retrieved_chunks = retrieve_text_chunks(query, k=5)
        if retrieved_chunks:
            retrieved_text = "\n\n".join(
                f"[Source: cluster {c['cluster_id']}] {c['text']}"
                for c in retrieved_chunks
            )
    except Exception:  # noqa: BLE001
        pass

    # RAG Step 2: Build prompt with retrieved context
    evidence = _trim_evidence(context)
    user_prompt = (
        f"USER QUESTION: {query}\n\n"
        "RETRIEVED CONTEXT (relevant visual trends found in the knowledge base):\n"
        + retrieved_text
        + "\n\n"
        "RETRIEVED CLUSTER DATA (JSON):\n"
        + json.dumps(evidence, indent=1, default=str)
    )

    # RAG Step 3: LLM generates answer grounded in retrieved context
    try:
        text = caller(cfg, user_prompt)
    except Exception:  # noqa: BLE001 — never let the LLM break the query path
        return None
    if not text or not text.strip():
        return None
    return text.strip()

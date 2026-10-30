"""
Gemini-only trend chat for the TrendLens chat interface.

This module replaces the old pipeline-fed chat path. It asks Gemini for the
latest food and fashion-photography trends, optionally grounding the answer in
Google Search, and asks Gemini's image model for illustrative "representative"
visuals. No other provider and no old pipeline artifact is used here.

Environment:
  TRENDLENS_GEMINI_API_KEY        Gemini API key (falls back to GEMINI_API_KEY,
                                  GOOGLE_API_KEY, then TRENDLENS_LLM_API_KEY)
  TRENDLENS_GEMINI_MODEL          text model (default: gemini-3.1-flash-lite)
  TRENDLENS_GEMINI_IMAGE_MODEL    image model (default: gemini-3.1-flash-image)
  TRENDLENS_GEMINI_IMAGE_COUNT    images per answer, 0-12 (default: 4)
  TRENDLENS_GEMINI_SEARCH_GROUNDING  1/0, use Google Search grounding (default: 1)
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Optional

import requests

import config

GEMINI_MODEL_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)
DEFAULT_TEXT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image"
IMAGE_MODEL_FALLBACKS = (
    "gemini-3.1-flash-image",
    "gemini-2.5-flash-image",
    "gemini-nano-banana-2.1",
)
SCOPE_LABEL = "food and fashion photography"

CHAT_SYSTEM_PROMPT = """You are TrendLens, a visual-trend analyst.

TrendLens exists to spot visual patterns before they have a name. Never give a
pattern a name, label, genre, or aesthetic title. Describe what the images look
like in plain language instead.

You report VISUALS ONLY: colour palette, tone, light, shadow, texture,
composition, framing, surfaces, props, editing, and visible treatment. You do
not give personal styling advice, wardrobe formulas, outfit pairings, shopping
recommendations, recipes, or serving instructions.

You answer questions ONLY about current trends in two areas:
1. Food photography and food styling.
2. Fashion photography and styling.

If the user asks about anything outside those two areas, decline in one short
sentence and invite a food or fashion-photography question instead.
Never answer maths, coding, politics, medical, legal, or general-knowledge
questions.

Use Google Search grounding when it is available to identify what is genuinely
current. Prefer patterns from the last 6-12 months. Describe the visible
grammar of each pattern (subject, light, colour palette, texture, framing,
surface or garment treatment, editing) instead of naming it. Never use or
invent named trends or aesthetic labels — for example, do not write
"Hyper-Textural Macro", "Deconstructed Chaos", or "Analog Nostalgia". Write
"extreme close-ups of wet, tactile ingredients" or "loose, scattered plating
with crumbs and spills" instead. Do not use title case for pattern headings.
If a pattern has no settled name, say so plainly.

Answer directly in 250-400 words of Markdown:

1. A one-sentence verdict describing the strongest current visual pattern for
   the question in plain language. Do not name it.
2. "Patterns to watch" with exactly 3 patterns. For each:
   **A short descriptive phrase** - category
   - What it is: one or two sentences describing the visible pattern — colour
     palette, light, texture, composition, surfaces, and visual treatment.
   - How to shoot it: 3 to 5 concrete steps covering lighting, framing,
     colour palette, texture, surface or garment treatment, and process.
   Do not include a "why it is landing now" line or any explanation of why the
   pattern is popular.

Visual-only rules:
- Never say what to wear with what, what to pair, what to buy, or how to build
  an outfit.
- For fashion, describe visible garment treatment (layering, silhouette,
  fabric, colour blocking, hardware) as it appears in the image. Do not
  recommend combinations.
- For food, describe plating, surfaces, colour, texture, and light. Do not
  give recipes or serving instructions.
- Colour names, palettes, tones, and lighting are encouraged. Prefer concrete
  visual detail over advice.
- Stay concrete and photographic: light, colour, framing, texture, surfaces,
  props, poses, editing.
- Do not invent names, statistics, engagement numbers, hashtags, or sources.
- Do not mention that you are an AI, a language model, or a search engine.
- Do not discuss the TrendLens pipeline, CLIP, embeddings, clusters, or
  retrieval.
- No preamble. Start with the verdict.

After the answer, add exactly these two blocks. Write 2 entries per pattern in
pattern order, for 6 entries total.

IMAGE QUERIES:
1. <3 to 6 plain words a stock-photo search could use for the first pattern>
2. <3 to 6 plain words for the same pattern, another angle>
3. <3 to 6 plain words for the second pattern>
4. <3 to 6 plain words for the same pattern, another angle>
5. <3 to 6 plain words for the third pattern>
6. <3 to 6 plain words for the same pattern, another angle>

IMAGE PROMPTS:
1. <a concise text-to-image prompt illustrating the first pattern>
2. <a second prompt for the first pattern, different composition>
3. <a prompt for the second pattern>
4. <a second prompt for the second pattern, different composition>
5. <a prompt for the third pattern>
6. <a second prompt for the third pattern, different composition>

Each image query must:
- name the visible subject and the category, not an abstract concept,
- read like a search phrase, for example "sunlit mediterranean food table",
- avoid invented trend names or aesthetic labels.

Each image prompt must:
- describe the visible pattern and category without naming it,
- describe a photorealistic editorial photograph that visually captures it,
- specify subject, setting, lighting, colour palette, composition, and styling,
- be 35 to 70 words,
- contain no text overlays, logos, watermarks, celebrities, or identifiable
  real people (anonymous models are fine),
- avoid anything unsafe or misleading.
"""
_PROMPTS_MARKER = re.compile(
    r"(?im)^\s*(?:\*\*|__)?\s*IMAGE\s*PROMPTS?\s*(?:\*\*|__)?\s*:?\s*$"
)
_PROMPTS_INLINE = re.compile(r"(?i)IMAGE\s*PROMPTS?\s*:\s*")
_QUERIES_MARKER = re.compile(
    r"(?im)^\s*(?:\*\*|__)?\s*IMAGE\s*QUERIES?\s*(?:\*\*|__)?\s*:?\s*$"
)
_QUERIES_INLINE = re.compile(r"(?i)IMAGE\s*QUERIES?\s*:\s*")
_LINE_PREFIX = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s*")
_WHY_LINE = re.compile(
    r"(?i)^\s*(?:[-*+•]\s*)?(?:\*\*)?\s*why\s+"
    r"(?:it(?:['’]s|\s+is)|\s*this\s+is)\s+landing(?:\s+now)?"
    r"(?:\*\*)?\s*[:\-–]?"
)
_ADAPT_HEADING = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*how\s+to\s+adapt\s+it"
    r"\s*(?:\*\*)?\s*:?\s*$"
)
_STYLING_ADVICE = re.compile(
    r"(?i)(?:"
    r"\b(?:pair|team|match|combine)\b[^.!?\n]{0,60}\bwith\b[^.!?\n]{0,60}"
    r"\b(?:skirt|top|shirt|blouse|jacket|coat|dress|trousers|pants|jeans|"
    r"shoes|boots|bag|accessor(?:y|ies)|outfit|look|garment|layer)\b"
    r"|\bwear\b[^.!?\n]{0,60}\bwith\b"
    r"|\b(?:goes|go|pairs|works)\s+well\s+with\b"
    r"|\b(?:complete|finish|elevate)\s+the\s+look\s+with\b"
    r"|\bstyle\s+(?:it|this|that)\b[^.!?\n]{0,30}\bwith\b"
    r"|\baccessori[sz]e\b[^.!?\n]{0,40}\bwith\b"
    r"|\boutfit\s+(?:formula|combination|pairing)\b"
    r")"
)
_FOOD_DOMAIN = re.compile(
    r"(?i)\b(?:food|recipe|cook|bake|chef|kitchen|eat|dish|coffee|latte|"
    r"cake|bread|dessert|restaurant|meal|drink|barista|pastry|chocolate|"
    r"fruit|vegetable|brunch|dinner|lunch|breakfast|pizza|pasta|sushi|"
    r"taco|wine|cocktail|matcha|espresso)\b"
)
_FASHION_DOMAIN = re.compile(
    r"(?i)\b(?:fashion|style|outfit|runway|model|wear|collection|designer|"
    r"couture|streetwear|vogue|garment|dress|suit|jacket|shoe|accessor|"
    r"styling)\b"
)


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or "").strip() or default


def gemini_config() -> dict[str, Any]:
    api_key = ""
    for var in (
        "TRENDLENS_GEMINI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "TRENDLENS_LLM_API_KEY",
    ):
        api_key = _env(var)
        if api_key:
            break

    model = _env(
        "TRENDLENS_GEMINI_MODEL",
        _env("TRENDLENS_LLM_MODEL", DEFAULT_TEXT_MODEL),
    )
    image_model = _env("TRENDLENS_GEMINI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)
    try:
        image_count = int(_env("TRENDLENS_GEMINI_IMAGE_COUNT", "4"))
    except ValueError:
        image_count = 4
    search_grounding = _env(
        "TRENDLENS_GEMINI_SEARCH_GROUNDING", "1"
    ).lower() not in {"0", "false", "no", "off"}
    return {
        "api_key": api_key,
        "model": model,
        "image_model": image_model,
        "image_count": max(0, min(12, image_count)),
        "search_grounding": search_grounding,
    }


def is_enabled() -> bool:
    return bool(gemini_config().get("api_key"))


def _request(model: str, api_key: str, payload: dict, timeout: int) -> dict:
    url = GEMINI_MODEL_URL.format(model=model, key=api_key)
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _extract_text(data: dict) -> str:
    for candidate in data.get("candidates") or []:
        parts = (candidate.get("content") or {}).get("parts") or []
        texts = [part.get("text") for part in parts if part.get("text")]
        if texts:
            return "\n".join(texts).strip()
    return ""


def _extract_sources(data: dict) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for candidate in data.get("candidates") or []:
        metadata = candidate.get("groundingMetadata") or {}
        for chunk in metadata.get("groundingChunks") or []:
            web = chunk.get("web") or {}
            url = web.get("uri")
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append({"title": web.get("title") or url, "url": url})
    return sources[:8]


def _call_text(
    cfg: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    grounding: bool,
) -> tuple[str, list[dict[str, str]]]:
    payload: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": float(_env("TRENDLENS_GEMINI_TEMPERATURE", "0.45")),
            "maxOutputTokens": int(_env("TRENDLENS_GEMINI_MAX_TOKENS", "4096")),
        },
    }
    use_grounding = bool(grounding and cfg.get("search_grounding"))
    if use_grounding:
        payload["tools"] = [{"google_search": {}}]

    try:
        data = _request(cfg["model"], cfg["api_key"], payload, timeout=90)
    except requests.HTTPError:
        if not use_grounding:
            raise
        payload.pop("tools", None)
        data = _request(cfg["model"], cfg["api_key"], payload, timeout=90)

    return _extract_text(data), _extract_sources(data)


def _clean_prompt_line(line: str) -> str:
    cleaned = _LINE_PREFIX.sub("", line.strip()).strip()
    cleaned = cleaned.strip('"\'`')
    return cleaned


def _parse_prompt_lines(block: str, min_words: int = 8) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        cleaned = _clean_prompt_line(line)
        if len(cleaned.split()) < min_words or cleaned in seen:
            continue
        seen.add(cleaned)
        prompts.append(cleaned)
    return prompts


def _find_marker(text: str, block: re.Pattern, inline: re.Pattern):
    return block.search(text) or inline.search(text)


def split_content(text: str) -> tuple[str, list[str], list[str]]:
    text = (text or "").strip()
    prompts_match = _find_marker(text, _PROMPTS_MARKER, _PROMPTS_INLINE)
    queries_match = _find_marker(text, _QUERIES_MARKER, _QUERIES_INLINE)
    matches = [m for m in (prompts_match, queries_match) if m]
    if not matches:
        return text.strip("`").strip(), [], []

    answer_end = min(m.start() for m in matches)
    answer = text[:answer_end].strip().strip("`").strip()

    queries: list[str] = []
    if queries_match:
        queries_end = (
            prompts_match.start()
            if prompts_match and prompts_match.start() > queries_match.end()
            else len(text)
        )
        queries = _parse_prompt_lines(
            text[queries_match.end() : queries_end], min_words=2
        )

    prompts: list[str] = []
    if prompts_match:
        prompts = _parse_prompt_lines(text[prompts_match.end() :])

    return answer, queries, prompts


def split_answer_and_prompts(text: str) -> tuple[str, list[str]]:
    answer, _queries, prompts = split_content(text)
    return answer, prompts


def _strip_why_lines(text: str) -> str:
    """Drop any "why it is landing now" lines the model slips in."""
    lines = [line for line in text.splitlines() if not _WHY_LINE.match(line)]
    return "\n".join(lines).strip()


def _strip_styling_advice(text: str) -> str:
    """Drop wardrobe pairing and other non-visual advice."""
    text = text or ""
    adapt = _ADAPT_HEADING.search(text)
    if adapt:
        text = text[: adapt.start()].rstrip()

    lines: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            lines.append(line)
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        kept = [part for part in parts if not _STYLING_ADVICE.search(part)]
        if kept:
            lines.append(" ".join(kept))
    return "\n".join(lines).strip()


_PATTERN_HEADING = re.compile(
    r"^\s*\*\*(.+?)\*\*\s*[-\u2013\u2014]\s*(.+?)\s*$",
    re.MULTILINE,
)


def _pattern_headings(answer: str) -> list[tuple[str, str]]:
    """Return (heading text, image-search query) for each named pattern."""
    headings: list[tuple[str, str]] = []
    for match in _PATTERN_HEADING.finditer(answer or ""):
        heading = match.group(1).strip(" .:")
        category = match.group(2).strip(" .:")
        if not heading:
            continue
        if "photography" not in category.lower():
            category = f"{category} photography"
        headings.append((heading, f"{heading} {category}"))
        if len(headings) >= 3:
            return headings

    if headings:
        return headings

    for line in (answer or "").splitlines():
        match = re.match(r"^\s*\*\*(.+?)\*\*\s*$", line)
        if not match:
            continue
        heading = match.group(1).strip(" .:")
        if not heading or heading.lower() in {"what it is", "how to shoot it"}:
            continue
        headings.append((heading, f"{heading} photography"))
        if len(headings) >= 3:
            break
    return headings


def _derive_image_prompts(
    cfg: dict[str, Any], query: str, answer: str, count: int
) -> list[str]:
    if count <= 0:
        return []
    system = (
        "You write safe, photorealistic prompts for an editorial image model. "
        "Return only a numbered list of prompts, one per line."
    )
    user = (
        f"The user asked: {query}\n\n"
        f"The visual-pattern answer was:\n{answer}\n\n"
        f"Write {count} image-prompts that visually illustrate the patterns in "
        "that answer. Do not invent or use named trends or aesthetic labels; "
        "describe only what is visible. Each prompt must describe food or "
        "fashion photography, including subject, setting, lighting, colour "
        "palette, composition, and styling. No text overlays, logos, "
        "watermarks, celebrities, or identifiable real people. Anonymous "
        "models are fine. 35-70 words each."
    )
    try:
        text, _sources = _call_text(cfg, system, user, grounding=False)
    except Exception:
        return []
    return _parse_prompt_lines(text)[:count]


def _queries_to_prompts(queries: list[str], count: int) -> list[str]:
    """Turn short image-search phrases into text-to-image prompts."""
    prompts: list[str] = []
    for query in queries:
        if len(prompts) >= count:
            break
        cleaned = (query or "").strip().rstrip(".")
        if not cleaned:
            continue
        prompts.append(
            f"Photorealistic editorial food or fashion photograph of "
            f"{cleaned}, styled set, natural window light, rich texture, "
            "shallow depth of field, 4:5 portrait composition."
        )
    return prompts


def _candidate_prompts(
    cfg: dict[str, Any],
    query: str,
    answer: str,
    prompts: list[str],
    queries: list[str],
    count: int,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    def add(items: list[str]) -> None:
        for item in items:
            if item and item not in seen:
                seen.add(item)
                result.append(item)

    add(prompts)
    if len(result) < count and len(answer.split()) >= 80:
        add(_derive_image_prompts(cfg, query, answer, count - len(result)))
    if len(result) < count:
        add(_queries_to_prompts(queries, count - len(result)))
    return result[:count]


def _image_digest(cfg: dict[str, Any], prompt: str) -> str:
    # v3 invalidates images generated by the older, less prompt-aligned path.
    return hashlib.sha256(
        f"chat-img-v3:{cfg['image_model']}:{prompt}".encode("utf-8")
    ).hexdigest()[:24]


def _cache_path(cfg: dict[str, Any], prompt: str) -> Optional[Any]:
    digest = _image_digest(cfg, prompt)
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        path = config.CHAT_IMAGES_DIR / f"{digest}{ext}"
        if path.is_file():
            return path
    return None


def _extract_inline_image(data: dict) -> Optional[tuple[bytes, str]]:
    for candidate in data.get("candidates") or []:
        parts = (candidate.get("content") or {}).get("parts") or []
        for part in parts:
            inline = part.get("inlineData") or part.get("inline_data") or {}
            raw = inline.get("data")
            if not raw:
                continue
            mime = (
                inline.get("mimeType")
                or inline.get("mime_type")
                or "image/png"
            )
            try:
                return base64.b64decode(raw), mime
            except (ValueError, TypeError):
                continue
    return None


POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}"
YANDEX_IMAGES_URL = "https://yandex.com/images/search"
YANDEX_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _save_image(digest: str, raw: bytes, mime: str) -> Optional[str]:
    ext = {
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/png": ".png",
    }.get((mime or "").lower(), ".png")
    path = config.CHAT_IMAGES_DIR / f"{digest}{ext}"
    try:
        path.write_bytes(raw)
    except OSError:
        return None
    return f"/api/chat-images?name={path.name}"


def _pollinations_image(prompt: str) -> Optional[tuple[bytes, str]]:
    """Free prompt-driven image service used before Gemini image generation."""
    from urllib.parse import quote

    url = POLLINATIONS_URL.format(prompt=quote(prompt, safe=""))
    try:
        response = requests.get(
            url,
            params={"width": 900, "height": 1125, "nologo": "true"},
            timeout=75,
        )
        response.raise_for_status()
    except Exception:
        return None
    mime = (response.headers.get("content-type") or "").split(";")[0].strip()
    if not mime.lower().startswith("image/") or len(response.content) < 1024:
        return None
    return response.content, mime


def _reference_digest(url: str) -> str:
    return hashlib.sha256(f"chat-ref-v1:{url}".encode("utf-8")).hexdigest()[:24]


def _reference_cache_path(url: str) -> Optional[Any]:
    digest = _reference_digest(url)
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        path = config.CHAT_IMAGES_DIR / f"{digest}{ext}"
        if path.is_file():
            return path
    return None


def _download_reference_image(url: str) -> Optional[str]:
    cached = _reference_cache_path(url)
    if cached is not None:
        return f"/api/chat-images?name={cached.name}"

    try:
        from io import BytesIO

        from PIL import Image, ImageOps

        response = requests.get(url, headers=YANDEX_HEADERS, timeout=45)
        response.raise_for_status()
        raw = response.content
        if len(raw) < 2048 or len(raw) > 12 * 1024 * 1024:
            return None

        image = Image.open(BytesIO(raw))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(
            image, (900, 1125), method=Image.Resampling.LANCZOS
        )
        out = BytesIO()
        image.save(out, "JPEG", quality=88, optimize=True)
        return _save_image(_reference_digest(url), out.getvalue(), "image/jpeg")
    except Exception:
        return None


def _is_public_image_url(url: str) -> bool:
    import ipaddress
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.strip().lower()
    if host in {"localhost", "0.0.0.0", "::1"} or host.endswith(".local"):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return address.is_global


def _yandex_image_urls(query: str, limit: int = 6) -> list[str]:
    import html

    try:
        response = requests.get(
            YANDEX_IMAGES_URL,
            params={"text": query},
            headers=YANDEX_HEADERS,
            timeout=30,
        )
        response.raise_for_status()
    except Exception:
        return []

    text = html.unescape(response.text)
    urls: list[str] = []
    for raw in re.findall(r'"origUrl":"(.*?)"', text):
        url = raw.replace("\\/", "/").replace("\\u0026", "&").strip()
        if (
            not url.startswith("http")
            or url in urls
            or not _is_public_image_url(url)
        ):
            continue
        urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def _yandex_reference_images(
    queries: list[str],
    prompts: list[str],
    limit: int,
    headings: Optional[list[tuple[str, str]]] = None,
) -> list[str]:
    if limit <= 0 or not prompts:
        return []

    prompt_groups = [
        prompts[start : start + 2] for start in range(0, len(prompts), 2)
    ][:3]
    query_groups = [
        queries[start : start + 2] for start in range(0, len(queries), 2)
    ]
    groups = []
    headings = headings or []
    for index, prompt_group in enumerate(prompt_groups):
        short_query = ""
        if index < len(query_groups) and query_groups[index]:
            short_query = query_groups[index][0]

        heading_text = ""
        heading_query = ""
        if index < len(headings):
            heading_text, heading_query = headings[index]

        if heading_query:
            query = heading_query
            text = (
                f"{heading_text} {short_query}".strip()
                if short_query
                else heading_query
            )
        elif short_query:
            query = short_query
            text = short_query
        else:
            query = prompt_group[0]
            text = prompt_group[0]

        if short_query and short_query != query:
            second = short_query
        elif prompt_group[0] and prompt_group[0][:240] != query:
            second = prompt_group[0][:240]
        else:
            second = ""

        groups.append(
            {
                "query": query,
                "second": second,
                "text": text,
                "prompt": prompt_group[0],
            }
        )
    if not groups:
        return []
    limit = min(limit, len(groups))

    searches: list[tuple[int, str]] = []
    for group_index, group in enumerate(groups):
        searches.append((group_index, group["query"]))
        if group["second"]:
            searches.append((group_index, group["second"]))

    try:
        with ThreadPoolExecutor(max_workers=min(3, len(searches))) as pool:
            search_results = list(
                pool.map(
                    lambda search: (
                        search[0],
                        _yandex_image_urls(search[1], 8),
                    ),
                    searches,
                )
            )

        url_lists: list[list[list[str]]] = [[] for _group in groups]
        for group_index, urls in search_results:
            if urls:
                url_lists[group_index].append(urls)

        tasks: list[tuple[int, str]] = []
        for group_index, lists in enumerate(url_lists):
            for urls in lists:
                for url in urls[:4]:
                    tasks.append((group_index, url))
        if not tasks:
            return []
        with ThreadPoolExecutor(max_workers=min(5, len(tasks))) as pool:
            downloaded = list(
                pool.map(
                    lambda task: (
                        task[0],
                        _download_reference_image(task[1]),
                    ),
                    tasks,
                )
            )
    except Exception:
        return []

    candidates: list[tuple[int, str]] = []
    seen: set[str] = set()
    for group_index, url in downloaded:
        if not url or url in seen:
            continue
        seen.add(url)
        candidates.append((group_index, url))
    if not candidates:
        return []

    scores: Optional[list[float]] = None
    paths = [_image_path_from_url(url) for _group, url in candidates]
    if all(path is not None for path in paths):
        scores = _clip_image_scores(
            [groups[group_index]["text"] for group_index, _url in candidates],
            paths,
        )

    if scores is None or len(scores) != len(candidates):
        chosen: list[int] = []
        seen_groups: set[int] = set()
        for index, (group_index, _url) in enumerate(candidates):
            if group_index in seen_groups:
                continue
            seen_groups.add(group_index)
            chosen.append(index)
            if len(chosen) >= limit:
                break
        for index in range(len(candidates)):
            if len(chosen) >= limit:
                break
            if index not in chosen:
                chosen.append(index)
        return [candidates[index][1] for index in chosen[:limit]]

    best_by_group: list[tuple[int, int, float]] = []
    for group_index in range(len(groups)):
        indices = [
            index
            for index, (candidate_group, _url) in enumerate(candidates)
            if candidate_group == group_index
        ]
        if not indices:
            continue
        best = max(indices, key=lambda index: scores[index])
        best_by_group.append((group_index, best, scores[best]))
    best_by_group.sort(key=lambda item: item[2], reverse=True)

    group_best = {
        group_index: score for group_index, _index, score in best_by_group
    }
    chosen = [index for _group, index, _score in best_by_group[:limit]]
    if len(chosen) < limit:
        remaining = [index for index in range(len(candidates)) if index not in chosen]
        for index in sorted(remaining, key=lambda item: scores[item], reverse=True):
            if len(chosen) >= limit:
                break
            group_index = candidates[index][0]
            if scores[index] < 0.9 * group_best.get(group_index, 0.0):
                continue
            chosen.append(index)

    chosen.sort(key=lambda index: (candidates[index][0], index))
    return [candidates[index][1] for index in chosen]


def _generate_one_image(cfg: dict[str, Any], prompt: str) -> Optional[str]:
    cached = _cache_path(cfg, prompt)
    if cached is not None:
        return f"/api/chat-images?name={cached.name}"

    digest = _image_digest(cfg, prompt)

    # Pollinations follows the prompt closely. Prefer it, then fall back to
    # Gemini image models when the service is unavailable.
    service_image = _pollinations_image(prompt)
    if service_image is not None:
        raw, mime = service_image
        url = _save_image(digest, raw, mime)
        if url:
            return url

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }

    models = [cfg["image_model"]]
    for fallback in IMAGE_MODEL_FALLBACKS:
        if fallback not in models:
            models.append(fallback)

    for model in models:
        try:
            data = _request(model, cfg["api_key"], payload, timeout=120)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429:
                break
            continue
        image = _extract_inline_image(data)
        if image is None:
            continue
        raw, mime = image
        url = _save_image(digest, raw, mime)
        if url:
            return url
    return None


def _generate_image_pairs(
    cfg: dict[str, Any], prompts: list[str]
) -> list[tuple[str, str]]:
    prompts = [p for p in prompts if p]
    if not prompts:
        return []
    with ThreadPoolExecutor(max_workers=min(3, len(prompts))) as pool:
        results = list(
            pool.map(lambda p: (p, _generate_one_image(cfg, p)), prompts)
        )
    return [(prompt, url) for prompt, url in results if url]


def generate_images(cfg: dict[str, Any], prompts: list[str]) -> list[str]:
    return [url for _prompt, url in _generate_image_pairs(cfg, prompts)]


_CORPUS_IMAGE_CACHE: Optional[dict[str, Any]] = None


def _load_corpus_image_index() -> dict[str, Any]:
    global _CORPUS_IMAGE_CACHE
    if _CORPUS_IMAGE_CACHE is not None:
        return _CORPUS_IMAGE_CACHE

    try:
        import numpy as np
        import pandas as pd

        embeddings_path = config.INSTAGRAM_DIR / "embeddings.npy"
        meta_path = config.INSTAGRAM_DIR / "embed_meta.parquet"
        if not embeddings_path.is_file() or not meta_path.is_file():
            _CORPUS_IMAGE_CACHE = {}
            return _CORPUS_IMAGE_CACHE

        embeddings = np.load(embeddings_path).astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms == 0, 1.0, norms)
        meta = pd.read_parquet(meta_path)

        names: list[Optional[str]] = []
        for post_id in meta["post_id"]:
            name = None
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                candidate = f"{post_id}{ext}"
                if (config.INSTAGRAM_IMAGES_DIR / candidate).is_file():
                    name = candidate
                    break
            names.append(name)

        captions = (
            meta["caption"].fillna("").astype(str)
            + " "
            + meta["hashtags"].astype(str)
        ).str.lower()
        _CORPUS_IMAGE_CACHE = {
            "embeddings": embeddings,
            "names": names,
            "captions": captions.tolist(),
        }
    except Exception:
        _CORPUS_IMAGE_CACHE = {}
    return _CORPUS_IMAGE_CACHE


def _domain_from_text(text: str) -> Optional[str]:
    food = len(_FOOD_DOMAIN.findall(text))
    fashion = len(_FASHION_DOMAIN.findall(text))
    if food > fashion:
        return "food"
    if fashion > food:
        return "fashion"
    return None


def _clip_text_features(prompts: list[str]) -> Optional[Any]:
    try:
        import torch

        from src.embeddings import load_clip
    except Exception:
        return None

    try:
        model, processor, device = load_clip()
        with torch.no_grad():
            inputs = processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            features = model.get_text_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype("float32")
    except Exception:
        return None


def _corpus_reference_images(
    prompts: list[str], limit: int, query: str = ""
) -> list[str]:
    """Retrieve actual corpus images that visually match the answer prompts."""
    if limit <= 0 or not prompts:
        return []
    index = _load_corpus_image_index()
    if not index:
        return []

    try:
        import numpy as np

        text_features = _clip_text_features(prompts)
        if text_features is None:
            return []

        embeddings = index["embeddings"]
        names = index["names"]
        captions = index["captions"]
        sims = text_features @ embeddings.T

        domain = _domain_from_text(" ".join(prompts) + " " + query)
        domain_re = _FOOD_DOMAIN if domain == "food" else _FASHION_DOMAIN
        mask = np.array(
            [bool(domain_re.search(caption)) for caption in captions],
            dtype=bool,
        )
        if domain is None or int(mask.sum()) < max(limit * 3, 12):
            mask = np.array([name is not None for name in names], dtype=bool)
        else:
            mask &= np.array([name is not None for name in names], dtype=bool)
        if not mask.any():
            return []

        sims = sims.copy()
        sims[:, ~mask] = -np.inf
        all_scores = sims.max(axis=0)

        groups = [
            list(range(start, min(start + 2, len(prompts))))
            for start in range(0, len(prompts), 2)
        ]
        chosen: list[int] = []
        for group in groups:
            if len(chosen) >= limit:
                break
            group_scores = sims[group].max(axis=0)
            group_scores = group_scores.copy()
            if chosen:
                group_scores[chosen] = -np.inf
            index_best = int(np.argmax(group_scores))
            if np.isfinite(group_scores[index_best]):
                chosen.append(index_best)

        while len(chosen) < limit:
            scores = all_scores.copy()
            if chosen:
                scores[chosen] = -np.inf
            index_best = int(np.argmax(scores))
            if not np.isfinite(scores[index_best]):
                break
            chosen.append(index_best)

        return [
            f"/api/instagram-images?name={names[index_best]}"
            for index_best in chosen
            if names[index_best]
        ]
    except Exception:
        return []


def _image_path_from_url(url: str) -> Optional[Any]:
    from urllib.parse import parse_qs, urlparse

    try:
        name = (parse_qs(urlparse(url).query).get("name") or [""])[0]
    except Exception:
        return None
    if not name:
        return None
    if url.startswith("/api/instagram-images"):
        path = config.INSTAGRAM_IMAGES_DIR / name
    else:
        path = config.CHAT_IMAGES_DIR / name
    return path if path.is_file() else None


def _clip_image_scores(prompts: list[str], paths: list[Any]) -> Optional[list[float]]:
    try:
        import torch
        from PIL import Image

        from src.embeddings import load_clip
    except Exception:
        return None

    try:
        text_features = _clip_text_features(prompts)
        if text_features is None:
            return None
        model, processor, device = load_clip()
        with torch.no_grad():
            images = [Image.open(path).convert("RGB") for path in paths]
            image_inputs = processor(images=images, return_tensors="pt")
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            image_features = model.get_image_features(**image_inputs)
            if hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            text_tensor = torch.from_numpy(text_features).to(device)
            scores = (image_features @ text_tensor.T).diagonal()
        return [float(score) for score in scores.cpu().tolist()]
    except Exception:
        return None


def _select_relevant_images(
    urls: list[str], prompts: list[str], limit: int
) -> list[str]:
    """Keep one image per pattern, then the best remaining candidates."""
    if limit <= 0 or not urls:
        return []
    urls = urls[: len(prompts)]
    if len(urls) <= limit:
        return urls

    scores: Optional[list[float]] = None
    relevance = _env("TRENDLENS_CHAT_IMAGE_RELEVANCE", "1").lower()
    paths = [_image_path_from_url(url) for url in urls]
    if (
        relevance not in {"0", "false", "no", "off"}
        and all(path is not None for path in paths)
        and all(path.stat().st_size > 1024 for path in paths)
    ):
        scores = _clip_image_scores(prompts[: len(urls)], paths)
    if scores is None or len(scores) != len(urls):
        return urls[:limit]

    groups = [
        list(range(start, min(start + 2, len(urls))))
        for start in range(0, len(urls), 2)
    ]
    chosen: list[int] = []
    remaining: list[int] = []
    for group in groups:
        best = max(group, key=lambda index: scores[index])
        chosen.append(best)
        remaining.extend(index for index in group if index != best)

    for index in sorted(remaining, key=lambda item: scores[item], reverse=True):
        if len(chosen) >= limit:
            break
        chosen.append(index)

    if len(chosen) > limit:
        chosen = sorted(chosen, key=lambda item: scores[item], reverse=True)[:limit]

    return [urls[index] for index in sorted(chosen)]


DISCLAIMER = (
    "TREND NOTES: This answer is generated from current trend signals and "
    "public web sources. Images are generated interpretations of the described "
    "looks; they are not posts from the TrendLens corpus."
)
REFERENCE_DISCLAIMER = (
    "TREND NOTES: This answer is generated from current trend signals and "
    "public web sources. Representative images illustrate the described "
    "looks and are sourced from public image search and the TrendLens corpus."
)


def _response(
    query: str,
    answer: str,
    *,
    answer_mode: str,
    degraded: bool = False,
    degraded_reason: Optional[str] = None,
    images: Optional[list[str]] = None,
    image_kind: str = "none",
    sources: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    return {
        "query": query,
        "answer": answer,
        "answerMode": answer_mode,
        "degraded": degraded,
        "degradedReason": degraded_reason,
        "inScope": True,
        "scopeReason": None,
        "scopeMethod": "gemini",
        "retrievedClusters": [],
        "supportingImages": images or [],
        "imageKind": image_kind,
        "totalClustersAnalyzed": 0,
        "disclaimer": (
            REFERENCE_DISCLAIMER if image_kind == "reference" else DISCLAIMER
        ),
        "sources": sources or [],
        "grounded": bool(sources),
        "mode": "gemini-direct",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _legacy_fallback(query: str, reason: str) -> Optional[dict[str, Any]]:
    """Answer from the legacy FAISS/retrieval pipeline when the chat model fails."""
    try:
        from src import rag

        result = rag.run_query(query, k=5)
    except Exception:
        return None
    if not isinstance(result, dict) or not result.get("answer"):
        return None

    result = dict(result)
    if result.get("answer"):
        result["answer"] = _strip_styling_advice(result["answer"])
    images = [
        url for url in (result.get("supportingImages") or [])
        if isinstance(url, str) and url
    ]
    if images:
        paths = [_image_path_from_url(url) for url in images]
        if all(path is not None for path in paths):
            text = (result.get("answer") or query)[:600]
            scores = _clip_image_scores([text] * len(paths), paths)
            if scores and len(scores) == len(paths):
                order = sorted(
                    range(len(images)),
                    key=lambda index: scores[index],
                    reverse=True,
                )
                images = [images[index] for index in order][:4]
    result["supportingImages"] = images
    result["imageKind"] = "reference" if images else "none"
    result["degraded"] = True
    result["degradedReason"] = reason
    result["fallback"] = "legacy-faiss"
    result["fallbackReason"] = reason
    result["mode"] = "legacy-fallback"
    return result


def chat_query(query: str) -> dict[str, Any]:
    cfg = gemini_config()
    if not cfg["api_key"]:
        reason = "Trend service API key is not configured"
        fallback = _legacy_fallback(query, reason)
        if fallback is not None:
            return fallback
        return _response(
            query,
            "The Lens is not connected to a trend source yet. Add the API key "
            "to the backend configuration and file the query again.",
            answer_mode="unavailable",
            degraded=True,
            degraded_reason=reason,
        )

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    user_prompt = (
        f"TODAY: {today}\n"
        f"ALLOWED TOPICS: {SCOPE_LABEL}\n"
        "REMINDER: Describe unnamed visual patterns only. Do not use or invent "
        "trend names, aesthetic labels, or title-case headings.\n\n"
        f"USER QUESTION: {query}"
    )

    try:
        text, sources = _call_text(
            cfg, CHAT_SYSTEM_PROMPT, user_prompt, grounding=True
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        fallback = _legacy_fallback(query, reason)
        if fallback is not None:
            return fallback
        return _response(
            query,
            "The Lens could not answer that just now. Check the API key, quota, "
            "and network connection, then file the query again.",
            answer_mode="unavailable",
            degraded=True,
            degraded_reason=reason,
        )

    if not text:
        reason = "empty trend-service response"
        fallback = _legacy_fallback(query, reason)
        if fallback is not None:
            return fallback
        return _response(
            query,
            "The Lens returned an empty answer. Try rephrasing the question.",
            answer_mode="unavailable",
            degraded=True,
            degraded_reason=reason,
        )

    answer, questions, prompts = split_content(text)
    if not answer:
        answer = text
    answer = _strip_why_lines(answer)
    answer = _strip_styling_advice(answer)

    target_images = cfg["image_count"]
    if target_images:
        candidates = _candidate_prompts(
            cfg,
            query,
            answer,
            prompts,
            questions,
            max(6, target_images),
        )
        images = _yandex_reference_images(
            questions,
            candidates,
            target_images,
            _pattern_headings(answer),
        )
        if len(images) < min(3, target_images):
            images = list(
                dict.fromkeys(
                    images
                    + _corpus_reference_images(
                        candidates, target_images, query
                    )
                )
            )[:target_images]
        reference_images = images
        image_kind = "reference" if reference_images else "none"
        if len(images) < min(3, target_images):
            pairs = _generate_image_pairs(cfg, candidates)
            generated = _select_relevant_images(
                [url for _prompt, url in pairs],
                [prompt for prompt, _url in pairs],
                target_images,
            )
            if generated:
                images = list(dict.fromkeys(images + generated))[:target_images]
                image_kind = "reference" if reference_images else "generated"
    else:
        images = []
        image_kind = "none"

    return _response(
        query,
        answer,
        answer_mode="gemini-direct",
        images=images,
        image_kind=image_kind,
        sources=sources,
    )

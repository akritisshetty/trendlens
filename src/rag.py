"""
rag.py
------
RAG-style retrieval + context assembly, grounded ONLY in real pipeline
artifacts.

Instagram data takes priority when available. The query path checks for
Instagram RAG index first, then falls back to the legacy pipeline.

SCOPE
-----
TrendLens tracks emerging visual trends from real social media data.
Questions outside this scope are rejected with a clear message.

INTEGRITY
---------
* Instagram data: real timestamps/engagement from public accounts
* Legacy data: synthetic demo labels (``config.SYNTHETIC_DATA_WARNING``)
* cluster names/descriptions are VLM interpretations, not ground truth
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import config
from src.style_tags import format_style_tags, taxonomy_record

_LIFECYCLE_EMOJI = {"Rising": "📈", "Stable": "📊", "Declining": "📉"}

# ──────────────────────────────────────────────────────────────────────────
# Artifact loading (cached)
# ──────────────────────────────────────────────────────────────────────────
_CACHE: dict[str, Any] = {}


def load_interpretations() -> list[dict[str, Any]]:
    if "interpretations" not in _CACHE:
        _CACHE["interpretations"] = json.loads(
            (config.CLUSTER_METADATA_DIR / "cluster_captions.json").read_text()
        )["interpretations"]
    return _CACHE["interpretations"]


def _total_clusters_safe() -> int:
    """Cluster count for response metadata; 0 when legacy artifacts are
    absent (e.g. Instagram-only deployments) instead of crashing refusals."""
    try:
        return len(load_interpretations())
    except Exception:  # noqa: BLE001 — refusal must never 500 on missing metadata
        return 0


def load_metrics() -> pd.DataFrame:
    if "metrics" not in _CACHE:
        _CACHE["metrics"] = pd.read_csv(
            config.CLUSTER_METADATA_DIR / "trend_metrics.csv"
        ).set_index("cluster_id")
    return _CACHE["metrics"]


def load_representatives() -> dict[int, list[dict[str, Any]]]:
    if "representatives" not in _CACHE:
        raw = json.loads(
            (config.CLUSTER_METADATA_DIR / "representatives.json").read_text()
        )
        _CACHE["representatives"] = {int(k): v for k, v in raw.items()}
    return _CACHE["representatives"]


def _load_retrieval():
    """Lazily load the CLIP model, FAISS index and cluster id mapping."""
    from src import retrieval

    if "retrieval" not in _CACHE:
        model, processor, device = retrieval.load_clip_text()
        embs = np.load(retrieval.TEXT_EMBEDDINGS_PATH, mmap_mode="r")
        meta = json.loads(retrieval.TEXT_CLUSTER_IDS_PATH.read_text())
        cluster_ids = list(meta["cluster_ids"])
        index = retrieval.load_index()
        _CACHE["retrieval"] = {
            "model": model,
            "processor": processor,
            "device": device,
            "index": index,
            "cluster_ids": cluster_ids,
        }
    return _CACHE["retrieval"]


# ──────────────────────────────────────────────────────────────────────────
# Instagram data loading
# ──────────────────────────────────────────────────────────────────────────
_INSTAGRAM_TRENDS_CACHE: Optional[dict[str, Any]] = None
_INSTAGRAM_CHUNKS_CACHE: list[dict[str, Any]] = []
_INSTAGRAM_RAG_MODEL = None
_INSTAGRAM_RAG_INDEX = None


def instagram_image_urls(theme_names: list[str], limit: int = 4) -> list[str]:
    """Representative image URLs for the named Instagram themes.

    Maps theme name -> representative_post_id via trends.json, checks the
    image exists on disk, and returns URLs served by /api/instagram-images.
    """
    trends = load_instagram_trends() or {}
    by_name = {
        t.get("name"): t.get("representative_post_id")
        for t in (trends.get("themes") or [])
    }
    img_dir = config.INSTAGRAM_IMAGES_DIR
    urls: list[str] = []
    for name in theme_names[:limit + 2]:
        pid = by_name.get(name)
        if not pid:
            continue
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            fname = f"{pid}{ext}"
            if (img_dir / fname).is_file():
                urls.append(f"/api/instagram-images?name={fname}")
                break
        if len(urls) >= limit:
            break
    return urls


def load_instagram_trends() -> Optional[dict[str, Any]]:
    """Load Instagram trends JSON (from data_collector.py)."""
    global _INSTAGRAM_TRENDS_CACHE
    if _INSTAGRAM_TRENDS_CACHE is not None:
        return _INSTAGRAM_TRENDS_CACHE
    path = config.INSTAGRAM_TRENDS_PATH
    if not path.exists():
        return None
    try:
        _INSTAGRAM_TRENDS_CACHE = json.loads(path.read_text())
        return _INSTAGRAM_TRENDS_CACHE
    except (OSError, json.JSONDecodeError):
        return None


def _load_instagram_rag_index() -> None:
    """Load or build the Instagram RAG index (sentence-transformer + FAISS)."""
    global _INSTAGRAM_RAG_MODEL, _INSTAGRAM_RAG_INDEX, _INSTAGRAM_CHUNKS_CACHE
    if _INSTAGRAM_RAG_MODEL is not None:
        return

    import faiss
    from sentence_transformers import SentenceTransformer

    chunks_path = config.INSTAGRAM_RAG_CHUNKS_PATH
    index_path = config.INSTAGRAM_RAG_INDEX_PATH
    if not chunks_path.exists() or not index_path.exists():
        return

    _INSTAGRAM_CHUNKS_CACHE = json.loads(chunks_path.read_text())
    if not _INSTAGRAM_CHUNKS_CACHE:
        return

    _INSTAGRAM_RAG_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    _INSTAGRAM_RAG_INDEX = faiss.read_index(str(index_path))


def retrieve_instagram_chunks(query: str, k: int = 5, min_score: float = 0.25) -> list[dict[str, Any]]:
    """Retrieve relevant Instagram cluster chunks via semantic search.

    Returns only chunks whose cosine similarity to the query exceeds
    *min_score*.  When no chunk clears the threshold the list is empty,
    which signals to the caller that Instagram data has no relevant
    match for this query.
    """
    _load_instagram_rag_index()
    if _INSTAGRAM_RAG_INDEX is None or not _INSTAGRAM_CHUNKS_CACHE:
        return []

    q_emb = _INSTAGRAM_RAG_MODEL.encode([query], normalize_embeddings=True)
    scores, indices = _INSTAGRAM_RAG_INDEX.search(
        q_emb.astype("float32"), min(k, len(_INSTAGRAM_CHUNKS_CACHE))
    )
    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        if float(score) < min_score:
            continue
        chunk = dict(_INSTAGRAM_CHUNKS_CACHE[idx])
        chunk["relevance_score"] = round(float(score), 4)
        results.append(chunk)
    return results


def format_instagram_trends_answer(query: str, trends: dict[str, Any]) -> str:
    """Concise, actionable answer from Instagram trend data.

    For each trend: what it is, and how to recreate it visually.
    No engagement metrics, no example captions — just the trend and how-to.
    """
    themes = trends.get("themes") or []
    if not themes:
        return "No trend data available yet. Run the pipeline first."

    themes = sorted(themes, key=lambda t: t.get("emerging_score", 0), reverse=True)

    # Filter by subject if user mentioned one
    subject_match = re.search(
        r"\b(?:about|in|for|of|related to|involving)\s+(.{2,40})", query, re.IGNORECASE
    )
    subject = subject_match.group(1).strip() if subject_match else None
    if subject:
        subj_tokens = set(re.findall(r"[a-z]{3,}", subject.lower()))
        def _matches(t: dict) -> bool:
            kw = {str(x).lower() for x in t.get("keywords", [])}
            name_tokens = set(re.findall(r"[a-z]{3,}", t.get("name", "").lower()))
            return bool(subj_tokens & (kw | name_tokens))
        relevant = [t for t in themes if _matches(t)]
        if relevant:
            themes = relevant

    def _dirs(t: dict, limit: int = 2) -> list[str]:
        out: list[str] = []
        for st in (t.get("style_tags") or [])[:limit + 1]:
            try:
                d = taxonomy_record(st["tag"]).get("direction")
            except (KeyError, TypeError):
                d = None
            if d and d not in out:
                out.append(d)
            if len(out) >= limit:
                break
        return out

    lines: list[str] = []

    # Decision-first opener from the strongest trending look
    lead = _dirs(themes[0], limit=1)
    if lead:
        lines.append(
            f"**What to shoot now:** {lead[0]} — this execution is winning across trending themes."
        )
    else:
        lead_kw = ", ".join(str(x) for x in (themes[0].get("keywords") or [])[:3])
        lines.append(f"**What to shoot now:** {lead_kw or themes[0].get('name', '')} is leading activity.")
    lines.append("")

    for i, t in enumerate(themes[:5], 1):
        name = t.get("name", "this visual style")
        dirs = _dirs(t)
        cue = "; ".join(dirs) if dirs else ", ".join(str(x) for x in (t.get("keywords") or [])[:3])
        cap = (t.get("blip_caption") or "").strip().rstrip(".")
        cap = re.sub(r"^(a|an|the)\s+", "", cap, flags=re.IGNORECASE)
        if cap:
            cue += f" — e.g. {cap}"
        lines.append(f"{i}. **{name}** — {cue}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# Query scope gating
# ──────────────────────────────────────────────────────────────────────────
# Decisive out-of-scope phrase patterns. These flip the decision to OUT no
# matter what the anchor gate says (full-phrase patterns only — a single
# token like "code" must never veto a genuine visual-trend question).
_DECISIVE_OUT_PATTERNS = [
    # programming / software
    r"write a .{0,40}(program|script|function|code|class|method|app)",
    r"(print|output|return|implement).{0,20}hello world",
    r"(python|javascript|java|c\+\+|c#|typescript|golang|rust|ruby|php|html|css|sql|bash|shell|kotlin|swift|dart)\b.{0,40}(code|program|script|function|debug|error|compile)",
    r"(syntax error|compile error|runtime error|debugging|debug this|fix my code|code review|leetcode|algorithm(ic)? problem|data structure|time complexity|regex for|how do i loop)",
    r"(unit test|test case|deploy|api endpoint|rest api|docker|kubernetes|git command|merge conflict)",
    # math
    r"(solve|what is the (value of|answer to|result of)).{0,30}(equation|algebra|integral|derivative|square root|quadratic|matrix|trigonom|calculus|logarithm|fraction|percentage|probability)",
    r"(x \+|x −|x -|x =|solve for x|pythagor|pascal|binomial|eigen)",
    # cooking
    r"(recipe for|ingredients for|how (do i|to) (make|bake|cook|prepare|fry|grill|boil|mix|stir)).{0,30}(cake|bread|pasta|soup|sauce|egg|chicken|rice|curry|pizza|salad|dessert|omelette|pancake|cookie|brownie|biryani|stew|noodle|sandwich|burger|taco|pie|toast|smoothie|muffin|fish|beef|pork|lamb|shrimp|lobster|tofu|dumpling|dosa|idli|samosa|kebab|tandoori|gravy|stock|broth)",
    r"((how long|what temperature).{0,30}(bake|cook|oven))",
    r"what should i (cook|make|eat)(.{0,15}(dinner|lunch|breakfast|tonight|today))?",
    r"(cook|make|bake).{0,12}(dinner|lunch|breakfast|tonight|today)",
    # science
    r"why is the .{1,20}(blue|green|red|wet|hot|cold|round|made of|so (big|small|hot|cold))",
    r"(how does|why do|why does).{0,25}(gravity|magnet|photosynthes|cell|atom|molecule|species evolve|evolution|weather form|rain form|ice form)",
    # news / current events / trivia
    r"(weather|forecast|rain|temperature).{0,20}(today|tomorrow|now|this week)",
    r"(stock market|share price|bitcoin|ethereum|nifty|sensex|crypto price)",
    r"(best|top|should i buy|should i invest in|is it a good idea to buy).{0,25}(crypto|bitcoin|stock|etf|mutual fund|share|bond|real estate)",
    r"(invest|investment).{0,20}(crypto|bitcoin|stock|fund|mutual|etf|real estate|gold)",
    r"(who won|match score|game score|final score|scoreboard|super bowl|world cup|champions league|ipl|nba|nfl).{0,30}(final|today|yesterday|last night|match|game)",
    r"(capital of|currency of|population of|largest (city|country|river|mountain) in)",
    r"(who (is|was) the (president|prime minister|king|queen|director|ceo|author|founder))",
    r"(who invented|who discovered|when was .{0,30} (invented|founded|born|died))",
    # translation
    r"translate .{1,60} (to|into) ",
    # health / legal / finance advice
    r"(dosage|dose of|medicine for|medical advice|symptom|diagnos|prescription|side effect)",
    r"(legal advice|lawyer|contract clause|visa application|immigration|tax return|loan interest|mortgage advice|insurance claim)",
    # academic homework
    r"(homework|exam question|essay on|summarize this article|book report|class assignment|project for school)",
]


def _keyword_scope(query: str) -> Optional[str]:
    """Return a reason string when the query decisively matches an
    out-of-scope pattern, else None."""
    for pat in _DECISIVE_OUT_PATTERNS:
        if re.search(pat, query, re.IGNORECASE):
            # which broad category is it?
            if re.search(r"program|code|python|java|syntax|debug|compile|javascript|html|css|sql|bash|api|docker|git ", query, re.I):
                return "This looks like a programming or software question."
            if re.search(r"equation|algebra|calculus|integral|derivative|square root|solve|matrix|quadratic|trigonom|probability", query, re.I):
                return "This looks like a math question."
            if re.search(r"recipe|bake|oven|ingredients|cook|dough|simmer|fry|grill", query, re.I):
                return "This looks like a cooking/recipe question."
            if re.search(r"weather|forecast|stock market|bitcoin|crypto|score|super bowl|world cup|news", query, re.I):
                return "This looks like a news, sports or markets question."
            if re.search(r"translate|translation", query, re.I):
                return "This looks like a translation request."
            if re.search(r"dosage|medicine|medical|legal|visa|mortgage|loan|tax ", query, re.I):
                return "This looks like a medical, legal or financial advice request."
            if re.search(r"homework|exam|essay on|book report|summarize", query, re.I):
                return "This looks like an academic/homework question."
            return "This is outside the visual-trend domain."
    return None


# Curated anchor questions — the kinds of things TrendLens CAN answer.
# Sentence-style anchors + short visual noun-phrases covering the actual
# cluster themes so short queries like "red flowers" or "moon" pass the gate.
SCOPE_DOMAIN_ANCHORS = [
    "what kind of dog photos are trending on social media",
    "food photography lighting and composition style",
    "fashion model outfit streetwear aesthetic photos",
    "coffee latte art photography",
    "sneakers shoe product photography",
    "interior design home decor photos",
    "nature landscape sunset photography",
    "portrait photography of people",
    "car and vehicle photography",
    "makeup nail art beauty photos",
    "travel photography destinations",
    "architecture building facade photography",
    "product photography commercial style",
    "street photography urban scenes",
    "wedding photography style",
    "pet and animal photography",
    "sky cloud and weather photography",
    "indoor plants garden photography",
    "city nightlife photography",
    "selfie and portrait posing style",
    "flat lay product styling photos",
    "abstract texture and pattern photography",
    "concert and music event photography",
    "fitness and gym workout photography",
    "baby and kids photography",
    "motorcycle bicycle photography",
    "aesthetic bedroom decor photos",
    "office desk setup tech gadgets photography",
    "what are the best photo styles for instagram",
    "engagement photos and social media visual style",
    # short visual noun phrases (match the 29 discovered cluster themes +
    # common photography subjects)
    "flower photography", "red flowers", "roses", "tulips", "plants",
    "trees", "leaves", "forest", "garden", "botanical",
    "moon", "night sky", "stars", "aurora", "rainbow", "clouds", "sky",
    "sunset", "sunrise", "sea", "ocean", "waves", "beach", "waterfall",
    "river", "lake", "mountain", "snow", "desert", "sand", "island",
    "coffee", "tea", "food photography", "fruit", "vegetables",
    "restaurant food", "pizza", "burger", "sushi", "dessert", "chocolate",
    "cake photography", "baking", "breakfast", "brunch",
    "cat", "cat photos", "dog", "puppy", "kitten", "bird", "fish", "horse",
    "tiger", "lion", "elephant", "rabbit", "butterfly", "bee", "duck",
    "penguin", "squirrel", "bear", "pet portraits",
    "sneakers", "shoes", "boots", "handbag", "watch", "jewelry", "glasses",
    "hat", "dress", "wedding dress", "suit", "jeans", "shirt", "jacket",
    "sunglasses", "fashion", "outfit", "streetwear", "lookbook",
    "portrait", "selfie", "face", "baby", "woman", "man", "couple",
    "family", "people", "wedding photography", "bride",
    "car", "truck", "motorcycle", "bicycle", "train", "airplane", "boat",
    "building", "house", "bridge", "city", "street", "skyline",
    "skyscraper", "castle", "church", "architecture",
    "laptop", "phone", "gadgets", "keyboard", "desk setup", "computer",
    "headphones", "camera", "tech product photography",
    "makeup", "nail art", "lipstick", "hair", "beauty", "skincare",
    "perfume", "mani", "cosmetics",
    "abstract", "texture", "pattern", "geometry", "colorful", "neon",
    "graffiti", "art", "painting", "sculpture", "digital art",
    "gym", "fitness", "yoga", "sports", "skateboard", "running", "dancing",
    "concert", "music", "guitar", "dj", "festival",
    "bedroom", "living room", "kitchen", "bathroom", "furniture", "sofa",
    "lamp", "candle", "vase", "pottery", "ceramic", "rug", "pillows",
    "interior", "minimalist", "cozy", "aesthetic",
    "camping", "hiking", "outdoor adventure", "wildlife", "safari",
    "dinosaur", "toys", "dolls", "plush", "lego",
]

# Out-of-scope anchor questions — the kinds of things TrendLens CANNOT answer.
SCOPE_OUT_ANCHORS = [
    "write a python program to print hello world",
    "explain the code syntax error and fix it",
    "what is the square root of 144 math problem",
    "solve this algebra equation step by step",
    "give me a recipe for chocolate cake",
    "how long to bake bread at 350 degrees",
    "who won the football world cup final",
    "latest football match scores",
    "stock market news today update",
    "weather forecast for tomorrow",
    "translate this sentence to french",
    "how does a car engine work physics",
    "movie plot summary and ending explained",
    "history of world war two timeline",
    "what medicine dosage should i take",
    "legal advice for a rental contract",
    "latest celebrity gossip news",
    "how to fix a leaky faucet plumbing",
    "give me interview questions for a job",
    "what is the capital of australia",
    "who is the current prime minister",
    "quantum computing explained for beginners",
    "how to lose weight fast diet plan",
    "best laptop to buy in 2026",
    "how to get a visa for canada",
    "what is the currency of japan",
    "how does photosynthesis work",
    "who invented the telephone",
    "what is 2 plus 2",
    "cure for headache",
]

# Anchor-gate thresholds (calibrated empirically on 32 probe queries):
#   in-scope  iff  domain_score >= _SCOPE_DOMAIN_MIN
#               and (domain_score - out_score) >= _SCOPE_MARGIN_MIN
SCOPE_DOMAIN_MIN = 0.22
SCOPE_MARGIN_MIN = -0.01

SCOPE_DOMAIN_EMB_PATH = config.EMBEDDINGS_DIR / "scope_domain_anchors.npy"
SCOPE_OUT_EMB_PATH = config.EMBEDDINGS_DIR / "scope_out_anchors.npy"
SCOPE_ANCHOR_META_PATH = config.EMBEDDINGS_DIR / "scope_anchors.json"


def _embed_anchors(anchors: list[str], path: Path, tag: str) -> np.ndarray:
    """Embed + cache an anchor set keyed on its exact text."""
    if path.exists() and SCOPE_ANCHOR_META_PATH.exists():
        meta = json.loads(SCOPE_ANCHOR_META_PATH.read_text())
        if meta.get(tag) == anchors:
            return np.load(path, mmap_mode="r")
    from src import retrieval

    rt = _load_retrieval()
    emb = retrieval.embed_texts(
        rt["model"], rt["processor"], anchors, device=rt["device"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    try:
        meta = json.loads(SCOPE_ANCHOR_META_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        meta = {}
    meta[tag] = anchors
    SCOPE_ANCHOR_META_PATH.write_text(json.dumps(meta))
    return emb


def _anchor_scores(query: str) -> tuple[float, float]:
    from src import retrieval

    rt = _load_retrieval()
    qe = retrieval.embed_texts(
        rt["model"], rt["processor"], [query], device=rt["device"]
    )[0]
    d_emb = _embed_anchors(
        SCOPE_DOMAIN_ANCHORS, SCOPE_DOMAIN_EMB_PATH, "domain"
    )
    o_emb = _embed_anchors(SCOPE_OUT_ANCHORS, SCOPE_OUT_EMB_PATH, "out")
    domain = float((qe @ d_emb.T).max())
    out = float((qe @ o_emb.T).max())
    return domain, out


def classify_scope(query: str) -> dict[str, Any]:
    """
    Decide whether ``query`` is a social-media visual-trend question.

    Returns:
      in_scope  bool
      method    "keywords" | "anchors" | "fallback"
      reason    str | None  (for out-of-scope refusals)
      domain_score / out_score (when anchor method used)
    """
    kw_reason = _keyword_scope(query)
    if kw_reason is not None:
        return {
            "in_scope": False,
            "method": "keywords",
            "reason": kw_reason,
            "domain_score": None,
            "out_score": None,
        }

    try:
        domain, out = _anchor_scores(query)
    except Exception:  # noqa: BLE001 — model/index unavailable: don't block
        return {
            "in_scope": True,
            "method": "fallback",
            "reason": None,
            "domain_score": None,
            "out_score": None,
        }

    in_scope = domain >= SCOPE_DOMAIN_MIN and (domain - out) >= SCOPE_MARGIN_MIN
    return {
        "in_scope": in_scope,
        "method": "anchors",
        "reason": None if in_scope else (
            "The query is not close enough to any social-media visual-trend "
            "topic (and is closer to general-purpose topics)."
        ),
        "domain_score": round(float(domain), 4),
        "out_score": round(float(out), 4),
    }


# ──────────────────────────────────────────────────────────────────────────
# Honest cluster record
# ──────────────────────────────────────────────────────────────────────────
def _image_url(image_path: Optional[str]) -> Optional[str]:
    if not image_path:
        return None
    from urllib.parse import quote

    return "/api/images?path=" + quote(str(image_path), safe="")


def _cluster_record(cluster_id: int, similarity: float, rank: int) -> dict[str, Any]:
    interpretations = {int(i["cluster_id"]): i for i in load_interpretations()}
    reps = load_representatives()
    it = interpretations.get(cluster_id, {})

    m = load_metrics().loc[cluster_id] if cluster_id in load_metrics().index else None
    rep_path = (reps.get(cluster_id) or [{}])[0].get("image_path")
    rec = {
        "id": f"cluster-{cluster_id}",
        "cluster_id": cluster_id,
        "rank": rank,
        "similarity_score": round(float(similarity), 4),
        "name": it.get("name", f"Cluster {cluster_id}"),
        "description": it.get("description", ""),
        "blip_caption": (it.get("sample_captions") or [""])[0],
        "characteristics": it.get("characteristics", []),
        "interpretation_confidence": it.get("confidence", None),
        "lifecycle": (m["lifecycle"] if m is not None and "lifecycle" in m else None),
        "n_posts": int(m["n_posts"]) if m is not None else None,
        "average_engagement": float(m["average_engagement"]) if m is not None else None,
        "recent_growth": float(m["recent_growth"]) if m is not None else None,
        "trend_score": float(m["trend_score_growth_size_stability"])
        if m is not None else None,
        "text_trend_score": float(m["text_trend_score"]) if m is not None else None,
        "representative_image": rep_path,
        "representative_image_url": _image_url(rep_path),
        # NOT MEASURED fields — omitted as null, never invented.
        "viral_rate": None,
        "geographic_hotspots": [],
        "keywords": [],
    }
    return rec


# ──────────────────────────────────────────────────────────────────────────
# Context assembly
# ──────────────────────────────────────────────────────────────────────────
def build_context(query: str, k: int = 5) -> dict[str, Any]:
    """
    Embed the query, retrieve top-k clusters via the Phase 6 FAISS index,
    and assemble an honest context for the response.
    """
    from src import retrieval

    rt = _load_retrieval()
    q_emb = retrieval.embed_texts(
        rt["model"], rt["processor"], [query], device=rt["device"]
    )[0]
    dists, idxs = retrieval.query_index(rt["index"], q_emb, k=k)
    cluster_ids = rt["cluster_ids"]

    records = [
        _cluster_record(
            cluster_ids[int(idx)], float(dist), rank=i + 1
        )
        for i, (dist, idx) in enumerate(zip(dists[0], idxs[0]))
    ]

    return {
        "query": query,
        "k": k,
        "disclaimer": config.SYNTHETIC_DATA_WARNING,
        "total_clusters_analyzed": len(load_interpretations()),
        "dataset": "CLIP clustering + BLIP interpretation of 5,000 sampled "
        "images (69,226 available); neutral synthetic timestamps/engagement",
        "retrieved_clusters": records,
    }


# ──────────────────────────────────────────────────────────────────────────
# RAG Knowledge Base — text chunks for retrieval-augmented generation
# ──────────────────────────────────────────────────────────────────────────
def _build_text_chunks() -> list[dict[str, Any]]:
    """Build text chunks from cluster metadata for RAG retrieval.

    Each chunk is a textual description of a visual trend cluster, containing
    the cluster name, description, visual characteristics, and representative
    caption. Pipeline internals (cluster IDs, engagement metrics, lifecycle
    labels) are excluded from the text used for retrieval.
    """
    interpretations = load_interpretations()
    chunks: list[dict[str, Any]] = []
    for it in interpretations:
        cid = int(it["cluster_id"])
        name = it.get("name", f"Cluster {cid}")
        desc = it.get("description", "")
        chars = it.get("characteristics", [])
        caps = it.get("sample_captions", [])

        parts = [f"Visual trend: \"{name}\"."]
        if desc:
            parts.append(desc)
        if chars:
            parts.append(f"Key visual elements: {', '.join(str(x) for x in chars[:8])}.")
        if caps:
            parts.append(f'Representative image caption: "{caps[0]}".')

        chunks.append({
            "cluster_id": cid,
            "text": " ".join(parts),
            "name": name,
        })
    return chunks


_RAG_TEXT_MODEL = None
_RAG_TEXT_INDEX = None
_RAG_TEXT_CHUNKS: list[dict[str, Any]] = []


def _load_rag_index() -> None:
    """Load or build the text RAG index (sentence-transformer + FAISS)."""
    global _RAG_TEXT_MODEL, _RAG_TEXT_INDEX, _RAG_TEXT_CHUNKS
    if _RAG_TEXT_MODEL is not None:
        return

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    _RAG_TEXT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    _RAG_TEXT_CHUNKS = _build_text_chunks()

    if not _RAG_TEXT_CHUNKS:
        return

    texts = [c["text"] for c in _RAG_TEXT_CHUNKS]
    embeddings = _RAG_TEXT_MODEL.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    dim = embeddings.shape[1]
    _RAG_TEXT_INDEX = faiss.IndexFlatIP(dim)
    _RAG_TEXT_INDEX.add(embeddings.astype("float32"))


def retrieve_text_chunks(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Retrieve the most relevant text chunks for a query via semantic search.

    Returns the top-k text chunks ranked by cosine similarity to the query.
    Each chunk contains the text, cluster_id, and name.
    """
    _load_rag_index()
    if _RAG_TEXT_INDEX is None or not _RAG_TEXT_CHUNKS:
        return []

    q_emb = _RAG_TEXT_MODEL.encode([query], normalize_embeddings=True)
    scores, indices = _RAG_TEXT_INDEX.search(q_emb.astype("float32"), min(k, len(_RAG_TEXT_CHUNKS)))
    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = dict(_RAG_TEXT_CHUNKS[idx])
        chunk["relevance_score"] = round(float(score), 4)
        results.append(chunk)
    return results


# ──────────────────────────────────────────────────────────────────────────
# Answer formatting (no LLM)
# ──────────────────────────────────────────────────────────────────────────
def _fmt_rate(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.2f}"


def _fmt_int(v: Optional[int]) -> str:
    return "—" if v is None else f"{v:,}"


def _lifecycle_str(c: dict[str, Any]) -> str:
    if not c["lifecycle"]:
        return "n/a"
    return f"{_LIFECYCLE_EMOJI.get(c['lifecycle'], '')} {c['lifecycle']} *(demo label — neutral synthetic timestamps, noise-level)*"


# ──────────────────────────────────────────────────────────────────────────
# Photography how-to guide mode
#
# When the query asks *how to shoot / frame / style a subject for
# engagement* ("post a picture of X — what should it look like?"), we
# answer with a step-by-step photo guide instead of a cluster listing.
# Every piece of advice is derived ONLY from the retrieved clusters'
# real keywords (BLIP characteristics) and captions, plus measured
# engagement/trend stats — nothing is invented.
# ──────────────────────────────────────────────────────────────────────────
_ADVICE_TRIGGERS = [
    r"post(ing)? a (picture|photo|pic|image)",
    r"(take|taking|click|clicking|shoot|shooting|make|making|frame|framing|compose|composing).{0,20}(a |the )?(picture|photo|pic|shot|image)",
    r"how (should|do|can|would) i (take|shoot|make|frame|compose|click|style)",
    r"what should the (visual|photo|picture|shot|image|look)",
    r"(for|to get|that gets|get|gets|with|maximis[ez]e) (max|maximum|more|the most|highest|better|best) (engagement|likes|reach|virality|viral|impact)",
    r"(background|composition|colou?rs?|lighting|angle|aesthetic).{0,30}(photo|picture|shot|image|look)",
]

_VISUAL_TOKEN = re.compile(
    r"\b(pictures?|photos?|pics?|images?|visuals?|shot|shots|shoot|click|frame|compose|photograph(ic|y)?)\b",
    re.IGNORECASE,
)


def _is_advice_intent(query: str) -> bool:
    if not _VISUAL_TOKEN.search(query):
        return False
    return any(re.search(p, query, re.IGNORECASE) for p in _ADVICE_TRIGGERS)


def _advice_subject(query: str, top_cluster: dict[str, Any]) -> str:
    """Extract the subject the user wants to photograph, else the cluster name."""
    m = re.search(
        r"\b(?:picture|photo|pic|image|shot) of (.{1,120})", query, re.IGNORECASE
    )
    if m:
        subject = m.group(1).strip()
        subject = re.split(
            r"\b(what should|what does|for max|to get|so that|and how|please| on | for | in | with | to | if |$)",
            subject,
            flags=re.IGNORECASE,
        )[0]
        subject = re.split(r"[?.!]", subject)[0].strip(" ,;:-")
        if subject:
            return subject
    return top_cluster.get("name") or "your subject"


def _feats(cluster: dict[str, Any]) -> list[str]:
    return [str(x) for x in cluster.get("characteristics", [])][:6]


def format_advice_answer(query: str, context: dict[str, Any]) -> str:
    clusters = context["retrieved_clusters"]
    if not clusters:
        return format_answer(query, context)

    top = clusters[0]
    subject = _advice_subject(query, top)
    feats = _feats(top)
    caption = top.get("blip_caption") or ""

    lines: list[str] = []
    lines.append(f"## 📸 How to shoot \"{subject}\"")
    lines.append(f"Closest real look in the index: **{top['name']}**")
    lines.append("")
    lines.append(
        "Every direction below comes **only from the keywords, captions and "
        "measured stats of these real clusters** — no invented styling advice."
    )
    lines.append("")

    # Execution guidance from measured style tags first, then keywords
    shoot: list[str] = []
    for st in top.get("style_tags") or []:
        try:
            d = taxonomy_record(st["tag"]).get("direction")
        except (KeyError, TypeError):
            d = None
        if d and d not in shoot:
            shoot.append(d)
        if len(shoot) >= 3:
            break
    if not shoot and feats:
        shoot = [f"make `{f}` the anchor of the frame" for f in feats[:3]]
    for s in shoot:
        lines.append(f"- {s}")
    if caption and not shoot:
        lines.append(f"- Reference shot: \"{caption}\"")
    lines.append("")

    if len(clusters) > 1:
        best_eng = max(clusters, key=lambda c: c.get("average_engagement") or -1)
        if best_eng["cluster_id"] == top["cluster_id"]:
            lines.append(
                "This look already has the **highest measured engagement** "
                "among the retrieved matches."
            )
        else:
            shared = set(feats) & set(_feats(best_eng))
            if shared:
                lines.append(
                    f"The \"{best_eng['name']}\" look has the highest measured engagement among the "
                    f"matches and shares `{', '.join(sorted(shared))}` — borrowing those elements is the "
                    "only engagement edge the data can point to."
                )
            else:
                lines.append(
                    f"The highest-engagement look is \"{best_eng['name']}\", but its keywords "
                    f"(`{', '.join(_feats(best_eng)[:3])}`) do **not** overlap your subject — copying it would "
                    "change the subject, not improve your shot. No engagement advantage to borrow."
                )
        lines.append("")

    lines.append("_Engagement figures are synthetic demo data; names/captions are VLM interpretations._")
    return "\n".join(lines)


def format_answer(query: str, context: dict[str, Any]) -> str:
    clusters = context["retrieved_clusters"]
    if not clusters:
        return "\n".join([
            "⚠️ **No matching clusters found**",
            "",
            "The TrendLens index could not find relevant clusters for this query.",
            "Try a visual/category keyword (e.g., coffee, dogs, sneakers, sky).",
        ])

    lines: list[str] = []
    lines.append(f"Visual themes closest to **\"{query}\"**:")
    lines.append("")

    for c in clusters[:6]:
        detail = ", ".join(str(x) for x in (c.get("characteristics") or [])[:4])
        if not detail:
            detail = c.get("blip_caption") or c.get("description") or ""
        lines.append(f"- **{c['name']}** — {detail}")

    return "\n".join(lines)


def _refusal_answer(query: str, scope: dict[str, Any]) -> str:
    reason = scope.get("reason") or (
        "The query is not about a visual trend."
    )
    return "\n".join([
        "I track emerging visual trends from real social media data — photography styles, engagement patterns, and early signals.",
        "",
        f"Your question doesn't fit that scope: *{query}*",
        "",
        f"**Reason:** {reason}",
        "",
        "**Try asking about:**",
        "- What photography aesthetics are rising?",
        "- Which visual styles get the most engagement?",
        "- What content patterns are emerging in [niche]?",
    ])


# ──────────────────────────────────────────────────────────────────────────
# Live trends (real Reddit data) — "what's trending right now?"
# ──────────────────────────────────────────────────────────────────────────
_LIVE_TREND_PATTERNS = [
    r"trending",
    r"what.{0,12}(is|are|'?s|'?re)\s*(trend|trending|hot|popular)",
    r"(what|which|top)\s.{0,15}(trends|aesthetic|trend)",
    r"(hot|popular|going viral|blowing up).{0,25}(right now|this week|today|currently|these days)",
    r"(right now|this week|currently|these days).{0,20}(trend|popular|hot)",
]


def _live_trend_intent(query: str) -> bool:
    return any(re.search(p, query, re.IGNORECASE) for p in _LIVE_TREND_PATTERNS)


def load_live_trends() -> Optional[dict[str, Any]]:
    path = config.LIVE_TRENDS_PATH
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# Aesthetic cues transfer across subjects (vs object-specific words like
# "coffee"/"laptop" that only fit their own scene). Used to pick which live
# themes are relevant to the user's subject.
_AESTHETIC_CUES = {
    "warm", "inviting", "bright", "minimal", "clean", "simple", "open",
    "soft", "cozy", "light", "natural", "rustic", "fresh", "vibrant",
    "colorful", "moody", "dark", "elegant", "modern", "airy", "organic",
    "neutral", "textured", "layered",
}

_STOPWORDS = {
    "picture", "photo", "pic", "image", "shot", "post", "with", "and",
    "for", "the", "my", "of", "your", "that", "this",
}


def _subject_tokens(subject: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z]{3,}", subject.lower())
        if w not in _STOPWORDS
    }


_NAME_NOISE = {"visual", "theme"}

# Concrete photo direction per real detected keyword. Every direction is
# derived from a keyword that actually appears in the theme, so the advice is
# grounded in the detected data — never invented styling.
_PHOTO_DIRECTIONS = {
    "warm": "Shoot in warm, natural-toned light",
    "inviting": "Style an inviting, welcoming composition",
    "open": "Keep the frame open — give the subject room to breathe",
    "bright": "Use bright, airy lighting",
    "soft": "Use soft, diffused light",
    "minimal": "Keep it minimal — one clear focal point",
    "clean": "Use a clean, uncluttered background",
    "natural": "Shoot in natural light",
    "rustic": "Add rustic textures — wood, linen, stone",
    "fresh": "Use fresh, colorful accents for a pop of color",
    "vibrant": "Push vibrant, saturated color",
    "moody": "Go moody — low light, deep shadows",
    "simple": "Keep the styling simple and focused",
    "cozy": "Build a cozy, warm scene",
    "dark": "Use a dark background to make the subject pop",
    "elegant": "Plate and frame elegantly — refined, deliberate",
    "modern": "Use a clean, modern aesthetic",
    "neutral": "Work in neutral tones — beige, cream, grey",
    "textured": "Add texture — linen, wood, natural grain",
    "layered": "Layer elements in the frame for depth",
    "scattered": "Scatter small props or ingredients around the plate",
    "features": "Include supporting elements around the main subject",
    "setup": "Style a full scene — not a bare subject",
    "table": "Shoot on a styled table surface",
}


def _photo_directions(keywords: list[Any], limit: int = 4) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for k in keywords:
        d = _PHOTO_DIRECTIONS.get(str(k).lower())
        if d and d not in seen:
            seen.add(d)
            out.append(d)
        if len(out) >= limit:
            break
    return out


def format_live_trends_answer(query: str, context: dict[str, Any]) -> str:
    """Concise, tailored answer from the real detected themes.

    When the query names a subject (e.g. "a pasta"), only themes relevant to
    that subject are shown — themes whose *name* has an unrelated object word
    (e.g. "coffee") are skipped. Each theme renders as a concrete "what to do"
    list derived from its real keywords. Provenance stays available as the API
    response `disclaimer` field.
    """
    trends = context.get("live_trends") or {}
    themes = trends.get("themes") or []
    if not themes:
        return format_answer(query, context)

    subject = _advice_subject(query, themes[0])
    has_subject = bool(
        re.search(r"\b(?:picture|photo|pic|image|shot) of (.{1,120})", query, re.IGNORECASE)
    )
    subj = re.sub(r"^(?:a|an|the)\s+", "", subject, flags=re.IGNORECASE).strip()
    subj_toks = _subject_tokens(subject) if has_subject else set()

    def kw_set(t: dict[str, Any]) -> set[str]:
        return {str(x).lower() for x in (t.get("keywords") or [])}

    def score(t: dict[str, Any]) -> int:
        kw = kw_set(t)
        return 2 * len(subj_toks & kw) + len(kw & _AESTHETIC_CUES)

    def scene_specific(t: dict[str, Any]) -> bool:
        # Theme name has a non-aesthetic object word (e.g. "coffee") that is
        # NOT the user's subject -> a scene-specific look, not transferable.
        name_toks = set(re.findall(r"[a-z]{3,}", (t.get("name") or "").lower()))
        return bool(name_toks - _AESTHETIC_CUES - subj_toks - _NAME_NOISE)

    scored = sorted(themes, key=score, reverse=True)
    if has_subject:
        # Keep only themes with some relevance that aren't scene-specific;
        # if none qualify, keep the closest look rather than nothing.
        relevant = [t for t in scored if score(t) > 0 and not scene_specific(t)]
        scored = relevant or scored[:1]

    lines: list[str] = []
    lines.append("## 🔥 Trending right now")
    if has_subject:
        lines.append(f"For {subj}, the trending look to borrow is:")
    lines.append("")

    for i, t in enumerate(scored[:5], 1):
        growth = t.get("growth_rate")
        if t.get("prior_posts", 0) == 0 and t.get("recent_posts", 0) > 0:
            growth_s = "brand new this window"
        elif growth is not None:
            growth_s = f"{'+' if growth >= 0 else ''}{growth * 100:.0f}% vs prior window"
        else:
            growth_s = ""
        head = f"{i}. **{t.get('name', 'theme')}**"
        tail = [
            s for s in (
                growth_s,
                f"{t.get('recent_posts')} recent posts"
                if t.get("recent_posts") is not None else "",
            ) if s
        ]
        if tail:
            head += " — " + " · ".join(tail)
        lines.append(head)

        dirs = _photo_directions(t.get("keywords") or [], limit=2)
        if dirs:
            lines.append("")
            lines.append("   What to do:")
            for d in dirs:
                lines.append(f"   - {d}")
        lines.append("")

    if has_subject and not any(
        subj_toks & {str(x).lower() for x in (t.get("keywords") or [])}
        for t in themes
    ):
        lines.append(
            f"_No {subj}-specific trend in the live feed yet "
            f"— the look above is the closest trending aesthetic._"
        )

    return "\n".join(lines).rstrip() + "\n"


# ──────────────────────────────────────────────────────────────────────────
# Query driver
# ──────────────────────────────────────────────────────────────────────────
_WANTS_IMAGES = re.compile(
    r"(show|see|display|visuali[sz]e|give me|include|with).{0,20}"
    r"(image|photo|pic|picture|shot|visual|representative|example)",
    re.IGNORECASE,
)


def _wants_images(query: str) -> bool:
    """Check if the user explicitly asks to see images."""
    return bool(_WANTS_IMAGES.search(query))


def run_query(query: str, k: int = 5) -> dict[str, Any]:
    scope = classify_scope(query)

    # ── Instagram data path (primary) ──
    ig_trends = load_instagram_trends()
    ig_chunks = retrieve_instagram_chunks(query, k=k) if ig_trends else []

    if ig_trends and (_live_trend_intent(query) or ig_chunks):
        # Instagram data is available — use it as the primary source
        answer = format_instagram_trends_answer(query, ig_trends)
        ig_themes = ig_trends.get("themes", [])
        ig_retrieved = [
            {
                "name": c.get("name", ""),
                "keywords": c.get("keywords", []),
                "style_tags": c.get("style_tags", []),
                "growth_rate": c.get("growth_rate"),
                "emerging_score": c.get("emerging_score", 0),
                "relevance_score": c.get("relevance_score", 0),
                "avg_likes": c.get("avg_likes", 0),
                "avg_comments": c.get("avg_comments", 0),
                "avg_views": c.get("avg_views", 0),
                "content_types": c.get("content_types", {}),
                "top_hashtags": c.get("top_hashtags", []),
            }
            for c in ig_chunks
        ]

        # Try LLM polish for natural language output
        answer_mode = "rule-based"
        try:
            from src import llm

            # Build a context bundle for the LLM from Instagram data
            ig_context = {
                "query": query,
                "total_clusters_analyzed": len(ig_themes),
                "dataset": "instagram",
                "disclaimer": ig_trends.get("disclaimer", config.INSTAGRAM_DATA_WARNING),
                "retrieved_clusters": [
                    {
                        "rank": i + 1,
                        "name": c.get("name", ""),
                        "description": c.get("blip_caption", ""),
                        "blip_caption": c.get("blip_caption", ""),
                        "characteristics": c.get("keywords", []),
                        "style_tags": c.get("style_tags", []),
                        "interpretation_confidence": None,
                    }
                    for i, c in enumerate(ig_chunks)
                ],
            }
            polished = llm.format_answer_with_llm(query, ig_context)
            if polished:
                answer, answer_mode = polished, f"llm-{llm.llm_config().get('provider')}"
        except Exception:  # noqa: BLE001 — fall back to rule-based, never break
            pass

        # Representative images for the retrieved themes (top themes as
        # fallback when nothing cleared the retrieval threshold)
        theme_names = [c.get("name") for c in ig_chunks if c.get("name")] or [
            t.get("name")
            for t in sorted(
                ig_trends.get("themes") or [],
                key=lambda t: t.get("emerging_score", 0),
                reverse=True,
            )[:4]
        ]

        return {
            "query": query,
            "answer": answer,
            "answerMode": answer_mode,
            "inScope": True,
            "scopeReason": None,
            "scopeMethod": "instagram-data",
            "retrievedClusters": ig_retrieved,
            "supportingImages": instagram_image_urls(theme_names),
            "totalClustersAnalyzed": len(ig_themes),
            "disclaimer": ig_trends.get("disclaimer", config.INSTAGRAM_DATA_WARNING),
            "sources": ["instagram"],
            "mode": "instagram-rag",
            "timestamp": pd.Timestamp.now("UTC").isoformat(),
        }

    # ── Legacy pipeline path (fallback) ──
    if not scope["in_scope"]:
        return {
            "query": query,
            "answer": _refusal_answer(query, scope),
            "answerMode": "rule-based",
            "inScope": False,
            "scopeReason": scope.get("reason"),
            "scopeMethod": scope.get("method"),
            "retrievedClusters": [],
            "supportingImages": [],
            "totalClustersAnalyzed": _total_clusters_safe(),
            "disclaimer": config.SYNTHETIC_DATA_WARNING,
            "sources": [],
            "mode": "faiss-only",
            "timestamp": pd.Timestamp.now("UTC").isoformat(),
        }

    context = build_context(query, k=k)
    if _live_trend_intent(query):
        live = load_live_trends()
        if live:
            context["live_trends"] = live
    records = context["retrieved_clusters"]
    images = [
        c["representative_image_url"]
        for c in records[:6]
        if c.get("representative_image_url")
    ]
    rule_answer = (
        format_advice_answer(query, context)
        if _is_advice_intent(query)
        else format_answer(query, context)
    )

    # Optional LLM writing layer (opt-in via env). It only rewrites the
    # retrieved evidence into fluent prose — never a knowledge source.
    answer, answer_mode = rule_answer, "rule-based"
    live_override = False
    if "live_trends" in context and context["live_trends"]:
        answer = format_live_trends_answer(query, context)
        live_override = True
    if not live_override:
        try:
            from src import llm

            polished = llm.format_answer_with_llm(query, context)
            if polished:
                answer, answer_mode = polished, f"llm-{llm.llm_config().get('provider')}"
        except Exception:  # noqa: BLE001 — fall back to rule-based, never break
            pass

    return {
        "query": query,
        "answer": answer,
        "answerMode": answer_mode,
        "inScope": True,
        "scopeReason": None,
        "scopeMethod": scope.get("method"),
        "retrievedClusters": records,
        "supportingImages": images if _wants_images(query) else [],
        "totalClustersAnalyzed": context["total_clusters_analyzed"],
        "disclaimer": context["disclaimer"],
        "sources": [],
        "mode": "faiss-only",
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "a cup of coffee"
    import json as _json

    print(_json.dumps(run_query(q), indent=1)[:4000])

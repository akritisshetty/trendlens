"""
style_tags.py
-------------
CLIP zero-shot *photography-style* tagging.

PROBLEM THIS SOLVES
-------------------
BLIP captions describe the SUBJECT of an image ("a wooden table with a
brunch spread") but not the EXECUTION — lighting, camera angle, depth of
field, color grading, process storytelling. When users ask "what food
photography styles are trending?", subject-only evidence makes answers
degenerate into a list of what people are shooting instead of how.

APPROACH
--------
Zero-shot scoring against a curated bank of style text prompts using the
SAME CLIP model as the image embeddings (one shared space). Because image
embeddings are already computed and stored by the pipeline, style scoring
is just a matrix product:

    style_scores = image_embeddings @ style_text_embeddings.T

No extra model passes, no image re-processing. Per-tag scores are the
mean cosine similarity over each tag's prompt ensemble; per-cluster style
profiles are the mean score over member images.

INTEGRITY
---------
* Scores are CLIP similarities (0..1 cosine) — model interpretations,
  not ground-truth labels. They are stored alongside BLIP captions and
  carry the same epistemic status.
* ``direction`` strings are fixed definitions attached to each tag —
  they describe what the tag MEANS photographically; they are never
  generated per cluster and never invented from data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

import config

# Bump when STYLE_TAXONOMY changes so cached prompt embeddings rebuild.
STYLE_VERSION = 1

STYLE_EMB_PATH = config.EMBEDDINGS_DIR / "style_prompt_embeddings.npy"
STYLE_META_PATH = config.EMBEDDINGS_DIR / "style_prompts_meta.json"

# Cosine similarities below this are CLIP noise for style prompts.
MIN_STYLE_SCORE = 0.18

# ──────────────────────────────────────────────────────────────────────────
# Style taxonomy — the "how it is shot" axis
# ──────────────────────────────────────────────────────────────────────────
STYLE_TAXONOMY: list[dict[str, Any]] = [
    # ── Framing / shot type ──
    {
        "tag": "tactile macro close-up",
        "aspect": "framing",
        "prompts": [
            "an extreme macro close-up photo showing fine food texture detail",
            "a close-up photo where surface texture fills the whole frame",
        ],
        "direction": "get close — fill the frame with texture via a macro-style crop",
    },
    {
        "tag": "top-down flat lay",
        "aspect": "framing",
        "prompts": [
            "a flat lay photo taken directly from above",
            "a top-down overhead shot of items arranged on a flat surface",
        ],
        "direction": "shoot straight down as a flat lay",
    },
    {
        "tag": "forty-five degree table angle",
        "aspect": "framing",
        "prompts": [
            "a photo taken from a forty five degree angle looking down at the table",
            "a three-quarter overhead diner's angle photo of food on a table",
        ],
        "direction": "shoot from a 45-degree diner's-eye angle",
    },
    {
        "tag": "eye-level straight-on",
        "aspect": "framing",
        "prompts": [
            "a straight-on eye level photo across the table",
            "a photo at eye level with the subject facing the camera head-on",
        ],
        "direction": "shoot straight-on at eye level",
    },
    # ── Lighting ──
    {
        "tag": "natural window light",
        "aspect": "lighting",
        "prompts": [
            "a photo lit by soft daylight coming from a window",
            "a naturally side-lit photo with gentle soft shadows",
        ],
        "direction": "use soft natural window light from the side",
    },
    {
        "tag": "harsh direct flash",
        "aspect": "lighting",
        "prompts": [
            "a photo taken with harsh direct on-camera flash",
            "a flash-lit snapshot with bright hotspots falling off into darkness",
        ],
        "direction": "use harsh direct flash for a raw snapshot look",
    },
    {
        "tag": "dark moody low-key",
        "aspect": "lighting",
        "prompts": [
            "a dark moody low-key photo with deep shadows",
            "a dramatically shadowed photo against a dark background",
        ],
        "direction": "go low-key — dark background, deep shadows",
    },
    {
        "tag": "bright airy high-key",
        "aspect": "lighting",
        "prompts": [
            "a bright airy high-key photo full of even light",
            "a bright white minimal photo lit evenly with no harsh shadows",
        ],
        "direction": "keep it bright and airy with even high-key light",
    },
    # ── Color / mood grading ──
    {
        "tag": "warm cozy amber tones",
        "aspect": "color mood",
        "prompts": [
            "a warm amber toned photo with golden color grading",
            "a warm nostalgic cozy toned photo in golden hues",
        ],
        "direction": "grade warm — amber golden tones for cozy nostalgia",
    },
    {
        "tag": "muted desaturated palette",
        "aspect": "color mood",
        "prompts": [
            "a muted desaturated photo with faded washed-out colors",
            "a soft neutral beige-toned photo with low saturation",
        ],
        "direction": "desaturate — muted faded neutrals, low contrast",
    },
    {
        "tag": "vibrant saturated colors",
        "aspect": "color mood",
        "prompts": [
            "a vibrant saturated colorful photo",
            "a bold vivid high-saturation photo that pops",
        ],
        "direction": "push vivid saturated color",
    },
    # ── Process / storytelling ──
    {
        "tag": "hands-in-frame action",
        "aspect": "process",
        "prompts": [
            "a photo of hands holding or preparing food",
            "hands in frame performing an action while cooking or eating",
        ],
        "direction": "put hands in the frame doing something",
    },
    {
        "tag": "messy in-progress making",
        "aspect": "process",
        "prompts": [
            "a messy cooking preparation scene still in progress",
            "an untidy behind-the-scenes making-of scene mid-process",
        ],
        "direction": "show the messy middle — in-progress unstyled moments",
    },
    # ── Composition ──
    {
        "tag": "minimal negative space",
        "aspect": "composition",
        "prompts": [
            "a minimalist photo with large empty negative space",
            "a sparse composition with one subject surrounded by clean empty space",
        ],
        "direction": "compose minimal — one hero subject, generous negative space",
    },
    {
        "tag": "abundant crowded spread",
        "aspect": "composition",
        "prompts": [
            "a table crowded with many dishes and food items",
            "an abundant overflowing spread filling the whole frame",
        ],
        "direction": "fill the frame with an abundant spread",
    },
]

TAGS = [s["tag"] for s in STYLE_TAXONOMY]
_TAG_INDEX = {s["tag"]: i for i, s in enumerate(STYLE_TAXONOMY)}

# Prompt list flattened in taxonomy order (tag i owns prompts[i_offsets]).
ALL_PROMPTS: list[str] = [p for s in STYLE_TAXONOMY for p in s["prompts"]]
_PROMPT_TAG_POS: list[int] = [
    ti for ti, s in enumerate(STYLE_TAXONOMY) for _ in s["prompts"]
]


def taxonomy_record(tag: str) -> dict[str, Any]:
    """Return the full taxonomy entry (tag, aspect, direction, prompts)."""
    return dict(STYLE_TAXONOMY[_TAG_INDEX[tag]])


# ──────────────────────────────────────────────────────────────────────────
# Pure math (no torch / model needed — unit-testable)
# ──────────────────────────────────────────────────────────────────────────
def scores_from_prompt_sims(prompt_sims: np.ndarray) -> np.ndarray:
    """
    Collapse (N, P) per-prompt cosine similarities into (N, T) per-tag
    scores by averaging each tag's prompt ensemble.
    """
    sims = np.asarray(prompt_sims, dtype="float32")
    if sims.ndim != 2 or sims.shape[1] != len(ALL_PROMPTS):
        raise ValueError(
            f"expected (N, {len(ALL_PROMPTS)}) prompt similarities, "
            f"got {sims.shape}"
        )
    n_tags = len(STYLE_TAXONOMY)
    out = np.zeros((sims.shape[0], n_tags), dtype="float32")
    pos = np.asarray(_PROMPT_TAG_POS)
    for ti in range(n_tags):
        mask = pos == ti
        out[:, ti] = sims[:, mask].mean(axis=1)
    return out


def aggregate_styles(
    style_scores: np.ndarray,
    indices: Optional[list[int]] = None,
    top_k: int = 3,
    min_score: float = MIN_STYLE_SCORE,
) -> list[dict[str, Any]]:
    """
    Mean style profile over rows of ``style_scores`` (optionally restricted
    to cluster member ``indices``), returned as ranked tags:

        [{"tag", "aspect", "score"}]  (descending by mean score)

    Tags below ``min_score`` are dropped — a weak signal must never be
    presented as a cluster's style.
    """
    scores = np.asarray(style_scores, dtype="float32")
    if scores.ndim == 1:
        scores = scores[None, :]
    if indices is not None:
        if len(indices) == 0:
            return []
        scores = scores[list(indices)]
    mean = scores.mean(axis=0)

    order = np.argsort(-mean)
    out: list[dict[str, Any]] = []
    for ti in order[:top_k]:
        s = float(mean[int(ti)])
        if s < min_score:
            break
        rec = STYLE_TAXONOMY[int(ti)]
        out.append({
            "tag": rec["tag"],
            "aspect": rec["aspect"],
            "score": round(s, 4),
        })
    return out


def format_style_tags(style_tags: list[Any], limit: int = 3) -> str:
    """Render style tags as a compact human phrase for answers/chunks."""
    names: list[str] = []
    for st in style_tags or []:
        tag = st.get("tag") if isinstance(st, dict) else str(st)
        if tag:
            names.append(str(tag))
        if len(names) >= limit:
            break
    return ", ".join(names)


# ──────────────────────────────────────────────────────────────────────────
# Model-dependent scoring
# ──────────────────────────────────────────────────────────────────────────
def _style_text_embeddings_cached() -> Optional[np.ndarray]:
    if not STYLE_EMB_PATH.exists() or not STYLE_META_PATH.exists():
        return None
    try:
        meta = json.loads(STYLE_META_PATH.read_text())
        if meta.get("version") != STYLE_VERSION or meta.get("prompts") != ALL_PROMPTS:
            return None
        emb = np.load(STYLE_EMB_PATH)
        if emb.shape != (len(ALL_PROMPTS), meta["dim"]):
            return None
        return emb.astype("float32")
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def style_text_embeddings(
    model=None,
    processor=None,
    device: str | None = None,
) -> tuple[np.ndarray, bool]:
    """
    L2-normalized CLIP text embeddings for every style prompt.
    Cached on disk; rebuilt when the taxonomy version changes.

    Returns (embeddings (P, D), from_cache).
    Loads the shared CLIP model via src.retrieval when not supplied.
    """
    cached = _style_text_embeddings_cached()
    if cached is not None:
        return cached, True

    from src import retrieval

    if model is None or processor is None:
        model, processor, device = retrieval.load_clip_text()
    embs = retrieval.embed_texts(model, processor, ALL_PROMPTS, device=device)
    embs = np.asarray(embs, dtype="float32")
    STYLE_EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(STYLE_EMB_PATH, embs)
    STYLE_META_PATH.write_text(json.dumps({
        "version": STYLE_VERSION,
        "model": "openai/clip-vit-base-patch32",
        "prompts": ALL_PROMPTS,
        "dim": int(embs.shape[1]),
    }))
    return embs, False


def compute_style_scores(
    embeddings: np.ndarray,
    text_embs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Zero-shot style scores for already-computed CLIP image embeddings.

    embeddings : (N, D) float32, L2-normalized (pipeline artifacts)
    text_embs  : optional precomputed style prompt embeddings (P, D)

    Returns (N, T) float32 — one column per taxonomy tag.
    """
    img = np.ascontiguousarray(np.asarray(embeddings, dtype="float32"))
    if img.ndim != 2:
        raise ValueError(f"image embeddings must be 2-D, got {img.ndim}D")
    if text_embs is None:
        text_embs, _ = style_text_embeddings()
    sims = img @ np.asarray(text_embs, dtype="float32").T
    return scores_from_prompt_sims(sims)


def summarize_style_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[int, list[dict[str, Any]]]:
    """Per-cluster style profiles for all clustered rows at once."""
    scores = compute_style_scores(embeddings)
    out: dict[int, list[dict[str, Any]]] = {}
    lbl = np.asarray(labels)
    for cid in sorted(set(int(l) for l in lbl.tolist()) - {-1}):
        idx = [int(i) for i in np.flatnonzero(lbl == cid)]
        out[cid] = aggregate_styles(scores, indices=idx)
    return out

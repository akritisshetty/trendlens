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
from typing import Any, Optional

import numpy as np

import config

# Bump when STYLE_TAXONOMY changes so cached prompt embeddings rebuild.
STYLE_VERSION = 1

STYLE_EMB_PATH = config.EMBEDDINGS_DIR / "style_prompt_embeddings.npy"
STYLE_META_PATH = config.EMBEDDINGS_DIR / "style_prompts_meta.json"

# Reporting threshold, applied to the 0-1 scale produced by
# normalized_style_scores. A tag must explain at least 55% of the maximum
# available style signal for this batch before it is presented as a cluster's
# style. The previous value (0.18) was an absolute CLIP cosine cut-off, which
# every tag cleared on every image and so filtered nothing.
MIN_STYLE_SCORE = 0.55

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

# ──────────────────────────────────────────────────────────────────────────
# Shot recipes
#
# `direction` (above) is a one-line label used in retrieval text. `steps` is the
# actionable form: concrete, checkable things a photographer can do. These are
# AUTHORED IN CODE, deliberately, not generated by the LLM writing layer.
#
# This matters for integrity: the LLM prompt forbids the model from inventing
# photography advice, because advice it invents is untraceable to the images. By
# keeping the tag -> instruction mapping here, every step the user is given
# traces back to a style tag that was actually MEASURED on the cluster's images
# by CLIP zero-shot scoring. The model selects and sequences; it does not invent.
#
# Each step is written to be observable in the final frame, so a reader can
# check whether the shot actually matches the advice.
# ──────────────────────────────────────────────────────────────────────────
SHOT_STEPS: dict[str, list[str]] = {
    # framing
    "tactile macro close-up": [
        "Move in until one texture (crumbs, foam, grain) fills the whole frame",
        "Crop tighter than feels natural — the subject should read as surface, not object",
        "Use the closest focus distance your lens allows and confirm the focal plane is sharp",
    ],
    "top-down flat lay": [
        "Mount the camera directly overhead so the lens axis is vertical to the surface",
        "Arrange items in a deliberate grid and leave even margins around the outside",
        "Shoot on a flat surface and check that no item's shadow breaks the frame edge",
    ],
    "forty-five degree table angle": [
        "Raise the camera to roughly 45 degrees above the table, not straight down",
        "Tilt the lens down toward the near edge of the surface",
        "Keep the table plane running across the lower third for depth",
    ],
    "eye-level straight-on": [
        "Put the lens at the subject's own eye height",
        "Shoot perpendicular to the face or front face of the subject",
        "Keep both eyes (or the front plane) in focus and parallel to the sensor",
    ],
    # lighting
    "natural window light": [
        "Place the subject beside a window with the light falling across it from the side",
        "Face the shadow side toward an open wall or reflector so the dark side keeps detail",
        "Shoot in the middle of the day for even brightness, or early for long soft shadows",
    ],
    "harsh direct flash": [
        "Fire the flash head-on at the subject, close to the lens axis",
        "Expect and keep hard-edged shadows falling straight back behind the subject",
        "Drop the ambient exposure so the flash hotspot reads as the brightest point",
    ],
    "dark moody low-key": [
        "Remove the subject from the background so the backdrop goes fully dark",
        "Light only one side and let the other fall into shadow",
        "Expose for the highlights and let the shadow areas go near-black",
    ],
    "bright airy high-key": [
        "Use a white or near-white surface as both background and fill",
        "Light evenly from the front with no hard shadow edges anywhere",
        "Lift the exposure until the whites sit just below clipping",
    ],
    # color mood
    "warm cozy amber tones": [
        "Warm the white balance toward amber and let skin tones stay natural",
        "Increase saturation in the yellows and reds, pull back the blues",
        "Keep the overall exposure warm rather than raising brightness",
    ],
    "muted desaturated palette": [
        "Drop overall saturation and pull contrast down",
        "Shift the palette toward beige, cream and grey",
        "Let blacks sit slightly lifted and soft, not crushed",
    ],
    "vibrant saturated colors": [
        "Raise vibrance and saturation together, checking that no channel clips",
        "Separate the subject from the background by hue as well as brightness",
        "Keep contrast high enough that the colors read as intentional, not oversaturated",
    ],
    # process
    "hands-in-frame action": [
        "Include the hands mid-action, not posed beside the food",
        "Catch the action at its peak — pouring, tearing, placing, stirring",
        "Make sure the hands read clearly and are not cropped by the frame edge",
    ],
    "messy in-progress making": [
        "Shoot mid-process with tools and scraps left in frame",
        "Keep the scene unstaged — visible flour, spills and half-work read as authentic",
        "Frame loosely so the mess has room around it",
    ],
    # composition
    "minimal negative space": [
        "Reduce the subject to a single hero element",
        "Leave generous empty space on at least one side of the frame",
        "Check that the empty area is a deliberate shape, not awkward leftover",
    ],
    "abundant crowded spread": [
        "Add enough items that the surface runs out of frame",
        "Overlap items so the spread reads as abundant rather than arranged",
        "Vary height so the composition has depth instead of one flat plane",
    ],
}

#: Order in which aspects should appear in a shot recipe, matching the order a
#: photographer actually works: light, then frame, then arrange, then process.
ASPECT_ORDER = ["lighting", "framing", "color mood", "composition", "process"]


def steps_for_tag(tag: str) -> list[str]:
    """Concrete shooting steps for a style tag (empty if unauthored)."""
    return list(SHOT_STEPS.get(tag, []))


def build_shot_recipe(
    style_tags: list[dict[str, Any]],
    max_tags: int = 4,
) -> list[dict[str, Any]]:
    """
    Turn a cluster's measured style tags into an ordered, deduplicated recipe.

    Only tags that exist in ``SHOT_STEPS`` contribute steps, so a recipe can
    never contain an instruction without a measured tag behind it. Steps are
    ordered by ``ASPECT_ORDER`` rather than by raw score: a recipe that opens
    with colour grading before telling you where to put the light is not
    followable.
    """
    picked = [t for t in (style_tags or []) if t.get("tag") in SHOT_STEPS][:max_tags]
    by_aspect: dict[str, dict[str, Any]] = {}
    for t in picked:
        aspect = t.get("aspect", "")
        if aspect not in by_aspect:
            by_aspect[aspect] = t

    recipe: list[dict[str, Any]] = []
    seen: set[str] = set()
    for aspect in ASPECT_ORDER:
        t = by_aspect.get(aspect)
        if t is None:
            continue
        steps = [s for s in SHOT_STEPS[t["tag"]] if s not in seen]
        if not steps:
            continue
        seen.update(steps)
        recipe.append(
            {
                "aspect": aspect,
                "tag": t["tag"],
                "score": t.get("score"),
                "steps": steps,
            }
        )
    # Any measured tag outside ASPECT_ORDER still gets included, at the end.
    for aspect, t in by_aspect.items():
        if aspect in ASPECT_ORDER:
            continue
        steps = [s for s in SHOT_STEPS[t["tag"]] if s not in seen]
        if steps:
            seen.update(steps)
            recipe.append(
                {"aspect": aspect, "tag": t["tag"], "score": t.get("score"), "steps": steps}
            )
    return recipe


def format_shot_recipe(recipe: list[dict[str, Any]]) -> str:
    """Render a shot recipe as compact text for RAG chunks / answers."""
    if not recipe:
        return ""
    parts = []
    for item in recipe:
        head = item["tag"].capitalize()
        parts.append(f"{head}: " + "; ".join(item["steps"]))
    return " | ".join(parts)


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
    Collapse (N, P) per-prompt cosine similarities into (N, T) per-tag scores.

    Mean-centering is essential here, not cosmetic. Raw CLIP image-text cosine
    similarities occupy a narrow band (roughly 0.15-0.30) because of the
    modality gap: every image is more similar to every style caption than to
    nothing, and the absolute level says more about CLIP's embedding geometry
    than about the photograph. Measured on the real corpus, raw scoring gave a
    global per-tag mean spanning only 0.1715-0.2082 across all 15 tags, with
    within-cluster variance exceeding between-cluster variance. The top-3 tags
    were near-ties for every cluster (e.g. 0.2203 / 0.2197 / 0.2143) and the
    same two tags — "eye-level straight-on" and "natural window light" — won on
    8 of 12 clusters. That is the tagger describing Instagram photographs in
    general, not the cluster in front of it.

    Subtracting each image's own mean across tags removes the modality gap, so a
    score means "how much more this tag fits this image than the average style
    description fits it". That is the quantity a style tag is supposed to
    express, and it is what makes clusters separable.

    Returns raw centred margins; use ``normalized_style_scores`` for the
    0-1 values used in reporting.
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
    return out - out.mean(axis=1, keepdims=True)


def normalized_style_scores(centered: np.ndarray) -> np.ndarray:
    """
    Map centred style margins onto a 0-1 scale for reporting.

    Divides by the maximum centred margin observed in the batch so the strongest
    tag for an image sits at 1.0. Purely a presentation transform: ranking is
    unaffected, but it lets ``MIN_STYLE_SCORE`` be expressed as "fraction of
    the strongest available signal" instead of an absolute cosine value whose
    meaning depends on the model.
    """
    scores = np.asarray(centered, dtype="float32")
    scale = float(np.abs(scores).max())
    if scale < 1e-9:
        return np.full_like(scores, 0.5)
    return np.clip(0.5 + 0.5 * (scores / scale), 0.0, 1.0)


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
        # The cache lives in the SAME vector space as the image embeddings it is
        # multiplied against, so it is only valid for the encoder that produced
        # it. Checking version+prompts alone let a stale cache survive an
        # encoder swap: the stored "dim" was compared against itself, so the
        # shape check passed and the mismatched vectors were used.
        if meta.get("model") != config.CLIP_MODEL:
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
        "model": config.CLIP_MODEL,
        "prompts": ALL_PROMPTS,
        "dim": int(embs.shape[1]),
    }))
    return embs, False


def compute_style_scores(
    embeddings: np.ndarray,
    text_embs: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Zero-shot style scores for already-computed CLIP image embeddings.

    embeddings : (N, D) float32, L2-normalized (pipeline artifacts)
    text_embs  : optional precomputed style prompt embeddings (P, D)
    normalize  : return the 0-1 presentation scale (default) rather than raw
                 centred margins

    Returns (N, T) float32 — one column per taxonomy tag.
    """
    img = np.ascontiguousarray(np.asarray(embeddings, dtype="float32"))
    if img.ndim != 2:
        raise ValueError(f"image embeddings must be 2-D, got {img.ndim}D")
    if text_embs is None:
        text_embs, _ = style_text_embeddings()
    sims = img @ np.asarray(text_embs, dtype="float32").T
    centered = scores_from_prompt_sims(sims)
    return normalized_style_scores(centered) if normalize else centered


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

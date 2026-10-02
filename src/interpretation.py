"""
interpretation.py
-----------------
Cluster interpretation (Stage 8).

Pipeline:
  representative images (Phase 3)
      -> VLM captioning (BLIP by default; API option pluggable)
      -> deterministic aggregation -> name, description, characteristics,
         confidence

INTEGRITY NOTE
--------------
Every generated name/description/characteristic is an INTERPRETATION
produced by a vision-language model + statistical aggregation. It is NOT
ground truth about the cluster's meaning. All outputs are labelled as such
in ``cluster_captions.json`` and the markdown report.

The VLM is injected/callable so the pipeline can run on Colab with BLIP-2
(GPU) or BLIP-1 (CPU), or swap to an API model without changing the code.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

import config

#: Single source of truth is ``config.BLIP_MODEL`` (see the model-selection
#: section there for how this checkpoint was chosen).
DEFAULT_BLIP_MODEL = config.BLIP_MODEL

_STOP = {
    "a", "an", "the", "this", "that", "these", "those", "is", "are", "was",
    "were", "be", "been", "being", "of", "in", "on", "at", "to", "for",
    "with", "from", "by", "and", "or", "but", "not", "no", "as", "it", "its",
    "his", "her", "their", "our", "your", "my", "you", "he", "she", "they",
    "we", "i", "there", "here", "has", "have", "had", "do", "does", "did",
    "will", "would", "can", "could", "should", "may", "might", "all", "some",
    "very", "just", "one", "two", "more", "most", "up", "down", "out", "off",
    "over", "under", "so", "also", "when", "while", "then", "than", "photo",
    "photograph", "picture", "image", "images", "pictures", "photographs",
    "showing", "shown", "taken", "man", "woman", "people", "group", "person",
    "lot", "lot", "many", "number", "view", "views",
}

_WORD_RE = re.compile(r"[a-z']+")


# ──────────────────────────────────────────────────────────────────────────
# VLM captioning
# ──────────────────────────────────────────────────────────────────────────
def load_blip(model_name: str = DEFAULT_BLIP_MODEL, device: str | None = None):
    """Load BLIP processor + model on the best available device."""
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    from src.embeddings import detect_device

    device = device or detect_device()
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, processor, device


def caption_image(
    model,
    processor,
    image,
    device: str | None = None,
    max_length: int = 40,
) -> str:
    """
    Caption one already-loaded PIL image. Returns the caption text.
    """
    import torch

    if device is None:
        device = next(model.parameters()).device.type
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model.generate(**inputs, max_new_tokens=max_length)
    return processor.decode(out[0], skip_special_tokens=True).strip()


def caption_representatives(
    model,
    processor,
    reps: dict[int, list[dict[str, Any]]],
    image_root: Path | None = None,
    k: int = 4,
    device: str | None = None,
    show_progress: bool = True,
    caption_fn: Callable | None = None,
) -> dict[int, list[str]]:
    """
    Caption up to ``k`` representative images per cluster.

    Returns {cluster_id: [caption, ...]} — captions whose image failed to
    load are skipped, so per-cluster lists may be shorter than k.

    ``caption_fn`` (default ``caption_image``) is injectable for tests.
    """
    from src import preprocessing as pp
    from tqdm.auto import tqdm

    caption_fn = caption_fn or caption_image
    image_root = image_root or config.IMAGE_ROOT
    results: dict[int, list[str]] = {}
    for c, rows in tqdm(reps.items(), desc="Captioning", disable=not show_progress):
        caps: list[str] = []
        for r in rows[:k]:
            try:
                img = pp.load_image(r["image_path"], image_root, resize=config.IMAGE_RESIZE)
                caps.append(caption_fn(model, processor, img, device=device))
            except (OSError, ValueError):
                continue
        results[int(c)] = caps
    return results


# ──────────────────────────────────────────────────────────────────────────
# Deterministic aggregation
# ──────────────────────────────────────────────────────────────────────────
def _keywords(captions: Iterable[str]) -> tuple[Counter, Counter]:
    uni: Counter = Counter()
    bi: Counter = Counter()
    for cap in captions:
        words = [w for w in _WORD_RE.findall(cap.lower()) if w not in _STOP and len(w) >= 3]
        uni.update(words)
        bi.update(" ".join(b) for b in zip(words[:-1], words[1:]) if b)
    return uni, bi


def interpret_cluster(
    captions: list[str],
    cluster_id: int,
    top_k: int = 6,
) -> dict[str, Any]:
    """
    Aggregate a cluster's captions into an interpretation.

    Returns:
      cluster_id     int
      name           top descriptive bigram (or top keyword fallback)
      description    most-repeated caption, else keyword-based sentence
      characteristics top keywords (unigrams)
      confidence     entropy-based agreement across captions (0..1)
      sample_captions raw VLM output (internal evidence)
    """
    captions = [c for c in captions if c]
    uni, bi = _keywords(captions)

    if not uni:
        return {
            "cluster_id": int(cluster_id),
            "name": f"Cluster {int(cluster_id)} (uninterpreted)",
            "description": "No interpretable caption could be generated.",
            "characteristics": [],
            "confidence": 0.0,
            "sample_captions": captions,
        }

    bigrams_top = [g for g, _ in bi.most_common(3)]
    keywords = [w for w, _ in uni.most_common(top_k)]
    name = bigrams_top[0] if bigrams_top else keywords[0]

    counter = Counter(captions)
    desc, n_desc = counter.most_common(1)[0]
    if n_desc <= 1:
        desc = "A visual cluster whose images are described by: " + ", ".join(keywords)

    # agreement: 1 - normalized entropy over unigrams
    total = sum(uni.values())
    probs = np.asarray(list(uni.values()), dtype="float64") / total
    entropy = float(-(probs * np.log(probs)).sum())
    max_entropy = float(np.log(len(uni)))
    agreement = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0

    return {
        "cluster_id": int(cluster_id),
        "name": name,
        "description": desc,
        "characteristics": keywords,
        "confidence": round(float(np.clip(agreement, 0.0, 1.0)), 4),
        "sample_captions": captions,
    }


def interpret_all_clusters(
    captions_by_cluster: dict[int, list[str]],
    top_k: int = 6,
) -> list[dict[str, Any]]:
    """Interpret every cluster; sorted by cluster_id."""
    return [
        interpret_cluster(caps, c, top_k=top_k)
        for c, caps in sorted(captions_by_cluster.items())
    ]


# ──────────────────────────────────────────────────────────────────────────
# Persistence + report
# ──────────────────────────────────────────────────────────────────────────
def save_interpretations(
    interpretations: list[dict[str, Any]],
    path: Path | None = None,
    extra: dict | None = None,
) -> Path:
    path = Path(path) if path else config.CLUSTER_METADATA_DIR / "cluster_captions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "disclaimer": config.SYNTHETIC_DATA_WARNING,
        "note": "Names/descriptions/characteristics are VLM-derived "
        "INTERPRETATIONS, not ground truth.",
        "interpretations": interpretations,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=1))
    return path


def write_report(
    interpretations: list[dict[str, Any]],
    path: Path | None = None,
) -> Path:
    path = Path(path) if path else config.CLUSTER_METADATA_DIR / "captions_report.md"
    lines = [
        "# TrendLens — Cluster Interpretation Report",
        "",
        "> " + config.SYNTHETIC_DATA_WARNING,
        "> **Interpretations, not ground truth.** Names/descriptions are "
        "VLM output + statistical aggregation.",
        "",
    ]
    for it in interpretations:
        lines += [
            f"## Cluster {it['cluster_id']} — {it['name']}",
            f"- **confidence:** {it['confidence']}",
            f"- **description:** {it['description']}",
            "- **characteristics:** " + ", ".join(it["characteristics"]),
            "- **sample captions:** " + " | ".join(it["sample_captions"][:4]),
            "",
        ]
    path.write_text("\n".join(lines))
    return path


# ──────────────────────────────────────────────────────────────────────────
# CLI driver
# ──────────────────────────────────────────────────────────────────────────
def run_pipeline(
    k_reps: int = 4,
    top_k: int = 6,
    model_name: str = DEFAULT_BLIP_MODEL,
    reps_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Phase 5 driver: load representatives (Phase 3), caption them with BLIP,
    interpret each cluster, save JSON + markdown report.
    """
    reps_path = Path(reps_path) if reps_path else (
        config.CLUSTER_METADATA_DIR / "representatives.json"
    )
    reps = {int(k): v for k, v in json.loads(reps_path.read_text()).items()}
    print(f"Clusters to interpret: {len(reps)} (k_reps={k_reps})")

    model, processor, device = load_blip(model_name)
    print(f"BLIP on {device}")
    caps = caption_representatives(model, processor, reps, k=k_reps, device=device)
    interpreted = interpret_all_clusters(caps, top_k=top_k)

    save_interpretations(interpreted)
    write_report(interpreted)

    print(f"\nInterpreted {len(interpreted)} clusters.")
    for it in interpreted[:5]:
        print(f"  c{it['cluster_id']:>3} {it['name']:<28} conf={it['confidence']}")
    return interpreted


if __name__ == "__main__":
    run_pipeline()

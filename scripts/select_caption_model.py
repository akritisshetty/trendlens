#!/usr/bin/env python3
"""
select_caption_model.py
-----------------------
Choose the VLM captioner by measured caption quality on real corpus images.

Why this is measured at all
---------------------------
``blip-image-captioning-base`` was chosen because it fits on a CPU. BLIP's
larger checkpoint produces noticeably better captions, but "noticeably better"
is an impression, and the pipeline's cluster NAMES are derived from these
captions, so caption quality is user-visible output rather than an internal
detail. It is measured here instead of assumed.

Metrics
-------
clipscore     cosine similarity between the image embedding and its own
              caption's text embedding under the same CLIP encoder production
              uses. A standard automatic caption metric, and it is the only one
              here that measures whether the caption describes THIS image rather
              than images in general.
content_words content words per caption after stopwording. Captions drive the
              cluster name and keywords, so an empty or one-word caption is
              wasted output. Specificity, not length, is the target.
distinct2     distinct bigrams / total bigrams across the sample. Catches
              degenerate captioners that emit the same phrasing for every
              image, which would make all clusters look alike in the UI.
degenerate    fraction of captions with an immediately repeated token. BLIP
              is known to do this ("...topped with a sp of dil dil dil"); the
              magnitude is worth knowing before it reaches a user.

Run:  venv/bin/python scripts/select_caption_model.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.interpretation import _STOP  # noqa: E402

OUT_CSV = ROOT / "caption_model_selection_results.csv"
OUT_JSON = ROOT / "caption_model_selection_examples.json"

DEFAULT_CANDIDATES = [
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large",
]

N_IMAGES = 60
_WORD_RE = re.compile(r"[a-z']+")


def sample_images(n: int, seed: int = 42) -> list[tuple[str, "object"]]:
    """Deterministic sample of local corpus images: (post_id, PIL image)."""
    from PIL import Image

    meta = pd.read_parquet(config.INSTAGRAM_DIR / "embed_meta.parquet")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(meta))[:n]
    out = []
    for i in idx:
        pid = str(meta["post_id"].iloc[int(i)])
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = config.INSTAGRAM_IMAGES_DIR / f"{pid}{ext}"
            if p.is_file():
                try:
                    out.append((pid, Image.open(p).convert("RGB")))
                except OSError:
                    pass
                break
    return out


def _clipscore_fn(clip_model: str):
    """Return fn(images, captions) -> list[float] using one fixed CLIP encoder."""
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(clip_model)
    model = AutoModel.from_pretrained(clip_model)
    model.eval()

    def score(images, captions):
        with torch.no_grad():
            i_in = processor(images=images, return_tensors="pt")
            i_out = model.get_image_features(**i_in)
            if not hasattr(i_out, "shape"):
                i_out = i_out.pooler_output
            i_emb = torch.nn.functional.normalize(i_out.float(), dim=-1)

            t_in = processor(
                text=list(captions), return_tensors="pt", padding=True, truncation=True
            )
            t_out = model.get_text_features(**t_in)
            if not hasattr(t_out, "shape"):
                t_out = t_out.pooler_output
            t_emb = torch.nn.functional.normalize(t_out.float(), dim=-1)

        return (i_emb * t_emb).sum(dim=-1).cpu().numpy().tolist()

    return score


def caption_metrics(captions: list[str], scores: list[float]) -> dict:
    content, bigrams, degenerate = [], set(), 0
    for cap in captions:
        words = [w for w in _WORD_RE.findall(cap.lower()) if w not in _STOP and len(w) > 2]
        content.append(len(words))
        bigrams.update(zip(words[:-1], words[1:]))
        toks = cap.split()
        if any(a.lower() == b.lower() for a, b in zip(toks[:-1], toks[1:])):
            degenerate += 1
    total_bigrams = max(1, sum(max(0, c - 1) for c in content))
    return {
        "clipscore": float(np.mean(scores)) if scores else float("nan"),
        "content_words": float(np.mean(content)) if content else float("nan"),
        "distinct2": len(bigrams) / total_bigrams,
        "degenerate_frac": degenerate / max(1, len(captions)),
        "empty_frac": sum(1 for c in captions if not c.strip()) / max(1, len(captions)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="*", default=DEFAULT_CANDIDATES)
    ap.add_argument("--n-images", type=int, default=N_IMAGES)
    ap.add_argument("--clip-model", default=getattr(config, "CLIP_MODEL", None))
    args = ap.parse_args()

    from src.interpretation import caption_image, load_blip

    clip_model = args.clip_model
    if not clip_model:
        raise SystemExit("config.CLIP_MODEL is not set; pass --clip-model explicitly")

    images = sample_images(args.n_images)
    print(f"{len(images)} corpus images; CLIPScore judged by {clip_model}\n")

    score_fn = _clipscore_fn(clip_model)
    rows: list[dict] = []
    examples: dict[str, list[str]] = {}

    for name in args.candidates:
        print(f"--- {name}")
        t0 = time.perf_counter()
        try:
            model, processor, device = load_blip(name)
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED to load: {type(exc).__name__}: {str(exc)[:80]}")
            continue

        caps, kept = [], []
        for pid, img in images:
            try:
                c = caption_image(model, processor, img, device=device)
            except Exception:  # noqa: BLE001 — one bad image must not kill the run
                continue
            caps.append(c)
            kept.append(img)
        seconds = time.perf_counter() - t0

        scores = score_fn(kept, caps) if caps else []
        m = caption_metrics(caps, scores)
        rows.append(
            {
                "model": name,
                "n_captions": len(caps),
                "seconds": round(seconds, 1),
                "img_per_s": round(len(caps) / max(seconds, 1e-9), 2),
                **m,
            }
        )
        examples[name] = caps[:12]
        print(
            f"    clipscore={m['clipscore']:.4f}  content_words={m['content_words']:.2f}"
            f"  distinct2={m['distinct2']:.3f}  degenerate={m['degenerate_frac']:.3f}"
            f"  {len(caps)/max(seconds,1e-9):.2f} img/s"
        )
        for c in caps[:3]:
            print(f"      > {c}")
        del model

    if rows:
        df = pd.DataFrame(rows).sort_values("clipscore", ascending=False)
        df.to_csv(OUT_CSV, index=False)
        OUT_JSON.write_text(json.dumps(examples, indent=1))
        print(f"\nBest by CLIPScore: {df.iloc[0]['model']}")
        print(f"Saved -> {OUT_CSV}\nExamples -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""
rebuild_embeddings.py
---------------------
Rebuild the Instagram CLIP image embeddings from scratch for EVERY post in
``all_posts.parquet`` that has a local image file.

This fixes data-staleness: previously ``embeddings.npy`` only covered a
fraction of the downloaded images (e.g. 152 of ~456), which starved the
cluster / trend pipeline (few clusters -> weak statistics).

Outputs (same filenames the rest of TrendLens reads):
  data/instagram/embeddings.npy      (N, 512) float32, L2-normalised, row-aligned
  data/instagram/embed_meta.parquet  (N) rows matching all_posts columns

Run:  venv/bin/python scripts/rebuild_embeddings.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.embeddings import load_clip, l2_normalize


def _local_image_path(post_id: str, image_url: str) -> Path:
    ext = Path(image_url.split("?")[0]).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    return config.INSTAGRAM_IMAGES_DIR / f"{post_id}{ext}"


def main() -> int:
    posts = pd.read_parquet(config.INSTAGRAM_DIR / "all_posts.parquet")

    rows, paths = [], []
    for _, row in posts.iterrows():
        p = _local_image_path(str(row["post_id"]), str(row.get("image_url") or ""))
        if p.is_file():
            rows.append(row)
            paths.append(p)

    print(f"[rebuild] {len(posts)} posts total, {len(paths)} with local images")

    model, processor, device = load_clip()
    import torch

    embs, keep_idx = [], []
    for i, p in enumerate(paths):
        try:
            import PIL.Image as Image

            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model.get_image_features(**inputs)
            if hasattr(out, "pooler_output"):
                out = out.pooler_output
            embs.append(out.detach().cpu().numpy()[0])
            keep_idx.append(i)

    if not embs:
        print("[rebuild] no images embedded!")
        return 1

    emb = l2_normalize(np.vstack(embs).astype("float32"))
    aligned = pd.DataFrame([rows[i] for i in keep_idx]).reset_index(drop=True)

    np.save(config.INSTAGRAM_EMBEDDINGS_PATH, emb)
    aligned.to_parquet(config.INSTAGRAM_DIR / "embed_meta.parquet", index=False)
    print(f"[rebuild] wrote {emb.shape} embeddings + {len(aligned)} metadata rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

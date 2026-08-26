"""
Benchmark CLIP at different input resolutions.

Compares encoding speed and embedding similarity across 128, 224, and 336 px
to confirm the optimal resize dimension for TrendLens.
"""

import glob
import time
import numpy as np
from PIL import Image

SIZES = [128, 224, 336]
N_IMAGES = 20

def load_sample_images(n: int) -> list[Image.Image]:
    paths = sorted(glob.glob("data/instagram/images/*.jpg"))
    imgs = []
    for p in paths[:n]:
        img = Image.open(p).convert("RGB")
        imgs.append(img)
    return imgs

def bench(imgs: list[Image.Image], size: int):
    from src.embeddings import load_clip, l2_normalize
    import torch

    model, processor, device = load_clip()
    resized = [img.resize((size, size), Image.BILINEAR) for img in imgs]

    # warmup
    with torch.no_grad():
        inputs = processor(images=resized[:2], return_tensors="pt").to(device)
        _ = model.get_image_features(**inputs)

    t0 = time.perf_counter()
    with torch.no_grad():
        inputs = processor(images=resized, return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs).float().cpu().numpy()
    elapsed = time.perf_counter() - t0
    embs = l2_normalize(feats)
    return embs, elapsed

if __name__ == "__main__":
    imgs = load_sample_images(N_IMAGES)
    print(f"Benchmarking {len(imgs)} images at {SIZES} px …\n")

    results = {}
    for sz in SIZES:
        embs, t = bench(imgs, sz)
        norms = np.linalg.norm(embs, axis=1)
        results[sz] = {"emb": embs, "time": t, "norm_ok": bool(np.allclose(norms, 1.0, atol=1e-4))}
        print(f"  {sz}×{sz}:  {t:.3f}s  ({len(imgs)/t:.1f} img/s)  L2-normalised: {results[sz]['norm_ok']}")

    # cosine similarity between 224 baseline and others
    base = results[224]["emb"]
    print("\nCosine similarity vs 224 baseline:")
    for sz in SIZES:
        cos = np.mean(np.einsum("ij,ij->i", base, results[sz]["emb"]))
        print(f"  {sz}×{sz}:  mean cosine = {cos:.4f}")

    # pairwise similarity within each size (cluster coherence)
    print("\nIntra-size mean cosine (higher = more coherent embeddings):")
    for sz in SIZES:
        e = results[sz]["emb"]
        sim = np.einsum("ij,kj->ik", e, e)
        np.fill_diagonal(sim, 0)
        mean_sim = sim.sum() / (len(e) * (len(e) - 1))
        print(f"  {sz}×{sz}:  {mean_sim:.4f}")

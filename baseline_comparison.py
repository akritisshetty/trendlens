#!/usr/bin/env python3
"""
baseline_comparison.py
-----------------------
End-to-end baseline comparison for the TrendLens pipeline.

Evaluates the full TrendLens system against popular alternatives at every stage:
  1. Embedding models:  CLIP ViT-B/32 (ours) vs ViT-L/14 vs DINOv2 vs ResNet50
  2. Dim reduction:     UMAP (ours) vs PCA vs t-SNE
  3. Clustering:        HDBSCAN (ours) vs KMeans vs DBSCAN vs Agglomerative
  4. Trend detection:   Multi-signal (ours) vs Simple growth vs Moving average
  5. Retrieval:         MiniLM-L6-v2 (ours) vs all-mpnet-base-v2 vs CLIP text

Run with:  venv/bin/python baseline_comparison.py
"""

import gc, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score,
    mean_squared_error, mean_absolute_error,
)
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
EMB_PATH = ROOT / "data" / "instagram" / "embeddings.npy"
META_PATH = ROOT / "data" / "instagram" / "embed_meta.parquet"
POSTS_PATH = ROOT / "data" / "instagram" / "all_posts.parquet"
IMG_DIR = ROOT / "data" / "instagram" / "images"
TRENDS_PATH = ROOT / "data" / "instagram" / "trends.json"

# ═════════════════════════════════════════════════════════════════════════════
# Results collector
# ═════════════════════════════════════════════════════════════════════════════
results = []

def record(stage, method, metric, value, unit="", note=""):
    entry = dict(stage=stage, method=method, metric=metric,
                 value=round(value, 6) if isinstance(value, float) else value,
                 unit=unit, note=note)
    results.append(entry)
    if isinstance(value, float):
        print(f"  [{method}] {metric}: {value:.4f} {unit}  {note}")
    else:
        print(f"  [{method}] {metric}: {value} {unit}  {note}")


# ═════════════════════════════════════════════════════════════════════════════
# Load ground truth and metadata
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  TRENDLENS BASELINE COMPARISON — Full Pipeline Evaluation")
print("=" * 70)

emb = np.load(str(EMB_PATH))
meta = pd.read_parquet(str(META_PATH))
posts = pd.read_parquet(str(POSTS_PATH))

# Align posts to embeddings
post_ids = meta["post_id"].astype(str).tolist()
posts_aligned = posts[posts["post_id"].astype(str).isin(post_ids)].copy()
posts_aligned = posts_aligned.drop_duplicates(subset="post_id")
posts_aligned = posts_aligned.set_index("post_id").loc[post_ids].reset_index()

print(f"\nDataset: {len(emb)} posts, {emb.shape[1]}-d embeddings")
print(f"         {meta['author'].nunique()} Instagram accounts, "
      f"{posts_aligned['content_type'].nunique()} content types")

# Use existing HDBSCAN labels as ground truth clustering
import sys
sys.path.insert(0, str(ROOT))
import config
from src.clustering import reduce_dimensions, run_hdbscan, cluster_report, _silhouette_score

# A deterministic UMAP-10d reduction shared by all stages (the TrendLens
# default reduction). min_cluster_size=50 is far too large for 152 samples
# (it collapsed everything to noise), so the reference HDBSCAN run uses a
# parameterisation that actually resolves the structure at this dataset size.
import umap as umap_mod
UMAP_REF = umap_mod.UMAP(
    n_components=10, metric="cosine", n_neighbors=15, min_dist=0.0,
    random_state=42,
).fit_transform(emb)
labels_gt, probs_gt, _ = run_hdbscan(
    UMAP_REF, min_cluster_size=10, min_samples=5,
    cluster_selection_method="eom",
)


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1: Embedding Model Comparison
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STAGE 1: Embedding Model Comparison")
print("=" * 70)

import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sample_images = sorted(IMG_DIR.glob("*.jpg"))[:50]
pil_images = [Image.open(p).convert("RGB") for p in sample_images]
print(f"  Using {len(pil_images)} sample images for embedding benchmark")

# --- CLIP ViT-B/32 (TrendLens) ---
from transformers import CLIPModel, CLIPProcessor
clip_b = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
clip_b_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def bench_clip(model, proc, imgs, label):
    proc.image_processor.size = {"height": 224, "width": 224}
    # Throughput
    t0 = time.perf_counter()
    with torch.no_grad():
        inp = proc(images=imgs, return_tensors="pt").to(DEVICE)
        out = model.get_image_features(**inp)
    elapsed = time.perf_counter() - t0
    imgs_per_sec = len(imgs) / elapsed
    feats = out if isinstance(out, torch.Tensor) else out.pooler_output if hasattr(out, "pooler_output") else out.last_hidden_state[:, 0, :]
    dim = feats.shape[1]
    # Single-image latency
    t0 = time.perf_counter()
    with torch.no_grad():
        inp = proc(images=imgs[:1], return_tensors="pt").to(DEVICE)
        model.get_image_features(**inp)
    single_lat = (time.perf_counter() - t0) * 1000
    record("Embedding", label, "throughput", round(imgs_per_sec, 1), "img/s")
    record("Embedding", label, "dim", dim, "dims")
    record("Embedding", label, "single_latency", round(single_lat, 1), "ms")
    return feats.float().cpu().numpy()

feats_clip_b = bench_clip(clip_b, clip_b_proc, pil_images, "CLIP ViT-B/32 (ours)")
del clip_b, clip_b_proc; gc.collect()
if DEVICE == "cuda": torch.cuda.empty_cache()

# --- CLIP ViT-L/14 ---
try:
    clip_l = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()
    clip_l_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    feats_clip_l = bench_clip(clip_l, clip_l_proc, pil_images, "CLIP ViT-L/14")
    del clip_l, clip_l_proc; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
except Exception as e:
    print(f"  [CLIP ViT-L/14] Skipped: {e}")
    feats_clip_l = None

# --- DINOv2 (ViT-B/14) ---
try:
    from transformers import AutoImageProcessor, AutoModel
    dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        inp = dino_proc(images=pil_images, return_tensors="pt").to(DEVICE)
        out = dino_model(**inp)
        feats_dino = out.last_hidden_state[:, 0, :]  # CLS token
    elapsed = time.perf_counter() - t0
    imgs_per_sec = len(pil_images) / elapsed
    t0 = time.perf_counter()
    with torch.no_grad():
        inp = dino_proc(images=pil_images[:1], return_tensors="pt").to(DEVICE)
        dino_model(**inp)
    single_lat = (time.perf_counter() - t0) * 1000
    record("Embedding", "DINOv2 ViT-B/14", "throughput", round(imgs_per_sec, 1), "img/s")
    record("Embedding", "DINOv2 ViT-B/14", "dim", feats_dino.shape[1], "dims")
    record("Embedding", "DINOv2 ViT-B/14", "single_latency", round(single_lat, 1), "ms")
    feats_dino = feats_dino.float().cpu().numpy()
    del dino_model, dino_proc; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
except Exception as e:
    print(f"  [DINOv2] Skipped: {e}")
    feats_dino = None

# --- ResNet50 (torchvision features) ---
try:
    from torchvision import transforms as T, models
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(DEVICE).eval()
    resnet_feat = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove FC
    resnet_transform = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def _resnet_prep(img):
        return resnet_transform(img.convert("RGB"))

    t0 = time.perf_counter()
    batch = torch.stack([_resnet_prep(img) for img in pil_images]).to(DEVICE)
    with torch.no_grad():
        feats_resnet = resnet_feat(batch).squeeze(-1).squeeze(-1)
    elapsed = time.perf_counter() - t0
    imgs_per_sec = len(pil_images) / elapsed
    t0 = time.perf_counter()
    with torch.no_grad():
        inp = _resnet_prep(pil_images[0]).unsqueeze(0).to(DEVICE)
        resnet_feat(inp)
    single_lat = (time.perf_counter() - t0) * 1000
    record("Embedding", "ResNet50", "throughput", round(imgs_per_sec, 1), "img/s")
    record("Embedding", "ResNet50", "dim", feats_resnet.shape[1], "dims")
    record("Embedding", "ResNet50", "single_latency", round(single_lat, 1), "ms")
    feats_resnet = feats_resnet.float().cpu().numpy()
    del resnet, resnet_feat; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
except Exception as e:
    print(f"  [ResNet50] Skipped: {e}")
    feats_resnet = None

# --- Embedding quality: intra-cluster coherence on known clusters ---
# Compare every embedding model on the SAME labelled posts so the index
# spaces line up (the earlier benchmark used a 50-image sample that did not
# share indices with the 152-post ground truth — that made the metric
# meaningless). We embed the local images of the labelled posts under each
# model and measure how coherent each model's clusters are.
from sklearn.metrics.pairwise import cosine_similarity

# row index (in emb) -> local image path, for labelled posts with a file
meta = pd.read_parquet(str(META_PATH))
row_path = {}
for row_idx, pid in enumerate(meta["post_id"].astype(str)):
    if labels_gt[row_idx] < 0:
        continue
    cand = IMG_DIR / f"{pid}.jpg"
    if cand.is_file():
        row_path[row_idx] = cand
label_rows = sorted(row_path.keys())
sel_labels = labels_gt[label_rows]
print(f"  Coherence evaluated on {len(label_rows)} labelled posts "
      f"({len(np.unique(sel_labels))} clusters)")

def coherence_mapstar(_pil):
    """Loads images, returns PIL list."""
    imgs = [Image.open(row_path[r]).convert("RGB") for r in label_rows]
    return imgs

def embedding_coherence(feats, labels):
    """Average intra-cluster cosine similarity — higher = better."""
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    sims = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        sub = feats[idx]
        cos = cosine_similarity(sub)
        np.fill_diagonal(cos, 0)
        sims.append(cos.sum() / (len(idx) * (len(idx) - 1)))
    return float(np.mean(sims)) if sims else 0.0

# CLIP ViT-B/32 (TrendLens) — use the precomputed 152-d embeddings
record("Embedding", "CLIP ViT-B/32 (ours)", "intra_cluster_coherence",
       round(embedding_coherence(emb[label_rows], sel_labels), 4), "cosine",
       f"on {len(label_rows)} labelled real posts")

coherence_images = coherence_mapstar(None)

if feats_clip_l is not None:
    # model was unloaded after the benchmark — reload for the coherence pass
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    proc.image_processor.size = {"height": 224, "width": 224}
    clip_l = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()
    with torch.no_grad():
        inp = proc(images=coherence_images, return_tensors="pt").to(DEVICE)
        out = clip_l.get_image_features(**inp)
    _o = out if isinstance(out, torch.Tensor) else (
        out.pooler_output if hasattr(out, "pooler_output")
        else out.last_hidden_state[:, 0, :])
    fl = _o.float().cpu().numpy()
    record("Embedding", "CLIP ViT-L/14", "intra_cluster_coherence",
           round(embedding_coherence(fl, sel_labels), 4), "cosine")
    del clip_l; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

if feats_dino is not None:
    from transformers import AutoImageProcessor, AutoModel
    dp = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dm = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()
    with torch.no_grad():
        inp = dp(images=coherence_images, return_tensors="pt").to(DEVICE)
        out = dm(**inp)
        fd = out.last_hidden_state[:, 0, :].float().cpu().numpy()
    record("Embedding", "DINOv2 ViT-B/14", "intra_cluster_coherence",
           round(embedding_coherence(fd, sel_labels), 4), "cosine")
    del dm; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

if feats_resnet is not None:
    from torchvision import transforms as T, models
    _rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(DEVICE).eval()
    _rn_feat = torch.nn.Sequential(*list(_rn.children())[:-1])
    _rn_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def _rn_prep(img):
        return _rn_tf(img.convert("RGB"))

    batch = torch.stack([_rn_prep(img) for img in coherence_images]).to(DEVICE)
    with torch.no_grad():
        fr = _rn_feat(batch).squeeze(-1).squeeze(-1).float().cpu().numpy()
    record("Embedding", "ResNet50", "intra_cluster_coherence",
           round(embedding_coherence(fr, sel_labels), 4), "cosine")
    del _rn, _rn_feat; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2: Dimensionality Reduction Comparison
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STAGE 2: Dimensionality Reduction Comparison")
print("=" * 70)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap as umap_mod

reducers = {
    "UMAP (ours)": lambda e: umap_mod.UMAP(
        n_components=10, metric="cosine", n_neighbors=15, min_dist=0.0, random_state=42
    ).fit_transform(e),
    "PCA": lambda e: PCA(n_components=10, random_state=42).fit_transform(e),
    "t-SNE": lambda e: TSNE(n_components=2, perplexity=30, random_state=42, metric="cosine").fit_transform(e),
}

reduced_embs = {}
for name, fn in reducers.items():
    t0 = time.time()
    reduced = fn(emb)
    elapsed = time.time() - t0
    reduced_embs[name] = reduced
    record("DimReduction", name, "time", round(elapsed, 2), "s", f"{emb.shape[1]}d -> {reduced.shape[1]}d")

    # HDBSCAN on reduced embeddings (workable params for this dataset size)
    cl = run_hdbscan(reduced, min_cluster_size=10, min_samples=5, cluster_selection_method="eom")
    labels_r = cl[0]
    mask_r = labels_r >= 0
    n_clusters = int(labels_r.max() + 1) if labels_r.max() >= 0 else 0
    noise_pct = float((labels_r == -1).mean())

    record("DimReduction", name, "n_clusters", n_clusters)
    record("DimReduction", name, "noise_pct", round(noise_pct, 4))

    if n_clusters >= 2 and mask_r.sum() > n_clusters:
        sil = silhouette_score(reduced[mask_r], labels_r[mask_r], metric="euclidean")
        record("DimReduction", name, "silhouette", round(sil, 4), "", "excl noise")
        ari = adjusted_rand_score(labels_gt, labels_r)
        nmi = normalized_mutual_info_score(labels_gt, labels_r)
        record("DimReduction", name, "ARI", round(ari, 4), "", "vs HDBSCAN ground truth")
        record("DimReduction", name, "NMI", round(nmi, 4), "", "vs HDBSCAN ground truth")
        hom = homogeneity_score(labels_gt, labels_r)
        comp = completeness_score(labels_gt, labels_r)
        vms = v_measure_score(labels_gt, labels_r)
        record("DimReduction", name, "V-measure", round(vms, 4))
    gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3: Clustering Algorithm Comparison
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STAGE 3: Clustering Algorithm Comparison (on UMAP-10d)")
print("=" * 70)

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Use UMAP-10d (TrendLens default reduction)
emb_umap = reduced_embs.get("UMAP (ours)")

clusterers = {
    "HDBSCAN (ours)": lambda e: run_hdbscan(e, min_cluster_size=10, min_samples=5, cluster_selection_method="eom")[0],
    "KMeans (k=4)": lambda e: KMeans(n_clusters=4, random_state=42, n_init=10).fit(e),
    "KMeans (k=5)": lambda e: KMeans(n_clusters=5, random_state=42, n_init=10).fit(e),
    "KMeans (k=8)": lambda e: KMeans(n_clusters=8, random_state=42, n_init=10).fit(e),
    "DBSCAN (eps=0.5)": lambda e: DBSCAN(eps=0.5, min_samples=5, metric="euclidean").fit(e),
    "DBSCAN (eps=1.0)": lambda e: DBSCAN(eps=1.0, min_samples=5, metric="euclidean").fit(e),
    "Agglomerative (k=4)": lambda e: AgglomerativeClustering(n_clusters=4, linkage="ward").fit(e),
}

for name, fn in clusterers.items():
    t0 = time.time()
    result = fn(emb_umap)
    elapsed = time.time() - t0

    if hasattr(result, "labels_"):
        labels_c = np.asarray(result.labels_)
    else:
        labels_c = result

    n_clusters = int(labels_c.max() + 1) if labels_c.max() >= 0 else 0
    noise_pct = float((labels_c == -1).mean()) if -1 in labels_c else 0.0

    record("Clustering", name, "time", round(elapsed, 3), "s")
    record("Clustering", name, "n_clusters", n_clusters)
    record("Clustering", name, "noise_pct", round(noise_pct, 4))

    mask_c = labels_c >= 0
    if n_clusters >= 2 and mask_c.sum() > n_clusters:
        sil = silhouette_score(emb_umap[mask_c], labels_c[mask_c], metric="euclidean")
        record("Clustering", name, "silhouette", round(sil, 4), "", "excl noise")

        # External validity vs HDBSCAN ground truth
        ari = adjusted_rand_score(labels_gt, labels_c)
        nmi = normalized_mutual_info_score(labels_gt, labels_c)
        hom = homogeneity_score(labels_gt, labels_c)
        comp = completeness_score(labels_gt, labels_c)
        vms = v_measure_score(labels_gt, labels_c)
        record("Clustering", name, "ARI", round(ari, 4))
        record("Clustering", name, "NMI", round(nmi, 4))
        record("Clustering", name, "V-measure", round(vms, 4))
    else:
        record("Clustering", name, "silhouette", "N/A")

    # Cluster balance (how evenly distributed)
    if n_clusters > 0:
        counts = np.bincount(labels_c[mask_c])
        balance = counts.min() / counts.max() if counts.max() > 0 else 0
        record("Clustering", name, "balance_ratio", round(balance, 4), "", "min/max cluster size")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4: Trend Detection Method Comparison
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STAGE 4: Trend Detection Scoring Comparison")
print("=" * 70)

# Build the post-clustered dataframe
meta_c = meta.copy()
meta_c["cluster_id"] = labels_gt
meta_c = meta_c[meta_c["cluster_id"] >= 0].copy()
meta_c["engagement"] = meta_c["likes"].fillna(0) + meta_c["comments"].fillna(0)

# Time-series per cluster
ts = meta_c.copy()
t = ts["timestamp"]
if getattr(t.dt, "tz", None) is not None:
    t = t.dt.tz_localize(None)
ts["period"] = t.dt.to_period("W")

agg = ts.groupby(["cluster_id", "period"]).agg(
    post_count=("post_id", "size"),
    mean_engagement=("engagement", "mean"),
    unique_users=("author", "nunique"),
).reset_index()

# Per-cluster time-series of post counts
from collections import defaultdict
cluster_ts = {}
for c in sorted(ts["cluster_id"].unique()):
    sub = agg[agg["cluster_id"] == c].set_index("period")
    full_idx = pd.period_range(start=sub.index.min(), end=sub.index.max(), freq="W")
    cluster_ts[c] = sub["post_count"].reindex(full_idx, fill_value=0).to_numpy(dtype="float64")

# Method 1: TrendLens multi-signal (growth x size x stability)
def trendlens_score(counts, window=2):
    n = len(counts)
    if n < 2 * window:
        return 0.0
    recent = counts[-window:].sum()
    prior = counts[-2*window:-window].sum()
    growth = (recent - prior) / (prior + 1.0)
    size = np.log1p(counts.sum())
    stability = counts.mean()
    g_norm = max(min(growth / (abs(growth) + 1.0), 1.0), -1.0) if growth != 0 else 0.0
    s_norm = min(size / 10.0, 1.0)
    stab_norm = min(stability / (counts.max() + 1e-9), 1.0)
    return g_norm * s_norm * stab_norm

# Method 2: Simple growth rate
def simple_growth(counts, window=2):
    if len(counts) < 2 * window:
        return 0.0
    recent = counts[-window:].sum()
    prior = counts[-2*window:-window].sum()
    return (recent - prior) / (prior + 1.0)

# Method 3: Moving average slope
def ma_slope(counts, window=3):
    if len(counts) < window:
        return 0.0
    ma = np.convolve(counts, np.ones(window)/window, mode="valid")
    if len(ma) < 2:
        return 0.0
    x = np.arange(len(ma))
    slope, _, _, _, _ = sp_stats.linregress(x, ma)
    return slope / (ma.mean() + 1e-9)

# Method 4: Linear regression slope (normalized)
def linear_slope(counts):
    if len(counts) < 2:
        return 0.0
    x = np.arange(len(counts))
    slope, _, _, _, _ = sp_stats.linregress(x, counts)
    return slope / (counts.mean() + 1e-9)

# Method 5: Holt-Winters style exponential weighted growth
def exp_growth(counts, alpha=0.3):
    if len(counts) < 2:
        return 0.0
    weights = np.array([(1-alpha)**i for i in range(len(counts)-1, -1, -1)])
    weights /= weights.sum()
    weighted = counts * weights
    recent_half = weighted[len(counts)//2:].sum()
    prior_half = weighted[:len(counts)//2].sum()
    return (recent_half - prior_half) / (prior_half + 1e-9)

# Method 6: Text tag frequency baseline (using hashtags as proxy)
def text_tag_growth(meta_df, cluster_id, window=2):
    """Hashtag frequency growth as a text-based baseline."""
    sub = meta_df[meta_df["cluster_id"] == cluster_id].copy()
    t = sub["timestamp"]
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    sub["period"] = t.dt.to_period("W")
    
    all_tags = []
    for _, row in sub.iterrows():
        tags = str(row.get("hashtags", "")).split(",")
        tags = [t.strip().lower() for t in tags if t.strip() and t.strip() != "nan"]
        all_tags.extend([(tag, row["period"]) for tag in tags])
    
    if not all_tags:
        return 0.0
    
    tag_df = pd.DataFrame(all_tags, columns=["tag", "period"])
    tag_freq = tag_df.groupby("period").size()
    full_idx = pd.period_range(start=tag_freq.index.min(), end=tag_freq.index.max(), freq="W")
    counts = tag_freq.reindex(full_idx, fill_value=0).to_numpy(dtype="float64")
    
    if len(counts) < 2 * window:
        return 0.0
    recent = counts[-window:].sum()
    prior = counts[-2*window:-window].sum()
    return (recent - prior) / (prior + 1.0)

methods = {
    "TrendLens (ours)": trendlens_score,
    "Simple growth": simple_growth,
    "MA slope": ma_slope,
    "Linear slope": linear_slope,
    "Exp. weighted growth": exp_growth,
}

# Evaluate trend scoring: how well does each method rank clusters by engagement?
# (A good trend score should correlate with actual engagement levels)
engagement_per_cluster = meta_c.groupby("cluster_id")["engagement"].mean().to_dict()
true_engagement = np.array([engagement_per_cluster.get(c, 0) for c in sorted(cluster_ts.keys())])

for name, fn in methods.items():
    scores = np.array([fn(cluster_ts[c]) for c in sorted(cluster_ts.keys())])
    
    # Spearman correlation with mean engagement
    rho, p = sp_stats.spearmanr(scores, true_engagement)
    record("Trend", name, "spearman_engagement", round(rho, 4), "", "correlation with avg engagement")
    
    # Rank correlation between this method and TrendLens
    tl_scores = np.array([trendlens_score(cluster_ts[c]) for c in sorted(cluster_ts.keys())])
    rho_tl, _ = sp_stats.spearmanr(scores, tl_scores)
    record("Trend", name, "spearman_vs_trendlens", round(rho_tl, 4), "", "agreement with TrendLens ranking")
    
    # Precision@k: top-k clusters by this method that have above-median engagement
    median_eng = np.median(true_engagement)
    for k in [2, 3]:
        top_k_idx = np.argsort(scores)[-k:]
        precision_k = (true_engagement[top_k_idx] > median_eng).mean()
        record("Trend", name, f"precision@{k}", round(precision_k, 4), "", f"above-median engagement")

# Text baseline
text_scores = {}
for c in sorted(cluster_ts.keys()):
    text_scores[c] = text_tag_growth(meta_c, c)
text_arr = np.array([text_scores[c] for c in sorted(text_scores.keys())])
rho_text, _ = sp_stats.spearmanr(text_arr, true_engagement)
record("Trend", "Text tags baseline", "spearman_engagement", round(rho_text, 4), "", "hashtag frequency growth")
tl_arr = np.array([trendlens_score(cluster_ts[c]) for c in sorted(cluster_ts.keys())])
rho_tl_text, _ = sp_stats.spearmanr(text_arr, tl_arr)
record("Trend", "Text tags baseline", "spearman_vs_trendlens", round(rho_tl_text, 4))


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5: RAG Retrieval Comparison
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STAGE 5: RAG Retrieval Model Comparison")
print("=" * 70)

from sentence_transformers import SentenceTransformer
import faiss

rag_chunks = json.loads(open(ROOT / "data" / "instagram" / "rag_chunks.json").read())
chunk_texts = [c.get("summary", c.get("blip_caption", "")) for c in rag_chunks]

eval_queries = [
    {"query": "What cafe aesthetic is rising this week?", "relevant_ids": [0, 1, 3]},
    {"query": "What kind of food photography is trending?", "relevant_ids": [0, 1, 2]},
    {"query": "What cooking styles are popular?", "relevant_ids": [0, 1]},
    {"query": "What fitness aesthetics are going viral?", "relevant_ids": [1, 2]},
    {"query": "How should I style food photos for social media?", "relevant_ids": [0, 1, 3]},
    {"query": "What moody photography styles work best?", "relevant_ids": [0, 2, 3]},
    {"query": "What visual themes are emerging on Instagram?", "relevant_ids": [0, 1, 2, 3]},
    {"query": "Which coffee photography gets the most engagement?", "relevant_ids": [0, 3]},
]

retrieval_models = {
    "MiniLM-L6-v2 (ours)": "all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "all-mpnet-base-v2",
    "MiniLM-L12-v2": "all-MiniLM-L12-v2",
}

k_values = [1, 3]

for model_name, model_id in retrieval_models.items():
    try:
        st_model = SentenceTransformer(model_id)
        
        chunk_embs = st_model.encode(chunk_texts, normalize_embeddings=True).astype("float32")
        dim = chunk_embs.shape[1]
        faiss_idx = faiss.IndexFlatIP(dim)
        faiss_idx.add(chunk_embs)
        
        record("Retrieval", model_name, "dim", dim, "dims")
        record("Retrieval", model_name, "corpus_size", len(chunk_texts))
        
        all_precision = {k: [] for k in k_values}
        all_recall = {k: [] for k in k_values}
        all_hit = {k: [] for k in k_values}
        all_mrr = []
        
        for eq in eval_queries:
            q_emb = st_model.encode([eq["query"]], normalize_embeddings=True).astype("float32")
            scores, indices = faiss_idx.search(q_emb, max(k_values))
            retrieved = [int(idx) for idx in indices[0] if idx >= 0]
            relevant = set(eq["relevant_ids"])
            
            rr = 0.0
            for pos, idx in enumerate(retrieved, start=1):
                if idx in relevant:
                    rr = 1.0 / pos
                    break
            all_mrr.append(rr)
            
            for k in k_values:
                retrieved_k = retrieved[:k]
                rel_k = len(set(retrieved_k) & relevant)
                precision_k = rel_k / k
                recall_k = rel_k / len(relevant) if len(relevant) > 0 else 0
                hit_k = 1 if rel_k > 0 else 0
                all_precision[k].append(precision_k)
                all_recall[k].append(recall_k)
                all_hit[k].append(hit_k)
        
        for k in k_values:
            record("Retrieval", model_name, f"precision@{k}", round(np.mean(all_precision[k]), 4), "", f"k={k}")
            record("Retrieval", model_name, f"recall@{k}", round(np.mean(all_recall[k]), 4), "", f"k={k}")
            record("Retrieval", model_name, f"hit@{k}", round(np.mean(all_hit[k]), 4), "", f"k={k}")
        record("Retrieval", model_name, "MRR", round(np.mean(all_mrr), 4))
        
        del st_model
        gc.collect()
    except Exception as e:
        print(f"  [{model_name}] Skipped: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 6: End-to-End Latency Summary
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STAGE 6: End-to-End Pipeline Latency")
print("=" * 70)

# Measure full pipeline latency (incremental run)
from src.clustering import load_embeddings, load_aligned_metadata
from src.embeddings import CLIPEmbeddingGenerator
from src.trends import aggregate_cluster_trends, growth_metrics, trend_scores, classify_lifecycle

# Embedding time (already have CLIP ViT-B/32 loaded above — use the measured throughput)
record("Pipeline", "CLIP embedding", "time_per_image",
       round(1000.0 / (len(pil_images) / [r for r in results if r["metric"] == "throughput" and r["method"] == "CLIP ViT-B/32 (ours)"][0]["value"]), 1), "ms")

# UMAP + HDBSCAN time
t0 = time.time()
reduced_full = umap_mod.UMAP(n_components=10, metric="cosine", n_neighbors=15, min_dist=0.0, random_state=42).fit_transform(emb)
umap_time = time.time() - t0
record("Pipeline", "UMAP 10d", "time", round(umap_time, 2), "s", f"{len(emb)} samples")

t0 = time.time()
labels_pipe, _, _ = run_hdbscan(reduced_full, min_cluster_size=50, min_samples=10)
hdbscan_time = time.time() - t0
record("Pipeline", "HDBSCAN", "time", round(hdbscan_time, 2), "s")

# Trend computation time
t0 = time.time()
meta_pipe = meta.copy()
meta_pipe["user_id"] = meta_pipe.get("author", pd.Series(index=meta_pipe.index)).astype(str)
meta_pipe["cluster_id"] = labels_pipe
meta_pipe = meta_pipe[meta_pipe["cluster_id"] >= 0].copy()
agg_pipe = aggregate_cluster_trends(meta_pipe, period="W")
metrics_pipe = growth_metrics(agg_pipe, period="W", window=2)
scored_pipe = trend_scores(metrics_pipe)
trend_time = time.time() - t0
record("Pipeline", "Trend scoring", "time", round(trend_time, 2), "s")

total = umap_time + hdbscan_time + trend_time
record("Pipeline", "Total (UMAP+HDBSCAN+trends)", "time", round(total, 2), "s")


# ═════════════════════════════════════════════════════════════════════════════
# Save results
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  RESULTS SAVED")
print("=" * 70)

df = pd.DataFrame(results)
out_path = ROOT / "baseline_comparison_results.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved {len(df)} metrics to {out_path}")

# Print summary table
print("\n" + "=" * 70)
print("  SUMMARY: TrendLens vs Baselines")
print("=" * 70)

for stage in df["stage"].unique():
    sub = df[df["stage"] == stage]
    print(f"\n--- {stage} ---")
    for method in sub["method"].unique():
        msub = sub[sub["method"] == method]
        metrics_str = " | ".join(
            f"{r.metric}={r.value}" for r in msub.itertuples()
            if r.metric not in ("dim", "corpus_size", "n_clusters")
        )
        print(f"  {method}: {metrics_str}")

print("\nDone.")

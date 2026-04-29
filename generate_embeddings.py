"""
generate_embeddings.py
----------------------
Generates CLIP (openai/clip-vit-base-patch32) image embeddings for all valid
images in the TrendLens dataset. Run after generate_metadata.py.

Features
────────
  • Resumes automatically from embeddings_checkpoint.npy if present.
  • Checkpoints every CHECKPOINT_EVERY batches to guard against interruptions.
  • L2-normalises all embeddings (unit-norm → dot product == cosine similarity).
  • Logs failed images to failed_images.txt.
  • Runs on CUDA if available, otherwise falls back to CPU.

Outputs (all in trendlens_outputs/)
────────────────────────────────────
  embeddings.npy             — float32 array, shape (N, 512), L2-normalised
  embeddings_checkpoint.npy  — latest checkpoint (mirrors final state on finish)
  metadata_checkpoint.csv    — metadata slice aligned with the checkpoint
  failed_images.txt          — paths of any images that could not be opened
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "trendlens_outputs"
META_CSV   = OUTPUT_DIR / "metadata.csv"
EMB_PATH   = OUTPUT_DIR / "embeddings.npy"
FAILED_LOG = OUTPUT_DIR / "failed_images.txt"
CKPT_EMB   = OUTPUT_DIR / "embeddings_checkpoint.npy"
CKPT_CSV   = OUTPUT_DIR / "metadata_checkpoint.csv"

# ── CLIP config ────────────────────────────────────────────────────────────────
CLIP_MODEL_ID    = "openai/clip-vit-base-patch32"
BATCH_SIZE       = 32
CHECKPOINT_EVERY = 200   # save a checkpoint every N batches

# ── Device ─────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Load metadata ──────────────────────────────────────────────────────────────
print(f"Loading metadata from {META_CSV} …")
df_final = pd.read_csv(META_CSV)
print(f"  Total images: {len(df_final):,}")

# Reconstruct absolute paths (abs_path col was dropped from CSV at save time)
if "abs_path" in df_final.columns:
    image_paths = df_final["abs_path"].tolist()
else:
    image_paths = [str(BASE_DIR / p) for p in df_final["image_path"].tolist()]

N = len(image_paths)

# ── Resume from checkpoint ─────────────────────────────────────────────────────
start_idx      = 0
all_embeddings = []

if CKPT_EMB.exists() and CKPT_CSV.exists():
    ckpt_arr  = np.load(CKPT_EMB)
    ckpt_df   = pd.read_csv(CKPT_CSV)
    start_idx = len(ckpt_df)
    all_embeddings = [ckpt_arr]
    print(f"  Resuming from checkpoint: {start_idx:,} / {N:,} done "
          f"({N - start_idx:,} remaining)")
else:
    print("  No checkpoint found — starting from scratch.")

if start_idx >= N:
    print("✓ All embeddings already completed (checkpoint covers full dataset).")
    if not EMB_PATH.exists():
        embeddings = np.vstack(all_embeddings)
        np.save(EMB_PATH, embeddings)
        print(f"✓ Saved final embeddings → {EMB_PATH}  shape={embeddings.shape}")
    else:
        print(f"  embeddings.npy already exists at {EMB_PATH}")
    raise SystemExit(0)

# ── Load CLIP model ────────────────────────────────────────────────────────────
print(f"\nLoading {CLIP_MODEL_ID} …")
clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
clip_model.eval()
print("✓ Model loaded.\n")


# ── Embedding helper ───────────────────────────────────────────────────────────
@torch.no_grad()
def embed_batch(pil_images: list) -> np.ndarray:
    """
    Run a list of PIL images through the CLIP vision encoder.
    Returns a float32 numpy array of shape (len(pil_images), 512),
    each row L2-normalised (unit norm).
    """
    inputs       = clip_processor(images=pil_images, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device)
    vision_out   = clip_model.vision_model(pixel_values=pixel_values)
    pooled       = vision_out.pooler_output              # [B, hidden_dim]
    projected    = clip_model.visual_projection(pooled)  # [B, 512]
    normed       = projected / projected.norm(dim=-1, keepdim=True)
    return normed.cpu().float().numpy()


# ── Main embedding loop ────────────────────────────────────────────────────────
failed_images = []
batch_paths   = []
batch_count   = 0


def flush_batch() -> None:
    """Process the current batch and optionally write a checkpoint."""
    global batch_paths, batch_count, all_embeddings

    pil_imgs = []
    for bp in batch_paths:
        try:
            pil_imgs.append(Image.open(bp).convert("RGB"))
        except Exception as exc:
            failed_images.append(f"{bp}\t{exc}")

    if pil_imgs:
        all_embeddings.append(embed_batch(pil_imgs))

    batch_paths.clear()
    batch_count += 1

    # Periodic checkpoint
    if batch_count % CHECKPOINT_EVERY == 0:
        ckpt = np.vstack(all_embeddings)
        np.save(CKPT_EMB, ckpt)
        df_final.iloc[: len(ckpt)].to_csv(CKPT_CSV, index=False)
        print(f"  [ckpt] {len(ckpt):,} embeddings saved at batch {batch_count}")


pbar = tqdm(total=N - start_idx, desc="Embedding", unit="img")
for i in range(start_idx, N):
    batch_paths.append(image_paths[i])
    pbar.update(1)
    if len(batch_paths) == BATCH_SIZE:
        flush_batch()

if batch_paths:   # flush any remainder
    flush_batch()
pbar.close()


# ── Stack and save final output ────────────────────────────────────────────────
embeddings = np.vstack(all_embeddings)
np.save(EMB_PATH, embeddings)
print(f"\n✓ Saved embeddings → {EMB_PATH}")
print(f"  Shape : {embeddings.shape}  |  dtype : {embeddings.dtype}")

# Overwrite checkpoint so it always mirrors the final state
np.save(CKPT_EMB, embeddings)
df_final.to_csv(CKPT_CSV, index=False)

if failed_images:
    FAILED_LOG.write_text("\n".join(failed_images))
    print(f"⚠  {len(failed_images)} failed images logged → {FAILED_LOG}")
else:
    print("✓ No failed images.")


# ── Quick sanity check ─────────────────────────────────────────────────────────
assert embeddings.ndim == 2,          f"Expected 2-D, got {embeddings.ndim}-D"
assert embeddings.shape[1] == 512,    f"Expected 512 dims, got {embeddings.shape[1]}"
assert embeddings.dtype == np.float32, f"Expected float32, got {embeddings.dtype}"
assert embeddings.shape[0] == len(df_final), (
    f"Row mismatch: embeddings={embeddings.shape[0]}, metadata={len(df_final)}")

self_sim  = float(embeddings[0] @ embeddings[0])
cross_sim = float(embeddings[0] @ embeddings[1])
assert abs(self_sim - 1.0) < 1e-4, f"Self-similarity off: {self_sim}"

print()
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  Embedding shape : {embeddings.shape}")
print(f"  dtype           : {embeddings.dtype}")
print(f"  Metadata rows   : {len(df_final):,}")
print(f"  Self-sim [0·0]  : {self_sim:.6f}  (should be ~1.0)")
print(f"  Cross-sim [0·1] : {cross_sim:.6f}")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("✓ All assertions passed. Embedding pipeline complete.")

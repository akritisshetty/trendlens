"""
embeddings.py
-------------
CLIP image-embedding pipeline.

Design guarantees (from the spec):
  * embeddings are float32, L2-normalized
  * batch size, checkpoint interval configurable
  * periodic checkpointing + resume — a disconnected Colab runtime never
    forces a full recompute
  * metadata row index == embedding row index (verified by assertions)
  * only valid images are embedded; invalid rows are dropped *before*
    generation so alignment never silently breaks mid-run
  * GPU (CUDA) when available, MPS on macOS, CPU fallback
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import config

# Heavy deps are imported lazily so tests (and `import src.embeddings`)
# do not require torch/transformers to be installed.

#: Single source of truth is ``config.CLIP_MODEL``; re-exported here because
#: this module is the conventional import site for it.
DEFAULT_CLIP_MODEL = config.CLIP_MODEL


# ──────────────────────────────────────────────────────────────────────────
# Device & model loading
# ──────────────────────────────────────────────────────────────────────────
def detect_device(prefer_cuda: bool = True) -> str:
    """Return 'cuda' | 'mps' | 'cpu' based on what is actually available."""
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


#: Process-wide CLIP handles, keyed by (model_name, device). `from_pretrained`
#: performs a blocking HuggingFace Hub metadata request on every call, so
#: callers that load CLIP per item — e.g. `src.data_collector._clip_text_vec`,
#: which loads once for every uncached prompt — turn the concept vocabulary
#: into one network round trip per concept and stall the rebuild stage.
_CLIP_CACHE: dict[tuple[str, str], tuple[Any, Any, str]] = {}


def load_clip(model_name: str = DEFAULT_CLIP_MODEL, device: str | None = None):
    """Load CLIPModel + CLIPProcessor. Returns (model, processor, device).

    Memoized per (model_name, resolved device): the weights are read from disk
    and moved to the device only on the first call for that pair. The cached
    model is shared and must not be mutated by callers.
    """
    resolved = device or detect_device()
    key = (model_name, resolved)
    cached = _CLIP_CACHE.get(key)
    if cached is not None:
        return cached

    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(resolved)
    model.eval()
    _CLIP_CACHE[key] = (model, processor, resolved)
    return _CLIP_CACHE[key]


# ──────────────────────────────────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────────────────────────────────
def l2_normalize(emb: np.ndarray) -> np.ndarray:
    """L2-normalize a (N, D) or (D,) float array in place on a copy."""
    emb = np.asarray(emb, dtype="float32")
    norm = np.linalg.norm(emb, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0  # avoid div-by-zero for all-zero vectors
    return emb / norm


def assert_alignment(embeddings: np.ndarray, df: pd.DataFrame) -> None:
    """
    Verify embedding row count matches the manifest and that embeddings are
    float32 + L2-normalized. Raises AssertionError on any violation.
    """
    assert embeddings.ndim == 2, f"embeddings must be 2-D, got {embeddings.ndim}D"
    assert len(embeddings) == len(df), (
        f"alignment broken: {len(embeddings)} embeddings vs {len(df)} metadata rows"
    )
    assert embeddings.dtype == np.float32, (
        f"embeddings must be float32, got {embeddings.dtype}"
    )
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), (
        f"{int((np.abs(norms - 1.0) > 1e-4).sum())} rows are not L2-normalized"
    )


# ──────────────────────────────────────────────────────────────────────────
# Manifest preparation
# ──────────────────────────────────────────────────────────────────────────
def prepare_embedding_manifest(
    df: pd.DataFrame,
    image_root: Path | None = None,
    output: Path | None = None,
) -> pd.DataFrame:
    """
    Drop rows whose image fails validation and reset the index, so every
    remaining row embeds successfully and stays index-aligned.

    Returns the filtered manifest and persists it to
    ``data/embeddings/manifest.parquet``.
    """
    from src import preprocessing as pp

    image_root = image_root or config.IMAGE_ROOT
    keep = []
    for p in tqdm(df["image_path"], desc="Validating for embedding"):
        keep.append(pp.validate_image(p, image_root)[0])

    out = df.loc[keep].copy().reset_index(drop=True)
    output = output or (config.EMBEDDINGS_DIR / "manifest.parquet")
    out.to_parquet(output, index=False)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Checkpoint helpers (pure logic, no torch needed)
# ──────────────────────────────────────────────────────────────────────────
def _checkpoint_paths(out_dir: Path):
    out_dir = Path(out_dir)
    return {
        "emb": out_dir / "embeddings_checkpoint.npy",
        "meta": out_dir / "metadata_checkpoint.parquet",
        "manifest": out_dir / "manifest.parquet",
        "final": out_dir / "embeddings.npy",
    }


def load_checkpoint(out_dir: Path) -> tuple[np.ndarray | None, pd.DataFrame | None]:
    """Load any existing checkpoint; returns (embeddings, metadata)."""
    p = _checkpoint_paths(out_dir)
    if not p["emb"].exists() or not p["meta"].exists():
        return None, None
    emb = np.load(p["emb"], mmap_mode="r")
    meta = pd.read_parquet(p["meta"])
    return emb, meta


def compute_resume_offset(
    manifest: pd.DataFrame, checkpoint_meta: pd.DataFrame | None, id_col: str = "post_id"
) -> int:
    """
    How many leading rows of ``manifest`` are already embedded?

    Resumes only when the completed rows are an exact contiguous prefix of
    the manifest (same ids in the same order). Any mismatch => restart at 0.
    """
    if checkpoint_meta is None or len(checkpoint_meta) == 0:
        return 0
    ids = manifest[id_col].astype(str).tolist()
    done = checkpoint_meta[id_col].astype(str).tolist()
    offset = 0
    for a, b in zip(ids, done):
        if a != b:
            return 0
        offset += 1
    return offset


# ──────────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────────
class CLIPEmbeddingGenerator:
    """
    Batched, checkpointed, resumable CLIP embedding generator.

    Subclass or inject ``encode_batch`` to override the encoder (used by
    tests with a stub so no model / GPU / network is required).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        device: str | None = None,
        batch_size: int = 32,
        checkpoint_every: int = 100,
        out_dir: Path | None = None,
        image_root: Path | None = None,
        show_progress: bool = True,
        model=None,
        processor=None,
        proj_dim: int | None = None,
        load_model: bool = True,
    ):
        self.model_name = model_name
        self.device = device or detect_device()
        self.batch_size = int(batch_size)
        self.checkpoint_every = int(checkpoint_every)
        self.out_dir = Path(out_dir) if out_dir else config.EMBEDDINGS_DIR
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.image_root = image_root or config.IMAGE_ROOT
        self.show_progress = show_progress
        self.model = model
        self.processor = processor
        self._proj_dim = proj_dim
        if load_model and self.model is None:
            self.model, self.processor, self.device = self._load()

    def _load(self):
        return load_clip(self.model_name, self.device)

    # -- encoder hook -----------------------------------------------------
    def encode_batch(self, images: list) -> np.ndarray:
        """Images -> (B, D) float32 CLIP embedding (normalized by caller)."""
        import torch

        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_image_features(**inputs)
            if hasattr(outputs, "pooler_output"):  # transformers >= 4.x full output
                feats = outputs.pooler_output
            else:
                feats = outputs
        feats = feats.float().cpu().numpy()
        return l2_normalize(feats)

    # -- main entry -------------------------------------------------------
    def generate(self, manifest: pd.DataFrame) -> Path:
        """
        Embed every image in ``manifest`` (order = row order) and write
        ``data/embeddings/embeddings.npy`` + aligned metadata parquet.
        Resumes from a checkpoint if the manifest matches.
        """
        from src import preprocessing as pp

        manifest = manifest.reset_index(drop=True)
        paths = _checkpoint_paths(self.out_dir)
        emb_cp, meta_cp = load_checkpoint(self.out_dir)

        offset = compute_resume_offset(manifest, meta_cp)
        if offset:
            print(f"Resuming from row {offset} (reusing {offset} embedded rows).")
        else:
            print(f"Starting fresh ({len(manifest)} rows).")

        n_rows = len(manifest)
        if n_rows == 0:
            raise ValueError("Manifest is empty — nothing to embed.")

        # allocate / reuse memmap of full final size
        mmap = self._init_memmap(paths["emb"], n_rows, offset)

        batch = []
        batch_idx: list[int] = []
        n_new = 0
        t0 = time.time()
        pbar = tqdm(
            range(offset, n_rows), desc="CLIP embeddings", disable=not self.show_progress
        )
        for i in pbar:
            try:
                img = pp.load_image(manifest["image_path"].iloc[i], self.image_root, resize=config.IMAGE_RESIZE)
            except (OSError, ValueError):
                # validated upfront; this is an unexpected failure -> fail loudly
                raise
            batch.append(img)
            batch_idx.append(i)
            if len(batch) == self.batch_size or i == n_rows - 1:
                feats = self.encode_batch(batch)
                assert feats.shape[0] == len(batch_idx)
                for j, row in zip(batch_idx, feats):
                    mmap[j] = row
                n_new += len(batch_idx)
                batch, batch_idx = [], []
                if n_new >= self.checkpoint_every:
                    self._checkpoint(mmap, manifest, offset + n_new, paths)
                    elapsed = time.time() - t0
                    pbar.set_postfix(
                        rows=f"{offset + n_new}/{n_rows}",
                        speed=f"{n_new / max(elapsed, 1e-9):.1f} im/s",
                    )

        self._checkpoint(mmap, manifest, n_rows, paths)
        final = self._finalize(paths, manifest, n_rows)
        return final

    # -- internals --------------------------------------------------------
    def _init_memmap(self, path: Path, n_rows: int, offset: int) -> np.ndarray:
        shape = (n_rows, self.proj_dim())
        if path.exists():
            old = np.load(path, mmap_mode="r+")  # read-write: we overwrite from `offset`
            if old.shape == shape:
                return old  # reuse existing file
            if old.shape[1] == shape[1] and old.shape[0] <= shape[0]:
                # grow (manifest got bigger): preserve already-computed rows.
                # Materialise (real copy!) BEFORE truncating, since
                # open_memmap("w+") resizes the file underneath the old mmap.
                old_data = np.array(old, copy=True)
                new = np.lib.format.open_memmap(
                    path, mode="w+", dtype="float32", shape=shape
                )
                new[: len(old_data)] = old_data
                return new
            path.unlink()  # incompatible (new model / smaller manifest) -> recreate
        return np.lib.format.open_memmap(
            path, mode="w+", dtype="float32", shape=shape
        )

    def proj_dim(self) -> int:
        """CLIP embedding dimension — never hard-coded to 512."""
        if self._proj_dim is not None:
            return int(self._proj_dim)
        if getattr(self.model, "config", None) is not None:
            return int(self.model.config.projection_dim)
        return int(np.load(config.EMBEDDINGS_DIR / "embeddings_checkpoint.npy", mmap_mode="r").shape[1]) if (
            config.EMBEDDINGS_DIR / "embeddings_checkpoint.npy"
        ).exists() else 512

    def _checkpoint(self, mmap, manifest, done_rows: int, paths: dict):
        mmap.flush()
        done = manifest.iloc[:done_rows]
        done.to_parquet(paths["meta"], index=False)

    def _finalize(self, paths: dict, manifest: pd.DataFrame, n_rows: int) -> Path:
        """Copy the checkpoint to the final artifact + write aligned metadata."""
        final_emb = paths["final"]
        emb = np.load(paths["emb"])  # full array (memmap view)
        np.save(final_emb, emb)
        manifest.to_parquet(paths["meta"].with_name("metadata.parquet"), index=False)
        assert_alignment(emb, manifest)
        print(f"Done: {emb.shape[0]} x {emb.shape[1]} embeddings at {final_emb}")
        return final_emb

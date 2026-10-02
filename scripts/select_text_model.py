#!/usr/bin/env python3
"""
select_text_model.py
--------------------
Choose the RAG text-embedding model by retrieval quality on TrendLens' OWN
data, not on a public leaderboard.

Why the obvious benchmark is not enough
---------------------------------------
``all-MiniLM-L6-v2`` is the incumbent because sentence-transformers' default is
that model, not because anything here measured it. Public retrieval benchmarks
(MTEB, BEIR) rank models on generic web-scale corpora; this index is 18 chunks
about Instagram photography, and a model tuned for web query/document
asymmetry is not automatically better on that.

How this is evaluated without circular labels
---------------------------------------------
For every visual cluster we caption a handful of its member images with BLIP
and use those captions as queries; the cluster's own chunk is the relevant
answer. Crucially the chunk's embedded caption is the ONE representative image
caption, and query images exclude that representative — so the query text is
not a verbatim copy of anything in the indexed chunk. The task is therefore
"given a fresh description of this visual trend, find its entry", which is
exactly what the chat interface asks of the retriever.

Metrics: Recall@1 (strictest), MRR, Recall@3, and Hit@5. With 18 chunks the
task is easy in absolute terms; what matters is the RANKING BETWEEN candidates,
so every candidate is scored on the identical query set.

Run:  venv/bin/python scripts/select_text_model.py
      venv/bin/python scripts/select_text_model.py --candidates sentence-transformers/all-mpnet-base-v2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

OUT_DIR = config.ARTIFACTS_DIR / "model_selection"
OUT_CSV = ROOT / "text_model_selection_results.csv"

#: MiniLM (incumbent) through the usual accuracy ladder. Every model here is
#: Apache-2.0 / MIT and runs on CPU, which matters because the deployment target
#: for this project has no GPU.
DEFAULT_CANDIDATES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
]

#: Models that expect an instruction prefix on the QUERY side only. Getting this
#: wrong silently degrades retrieval rather than erroring, so it is declared
#: rather than guessed.
QUERY_PREFIX = {
    "intfloat/e5-base-v2": "query: ",
    "intfloat/e5-small-v2": "query: ",
    "intfloat/multilingual-e5-base": "query: ",
}
PASSAGE_PREFIX = {
    "intfloat/e5-base-v2": "passage: ",
    "intfloat/e5-small-v2": "passage: ",
    "intfloat/multilingual-e5-base": "passage: ",
}

IMAGES_PER_CLUSTER = 4


def load_state() -> tuple[pd.DataFrame, list[dict], np.ndarray]:
    """Cluster assignment, RAG chunks and the embedding matrix they came from."""
    emb = np.load(config.INSTAGRAM_EMBEDDINGS_PATH)
    meta = pd.read_parquet(config.INSTAGRAM_DIR / "embed_meta.parquet")
    assert len(emb) == len(meta), "embeddings/metadata misaligned"

    labels_path = config.CLUSTER_MODELS_DIR / "labels_instagram.npy"
    if not labels_path.exists():
        raise SystemExit(
            f"{labels_path} missing — run the pipeline "
            "(venv/bin/python scripts/rebuild_trends.py) first."
        )
    labels = np.load(labels_path)
    if len(labels) != len(meta):
        raise SystemExit("cached labels do not match the current embedding matrix")

    chunks = json.loads(config.INSTAGRAM_RAG_CHUNKS_PATH.read_text())
    if not chunks:
        raise SystemExit("no RAG chunks — run rebuild_trends.py first")
    return meta, chunks, labels


def resolve_image(post_id: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = config.INSTAGRAM_IMAGES_DIR / f"{post_id}{ext}"
        if p.is_file():
            return p
    return None


def build_queries(meta: pd.DataFrame, chunks: list[dict], labels: np.ndarray) -> list[dict]:
    """
    BLIP captions of cluster member images -> retrieval queries.

    The representative image whose caption is embedded in the chunk is excluded,
    so no query is a verbatim copy of indexed text.
    """
    from PIL import Image
    from src.interpretation import caption_image, load_blip

    # One captioner for every candidate, so the comparison isolates the text
    # encoder. Swapping the captioner here would confound the two.
    model, processor, device = load_blip()

    # Which post each chunk used as its embedded caption.
    rep_posts: set[str] = set()
    for c in chunks:
        for key in ("representative_post_id", "post_id"):
            if c.get(key):
                rep_posts.add(str(c[key]))
                break

    rng = np.random.default_rng(config.RANDOM_SEED)
    queries: list[dict] = []
    for c in chunks:
        cid = int(c["cluster_id"])
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        # Prefer cluster members that are not the chunk's representative.
        prefer = [i for i in idx if str(meta["post_id"].iloc[i]) not in rep_posts]
        pool = prefer if len(prefer) >= IMAGES_PER_CLUSTER else list(idx)
        pick = rng.choice(pool, size=min(IMAGES_PER_CLUSTER, len(pool)), replace=False)

        for i in sorted(int(x) for x in pick):
            path = resolve_image(str(meta["post_id"].iloc[i]))
            if path is None:
                continue
            try:
                img = Image.open(path).convert("RGB")
            except OSError:
                continue
            cap = str(caption_image(model, processor, img, device=device) or "").strip()
            if not cap:
                continue
            queries.append(
                {
                    "cluster_id": cid,
                    "post_id": str(meta["post_id"].iloc[i]),
                    "query": cap,
                    "is_representative": str(meta["post_id"].iloc[i]) in rep_posts,
                }
            )
    return queries


def evaluate(name: str, chunk_texts: list[str], queries: list[dict]) -> dict:
    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    model = SentenceTransformer(name)
    load_s = time.perf_counter() - t0

    qp = QUERY_PREFIX.get(name, "")
    pp = PASSAGE_PREFIX.get(name, "")

    t1 = time.perf_counter()
    doc_emb = model.encode(
        [pp + t for t in chunk_texts],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    q_emb = model.encode(
        [qp + q["query"] for q in queries],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    encode_s = time.perf_counter() - t1

    chunk_ids = [int(c["cluster_id"]) for c in json.loads(
        config.INSTAGRAM_RAG_CHUNKS_PATH.read_text()
    )]

    sims = q_emb @ doc_emb.T
    # rank of the correct chunk for each query (1-based)
    ranks: list[int] = []
    for row, q in zip(sims, queries):
        order = np.argsort(-row)
        pos = [k for k, o in enumerate(order) if chunk_ids[int(o)] == q["cluster_id"]]
        ranks.append((pos[0] + 1) if pos else len(order) + 1)

    ranks_arr = np.array(ranks, dtype=float)
    return {
        "model": name,
        "dim": int(doc_emb.shape[1]),
        "n_queries": len(ranks),
        "recall_at_1": float((ranks_arr <= 1).mean()),
        "recall_at_3": float((ranks_arr <= 3).mean()),
        "mrr": float((1.0 / ranks_arr).mean()),
        "hit_at_5": float((ranks_arr <= 5).mean()),
        "load_seconds": round(load_s, 1),
        "encode_seconds": round(encode_s, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="*", default=DEFAULT_CANDIDATES)
    args = ap.parse_args()

    meta, chunks, labels = load_state()
    chunk_texts = [c["text"] for c in chunks]
    print(f"{len(chunks)} chunks, {len(meta)} posts, "
          f"{len(np.unique(labels[labels >= 0]))} clusters\n")

    qcache = OUT_DIR / "rag_queries.json"
    if qcache.exists():
        queries = json.loads(qcache.read_text())
        print(f"Using cached query set ({len(queries)} queries) -> {qcache}")
    else:
        print("Captioning cluster members with BLIP to build queries …")
        queries = build_queries(meta, chunks, labels)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        qcache.write_text(json.dumps(queries, indent=1))
        print(f"Built {len(queries)} queries -> {qcache}")

    n_rep = sum(1 for q in queries if q["is_representative"])
    print(f"({n_rep} queries are representative images and were meant to be "
          f"excluded; their presence would make the task trivial)\n")

    rows = []
    print("=" * 88)
    print(f"{'model':<52}{'dim':>5}{'R@1':>8}{'R@3':>8}{'MRR':>8}{'enc s':>8}")
    print("=" * 88)
    for name in args.candidates:
        try:
            r = evaluate(name, chunk_texts, queries)
        except Exception as exc:  # noqa: BLE001 — a bad candidate must not kill the sweep
            print(f"{name:<52} FAILED: {type(exc).__name__}: {str(exc)[:60]}")
            continue
        rows.append(r)
        print(
            f"{name:<52}{r['dim']:>5}{r['recall_at_1']:>8.3f}"
            f"{r['recall_at_3']:>8.3f}{r['mrr']:>8.3f}{r['encode_seconds']:>8.2f}",
            flush=True,
        )

    if rows:
        df = pd.DataFrame(rows).sort_values("mrr", ascending=False)
        df.to_csv(OUT_CSV, index=False)
        print(f"\nBest by MRR: {df.iloc[0]['model']} (MRR {df.iloc[0]['mrr']:.3f})")
        print(f"Saved -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Evaluation metrics for the TrendLens pipeline.

Computes:
  1. Clustering quality: Silhouette, ARI, NMI, Homogeneity, Completeness, V-measure
  2. Scope gate classification: Accuracy, Precision, Recall, F1
  3. RAG retrieval: Precision@k, Recall@k, F1@k, MRR
  4. Lifecycle classification: Accuracy, Precision, Recall, F1

Run with: venv/bin/python evaluate_system.py
"""

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

results = []


def record(section, metric, value, unit="", note=""):
    results.append(dict(section=section, metric=metric,
                        value=round(value, 6) if isinstance(value, float) else value,
                        unit=unit, note=note))
    print(f"  {metric}: {value} {unit}  {note}")


# ═════════════════════════════════════════════════════════════════════════════
# 1. Clustering Quality Metrics (synthetic evaluation)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 1. Clustering Quality Metrics")
print("=" * 60)

from hdbscan import HDBSCAN
from sklearn.cluster import KMeans

np.random.seed(42)
X_gt, y_gt = make_blobs(n_samples=500, n_features=512, centers=5, cluster_std=1.0, random_state=42)
record("Clustering", "n_samples", 500, "samples", "synthetic 512-dim")
record("Clustering", "n_ground_truth_clusters", 5, "clusters", "")

clusterer = HDBSCAN(min_cluster_size=25, min_samples=5, metric="euclidean", cluster_selection_method="eom")
labels_pred = clusterer.fit_predict(X_gt)
n_found = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
n_noise = int((labels_pred == -1).sum())

record("Clustering", "hdbscan_clusters_found", n_found, "clusters", "")
record("Clustering", "hdbscan_noise_points", n_noise, "points", f"{n_noise/len(labels_pred):.1%}")

mask = labels_pred >= 0
if mask.sum() > 2 and len(np.unique(labels_pred[mask])) >= 2:
    sil = silhouette_score(X_gt[mask], labels_pred[mask], metric="euclidean")
    record("Clustering", "silhouette_score", round(sil, 4), "", "excluding noise, euclidean")

ari = adjusted_rand_score(y_gt, labels_pred)
nmi = normalized_mutual_info_score(y_gt, labels_pred)
hom = homogeneity_score(y_gt, labels_pred)
comp = completeness_score(y_gt, labels_pred)
vms = v_measure_score(y_gt, labels_pred)

record("Clustering", "adjusted_rand_index", round(ari, 4), "", "vs ground truth")
record("Clustering", "normalized_mutual_info", round(nmi, 4), "", "vs ground truth")
record("Clustering", "homogeneity", round(hom, 4), "")
record("Clustering", "completeness", round(comp, 4), "")
record("Clustering", "v_measure", round(vms, 4), "")

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_km = kmeans.fit_predict(X_gt)
sil_km = silhouette_score(X_gt, labels_km, metric="euclidean")
ari_km = adjusted_rand_score(y_gt, labels_km)
nmi_km = normalized_mutual_info_score(y_gt, labels_km)

record("Clustering", "kmeans_silhouette", round(sil_km, 4), "", "KMeans baseline")
record("Clustering", "kmeans_ari", round(ari_km, 4), "", "KMeans baseline")
record("Clustering", "kmeans_nmi", round(nmi_km, 4), "", "KMeans baseline")


# ═════════════════════════════════════════════════════════════════════════════
# 2. Scope Gate Classification (keyword-only, no model loads)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 2. Scope Gate Classification (In-Scope vs Out-of-Scope)")
print("=" * 60)

from src.rag import _keyword_scope

# Test queries: manually label expected outcomes
in_scope_queries = [
    "What cafe aesthetic is rising this week?",
    "What kind of latte art gets the most engagement?",
    "What makeup looks are trending?",
    "Which photography styles are going viral?",
    "How should I style food photos for social media?",
    "What fashion aesthetics are popular?",
    "What interior design styles are trending?",
    "What moody photography styles work best?",
    "Are neon signs good for brand shoots?",
    "Which wedding photo presets are most liked?",
    "What nail art colors are in season?",
    "Best lighting for product flat-lays?",
]

out_scope_queries = [
    "Write a python program to print hello world",
    "Explain the code syntax error and fix it",
    "What is the square root of 144?",
    "Who won the football world cup final?",
    "Stock market news today",
    "Weather forecast for tomorrow",
    "How does a car engine work?",
    "Give me interview questions for a job",
    "What is the capital of Australia?",
    "How does photosynthesis work?",
]

y_true = []
y_pred = []

for q in in_scope_queries:
    result = _keyword_scope(q)
    # _keyword_scope returns None if in-scope, a reason string if out-of-scope
    is_rejected = result is not None
    y_true.append(1)   # should be in-scope
    y_pred.append(0 if is_rejected else 1)

for q in out_scope_queries:
    result = _keyword_scope(q)
    is_rejected = result is not None
    y_true.append(0)   # should be out-of-scope
    y_pred.append(0 if is_rejected else 1)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

record("Scope Gate", "accuracy", round(acc, 4), "", f"n={len(y_true)} queries")
record("Scope Gate", "precision", round(prec, 4), "", "in-scope class")
record("Scope Gate", "recall", round(rec, 4), "", "in-scope class")
record("Scope Gate", "f1_score", round(f1, 4), "", "in-scope class")

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
record("Scope Gate", "true_negatives", int(tn), "", "correctly rejected")
record("Scope Gate", "false_positives", int(fp), "", "passed but should reject")
record("Scope Gate", "false_negatives", int(fn), "", "rejected but should pass")
record("Scope Gate", "true_positives", int(tp), "", "correctly passed")

spec = tn / (tn + fp) if (tn + fp) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
record("Scope Gate", "specificity", round(spec, 4), "", "TN/(TN+FP)")
record("Scope Gate", "false_positive_rate", round(fpr, 4), "", "FP/(FP+TN)")

print(f"\n  Classification Report (Scope Gate):")
print(classification_report(y_true, y_pred, target_names=["Out-of-Scope", "In-Scope"], zero_division=0))


# ═════════════════════════════════════════════════════════════════════════════
# 3. RAG Retrieval Metrics
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 3. RAG Retrieval Metrics (Precision@k, Recall@k, F1@k, MRR)")
print("=" * 60)

import faiss
from sentence_transformers import SentenceTransformer

sample_texts = [
    "minimalist latte art on white ceramic cup with natural lighting",
    "rustic wooden brunch spread with flowers and coffee",
    "warm-toned streetwear fashion photography urban background",
    "moody portrait photography with dramatic side lighting",
    "flat lay product styling with pastel colors and dried flowers",
    "neon-lit city nightlife photography with rain reflections",
    "vintage film grain aesthetic with warm amber tones",
    "clean beauty product shot with soft diffused lighting",
    "outdoor adventure photography mountain landscape golden hour",
    "cozy indoor plant corner with warm string lights aesthetic",
    "dark academia aesthetic library books candlelight",
    "cottagecore baking sourdough bread kitchen rustic",
    "brutalist architecture concrete building angular shadows",
    "soft pastel dessert table styling macarons cupcakes",
    "dramatic food photography dark moody styling steak wine",
]

eval_queries = [
    {"query": "What cafe aesthetic is rising this week?", "relevant_ids": [0, 1, 4]},
    {"query": "What kind of latte art gets the most engagement?", "relevant_ids": [0, 6]},
    {"query": "What makeup looks are trending?", "relevant_ids": [7, 4]},
    {"query": "Which photography styles are going viral?", "relevant_ids": [3, 5, 8]},
    {"query": "How should I style food photos for social media?", "relevant_ids": [1, 14, 4]},
    {"query": "What fashion aesthetics are popular?", "relevant_ids": [2, 10, 11]},
    {"query": "What interior design styles are trending?", "relevant_ids": [9, 10, 12]},
    {"query": "What moody photography styles work best?", "relevant_ids": [3, 5, 14]},
]

st_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embs = st_model.encode(sample_texts, normalize_embeddings=True).astype("float32")
dim = chunk_embs.shape[1]

faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(chunk_embs)

record("RAG Retrieval", "corpus_size", len(sample_texts), "documents", "")
record("RAG Retrieval", "embedding_dim", dim, "dims", "")
record("RAG Retrieval", "n_eval_queries", len(eval_queries), "queries", "")

k_values = [1, 3, 5]
all_precision = {k: [] for k in k_values}
all_recall = {k: [] for k in k_values}
all_f1 = {k: [] for k in k_values}
all_mrr = []
all_hit = {k: [] for k in k_values}

for eq in eval_queries:
    q_emb = st_model.encode([eq["query"]], normalize_embeddings=True).astype("float32")
    scores, indices = faiss_index.search(q_emb, max(k_values))

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
        f1_k = 2 * precision_k * recall_k / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0
        hit_k = 1 if rel_k > 0 else 0

        all_precision[k].append(precision_k)
        all_recall[k].append(recall_k)
        all_f1[k].append(f1_k)
        all_hit[k].append(hit_k)

for k in k_values:
    record("RAG Retrieval", f"precision@{k}", round(np.mean(all_precision[k]), 4), "", f"k={k}")
    record("RAG Retrieval", f"recall@{k}", round(np.mean(all_recall[k]), 4), "", f"k={k}")
    record("RAG Retrieval", f"f1@{k}", round(np.mean(all_f1[k]), 4), "", f"k={k}")
    record("RAG Retrieval", f"hit@{k}", round(np.mean(all_hit[k]), 4), "", f"k={k}")

record("RAG Retrieval", "mrr", round(np.mean(all_mrr), 4), "", "Mean Reciprocal Rank")


# ═════════════════════════════════════════════════════════════════════════════
# 4. Lifecycle Classification Metrics
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 4. Lifecycle Classification Metrics")
print("=" * 60)

from src.trends import classify_lifecycle

test_data = pd.DataFrame({
    "cluster_id": list(range(20)),
    "recent_growth": [0.5, 0.3, 0.1, -0.1, -0.3, -0.5, 0.6, 0.25, -0.25, 0.0,
                      0.4, 0.2, -0.2, -0.4, 0.7, 0.15, -0.15, 0.35, -0.6, 0.05],
    "n_posts": [100, 80, 60, 40, 30, 20, 120, 90, 50, 70,
                110, 75, 45, 25, 130, 65, 35, 95, 15, 55],
})

expected_lc = []
for g in test_data["recent_growth"]:
    if g >= 0.25:
        expected_lc.append("Rising")
    elif g <= -0.25:
        expected_lc.append("Declining")
    else:
        expected_lc.append("Stable")

classified = classify_lifecycle(test_data, rising_threshold=0.25, declining_threshold=-0.25)
predicted_lc = classified["lifecycle"].tolist()

acc_lc = accuracy_score(expected_lc, predicted_lc)
prec_lc = precision_score(expected_lc, predicted_lc, average="weighted", zero_division=0)
rec_lc = recall_score(expected_lc, predicted_lc, average="weighted", zero_division=0)
f1_lc = f1_score(expected_lc, predicted_lc, average="weighted", zero_division=0)

record("Lifecycle", "accuracy", round(acc_lc, 4), "", f"n={len(expected_lc)} clusters")
record("Lifecycle", "precision_weighted", round(prec_lc, 4), "", "weighted avg")
record("Lifecycle", "recall_weighted", round(rec_lc, 4), "", "weighted avg")
record("Lifecycle", "f1_weighted", round(f1_lc, 4), "", "weighted avg")

print(f"\n  Classification Report (Lifecycle):")
print(classification_report(expected_lc, predicted_lc, zero_division=0))

cm_lc = confusion_matrix(expected_lc, predicted_lc, labels=["Rising", "Stable", "Declining"])
print("  Confusion Matrix (Lifecycle):")
print(f"  {'':>15} {'Pred Rising':>12} {'Pred Stable':>12} {'Pred Declining':>15}")
for i, label in enumerate(["Rising", "Stable", "Declining"]):
    print(f"  {label:>15} {cm_lc[i][0]:>12} {cm_lc[i][1]:>12} {cm_lc[i][2]:>15}")


# ═════════════════════════════════════════════════════════════════════════════
# Save results
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" Evaluation Complete")
print("=" * 60)

df = pd.DataFrame(results)
out_path = ROOT / "evaluation_results.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved {len(df)} metrics to {out_path}")
print(df.to_string(index=False))

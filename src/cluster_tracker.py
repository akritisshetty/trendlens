"""
cluster_tracker.py
------------------
Stable, incremental cluster tracking for Instagram visual trends.

Solves the "vector drift" problem: instead of re-clustering from scratch
on every pipeline run (which shuffles cluster IDs and breaks time-series
history), this module:

  1. Locks baseline cluster centroids from the initial HDBSCAN run.
  2. Assigns new images to existing clusters via FAISS KNN search.
  3. Detects emerging micro-clusters when images are too far from all
     existing centroids.

Stable cluster IDs persist across runs, enabling genuine time-series
tracking of visual trends.

Usage::

    from src.cluster_tracker import ClusterRegistry

    registry = ClusterRegistry.load()

    # After baseline HDBSCAN:
    registry.init_from_hdbscan(labels, embeddings, metadata)

    # On incremental runs:
    assignments = registry.assign_new_images(new_embeddings)
    emerging = registry.detect_emerging(new_embeddings, new_metadata)
    registry.save()
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

import config

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_ASSIGNMENT_THRESHOLD = 0.25   # cosine similarity min to assign
DEFAULT_MICRO_CLUSTER_MIN = 3         # min unassigned to trigger HDBSCAN
DEFAULT_CENTROID_FAISS_PATH = config.ARTIFACTS_DIR / "centroid_index.faiss"
DEFAULT_REGISTRY_PATH = config.ARTIFACTS_DIR / "cluster_registry.json"


def _new_stable_id() -> str:
    return f"cls_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# ClusterRegistry
# ---------------------------------------------------------------------------
class ClusterRegistry:
    """Persistent registry of clusters with locked centroids and stable IDs.

    Attributes
    ----------
    clusters : dict[str, dict]
        Mapping from stable_id -> cluster record.
    assignment_threshold : float
        Minimum cosine similarity to assign a new image to an existing cluster.
    micro_cluster_min_size : int
        Minimum unassigned images needed to trigger HDBSCAN for new clusters.
    pending_candidates : list[dict]
        Unassigned images waiting to form a new micro-cluster.
    baseline_date : str
        ISO timestamp of the initial baseline run.
    """

    def __init__(
        self,
        clusters: Optional[dict[str, dict]] = None,
        assignment_threshold: float = DEFAULT_ASSIGNMENT_THRESHOLD,
        micro_cluster_min_size: int = DEFAULT_MICRO_CLUSTER_MIN,
        pending_candidates: Optional[list[dict]] = None,
        baseline_date: str = "",
        noise_history: Optional[list[dict]] = None,
    ) -> None:
        self.clusters: dict[str, dict] = clusters or {}
        self.assignment_threshold = assignment_threshold
        self.micro_cluster_min_size = micro_cluster_min_size
        self.pending_candidates: list[dict] = pending_candidates or []
        self.baseline_date = baseline_date or datetime.now(timezone.utc).isoformat()
        self.noise_history: list[dict] = noise_history or []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> None:
        """Save registry to JSON."""
        path = path or DEFAULT_REGISTRY_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "baseline_date": self.baseline_date,
            "assignment_threshold": self.assignment_threshold,
            "micro_cluster_min_size": self.micro_cluster_min_size,
            "n_clusters": len(self.clusters),
            "clusters": {},
            "pending_candidates": self.pending_candidates,
            "noise_history": self.noise_history,
        }
        for sid, rec in self.clusters.items():
            c = dict(rec)
            # Convert ndarray centroid to list for JSON
            if isinstance(c.get("centroid"), np.ndarray):
                c["centroid"] = c["centroid"].tolist()
            data["clusters"][sid] = c

        path.write_text(json.dumps(data, indent=1, default=str))
        print(f"[tracker] saved registry: {len(self.clusters)} clusters, "
              f"{len(self.pending_candidates)} pending candidates → {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ClusterRegistry":
        """Load registry from JSON. Returns empty registry if file missing."""
        path = path or DEFAULT_REGISTRY_PATH
        if not path.exists():
            print("[tracker] no existing registry — will create on first run")
            return cls()

        data = json.loads(path.read_text())
        clusters = data.get("clusters", {})
        # Convert centroid lists back to numpy
        for sid, rec in clusters.items():
            if isinstance(rec.get("centroid"), list):
                rec["centroid"] = np.array(rec["centroid"], dtype="float32")

        return cls(
            clusters=clusters,
            assignment_threshold=data.get("assignment_threshold", DEFAULT_ASSIGNMENT_THRESHOLD),
            micro_cluster_min_size=data.get("micro_cluster_min_size", DEFAULT_MICRO_CLUSTER_MIN),
            pending_candidates=data.get("pending_candidates", []),
            baseline_date=data.get("baseline_date", ""),
            noise_history=data.get("noise_history", []),
        )

    # ------------------------------------------------------------------
    # Baseline initialization from HDBSCAN output
    # ------------------------------------------------------------------
    def init_from_hdbscan(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray,
        metadata: dict[int, dict[str, Any]],
    ) -> None:
        """Create the baseline cluster registry from HDBSCAN results.

        Parameters
        ----------
        labels : np.ndarray
            HDBSCAN labels (-1 = noise, 0..N-1 = clusters).
        embeddings : np.ndarray
            CLIP embeddings, shape (n, dim), L2-normalized.
        metadata : dict[int, dict]
            Per-cluster metadata: {cluster_id: {name, keywords, blip_caption, ...}}.
        """
        self.clusters = {}
        self.pending_candidates = []
        self.baseline_date = datetime.now(timezone.utc).isoformat()

        unique_labels = set(labels.tolist())
        unique_labels.discard(-1)  # ignore noise

        for old_id in sorted(unique_labels):
            mask = labels == old_id
            member_embs = embeddings[mask]
            centroid = member_embs.mean(axis=0)
            # Re-normalize centroid
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            meta = metadata.get(int(old_id), {})
            stable_id = _new_stable_id()

            self.clusters[stable_id] = {
                "stable_id": stable_id,
                "original_hdbscan_id": int(old_id),
                "centroid": centroid.astype("float32"),
                "n_members": int(mask.sum()),
                "first_seen": self.baseline_date,
                "last_updated": self.baseline_date,
                "name": meta.get("name", f"Theme {stable_id}"),
                "keywords": meta.get("keywords", []),
                "blip_caption": meta.get("blip_caption", ""),
                "lifecycle": "New",
                "total_posts_all_time": int(mask.sum()),
            }

        print(f"[tracker] baseline initialized: {len(self.clusters)} clusters "
              f"from {len(labels)} images, {(labels == -1).sum()} noise points")

        # Track noise volume for rupture detection
        n_noise = int((labels == -1).sum())
        self.track_noise(n_noise, len(labels), len(self.clusters))

    # ------------------------------------------------------------------
    # FAISS centroid index
    # ------------------------------------------------------------------
    def build_centroid_index(self, path: Optional[Path] = None) -> Any:
        """Build a FAISS index over locked cluster centroids for fast KNN.

        Returns the FAISS index object.
        """
        import faiss

        path = path or DEFAULT_CENTROID_FAISS_PATH
        if not self.clusters:
            print("[tracker] no clusters to index")
            return None

        stable_ids = list(self.clusters.keys())
        centroids = np.array(
            [self.clusters[sid]["centroid"] for sid in stable_ids],
            dtype="float32",
        )

        dim = centroids.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalized vectors
        index.add(np.ascontiguousarray(centroids))

        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))

        # Also save the stable_id mapping (position in index → stable_id)
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps({
            "stable_ids": stable_ids,
            "n_clusters": len(stable_ids),
            "dim": dim,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }, indent=1))

        print(f"[tracker] centroid FAISS index built: {len(stable_ids)} centroids, dim={dim}")
        return index

    def load_centroid_index(self, path: Optional[Path] = None) -> tuple[Any, list[str]]:
        """Load the centroid FAISS index and stable_id mapping.

        Returns (faiss_index, stable_ids_list).
        """
        import faiss

        path = path or DEFAULT_CENTROID_FAISS_PATH
        meta_path = path.with_suffix(".json")

        if not path.exists() or not meta_path.exists():
            return None, []

        index = faiss.read_index(str(path))
        meta = json.loads(meta_path.read_text())
        stable_ids = meta["stable_ids"]

        return index, stable_ids

    # ------------------------------------------------------------------
    # KNN assignment for new images
    # ------------------------------------------------------------------
    def assign_new_images(
        self,
        new_embeddings: np.ndarray,
        existing_assignments: Optional[np.ndarray] = None,
    ) -> list[dict[str, Any]]:
        """Assign new images to existing clusters via FAISS KNN.

        Parameters
        ----------
        new_embeddings : np.ndarray
            L2-normalized CLIP embeddings of new images, shape (n, dim).
        existing_assignments : np.ndarray, optional
            Pre-existing assignments (from previous runs). If provided,
            already-assigned images are skipped.

        Returns
        -------
        list of dict
            One entry per new image:
            {stable_id, post_idx, similarity, assigned: bool}
        """
        index, stable_ids = self.load_centroid_index()
        if index is None or not stable_ids:
            print("[tracker] no centroid index — all images will be candidates")
            return [
                {"stable_id": None, "post_idx": i, "similarity": 0.0, "assigned": False}
                for i in range(len(new_embeddings))
            ]

        # Search for nearest centroid per new image
        k = min(1, len(stable_ids))  # 1-NN
        scores, indices = index.search(
            np.ascontiguousarray(new_embeddings.astype("float32")), k
        )

        results = []
        for i in range(len(new_embeddings)):
            sim = float(scores[i][0])
            idx = int(indices[i][0])

            if idx < 0 or idx >= len(stable_ids):
                results.append({"stable_id": None, "post_idx": i, "similarity": 0.0, "assigned": False})
                continue

            if sim >= self.assignment_threshold:
                sid = stable_ids[idx]
                results.append({
                    "stable_id": sid,
                    "post_idx": i,
                    "similarity": round(sim, 4),
                    "assigned": True,
                })
                # Update cluster stats
                self.clusters[sid]["n_members"] += 1
                self.clusters[sid]["total_posts_all_time"] += 1
                self.clusters[sid]["last_updated"] = datetime.now(timezone.utc).isoformat()
            else:
                results.append({
                    "stable_id": None,
                    "post_idx": i,
                    "similarity": round(sim, 4),
                    "assigned": False,
                })

        n_assigned = sum(1 for r in results if r["assigned"])
        n_unassigned = len(results) - n_assigned
        print(f"[tracker] assigned {n_assigned}/{len(results)} new images, "
              f"{n_unassigned} unassigned (below threshold {self.assignment_threshold})")

        return results

    # ------------------------------------------------------------------
    # Micro-cluster detection
    # ------------------------------------------------------------------
    def detect_emerging(
        self,
        new_embeddings: np.ndarray,
        new_metadata: list[dict[str, Any]],
        assignments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Detect emerging micro-clusters from unassigned images.

        When enough unassigned images accumulate, runs HDBSCAN on them
        to form new clusters with new stable IDs.

        Returns
        -------
        list of dict
            New cluster records created from emerging micro-clusters.
        """
        import hdbscan

        unassigned_indices = [
            a["post_idx"] for a in assignments if not a["assigned"]
        ]

        if len(unassigned_indices) < self.micro_cluster_min_size:
            print(f"[tracker] {len(unassigned_indices)} unassigned — "
                  f"need {self.micro_cluster_min_size} for micro-cluster detection")
            return []

        unassigned_embs = new_embeddings[unassigned_indices]
        unassigned_meta = [new_metadata[i] for i in unassigned_indices]

        # Run HDBSCAN on unassigned images only
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(2, len(unassigned_indices) // 3),
            min_samples=1,
            metric="euclidean",
            cluster_selection_epsilon=0.6,
        )
        labels = clusterer.fit_predict(
            np.ascontiguousarray(unassigned_embs.astype("float32"))
        )

        new_clusters = []
        unique_labels = set(labels.tolist())
        unique_labels.discard(-1)

        for new_label in sorted(unique_labels):
            mask = labels == new_label
            member_embs = unassigned_embs[mask]
            member_meta = [unassigned_meta[j] for j in range(len(unassigned_meta)) if mask[j]]

            centroid = member_embs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            # Derive name from keywords in member metadata
            all_keywords = []
            for m in member_meta:
                all_keywords.extend(m.get("keywords", []))
            # Deduplicate preserving order
            seen = set()
            unique_kw = []
            for kw in all_keywords:
                if kw.lower() not in seen:
                    seen.add(kw.lower())
                    unique_kw.append(kw)
            name = " ".join(unique_kw[:3]) if unique_kw else "Emerging theme"

            stable_id = _new_stable_id()
            now = datetime.now(timezone.utc).isoformat()

            self.clusters[stable_id] = {
                "stable_id": stable_id,
                "original_hdbscan_id": None,
                "centroid": centroid.astype("float32"),
                "n_members": int(mask.sum()),
                "first_seen": now,
                "last_updated": now,
                "name": name,
                "keywords": unique_kw[:10],
                "blip_caption": "",
                "lifecycle": "Emerging",
                "total_posts_all_time": int(mask.sum()),
            }

            new_clusters.append(self.clusters[stable_id])

        # Remove assigned-to-micro-cluster images from pending
        assigned_to_micro = [i for i, l in zip(range(len(labels)), labels) if l >= 0]
        remaining = [
            c for i, c in enumerate(self.pending_candidates)
            if i not in assigned_to_micro
        ]
        self.pending_candidates = remaining

        if new_clusters:
            print(f"[tracker] detected {len(new_clusters)} emerging micro-clusters "
                  f"from {len(unassigned_indices)} unassigned images")
        return new_clusters

    def add_pending(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Add unassigned images to the pending candidates pool."""
        for i in range(len(embeddings)):
            self.pending_candidates.append({
                "embedding": embeddings[i].tolist() if isinstance(embeddings[i], np.ndarray) else embeddings[i],
                "metadata": metadata[i] if i < len(metadata) else {},
                "added_at": datetime.now(timezone.utc).isoformat(),
            })
        print(f"[tracker] added {len(embeddings)} images to pending pool "
              f"({len(self.pending_candidates)} total)")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_centroids_array(self) -> tuple[np.ndarray, list[str]]:
        """Return (centroids_array, stable_ids) for all clusters."""
        if not self.clusters:
            return np.array([], dtype="float32"), []
        stable_ids = list(self.clusters.keys())
        centroids = np.array(
            [self.clusters[sid]["centroid"] for sid in stable_ids],
            dtype="float32",
        )
        return centroids, stable_ids

    def get_cluster(self, stable_id: str) -> Optional[dict]:
        return self.clusters.get(stable_id)

    def list_clusters(self) -> list[dict]:
        return list(self.clusters.values())

    def summary(self) -> str:
        lines = [f"ClusterRegistry: {len(self.clusters)} clusters, "
                 f"{len(self.pending_candidates)} pending"]
        for sid, rec in self.clusters.items():
            lines.append(f"  {sid}: {rec['name']} ({rec['n_members']} posts, "
                        f"lifecycle={rec.get('lifecycle', '?')})")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Noise tracking & aesthetic rupture detection
    # ------------------------------------------------------------------
    def track_noise(
        self,
        n_noise: int,
        n_total: int,
        n_clusters: int,
    ) -> None:
        """Record noise volume for this run.

        Call this after each pipeline run to build a historical record of
        how many images are classified as HDBSCAN noise (-1 label) vs
        structured cluster members.

        A rising noise-to-cluster ratio signals that the visual landscape
        is shifting away from established trends — an "aesthetic rupture".
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_noise": n_noise,
            "n_total": n_total,
            "n_clusters": n_clusters,
            "noise_ratio": round(n_noise / max(n_total, 1), 4),
        }
        self.noise_history.append(entry)
        print(f"[tracker] noise snapshot: {n_noise}/{n_total} "
              f"({entry['noise_ratio']:.1%}), {n_clusters} clusters")

    def detect_aesthetic_rupture(self, window: int = 3) -> dict[str, Any]:
        """Detect if noise is growing faster than structured clusters.

        Compares the last `window` noise snapshots. If the noise ratio
        is accelerating upward while cluster count is stagnant or
        declining, signals an aesthetic rupture.

        Returns
        -------
        dict with keys:
            is_rupture : bool
            noise_trend : "rising" | "falling" | "stable"
            cluster_trend : "rising" | "falling" | "stable"
            noise_ratio_change : float (latest vs previous)
            message : str (human-readable assessment)
        """
        if len(self.noise_history) < 2:
            return {
                "is_rupture": False,
                "noise_trend": "stable",
                "cluster_trend": "stable",
                "noise_ratio_change": 0.0,
                "message": "Insufficient history for rupture detection (need 2+ runs).",
            }

        recent = self.noise_history[-window:]
        old = self.noise_history[:-(window)] if len(self.noise_history) > window else self.noise_history[:1]

        # Compare average noise ratio
        recent_noise_avg = sum(r["noise_ratio"] for r in recent) / len(recent)
        old_noise_avg = sum(r["noise_ratio"] for r in old) / len(old)
        noise_change = recent_noise_avg - old_noise_avg

        # Compare cluster count
        recent_clusters = [r["n_clusters"] for r in recent]
        old_clusters = [r["n_clusters"] for r in old]
        recent_cluster_avg = sum(recent_clusters) / len(recent_clusters)
        old_cluster_avg = sum(old_clusters) / len(old_clusters)
        cluster_change = recent_cluster_avg - old_cluster_avg

        # Determine trends
        if noise_change > 0.05:
            noise_trend = "rising"
        elif noise_change < -0.05:
            noise_trend = "falling"
        else:
            noise_trend = "stable"

        if cluster_change > 0.5:
            cluster_trend = "rising"
        elif cluster_change < -0.5:
            cluster_trend = "falling"
        else:
            cluster_trend = "stable"

        # Rupture = noise rising + clusters not rising
        is_rupture = (
            noise_trend == "rising"
            and cluster_trend != "rising"
            and noise_change > 0.10
        )

        if is_rupture:
            message = (
                f"AESTHETIC RUPTURE DETECTED: Noise ratio rose "
                f"{noise_change:+.1%} while clusters are {cluster_trend}. "
                f"The visual landscape is shifting away from established trends."
            )
        elif noise_trend == "rising":
            message = (
                f"Noise ratio rising ({noise_change:+.1%}) but clusters still "
                f"growing. Monitor for potential rupture."
            )
        else:
            message = (
                f"Stable. Noise ratio change: {noise_change:+.1%}, "
                f"clusters: {cluster_trend}."
            )

        return {
            "is_rupture": is_rupture,
            "noise_trend": noise_trend,
            "cluster_trend": cluster_trend,
            "noise_ratio_change": round(noise_change, 4),
            "message": message,
        }

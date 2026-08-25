"""
api.py
------
Minimal dependency JSON API for TrendLens (Stage 13 backend).

Runs on Python's stdlib ``http.server`` (Colab / free-tier friendly; no
FastAPI/uvicorn install needed). Exposes honest endpoints over the real
pipeline artifacts:

  GET  /api/health              service status
  POST /api/rag-query           {query} -> {answer, retrievedClusters, ...}
  GET  /api/trends              top trends from trend_metrics.csv
  GET  /api/clusters            all cluster interpretations + metrics
  GET  /api/images?path=...     serve a whitelisted representative image
  GET  /api/live-trends         real Reddit emerging trends (REAL data)
  GET  /api/live-images?name=   serve a downloaded live post image
  POST /api/predict-popularity  honest demo: observed engagement reference,
                                NO fabricated predictions

CORS is wide open for local development. The React frontend's Express
server proxies /api/* here.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import config


def _json(data: Any) -> tuple[bytes, int]:
    return json.dumps(data, indent=1, default=str).encode("utf-8"), 200


def handle_health() -> dict[str, Any]:
    """Service status. Never 500s on missing optional artifacts: reports
    whichever corpus is actually available (legacy SMPD sample or live
    Instagram data) so Instagram-only deployments don't look dead."""
    import pandas as pd

    n_clusters = 0
    dataset = "5K sampled images of 69,226 available"
    timestamps = "neutral synthetic (demo)"
    try:
        metrics = pd.read_csv(config.CLUSTER_METADATA_DIR / "trend_metrics.csv")
        n_clusters = int(len(metrics))
    except Exception:  # noqa: BLE001 — legacy artifacts absent
        from src.rag import load_instagram_trends

        trends = load_instagram_trends()
        if trends:
            n_clusters = int(
                trends.get("n_themes") or len(trends.get("themes") or [])
            )
            dataset = f"real Instagram posts ({int(trends.get('n_posts') or 0)} fetched)"
            timestamps = "real Instagram timestamps"

    return {
        "status": "ok",
        "service": "TrendLens Python backend (FAISS-only, no LLM)",
        "clustersAnalyzed": n_clusters,
        "totalClustersAnalyzed": n_clusters,
        "dataset": dataset,
        "timestamps": timestamps,
        "mode": "faiss-only",
        "llmEnabled": False,
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
    }


def handle_rag_query(payload: dict) -> dict[str, Any]:
    from src import rag

    query = payload.get("query")
    if not query or not isinstance(query, str):
        raise ValueError("Valid query text is required")
    k = payload.get("k", 5)
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 5
    return rag.run_query(query, k=k)


def handle_trends() -> dict[str, Any]:
    import pandas as pd

    from src import rag

    metrics = pd.read_csv(config.CLUSTER_METADATA_DIR / "trend_metrics.csv")
    metrics = metrics.sort_values(
        "trend_score_growth_size_stability", ascending=False
    ).head(20)
    interpretations = {int(i["cluster_id"]): i for i in rag.load_interpretations()}
    reps = rag.load_representatives()
    records = []
    for _, row in metrics.iterrows():
        cid = int(row["cluster_id"])
        it = interpretations.get(cid, {})
        rep_path = (reps.get(cid) or [{}])[0].get("image_path")
        records.append(
            {
                **row.to_dict(),
                "name": it.get("name"),
                "description": it.get("description"),
                "blip_caption": (it.get("sample_captions") or [""])[0],
                "representative_image": rep_path,
                "representative_image_url": rag._image_url(rep_path),
            }
        )
    return {
        "disclaimer": config.SYNTHETIC_DATA_WARNING,
        "trends": records,
    }


def handle_clusters() -> dict[str, Any]:
    import pandas as pd

    from src import rag

    interpretations = {int(i["cluster_id"]): i for i in rag.load_interpretations()}
    reps = rag.load_representatives()
    metrics = pd.read_csv(config.CLUSTER_METADATA_DIR / "trend_metrics.csv")
    records = []
    for _, row in metrics.iterrows():
        cid = int(row["cluster_id"])
        it = interpretations.get(cid, {})
        rep_path = (reps.get(cid) or [{}])[0].get("image_path")
        records.append(
            {
                "cluster_id": cid,
                "name": it.get("name"),
                "description": it.get("description"),
                "characteristics": it.get("characteristics", []),
                "confidence": it.get("confidence"),
                "blip_caption": (it.get("sample_captions") or [""])[0],
                "n_posts": int(row["n_posts"]),
                "lifecycle": row["lifecycle"],
                "average_engagement": float(row["average_engagement"]),
                "recent_growth": float(row["recent_growth"]),
                "trend_score": float(row["trend_score_growth_size_stability"]),
                "representative_image": rep_path,
                "representative_image_url": rag._image_url(rep_path),
            }
        )
    return {
        "disclaimer": config.SYNTHETIC_DATA_WARNING,
        "clusters": records,
    }


_ALLOWED_PATHS_CACHE: set[str] | None = None


def _allowed_representative_paths() -> set[str]:
    """Whitelist of image paths we are willing to serve (representatives only)."""
    global _ALLOWED_PATHS_CACHE
    if _ALLOWED_PATHS_CACHE is not None:
        return _ALLOWED_PATHS_CACHE
    import json as _json

    from src import rag

    reps = rag.load_representatives()
    allowed: set[str] = set()
    for rows in reps.values():
        for r in rows:
            p = r.get("image_path")
            if p:
                allowed.add(str(p))
    _ALLOWED_PATHS_CACHE = allowed
    return allowed


def handle_image(path: str):
    """
    Serve a whitelisted representative image.

    Only paths that appear in ``representatives.json`` are served — no
    arbitrary file access. ``image_path`` values are stored relative to the
    project root (e.g. ``train/59@N75/775.jpg``), so resolution is anchored
    to the project root and then verified to fall under ``IMAGE_ROOT``;
    path traversal is rejected up front.
    """
    import mimetypes

    path = (path or "").strip().lstrip("/")
    if path not in _allowed_representative_paths():
        return None, 404
    rel = path[len("train/"):] if path.startswith("train/") else path
    resolved = (config.IMAGE_ROOT / rel).resolve()
    image_root = config.IMAGE_ROOT.resolve()
    if not str(resolved).startswith(str(image_root)):
        return None, 403
    if not resolved.is_file():
        return None, 404
    ctype = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
    return resolved.read_bytes(), ctype


def handle_predict(payload: dict) -> dict[str, Any]:
    """
    Honest demo: no prediction model exists yet. Returns the observed
    engagement of the requested cluster as a reference and marks everything
    NOT EVALUATED instead of inventing metrics.
    """
    import pandas as pd

    metrics = pd.read_csv(config.CLUSTER_METADATA_DIR / "trend_metrics.csv")
    cluster_id = payload.get("clusterId")
    row = None
    if cluster_id is not None:
        match = metrics[metrics["cluster_id"] == int(cluster_id)]
        if len(match):
            row = match.iloc[0]

    return {
        "clusterId": None if row is None else int(row["cluster_id"]),
        "observedMeanEngagement": None if row is None else float(row["average_engagement"]),
        "observedPostCount": None if row is None else int(row["n_posts"]),
        "lifecycle": None if row is None else row["lifecycle"],
        "predictedLikes": None,
        "predictedComments": None,
        "predictedTotalEngagement": None,
        "nMseScore": None,
        "status": "NOT EVALUATED",
        "note": "Status: NOT EVALUATED. Popularity prediction model is not "
        "implemented. These are observed cluster statistics from the 5K demo "
        "sample (synthetic engagement labels), provided as a reference — no "
        "prediction is made.",
    }


def handle_instagram_trends() -> dict[str, Any]:
    """Real Instagram-sourced emerging trends."""
    from src.rag import load_instagram_trends

    trends = load_instagram_trends()
    if trends is None:
        return {
            "disclaimer": config.INSTAGRAM_DATA_WARNING,
            "error": "No Instagram trends yet — run `python -m src.data_collector` first.",
            "themes": [],
        }
    return trends


def handle_instagram_image(name: str) -> tuple[bytes | None, str | int]:
    """Serve a downloaded Instagram post image."""
    import mimetypes

    name = (name or "").strip()
    if not name or name != Path(name).name:
        return None, 404
    path = (config.INSTAGRAM_IMAGES_DIR / name).resolve()
    img_dir = config.INSTAGRAM_IMAGES_DIR.resolve()
    if not str(path).startswith(str(img_dir)) or not path.is_file():
        return None, 404
    ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return path.read_bytes(), ctype


def handle_live_trends() -> dict[str, Any]:
    """Real Reddit-sourced emerging trends (labelled REAL, unlike SMPD demo)."""
    from src.live import load_trends

    trends = load_trends()
    if trends is None:
        return {
            "disclaimer": config.LIVE_DATA_WARNING,
            "error": "No live trends yet — run `python -m src.live` first.",
            "themes": [],
        }
    return trends


def handle_live_image(name: str) -> tuple[bytes | None, str | int]:
    """
    Serve a downloaded live post image. ``name`` must be a plain basename
    inside ``config.LIVE_IMAGES_DIR`` — no directories, no traversal.
    """
    import mimetypes

    name = (name or "").strip()
    if not name or name != Path(name).name:
        return None, 404
    path = (config.LIVE_IMAGES_DIR / name).resolve()
    live_dir = config.LIVE_IMAGES_DIR.resolve()
    if not str(path).startswith(str(live_dir)) or not path.is_file():
        return None, 404
    ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return path.read_bytes(), ctype


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return  # keep console quiet; optional

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # CORS preflight
        self._send(200, b"")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _serve_image(self, path: str):
        from urllib.parse import parse_qs, urlparse

        query = parse_qs(urlparse(path).query)
        img_path = (query.get("path") or [""])[0]
        body, ctype = handle_image(img_path)
        if body is None:
            msg = json.dumps({"error": "Image not found"}).encode("utf-8")
            self._send(404 if ctype == 404 else 403, msg)
            return
        self._send(200, body, ctype)

    def _serve_live_image(self, path: str):
        from urllib.parse import parse_qs, urlparse

        query = parse_qs(urlparse(path).query)
        name = (query.get("name") or [""])[0]
        body, ctype = handle_live_image(name)
        if body is None:
            self._send(404, json.dumps({"error": "Live image not found"}).encode("utf-8"))
            return
        self._send(200, body, ctype)

    def _serve_instagram_image(self, path: str):
        from urllib.parse import parse_qs, urlparse

        query = parse_qs(urlparse(path).query)
        name = (query.get("name") or [""])[0]
        body, ctype = handle_instagram_image(name)
        if body is None:
            self._send(404, json.dumps({"error": "Instagram image not found"}).encode("utf-8"))
            return
        self._send(200, body, ctype)

    def do_GET(self):
        try:
            if self.path.startswith("/api/instagram-images"):
                self._serve_instagram_image(self.path)
            elif self.path.startswith("/api/instagram-trends"):
                body, code = _json(handle_instagram_trends())
                self._send(code, body)
            elif self.path.startswith("/api/images"):
                self._serve_image(self.path)
            elif self.path.startswith("/api/live-images"):
                self._serve_live_image(self.path)
            elif self.path.startswith("/api/live-trends"):
                body, code = _json(handle_live_trends())
                self._send(code, body)
            elif self.path.startswith("/api/health"):
                body, code = _json(handle_health())
                self._send(code, body)
            elif self.path.startswith("/api/trends"):
                body, code = _json(handle_trends())
                self._send(code, body)
            elif self.path.startswith("/api/clusters"):
                body, code = _json(handle_clusters())
                self._send(code, body)
            else:
                body, code = _json({"error": "Not found"}), 404
                self._send(code, body)
        except Exception as e:  # noqa: BLE001
            self._send(500, _json({"error": f"Internal error: {e} (honest: no data was invented)"})[0])

    def do_POST(self):
        try:
            payload = self._read_json()
            if self.path.startswith("/api/rag-query"):
                body, code = _json(handle_rag_query(payload))
            elif self.path.startswith("/api/predict-popularity"):
                body, code = _json(handle_predict(payload))
            else:
                body, code = _json({"error": "Not found"}), 404
            self._send(code, body)
        except Exception as e:  # noqa: BLE001
            self._send(500, _json({"error": f"Internal error: {e} (honest: no data was invented)"})[0])


def main():
    host = os.environ.get("TRENDLENS_API_HOST", "0.0.0.0")
    port = int(os.environ.get("TRENDLENS_API_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"TrendLens Python backend on http://{host}:{port} (FAISS-only)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

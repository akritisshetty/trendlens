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
import re
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


# ──────────────────────────────────────────────────────────────────────────
# Auth — SQLite-backed accounts (stdlib only)
#   users(id, email UNIQUE, password_hash, salt, name, created_at)
#   sessions(token, email, created_at)
# Passwords: PBKDF2-HMAC-SHA256, 200k iterations, per-user random salt.
# ──────────────────────────────────────────────────────────────────────────

_PBKDF2_ITERATIONS = 200_000
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _auth_conn():
    import sqlite3

    conn = sqlite3.connect(config.AUTH_DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
               id            INTEGER PRIMARY KEY AUTOINCREMENT,
               email         TEXT    UNIQUE NOT NULL,
               password_hash TEXT    NOT NULL,
               salt          TEXT    NOT NULL,
               name          TEXT    NOT NULL DEFAULT '',
               created_at    TEXT    NOT NULL
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS sessions (
               token      TEXT PRIMARY KEY,
               email      TEXT NOT NULL,
               created_at TEXT NOT NULL
           )"""
    )
    conn.commit()
    return conn


def _hash_password(password: str, salt_hex: str) -> str:
    import hashlib

    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex),
        _PBKDF2_ITERATIONS,
    ).hex()


def _issue_session(email: str) -> str:
    import secrets as _secrets
    from datetime import datetime, timezone

    token = _secrets.token_urlsafe(32)
    with _auth_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (token, email, created_at) VALUES (?, ?, ?)",
            (token, email, datetime.now(timezone.utc).isoformat()),
        )
    return token


def handle_auth_signup(payload: dict) -> dict[str, Any]:
    import secrets as _secrets
    from datetime import datetime, timezone

    email = str(payload.get("email") or "").strip().lower()
    password = str(payload.get("password") or "")
    name = str(payload.get("name") or "").strip()[:80]

    if not _EMAIL_RE.match(email):
        return {"status": "error", "error": "Enter a valid email address."}
    if len(password) < 6:
        return {"status": "error", "error": "Password must be at least 6 characters."}

    salt = _secrets.token_bytes(16).hex()
    pw_hash = _hash_password(password, salt)
    now = datetime.now(timezone.utc).isoformat()

    try:
        with _auth_conn() as conn:
            conn.execute(
                "INSERT INTO users (email, password_hash, salt, name, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (email, pw_hash, salt, name, now),
            )
    except Exception:  # noqa: BLE001 — unique constraint on email
        return {
            "status": "exists",
            "error": "That email is already registered — log in instead.",
        }

    return {
        "status": "ok",
        "user": {"email": email, "name": name},
        "token": _issue_session(email),
    }


def handle_auth_login(payload: dict) -> dict[str, Any]:
    email = str(payload.get("email") or "").strip().lower()
    password = str(payload.get("password") or "")

    if not _EMAIL_RE.match(email) or not password:
        return {"status": "error", "error": "Email and password are required."}

    with _auth_conn() as conn:
        row = conn.execute(
            "SELECT password_hash, salt, name FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    if row is None:
        return {"status": "no-user", "error": "No account with that email — sign up first."}
    stored_hash, salt, name = row
    if _hash_password(password, salt) != stored_hash:
        return {"status": "bad-password", "error": "Wrong password for that account."}

    return {
        "status": "ok",
        "user": {"email": email, "name": name},
        "token": _issue_session(email),
    }


def handle_auth_logout(payload: dict) -> dict[str, Any]:
    token = str(payload.get("token") or "")
    if token:
        with _auth_conn() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    return {"status": "ok"}


def handle_feedback(payload: dict) -> dict[str, Any]:
    """
    Email the user's thought/feedback to the project inbox.

    The recipient address lives ONLY on the server (.env) — it is never
    exposed to the frontend. Sending uses Gmail SMTP with an app password:

      TRENDLENS_FEEDBACK_EMAIL     recipient (default: majorproject.2627@gmail.com)
      TRENDLENS_EMAIL_USER         Gmail account used to SEND
      TRENDLENS_EMAIL_APP_PASSWORD Gmail app password (needs 2FA enabled)
    """
    import os
    import re
    import smtplib
    from datetime import datetime, timezone
    from email.message import EmailMessage

    message = str(payload.get("message") or "").strip()
    contact = str(payload.get("contact") or "").strip()
    source = str(payload.get("source") or "website").strip()

    if not message:
        return {"status": "error", "error": "Empty message — nothing to send."}
    if len(message) > 5000:
        message = message[:5000]
    # Reply-To must look like an email if provided; otherwise drop it.
    if contact and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", contact):
        contact = ""

    user = os.environ.get("TRENDLENS_EMAIL_USER", "").strip()
    password = os.environ.get("TRENDLENS_EMAIL_APP_PASSWORD", "").strip()
    recipient = (
        os.environ.get("TRENDLENS_FEEDBACK_EMAIL", "").strip()
        or "majorproject.2627@gmail.com"
    )
    if not user or not password:
        return {
            "status": "email-not-configured",
            "error": "Server cannot send mail yet — SMTP credentials are not set.",
        }

    try:
        msg = EmailMessage()
        msg["From"] = user
        msg["To"] = recipient
        if contact:
            msg["Reply-To"] = contact
        msg["Subject"] = f"TrendLens feedback — {source}"
        msg.set_content(
            f"{message}\n\n"
            f"─────\n"
            f"Sent from the TrendLens {source} form\n"
            f"Reply-to: {contact or '(not provided)'}\n"
            f"Time: {datetime.now(timezone.utc).isoformat()}\n"
        )

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as smtp:
            smtp.login(user, password)
            smtp.send_message(msg)
    except smtplib.SMTPAuthenticationError:
        return {
            "status": "auth-failed",
            "error": "Mail server rejected the credentials — check the app password.",
        }
    except Exception as e:  # noqa: BLE001 — never leak stack traces to clients
        return {"status": "send-failed", "error": f"Could not send right now ({e})."}

    return {"status": "sent"}


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


def handle_instagram_tiles(limit: int = 24) -> dict[str, Any]:
    """
    Tiles for the frontend trend wall: real downloaded Instagram images,
    labelled ONLY from their own post metadata (Apify caption + author +
    account niche). Never labelled with cluster/theme interpretations —
    those describe clusters, not individual images.
    """
    img_dir = config.INSTAGRAM_IMAGES_DIR
    if not img_dir.is_dir():
        return {"tiles": []}

    meta = _load_post_meta()

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )

    tiles: list[dict[str, Any]] = []
    for path in files[:limit]:
        info = meta.get(path.stem) or {}
        caption_title = _caption_title(info.get("caption"))
        author = str(info.get("author") or "").strip()
        tiles.append({
            "id": path.stem,
            # honest label: what THIS post actually says it is
            "title": caption_title or "Post from the live feed",
            "category": _niche_for(author),
            "author": author,
            "url": f"/api/instagram-images?name={path.name}",
        })
    return {"tiles": tiles}


# ── post metadata cache for tiles ────────────────────────────────────────
_POST_META_CACHE: Optional[dict[str, dict[str, str]]] = None


def _load_post_meta() -> dict[str, dict[str, str]]:
    """post_id -> {caption, author} from the collected Apify data."""
    global _POST_META_CACHE
    if _POST_META_CACHE is not None:
        return _POST_META_CACHE
    meta: dict[str, dict[str, str]] = {}
    try:
        import pandas as pd

        df = pd.read_parquet(
            config.PROCESSED_DIR.parent / "instagram" / "all_posts.parquet",
            columns=["post_id", "caption", "author"],
        )
        for row in df.itertuples(index=False):
            meta[str(row.post_id)] = {
                "caption": str(row.caption or ""),
                "author": str(row.author or ""),
            }
    except Exception:  # noqa: BLE001 — tiles degrade to generic labels
        meta = {}
    _POST_META_CACHE = meta
    return meta


def _caption_title(caption: str, limit: int = 52) -> Optional[str]:
    """Short honest headline from the post's own caption."""
    import re

    text = re.sub(r"\s+", " ", caption or "").strip()
    text = re.sub(r"#\w+", "", text)          # drop hashtags
    text = re.sub(r"@\w+", "", text)          # drop mentions
    text = text.strip(" \n.#@-–—|*")
    text = re.sub(r"\s{2,}", " ", text).strip()
    if not text:
        return None
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:-–—") + "…"
    return text


_ACCOUNT_NICHES = {
    # coffee
    "baristamagazine": "Coffee", "sprudge": "Coffee",
    "tannercolsoncoffee": "Coffee", "baristadaz": "Coffee",
    "barista_jennyeah": "Coffee", "frenchpress.latteart": "Coffee",
    "latteartshow": "Coffee", "coffee": "Coffee",
    "jwc.coffeeacademy": "Coffee", "dnacoffee_improvementhub": "Coffee",
    # desserts / baking
    "chelsweets": "Desserts", "milkbarstore": "Desserts",
    "dominiqueansel": "Desserts", "thesweetimpact": "Desserts",
    "crumblcookies": "Desserts", "tartinebaker": "Baking",
    "theperfectloaf": "Baking", "halfbakedharvest": "Baking",
    "flatlays": "Food styling",
    # fashion
    "voguemagazine": "Fashion", "voguerunway": "Fashion",
    "highsnobiety": "Fashion", "styledumonde": "Fashion",
    "matildadjerf": "Fashion", "tokyofashion": "Fashion",
    # photography
    "natgeo": "Photography", "magnumphotos": "Photography",
    "jordi.koalitic": "Photography", "alan_schaller": "Photography",
    "moodygrams": "Photography",
    # beauty
    "hudabeauty": "Beauty", "rarebeauty": "Beauty",
    "glossier": "Beauty", "ctilburymakeup": "Beauty",
    "theordinary": "Skincare",
}

_NICHE_KEYWORDS = [
    ("coffee", "Coffee"), ("barista", "Coffee"), ("latte", "Coffee"),
    ("sweet", "Desserts"), ("bake", "Baking"), ("cake", "Desserts"),
    ("dessert", "Desserts"), ("cookie", "Desserts"),
    ("vogue", "Fashion"), ("fashion", "Fashion"), ("style", "Fashion"),
    ("wear", "Fashion"), ("outfit", "Fashion"),
    ("photo", "Photography"), ("lens", "Photography"), ("shutter", "Photography"),
    ("beauty", "Beauty"), ("makeup", "Beauty"), ("skin", "Skincare"),
]


def _niche_for(author: str) -> str:
    """Category from the ACCOUNT a post came from (never guessed from pixels)."""
    a = (author or "").lower().strip()
    if a in _ACCOUNT_NICHES:
        return _ACCOUNT_NICHES[a]
    for kw, niche in _NICHE_KEYWORDS:
        if kw in a:
            return niche
    return "Food" if a else "Instagram"


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
            elif self.path.startswith("/api/instagram-tiles"):
                body, code = _json(handle_instagram_tiles())
                self._send(code, body)
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
            elif self.path.startswith("/api/auth/signup"):
                body, code = _json(handle_auth_signup(payload))
            elif self.path.startswith("/api/auth/login"):
                body, code = _json(handle_auth_login(payload))
            elif self.path.startswith("/api/auth/logout"):
                body, code = _json(handle_auth_logout(payload))
            elif self.path.startswith("/api/feedback"):
                body, code = _json(handle_feedback(payload))
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

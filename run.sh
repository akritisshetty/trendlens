#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# TrendLens — One-shot launcher
# Runs the data pipeline (incremental by default), starts backend + frontend,
# and opens the browser.
#
# Usage:
#   ./run.sh              # incremental pipeline (default), 10-day window
#   ./run.sh 7            # incremental pipeline, 7-day window
#   ./run.sh --baseline   # full re-cluster from scratch, 10-day window
#   ./run.sh --fast       # SKIP data pipeline — start servers immediately on existing data
#
# Speed/cost knobs (put in .env):
#   TRENDLENS_MAX_ACCOUNTS=15        # fetch fewer accounts (of 81)
#   TRENDLENS_POSTS_PER_ACCOUNT=10   # default is 50 — biggest speed lever
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

# ── Parse arguments ──────────────────────────────────────────────────────────
DAYS=10
BASELINE_FLAG=""
FAST=0

for arg in "$@"; do
  case "$arg" in
    --baseline) BASELINE_FLAG="--baseline" ;;
    --fast)     FAST=1 ;;
    [0-9]*)     DAYS="$arg" ;;
    *)          echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ── Kill stale processes on our ports ────────────────────────────────────────
for port in 8000 3000 24678; do
  pids=$(lsof -ti:"$port" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "$pids" | xargs kill -9 2>/dev/null || true
  fi
done

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { printf "${CYAN}[trendlens]${NC} %s\n" "$*"; }
ok()    { printf "${GREEN}[trendlens]${NC} %s\n" "$*"; }
warn()  { printf "${YELLOW}[trendlens]${NC} %s\n" "$*"; }
die()   { printf "${RED}[trendlens]${NC} %s\n" "$*" >&2; exit 1; }

# ── Preflight checks (auto-install anything missing) ────────────────────────
if [ ! -f venv/bin/activate ]; then
  warn "Python venv not found — creating one ..."
  python3 -m venv venv || die "Could not create venv. Install python3-venv and retry."
fi

source venv/bin/activate

if ! python -c "import fastcluster, faiss, transformers" >/dev/null 2>&1; then
  warn "Python dependencies missing — installing requirements.txt (this can take a few minutes) ..."
  pip install -q --upgrade pip
  pip install -q -r requirements.txt || die "pip install failed — see output above."
fi

if [ ! -d frontend/node_modules ]; then
  warn "Frontend deps not found — running npm install ..."
  (cd frontend && npm install) || die "npm install failed — see output above."
fi

[ -f .env ] || { warn ".env not found — copying from .env.example"; cp .env.example .env 2>/dev/null \
  || warn "No .env or .env.example — Instagram scraping will fail until APIFY_API_TOKEN is set."; }

# ── Step 1: Run the data pipeline (skipped with --fast; non-fatal otherwise) ─
if [ "$FAST" -eq 1 ]; then
  info "Skipping data pipeline (--fast) — using previously collected data."
else
  if [ -n "$BASELINE_FLAG" ]; then
    info "Running data BASELINE (${DAYS}-day window) — full re-cluster ..."
    if ! python -m src.data_collector --days "$DAYS" --baseline; then
      warn "Pipeline failed (Apify quota/payment or network?) — continuing with previously collected data."
    fi
  else
    info "Running data pipeline INCREMENTAL (${DAYS}-day window) ..."
    if ! python -m src.data_collector --days "$DAYS"; then
      warn "Pipeline failed (Apify quota/payment or network?) — continuing with previously collected data."
    fi
  fi
fi
echo

# ── Step 2: Start the Python backend (port 8000) ────────────────────────────
info "Starting Python backend on :8000 ..."
python -m src.api &
API_PID=$!
trap "kill $API_PID 2>/dev/null || true" EXIT

# Wait for backend to be ready
for i in $(seq 1 15); do
    if curl -sf http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
        ok "Backend ready"
        break
    fi
    sleep 1
done

# ── Step 3: Start the React frontend (port 3000) ────────────────────────────
info "Starting React frontend on :3000 ..."
cd frontend
npx tsx server.ts &
FRONTEND_PID=$!
trap "kill $API_PID $FRONTEND_PID 2>/dev/null || true" EXIT
cd ..

# Wait for frontend to be ready
for i in $(seq 1 15); do
    if curl -sf http://127.0.0.1:3000 >/dev/null 2>&1; then
        ok "Frontend ready"
        break
    fi
    sleep 1
done

# ── Step 4: Open browser ────────────────────────────────────────────────────
URL="http://localhost:3000"
info "Opening $URL ..."
if command -v xdg-open   >/dev/null 2>&1; then xdg-open "$URL"   2>/dev/null &
elif command -v open      >/dev/null 2>&1; then open "$URL"       2>/dev/null &
elif command -v wslview    >/dev/null 2>&1; then wslview "$URL"   2>/dev/null &
else warn "Could not detect browser command — open $URL manually"; fi

echo
ok "═══════════════════════════════════════════════════════════════"
ok "  TrendLens is running!"
ok "  Frontend:  http://localhost:3000"
ok "  Backend:   http://localhost:8000/api/health"
ok "  Press Ctrl+C to stop both servers."
ok "═══════════════════════════════════════════════════════════════"

# Keep script alive until Ctrl+C
wait

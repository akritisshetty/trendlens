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
#   ./run.sh 7 --baseline # full re-cluster, 7-day window
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

# ── Parse arguments ──────────────────────────────────────────────────────────
DAYS=10
BASELINE_FLAG=""

for arg in "$@"; do
  case "$arg" in
    --baseline) BASELINE_FLAG="--baseline" ;;
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

# ── Preflight checks ────────────────────────────────────────────────────────
[ -f venv/bin/activate ]          || die "Python venv not found. Run:  python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
[ -f frontend/node_modules/.package-lock.json ] || die "Frontend deps not installed. Run:  cd frontend && npm install"
[ -f .env ]                       || { warn ".env not found — copying from .env.example"; cp .env.example .env; }

source venv/bin/activate

# ── Step 1: Run the data pipeline ────────────────────────────────────────────
if [ -n "$BASELINE_FLAG" ]; then
  info "Running data BASELINE (${DAYS}-day window) — full re-cluster ..."
  python -m src.data_collector --days "$DAYS" --baseline
else
  info "Running data pipeline INCREMENTAL (${DAYS}-day window) ..."
  python -m src.data_collector --days "$DAYS"
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

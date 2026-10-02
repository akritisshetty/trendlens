#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# TrendLens — one-command pipeline, evaluation, and (optionally) live stack.
#
#   ./run.sh                 offline pipeline + real-data evaluation
#                            (safe & idempotent: reuses cached embeddings)
#   ./run.sh --fetch         ALSO pull fresh posts from Apify first
#                            (requires APIFY_API_TOKEN in .env)
#   ./run.sh --embed         also re-run CLIP embeddings over local images
#   ./run.sh --sweep         also re-run the hyperparameter sweep
#   ./run.sh --baselines     also re-run the (legacy) baseline comparison
#   ./run.sh --server        after the pipeline, launch backend :8000 + frontend :3000
#   ./run.sh --skip-rebuild  do not regenerate trends.json / RAG / history
#   ./run.sh --skip-eval     do not run the evaluation harness
#   ./run.sh all             shorthand for --fetch --embed --sweep --baselines
#   ./run.sh --help
#
# Order: venv → (fetch) → (embed) → rebuild → evaluate → (sweep) → (baselines)
#        → (server). Every stage prints what it did and what it skipped.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"

FETCH=0 EMBED=0 SWEEP=0 BASELINES=0 SERVER=0 SKIP_REBUILD=0 SKIP_EVAL=0

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

for arg in "$@"; do
  case "$arg" in
    --fetch) FETCH=1 ;;
    --embed) EMBED=1 ;;
    --sweep) SWEEP=1 ;;
    --baselines) BASELINES=1 ;;
    --server) SERVER=1 ;;
    --skip-rebuild) SKIP_REBUILD=1 ;;
    --skip-eval) SKIP_EVAL=1 ;;
    all) FETCH=1; EMBED=1; SWEEP=1; BASELINES=1 ;;
    -h|--help|help) usage ;;
    *) echo "run.sh: unknown argument '$arg'" >&2; usage >&2; exit 1 ;;
  esac
done

ask() { printf '%s [y/N] ' "$1"; read -r a; [[ "$a" =~ ^([yY]|[yY][eE][sS])$ ]]; }

# ── Stage 0: virtualenv ─────────────────────────────────────────────────────
if [ ! -x venv/bin/python ]; then
  echo "── Stage 0: creating venv + installing requirements ──"
  python3 -m venv venv
  venv/bin/pip install -q -r requirements.txt
fi
source venv/bin/activate

has_apify_token() {
  [ -n "${APIFY_API_TOKEN:-}" ] && return 0
  [ -f .env ] && grep -qE '^[[:space:]]*APIFY_API_TOKEN=[^[:space:]]' .env && return 0
  return 1
}

# ── Stage 1: fetch ──────────────────────────────────────────────────────────
if [ "$FETCH" -eq 1 ]; then
  if has_apify_token; then
    echo "── Stage 1: fetching fresh Instagram posts via Apify ──"
    days="${TRENDLENS_INSTAGRAM_DAYS:-10}"
    python -m src.data_collector --days "$days"
    # The collector already rebuilt trends.json + history snapshot, so the
    # separate rebuild stage below is a no-op re-run (same corpus → rejected).
    SKIP_REBUILD=1
  else
    echo "── Stage 1: --fetch requested but APIFY_API_TOKEN is not set ──" >&2
    echo "    Add it to .env (see .env.example) or drop --fetch." >&2
    exit 1
  fi
else
  echo "── Stage 1: skipping fetch (no --fetch; reusing existing posts) ──"
fi

# ── Stage 2: embeddings ─────────────────────────────────────────────────────
if [ "$EMBED" -eq 1 ]; then
  if [ -d data/instagram/images ]; then
    echo "── Stage 2: re-embedding local images with CLIP ──"
    python scripts/rebuild_embeddings.py
  else
    echo "── Stage 2: --embed requested but no local images ──" >&2
    echo "    Run --fetch first (live downloads) or add images to data/instagram/images." >&2
    exit 1
  fi
else
  if [ -f data/instagram/embeddings.npy ] && [ -f data/instagram/embed_meta.parquet ]; then
    echo "── Stage 2: reusing cached embeddings (n=$(python - <<'PY'
import numpy as np
try:
    print(np.load('data/instagram/embeddings.npy').shape[0])
except Exception:
    print('?')
PY
)) ──"
  else
    echo "── Stage 2: no cached embeddings — running CLIP over local images ──"
    python scripts/rebuild_embeddings.py
  fi
fi

# ── Stage 3: rebuild production artifacts ───────────────────────────────────
if [ "$SKIP_REBUILD" -ne 1 ]; then
  echo "── Stage 3: regenerating trends.json, RAG index, history snapshots ──"
  python scripts/rebuild_trends.py
else
  echo "── Stage 3: skipping rebuild (kept: trends.json / RAG / history) ──"
fi

# ── Stage 4: real-data evaluation ───────────────────────────────────────────
if [ "$SKIP_EVAL" -ne 1 ]; then
  echo "── Stage 4: real-data evaluation vs baselines (evaluate_real.py) ──"
  python scripts/evaluate_real.py
else
  echo "── Stage 4: skipping evaluation ──"
fi

# ── Stage 5: optional sweeps ────────────────────────────────────────────────
if [ "$SWEEP" -eq 1 ]; then
  echo "── Stage 5: hyperparameter sweep (clustering + retrieval) ──"
  python scripts/sweep_hyperparameters.py
else
  echo "── Stage 5: skipping hyperparameter sweep (--sweep to run) ──"
fi

if [ "$BASELINES" -eq 1 ]; then
  echo "── Stage 6: legacy baseline comparison (pre-sweep hyperparameters) ──"
  python scripts/compare_trend_baselines.py
else
  echo "── Stage 6: skipping legacy baseline comparison (--baselines to run) ──"
fi

# ── Report ──────────────────────────────────────────────────────────────────
echo
echo "──────────────────────────────────────────────────────────────"
echo " Pipeline complete. Summary:"
python - <<'PY'
import json
try:
    t = json.load(open("data/instagram/trends.json"))
    themes = t.get("themes") or []
    rising = [x for x in themes if x.get("classification") == "Rising"]
    print(f"  trends.json       : {len(themes)} themes, {len(rising)} Rising, "
          f"{t.get('n_insufficient', 0)} InsufficientData, {t.get('n_posts', 0)} posts")
    for th in sorted(rising, key=lambda x: -float(x.get("emerging_score", 0)))[:3]:
        print(f"      rising: {th.get('name')} (priority {float(th.get('emerging_score', 0)):.2f})")
except FileNotFoundError:
    print("  trends.json      : (missing — run without --skip-rebuild)")
try:
    from src.trend_definition import TrendHistory
    st = TrendHistory().history_status()
    print(f"  history store    : {st['n_snapshots']} snapshots, "
          f"longitudinal {st['supports_longitudinal']}")
except Exception:
    print("  history store    : (unavailable)")
try:
    import csv
    rows = list(csv.DictReader(open("evaluation_real_results.csv")))
    for r in [x for x in rows if x.get("metric") == "spearman_pred_vs_actual"][:3]:
        print(f"  eval             : {r['detector']:<32} spearman={r['value']}")
except FileNotFoundError:
    print("  eval             : (missing — run without --skip-eval)")
PY
echo
echo "  Evaluation detail : evaluation_real_results.csv"
echo "  Trend data        : data/instagram/trends.json"
echo "  History           : data/trend_history.db"
echo "──────────────────────────────────────────────────────────────"

# ── Stage 7: optional live stack ────────────────────────────────────────────
if [ "$SERVER" -eq 1 ]; then
  echo "── Stage 7: launching backend :8000 + frontend :3000 ──"
  exec scripts/run_all.sh
fi

echo "Done. Add --server to launch the live stack, or ./scripts/run_all.sh."
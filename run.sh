#!/usr/bin/env bash
# Starts both AI workers (CLIP and Gemma).
#
# Usage:
#   ./run.sh           # production mode
#   ./run.sh --dev     # dev mode — enables /docs and /openapi.yaml on each worker
#
# Environment variables (optional overrides):
#   CLIP_PORT       default 8001
#   GEMMA_PORT      default 8002
#   CLIP_MODEL      default openai/clip-vit-base-patch32
#   GEMMA_MODEL     default google/gemma-4-26B-A4B-it
#   PYTHON          default python3 (or .venv/bin/python if venv exists)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CLIP_PORT="${CLIP_PORT:-8001}"
GEMMA_PORT="${GEMMA_PORT:-8002}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-base-patch32}"
GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-4-26B-A4B-it}"

DEV_FLAG=""
for arg in "$@"; do
  [ "$arg" = "--dev" ] && DEV_FLAG="--dev"
done

# ── Python interpreter & venv bootstrap ──────────────────────────────────────

# Keep venv on the Linux filesystem — /mnt/c blocks certain file ops in WSL
VENV="${STYLORA_VENV:-$HOME/.venvs/stylora-ai}"

if command -v uv &>/dev/null; then
  # Fast path: uv handles venv + pip with no system-package headaches
  if [ ! -f "$VENV/bin/python" ]; then
    echo "Creating virtual environment (uv)..."
    uv venv "$VENV"
  fi
  PYTHON="$VENV/bin/python"
  echo "Installing dependencies..."
  VIRTUAL_ENV="$VENV" UV_LINK_MODE=copy uv pip install --quiet -r "$SCRIPT_DIR/clothing-validation/requirements.txt"
  VIRTUAL_ENV="$VENV" UV_LINK_MODE=copy uv pip install --quiet -r "$SCRIPT_DIR/outfit-chat/requirements.txt"
else
  # Fallback: system python + pip --user (no venv)
  if command -v python3 &>/dev/null; then
    PYTHON="python3"
  else
    echo "Error: python3 not found" >&2; exit 1
  fi
  if ! "$PYTHON" -m pip --version &>/dev/null 2>&1; then
    echo "Error: pip not found. Install uv (recommended): curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
  fi
  echo "Installing dependencies (pip --user)..."
  "$PYTHON" -m pip install --quiet --user -r "$SCRIPT_DIR/clothing-validation/requirements.txt"
  "$PYTHON" -m pip install --quiet --user -r "$SCRIPT_DIR/outfit-chat/requirements.txt"
fi
echo "Dependencies ready."

# ── Graceful shutdown ─────────────────────────────────────────────────────────

CLIP_PID=""
GEMMA_PID=""

cleanup() {
  echo ""
  echo "Shutting down workers..."
  [ -n "$CLIP_PID" ]  && kill "$CLIP_PID"  2>/dev/null && echo "  CLIP stopped"
  [ -n "$GEMMA_PID" ] && kill "$GEMMA_PID" 2>/dev/null && echo "  Gemma stopped"
  exit 0
}

trap cleanup INT TERM

# ── Start workers ─────────────────────────────────────────────────────────────

echo "Starting CLIP embedding worker on port $CLIP_PORT..."
"$PYTHON" "$SCRIPT_DIR/clothing-validation/clip_image_embedding_worker.py" \
  --port "$CLIP_PORT" \
  --model-id "$CLIP_MODEL" \
  $DEV_FLAG &
CLIP_PID=$!

echo "Starting Gemma intent worker on port $GEMMA_PORT..."
"$PYTHON" "$SCRIPT_DIR/outfit-chat/gemma_intent_worker.py" \
  --port "$GEMMA_PORT" \
  --model-id "$GEMMA_MODEL" \
  $DEV_FLAG &
GEMMA_PID=$!

# ── Print URLs ────────────────────────────────────────────────────────────────

echo ""
echo "Workers started (Gemma model load takes a few minutes)"
echo ""
echo "  CLIP   health:  http://localhost:$CLIP_PORT/health"
echo "  Gemma  health:  http://localhost:$GEMMA_PORT/health"

if [ -n "$DEV_FLAG" ]; then
  echo ""
  echo "  CLIP   Swagger: http://localhost:$CLIP_PORT/docs"
  echo "  Gemma  Swagger: http://localhost:$GEMMA_PORT/docs"
fi

echo ""
echo "Press Ctrl+C to stop both workers."

# ── Wait ──────────────────────────────────────────────────────────────────────

wait "$CLIP_PID" "$GEMMA_PID"

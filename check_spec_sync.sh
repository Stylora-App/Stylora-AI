#!/usr/bin/env bash
# Tests that the running AI workers match their OpenAPI specs using schemathesis.
#
# Requirements:
#   pip install schemathesis
#
# Usage:
#   ./check_spec_sync.sh                          # test both workers at default ports
#   CLIP_URL=http://localhost:9001 ./check_spec_sync.sh  # custom ports
#
# The workers must already be running with --dev (so /openapi.yaml is served).
# Start them with:
#   python clothing-validation/clip_image_embedding_worker.py --port 8001 --model-id openai/clip-vit-base-patch32 --dev
#   python outfit-chat/gemma_intent_worker.py --port 8002 --dev
set -euo pipefail

CLIP_URL="${CLIP_URL:-http://localhost:8001}"
GEMMA_URL="${GEMMA_URL:-http://localhost:8002}"
FAILED=0

# ── Dependency check ──────────────────────────────────────────────────────────

if ! command -v schemathesis &>/dev/null; then
  echo "schemathesis not found. Installing..."
  pip install schemathesis --quiet
fi

# ── Reachability check ────────────────────────────────────────────────────────

wait_ready() {
  local url="$1"
  local name="$2"
  local attempts=10
  echo "Waiting for $name at $url/health..."
  for i in $(seq 1 $attempts); do
    if curl -sf "$url/health" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('ready') else 1)" 2>/dev/null; then
      echo "  OK: $name is ready"
      return 0
    fi
    sleep 1
  done
  echo "  ERROR: $name did not become ready in ${attempts}s" >&2
  return 1
}

wait_ready "$CLIP_URL"  "CLIP worker"  || { FAILED=1; }
wait_ready "$GEMMA_URL" "Gemma worker" || { FAILED=1; }

[ "$FAILED" -eq 0 ] || { echo "One or more workers are not reachable. Aborting." >&2; exit 1; }

# ── Spec sync checks ──────────────────────────────────────────────────────────

run_check() {
  local name="$1"
  local spec_url="$2"
  local base_url="$3"

  echo ""
  echo "==> Checking $name spec sync..."
  echo "    spec:   $spec_url"
  echo "    target: $base_url"

  # --checks all         → validate status codes, response schemas, headers
  # --validate-schema    → also validate the spec itself
  # --hypothesis-phases  → use only explicit examples (fast, no fuzzing)
  if schemathesis run "$spec_url" \
      --base-url "$base_url" \
      --checks all \
      --validate-schema true \
      --hypothesis-phases explicit \
      2>&1; then
    echo "    OK: $name in sync with spec"
  else
    echo "    FAIL: $name has diverged from spec" >&2
    FAILED=1
  fi
}

run_check "CLIP"  "$CLIP_URL/openapi.yaml"  "$CLIP_URL"
run_check "Gemma" "$GEMMA_URL/openapi.yaml" "$GEMMA_URL"

# ── Result ────────────────────────────────────────────────────────────────────

echo ""
if [ "$FAILED" -eq 0 ]; then
  echo "All workers are in sync with their specs."
else
  echo "Spec sync check failed." >&2
  exit 1
fi

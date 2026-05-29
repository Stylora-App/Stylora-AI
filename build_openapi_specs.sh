#!/usr/bin/env bash
# Validates all OpenAPI spec files in docs/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCS_DIR="$SCRIPT_DIR/docs"
FAILED=0

validate_yaml() {
  local file="$1"
  if python3 -c "
import sys, yaml
try:
    doc = yaml.safe_load(open('$file'))
    if not isinstance(doc, dict) or 'openapi' not in doc or 'paths' not in doc:
        print('  missing required openapi/paths keys')
        sys.exit(1)
except yaml.YAMLError as e:
    print(f'  {e}')
    sys.exit(1)
" 2>&1; then
    echo "  OK"
  else
    FAILED=1
  fi
}

echo "Validating OpenAPI specs..."
echo ""

for spec in \
  "$DOCS_DIR/openapi.yaml" \
  "$DOCS_DIR/clip-openapi.yaml" \
  "$DOCS_DIR/gemma-openapi.yaml"; do
  echo ">> $spec"
  validate_yaml "$spec"
done

echo ""
if [ "$FAILED" -eq 0 ]; then
  echo "All specs valid."
else
  echo "One or more specs are invalid." >&2
  exit 1
fi

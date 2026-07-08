#!/usr/bin/env bash
# P5 deliverable 1: re-run EVERY P1/P2/P2.5 acceptance check INSIDE the pulled image.
# The image bundles python3 + agent + deps but not scripts/ or dataset.json, so we
# mount them read-only and override the entrypoint. This proves the packaged runtime
# (not just the host venv) passes the same acceptance suite. Self-contained: the
# suites spin their own mock Fireworks on 127.0.0.1 and force the stub, no network.
#
# Run AFTER build_and_size.sh has pulled the image. Override IMAGE= to match.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-localhost:5000/floor-cl:test}"
PLATFORM="linux/amd64"

for mod in scripts.acceptance_p1 scripts.acceptance_p2 scripts.acceptance_p25; do
  echo "== in-container: $mod =="
  docker run --rm --platform "$PLATFORM" \
    --entrypoint python3 \
    -e PYTHONPATH=/app \
    -e AGENT_FORCE_STUB=1 \
    -v "$ROOT/scripts:/app/scripts:ro" \
    -v "$ROOT/dataset.json:/app/dataset.json:ro" \
    "$IMAGE" -m "$mod"
done

echo "== PASS: P1 + P2 + P2.5 acceptance all green inside the container =="

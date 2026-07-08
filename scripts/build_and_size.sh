#!/usr/bin/env bash
# P4 acceptance: build -> assert <10GB compressed -> push/pull public registry
# -> inspect linux/amd64 manifest -> smoke-run the PULLED image BOTH ways.
# Override IMAGE=<your public repo> to use a real registry; default spins a
# throwaway local registry so this runs end-to-end offline.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-localhost:5000/floor-cl:test}"
PLATFORM="linux/amd64"
MAX_BYTES=$((10 * 1000 * 1000 * 1000))   # 10 GB compressed
LOCAL_REG=""

cleanup() { [ -n "$LOCAL_REG" ] && docker rm -f "$LOCAL_REG" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# 1. build
echo "== build ($PLATFORM) =="
docker build --platform "$PLATFORM" -t "$IMAGE" "$ROOT"

# 2. compressed size gate
echo "== compressed size =="
BYTES=$(docker save "$IMAGE" | gzip -c | wc -c)
printf 'compressed=%s bytes (%.2f GB)\n' "$BYTES" "$(echo "scale=2;$BYTES/1000000000"|bc)"
[ "$BYTES" -lt "$MAX_BYTES" ] || { echo "FAIL: image >= 10GB compressed"; exit 1; }

# 3. push + 4. pull (spin a local registry if IMAGE points at localhost:5000)
if [ "${IMAGE#localhost:5000/}" != "$IMAGE" ]; then
  LOCAL_REG=$(docker run -d -p 5000:5000 --name floorcl-reg registry:2)
  sleep 1
fi
echo "== push/pull =="
docker push "$IMAGE"
docker rmi "$IMAGE" >/dev/null 2>&1 || true
docker pull --platform "$PLATFORM" "$IMAGE"

# 4b. manifest includes linux/amd64
echo "== manifest inspect =="
docker image inspect "$IMAGE" --format '{{.Os}}/{{.Architecture}}' | grep -qx "linux/amd64" \
  || { echo "FAIL: no linux/amd64 manifest"; exit 1; }

# 5. smoke-run the PULLED image both ways
smoke() {
  local mode="$1" enabled="$2"
  local work; work="$(mktemp -d)"
  mkdir -p "$work/input" "$work/output"
  cat > "$work/input/tasks.json" <<'JSON'
[{"task_id":"t1","prompt":"What is 2 + 2?"},
 {"task_id":"t2","prompt":"Summarize: the cat sat on the mat."}]
JSON
  # Self-contained smoke: m=0/B=0 disables the remote path (zero tokens, no net).
  # awk, not pyyaml, so this runs on any build host. Flips only the enabled key
  # inside the local_candidate block (batching.enabled is left alone).
  local en=false; [ "$enabled" = "1" ] && en=true
  awk -v en="$en" '
    /^local_candidate:/ {inlc=1}
    /^[^ ]/ && !/^local_candidate:/ {inlc=0}
    /^token_budget:/ {print "token_budget: 0"; next}
    /^mandatory_remote:/ {print "mandatory_remote: 0"; next}
    inlc && /^  enabled:/ {print "  enabled: " en; next}
    {print}
  ' "$ROOT/agent/config.yaml" > "$work/config.yaml"
  echo "== smoke: $mode (enabled=$enabled) =="
  docker run --rm --platform "$PLATFORM" \
    -e CONFIG_PATH=/cfg/config.yaml \
    ${SMOKE_FORCE_STUB:+-e AGENT_FORCE_STUB=1} \
    -v "$work/input:/input:ro" -v "$work/output:/output" \
    -v "$work/config.yaml:/cfg/config.yaml:ro" \
    "$IMAGE"
  python3 -c "import json,sys; d=json.load(open('$work/output/results.json')); \
    ids={x['task_id'] for x in d}; assert ids=={'t1','t2'}, ids; print('  valid results.json', ids)"
  rm -rf "$work"
}
smoke "Floor-C"  0
smoke "Floor-CL" 1

echo "== PASS: build_and_size =="

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

docker_host_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -w "$1"
  else
    printf '%s' "$1"
  fi
}

cleanup() { [ -n "$LOCAL_REG" ] && docker rm -f "$LOCAL_REG" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# 1. build
echo "== build ($PLATFORM) =="
docker build --platform "$PLATFORM" -t "$IMAGE" "$ROOT"

# 2. compressed size gate
echo "== compressed size =="
BYTES=$(docker save "$IMAGE" | gzip -c | wc -c)
GB="$(awk -v bytes="$BYTES" 'BEGIN { printf "%.2f", bytes / 1000000000 }')"
printf 'compressed=%s bytes (%s GB)\n' "$BYTES" "$GB"
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
 {"task_id":"t2","prompt":"What is a GPU?"}]
JSON
  # Self-contained smoke: m=0/B=0 disables the remote path (zero tokens, no net).
  # awk, not pyyaml, so this runs on any build host. Floor-CL enables one local
  # category and uses the deterministic stub, so the smoke exercises local routing
  # without paying real llama-server warmup.
  local en=false; [ "$enabled" = "1" ] && en=true
  awk -v en="$en" '
    /^local_candidate:/ {inlc=1}
    /^[^ ]/ && !/^local_candidate:/ {inlc=0; incats=0}
    /^token_budget:/ {print "token_budget: 0"; next}
    /^mandatory_remote:/ {print "mandatory_remote: 0"; next}
    inlc && /^  enabled:/ {print "  enabled: " en; next}
    inlc && /^  categories:/ {incats=1; print; next}
    inlc && incats && /^    actual_qa:/ {print "    actual_qa: " en; next}
    {print}
  ' "$ROOT/agent/config.yaml" > "$work/config.yaml"
  local input_mount output_mount config_mount
  input_mount="$(docker_host_path "$work/input")"
  output_mount="$(docker_host_path "$work/output")"
  config_mount="$(docker_host_path "$work/config.yaml")"
  echo "== smoke: $mode (enabled=$enabled) =="
  MSYS_NO_PATHCONV=1 docker run --rm --platform "$PLATFORM" \
    -e CONFIG_PATH=/cfg/config.yaml \
    -e AGENT_FORCE_STUB=1 \
    -v "$input_mount:/input:ro" -v "$output_mount:/output" \
    -v "$config_mount:/cfg/config.yaml:ro" \
    "$IMAGE"
  local results_path
  results_path="$(docker_host_path "$work/output/results.json")"
  python3 - "$results_path" "$enabled" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1]))
ids = {x["task_id"] for x in d}
assert ids == {"t1", "t2"}, ids
if sys.argv[2] == "1":
    by_id = {x["task_id"]: x["answer"] for x in d}
    assert by_id["t2"].strip(), by_id
print("  valid results.json", ids)
PY
  rm -rf "$work"
}
smoke "Floor-C"  0
smoke "Floor-CL" 1

echo "== PASS: build_and_size =="

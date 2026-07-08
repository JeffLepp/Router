#!/bin/sh
# Start llama-server only when Tier 1.5 is enabled (Floor-CL); otherwise skip
# it for a faster cold start (Floor-C). Then exec the router.
set -e

CONFIG="${CONFIG_PATH:-/app/agent/config.yaml}"

ENABLED="$(python3 - "$CONFIG" <<'PY'
import sys, yaml
try:
    cfg = yaml.safe_load(open(sys.argv[1])) or {}
    print("1" if cfg.get("local_candidate", {}).get("enabled") else "0")
except Exception:
    print("0")
PY
)"

if [ "$ENABLED" = "1" ]; then
  echo "entrypoint: Floor-CL, starting llama-server" >&2
  GPU_ARGS=""
  if [ "${LLAMA_N_GPU_LAYERS:-0}" != "0" ]; then
    GPU_ARGS="--n-gpu-layers ${LLAMA_N_GPU_LAYERS}"
  fi
  llama-server --model "$MODEL_GGUF" --host 127.0.0.1 --port "$LLAMA_PORT" \
    --ctx-size 4096 $GPU_ARGS >/tmp/llama.log 2>&1 &
  # Wait for readiness (up to ~30s); the accept-gate defers to Fireworks if it
  # never comes up, so this only trims a startup race, never blocks the wall.
  i=0
  while [ "$i" -lt 60 ]; do
    if python3 -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:'+__import__('os').environ['LLAMA_PORT']+'/health',timeout=1)" 2>/dev/null; then
      echo "entrypoint: llama-server ready" >&2; break
    fi
    i=$((i+1)); sleep 0.5
  done
else
  echo "entrypoint: Floor-C, llama-server skipped" >&2
fi

exec python3 -m agent.main

#!/bin/sh
# Start llama-server only when Tier 1.5 is enabled (Floor-CL); otherwise skip
# it for a faster cold start (Floor-C). Then exec the router.
#
# The shipped image is Floor-C and bundles NO llama-server and NO GGUF (see Dockerfile).
# The block below is kept so re-adding the model stages is a one-line config flip; it degrades
# to a warning if the config asks for a local model that is not in the image.
set -e

CONFIG="${CONFIG_PATH:-/app/agent/config.yaml}"

START_LLAMA="$(python3 - "$CONFIG" <<'PY'
import sys, yaml
try:
    cfg = yaml.safe_load(open(sys.argv[1])) or {}
    local = cfg.get("local_candidate", {}) or {}
    categories = local.get("categories", {}) or {}
    def truthy(value):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    effective = truthy(local.get("enabled")) and any(truthy(v) for v in categories.values())
    print("1" if effective else "0")
except Exception:
    print("0")
PY
)"

case "${AGENT_FORCE_STUB:-0}" in
  1|true|TRUE|yes|YES) FORCE_STUB=1 ;;
  *) FORCE_STUB=0 ;;
esac

if [ "$FORCE_STUB" = "1" ]; then
  echo "entrypoint: local stub forced, llama-server skipped" >&2
elif [ "$START_LLAMA" = "1" ] && ! command -v llama-server >/dev/null 2>&1; then
  echo "entrypoint: config enables a local model but this image bundles none;" >&2
  echo "entrypoint: continuing remote-only (the accept-gates defer to Fireworks)" >&2
elif [ "$START_LLAMA" = "1" ]; then
  echo "entrypoint: Floor-CL, starting llama-server" >&2
  GPU_ARGS=""
  if [ "${LLAMA_N_GPU_LAYERS:-0}" != "0" ]; then
    GPU_ARGS="--n-gpu-layers ${LLAMA_N_GPU_LAYERS}"
  fi
  CTX_SIZE="${LLAMA_CTX_SIZE:-512}"
  THREADS="${LLAMA_THREADS:-2}"
  llama-server --model "$MODEL_GGUF" --host 127.0.0.1 --port "$LLAMA_PORT" \
    --ctx-size "$CTX_SIZE" --threads "$THREADS" $GPU_ARGS >/tmp/llama.log 2>&1 &
  LLAMA_PID="$!"
  # Wait briefly; the accept-gate defers to Fireworks if llama is still warming.
  i=0
  WAIT_S="${LLAMA_STARTUP_WAIT_S:-10}"
  while [ "$i" -lt "$WAIT_S" ]; do
    if python3 -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:'+__import__('os').environ['LLAMA_PORT']+'/health',timeout=1)" 2>/dev/null; then
      echo "entrypoint: llama-server ready" >&2; break
    fi
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
      echo "entrypoint: llama-server exited during startup" >&2
      tail -50 /tmp/llama.log >&2 || true
      break
    fi
    i=$((i+1)); sleep 1
  done
  if [ "$i" -ge "$WAIT_S" ]; then
    echo "entrypoint: llama-server not ready after ${WAIT_S}s; continuing with fallback routing" >&2
  fi
else
  echo "entrypoint: Floor-C/no local categories, llama-server skipped" >&2
fi

exec python3 -m agent.main

# syntax=docker/dockerfile:1
# Floor-C router image: deterministic gate + Fireworks, no local model.
#
# The 1.5B local model was measured and removed (2026-07-09, see HANDOFF.md). On the 2 vCPU
# grader it absorbed only actual_qa and sentiment_analysis, and got every one of them wrong --
# a knowledge-free model cannot RECALL facts, only TRANSFORM given text. Local summarization,
# the one thing it did well, ran 5-60s/task and broke the 30s/request cap. Dropping the GGUF
# and llama-server takes ~1.2GB off the image and removes the llama warm-up from cold start.
# To restore it: re-add the two build stages below from git history and flip
# local_candidate.enabled in the config; agent/local_gate.py is unchanged and still self-checks.
#
# Target: linux/amd64, << 10GB compressed, ready < 60s.

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

COPY agent/ /app/agent/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

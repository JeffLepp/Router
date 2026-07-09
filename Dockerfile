# syntax=docker/dockerfile:1
# Floor-CL router image. One image, two modes: local_candidate.enabled flips
# Floor-C (Tier 1.5 off) <-> Floor-CL (Tier 1.5 on) at runtime, no rebuild.
# Target: linux/amd64, < 10GB compressed (14B Q4 GGUF = 9.0GB + ~0.4GB runtime).
#
# llama-server is built with the Vulkan backend so the 48GB-VRAM GPU on the
# hackathon instance is used when present; with no GPU device it degrades to
# CPU and the local gate defers to Fireworks, so GPU is never *required*.

# ---- Stage 1: build llama-server (Vulkan + CPU fallback, static) ----
FROM ubuntu:24.04 AS build
ARG LLAMA_REF=b9934
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake build-essential ca-certificates libvulkan-dev glslc \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${LLAMA_REF} https://github.com/ggml-org/llama.cpp /src \
    || git clone --depth 1 https://github.com/ggml-org/llama.cpp /src
WORKDIR /src
# Static llama-server: no shared-lib copying, survives llama.cpp layout changes.
RUN cmake -B build -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
      -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target llama-server -j "$(nproc)"

# ---- Stage 2: fetch the baked GGUF ----
FROM ubuntu:24.04 AS model
# Qwen3-14B Q4_K_M = 9.0GB. If the size gate ever trips, rebuild with
# MODEL_GGUF_URL=.../Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q6_K.gguf (6.7GB).
ARG MODEL_GGUF_URL=https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fSL "${MODEL_GGUF_URL}" -o /model.gguf

# ---- Stage 3: runtime ----
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip libgomp1 ca-certificates libvulkan1 mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

COPY --from=build /src/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=model /model.gguf /models/model.gguf
COPY agent/ /app/agent/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV MODEL_GGUF=/models/model.gguf \
    LLAMA_PORT=8080 \
    LLAMA_CTX_SIZE=8192 \
    LLAMA_PARALLEL=4 \
    LLAMA_THREADS=2 \
    LLAMA_STARTUP_WAIT_S=45 \
    INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# syntax=docker/dockerfile:1
# Floor-CL router image. One image, two modes: local_candidate.enabled flips
# Floor-C (Tier 1.5 off) <-> Floor-CL (Tier 1.5 on) at runtime, no rebuild.
# Target: linux/amd64, < 10GB compressed (3B Q4 GGUF ~2GB).

# ---- Stage 1: build llama-server (CPU/AVX2 + Vulkan, never ROCm/CUDA) ----
FROM debian:bookworm-slim AS build
ARG LLAMA_REF=b4000
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake build-essential libcurl4-openssl-dev \
      libvulkan-dev glslang-tools glslc ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${LLAMA_REF} https://github.com/ggml-org/llama.cpp /src \
    || git clone --depth 1 https://github.com/ggml-org/llama.cpp /src
WORKDIR /src
# Vulkan on; AVX2 is default for x86-64 CPU backend. CUDA/ROCm stay OFF.
RUN cmake -B build -DGGML_VULKAN=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target llama-server -j "$(nproc)"

# ---- Stage 2: fetch the baked GGUF ----
FROM debian:bookworm-slim AS model
ARG MODEL_GGUF_URL=https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fSL "${MODEL_GGUF_URL}" -o /model.gguf

# ---- Stage 3: runtime ----
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip libvulkan1 libgomp1 ca-certificates \
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
    INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

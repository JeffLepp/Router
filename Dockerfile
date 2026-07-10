# syntax=docker/dockerfile:1
# Default target: Floor-C (deterministic gate + Fireworks, no local model).
# Candidate target: --target floor-cl, with verified model/license build arguments.

FROM debian:bookworm-slim AS llama-build
ARG LLAMA_REF=b9637
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${LLAMA_REF}" https://github.com/ggml-org/llama.cpp /src
WORKDIR /src
# Static, portable linux/amd64 CPU build. No CUDA, ROCm, Vulkan, or host-native tuning.
RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DGGML_NATIVE=OFF \
      -DGGML_CUDA=OFF \
      -DGGML_HIP=OFF \
      -DGGML_VULKAN=OFF \
      -DGGML_OPENCL=OFF \
      -DLLAMA_CURL=OFF \
    && cmake --build build --target llama-server -j "$(nproc)"

FROM debian:bookworm-slim AS verified-model
ARG MODEL_GGUF_URL
ARG MODEL_GGUF_SHA256
ARG MODEL_LICENSE_URL
ARG MODEL_LICENSE_SHA256
RUN test -n "${MODEL_GGUF_URL}" \
    && test -n "${MODEL_GGUF_SHA256}" \
    && test -n "${MODEL_LICENSE_URL}" \
    && test -n "${MODEL_LICENSE_SHA256}"
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /models /licenses \
    && curl --fail --location --retry 4 --retry-all-errors \
         "${MODEL_GGUF_URL}" --output /models/model.gguf \
    && echo "${MODEL_GGUF_SHA256}  /models/model.gguf" | sha256sum --check --strict \
    && curl --fail --location --retry 4 --retry-all-errors \
         "${MODEL_LICENSE_URL}" --output /licenses/MODEL-LICENSE.txt \
    && echo "${MODEL_LICENSE_SHA256}  /licenses/MODEL-LICENSE.txt" | sha256sum --check --strict

FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip libgomp1 ca-certificates \
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

# Explicit candidate image. MODEL_* arguments intentionally have no defaults: a caller must
# choose one exact candidate and its license, and both downloads are SHA-256 verified.
FROM runtime AS floor-cl
ARG MODEL_SYSTEM_PROMPT=""
COPY --from=llama-build /src/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=verified-model /models/model.gguf /models/model.gguf
COPY --from=verified-model /licenses/MODEL-LICENSE.txt /licenses/MODEL-LICENSE.txt
ENV MODEL_GGUF=/models/model.gguf \
    LLAMA_PORT=8080 \
    LLAMA_CTX_SIZE=1024 \
    LLAMA_THREADS=2 \
    LLAMA_PARALLEL=1 \
    LLAMA_STARTUP_WAIT_S=45 \
    LOCAL_SYSTEM_PROMPT=${MODEL_SYSTEM_PROMPT} \
    CONFIG_PATH=/app/agent/config.track2.yaml

# Keep this last: an ordinary `docker build` must stay the small, proven Floor-C image.
FROM runtime AS floor-c

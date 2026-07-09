# Floor-CL Router

Our Track 1 entry for the AMD Developer Hackathon ACT II. It answers a fixed set
of tasks while burning as few Fireworks tokens as possible — the competition
ranks you on token count *once you clear the accuracy gate*, so every task we can
answer locally for free is a win.

**New to the repo? Read this file, then:**
- [ARCHITECTURE_AND_RESULTS.md](ARCHITECTURE_AND_RESULTS.md) — how the pipeline
  works + latest benchmark numbers.
- [AGENTS.md](AGENTS.md) — the hard competition rules + agent/dev quickstart.
- [BLOCKERS.md](BLOCKERS.md) — open problems and known failure cases.

## Where we stand

Latest judged live run (`benchmark_runs/live-official-judged-floor-c-full80`):
**86.25% accuracy, 8,857 Fireworks tokens, 0 errors, 11s.** Half the tasks (40/80)
are answered locally at zero token cost. Full breakdown in
[ARCHITECTURE_AND_RESULTS.md](ARCHITECTURE_AND_RESULTS.md).

## How it works (the short version)

Each task flows through tiers, stopping at the first that can answer:

1. **Classify** — regex cascade sorts the prompt into one of 8 categories.
2. **Deterministic gate** — proof solvers answer safe tasks for **zero tokens**.
3. **Local 3B candidate** *(Floor-CL, off by default)* — optional CPU model tier.
4. **Fireworks fallback** — one compact remote call for whatever's left.

Default mode is **Floor-C**: tier 3 disabled, so there's no local server to start
and the container stays portable on a CPU-only grader. See the architecture doc
for the full diagram and module map.

### Input / output contract

The container reads `/input/tasks.json` and writes `/output/results.json`:

```json
// in                                    // out
[{"task_id": "t1", "prompt": "2 + 2?"}]  [{"task_id": "t1", "answer": "4"}]
```

Fireworks calls go through `FIREWORKS_BASE_URL` and use only models from
`ALLOWED_MODELS` — both injected by the harness at runtime, never hardcoded.

## Getting set up

You need Python 3 and (for container work) Docker. Install deps:

```bash
pip install -r requirements.txt
```

For live runs, put your Fireworks creds in `.env.local` (gitignored, never
commit) or export them:

```bash
export FIREWORKS_API_KEY=...
export FIREWORKS_BASE_URL=...
export ALLOWED_MODELS=...
```

## Running it

**Offline smoke test** (no key, uses the mock server) — do this first, always:

```bash
python -m scripts.live_benchmark --mock --limit 16
```

**Local checks** before you claim a change is safe:

```bash
python agent/local_gate.py
PYTHONPATH=. python -m scripts.acceptance_p2      # gate precision/recall
PYTHONPATH=. python -m scripts.acceptance_p25     # local candidate gate
python agent/solvers/ner_solve.py
```

**Live Fireworks benchmark** (spends tokens — get the go-ahead first):

```bash
python -m scripts.live_benchmark --floor floor-c --score-judge --env-file .env.local
```

`--score-judge` scores the summarization/judge tasks too; without it they're
marked unscored so you don't silently spend judge tokens. Add `--holdout` to
include the `eval/devset` variants.

## Building & testing the container

```bash
export IMAGE=localhost:5000/floor-cl:test
bash scripts/build_and_size.sh              # build + compressed-size check
bash scripts/acceptance_in_container.sh     # I/O contract inside the image
```

On Windows use Git Bash explicitly:

```cmd
cd /d "C:\Users\jeffe\Desktop\summer26\AMD"
set "IMAGE=localhost:5000/floor-cl:test"
"C:\Program Files\Git\bin\bash.exe" scripts/build_and_size.sh
"C:\Program Files\Git\bin\bash.exe" scripts/acceptance_in_container.sh
```

Run the built image against live Fireworks:

```bash
python -m scripts.live_benchmark --container-image localhost:5000/floor-cl:test --holdout
```

## Working on the local model (Floor-CL)

The baked llama.cpp server is **CPU-only** — Track 1 advertises a 2 vCPU / 4GB
grading target with no promised GPU driver, so a GPU-required image can fail
before the router even starts.

The entrypoint only starts llama-server when `local_candidate.enabled` is true in
`agent/config.yaml` *and* at least one local category is enabled. Useful knobs:

- `AGENT_FORCE_STUB=1` — skip the real model for fast container smoke tests.
- `scripts/bench_local.py` — real local benchmarking.
- `LLAMA_CTX_SIZE`, `LLAMA_THREADS`, `LLAMA_STARTUP_WAIT_S`, `LLAMA_N_GPU_LAYERS`
  — tune startup/perf per environment.

## Track 2 (local models allowed)

Track 2 permits local inference with no Fireworks-only restriction, but the
grading box is the same 2 vCPU / 4GB with no model runtime pre-installed — so
weights ship inside the image (they already do; `MODEL_GGUF_URL` is a build
arg). The 3B Q4 fits RAM (~2GB) but generates ~3 tok/s on 2 vCPUs and can't
clear the latency cap; a **1.5B Q4 (~1GB)** roughly halves latency and leaves
~3GB for llama.cpp KV cache + agent code. Build and run the Track 2 image:

```bash
docker build \
  --build-arg MODEL_GGUF_URL=https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -t router:track2 .

docker run --rm \
  -v "$PWD/sample_input:/input:ro" -v "$PWD/out:/output" \
  -e CONFIG_PATH=/app/agent/config.track2.yaml \
  -e LLAMA_CTX_SIZE=1024 -e LLAMA_STARTUP_WAIT_S=30 \
  -e FIREWORKS_API_KEY -e FIREWORKS_BASE_URL -e ALLOWED_MODELS \
  router:track2
```

`agent/config.track2.yaml` enables the local tier for the short-output gated
categories (k=1, 10s latency cap); summarization and code generation still go
remote. Everything the local gate rejects falls through to Fireworks as usual.
Before trusting it on the grader, verify latency in-container with
`--cpus=2 --memory=4g` and `scripts/bench_local.py`.

## Before you submit

- [ ] Image builds for `linux/amd64` and passes `scripts/build_and_size.sh`
      (compressed size well under 10GB — we're ~2.1GB).
- [ ] Container writes `/output/results.json` and exits 0.
- [ ] No keys, base URLs, `.env`, or model IDs baked into the image.
- [ ] **Push the image to a public, pullable registry** (the local `localhost:5000`
      tag won't work for graders).
- [ ] lablab.ai listing complete (title, descriptions, tags, cover, video, slides,
      public repo, demo URL). Full rule list in [AGENTS.md](AGENTS.md).

## Repo layout

Full module-by-module map is in [AGENTS.md](AGENTS.md). The short version:
`agent/` is the runtime router, `agent/solvers/` + `agent/verify/` are the
zero-token deterministic tiers, `eval/` and `scripts/` are dev/benchmark helpers
(not shipped in the runtime image), and `benchmark_runs/` holds generated results.

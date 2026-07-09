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
3. **Local 14B candidate** *(Floor-CL, on by default)* — Qwen3-14B via llama.cpp, GPU-offloaded when a GPU is present, verified by the local gate.
4. **Fireworks fallback** — one compact remote call for whatever's left.

Default mode is now **Floor-CL**: tier 3 answers verified categories for zero
tokens. With no GPU device the server runs on CPU and the local gate's latency
cap defers everything to Fireworks, so a CPU-only grader still passes. See the
architecture doc for the full diagram and module map.

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

The baked llama.cpp server is built with the **Vulkan backend** and bakes
**Qwen3-14B Q4_K_M** (9.0GB) — the hackathon instance has ~48GB of VRAM, so the
whole model plus KV cache fits on-GPU. The entrypoint autodetects a GPU
(`/dev/dri/renderD*`) and offloads all layers; with no GPU it runs CPU-only and
the local gate simply defers to Fireworks, so GPU is never required.

The entrypoint only starts llama-server when `local_candidate.enabled` is true in
`agent/config.yaml` *and* at least one local category is enabled. Useful knobs:

- `AGENT_FORCE_STUB=1` — skip the real model for fast container smoke tests.
- `scripts/bench_local.py` — real local benchmarking.
- `LLAMA_CTX_SIZE`, `LLAMA_THREADS`, `LLAMA_STARTUP_WAIT_S`, `LLAMA_N_GPU_LAYERS`
  — tune startup/perf per environment.

## Before you submit

- [ ] Image builds for `linux/amd64` and passes `scripts/build_and_size.sh`
      (compressed size under 10GB — ~9.4GB with the 14B GGUF baked in).
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

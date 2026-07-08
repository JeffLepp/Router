# Floor-CL Router

Track 1 router for the AMD Developer Hackathon ACT II. The agent minimizes
Fireworks token use while preserving accuracy by answering only when a local
deterministic verifier can prove the result, then deferring the rest to an
allowed Fireworks model.

## Runtime Contract

The container reads tasks from `/input/tasks.json` and writes valid JSON to
`/output/results.json`.

```json
[
  {"task_id": "t1", "prompt": "What is 2 + 2?"}
]
```

Output:

```json
[
  {"task_id": "t1", "answer": "4"}
]
```

Fireworks calls are routed through `FIREWORKS_BASE_URL` and use only models
listed in `ALLOWED_MODELS`.

## Routing Strategy

- Tier 1: deterministic proof gate in `agent/gate.py`.
- Tier 1.5: optional local 3B candidate in `agent/local_gate.py`, disabled by
  default in `agent/config.yaml`.
- Tier 2: Fireworks fallback in `agent/remote.py`.

The default submission mode is Floor-C: local candidate disabled and one remote
call for every task the proof gate cannot solve.

## Prototype 1 Baseline

Prototype 1 is the current Floor-C router: deterministic proof solvers answer
safe tasks for zero Fireworks tokens, and every unresolved task gets one compact
Fireworks call. The local llama.cpp candidate tier is present but off by default
so the container stays portable in a CPU-only grading environment.

Latest trusted live run:
`benchmark_runs/live-official-accessible-floor-c-full80-final`

| Metric | Result |
|---|---:|
| Scored accuracy | 46/70 (65.71%) |
| Unscored summarization tasks | 10 |
| Fireworks requests | 42 |
| Fireworks tokens | 7,719 |
| Remote errors | 0 |
| Wall time | 10.94s |
| Docker smoke image size | 2.13 GB compressed |

Interpretation: Prototype 1 is infrastructure-stable and token-light enough for
iteration, but not yet accuracy-competitive. All scored failures in the trusted
run came from the remote path, so later prototypes should compare against this
baseline by improving remote answer shape and category routing before chasing
more token cuts.

## Change Tracking

Log each live run's local score + what it proved. **Caveat:** the local scorer
does exact/format matching and cannot run `unit_tests`/`schema_match` or make
LLM judgements, so it *undercounts* code and terse-QA answers that pass under the
real harness. Read the failures.csv, don't trust the raw %.

| Run | Scope | Local acc | Gate | Remote | Tokens | Errors |
|---|---|---:|---:|---:|---:|---:|
| `live-official-...-full80-final` | full 70 | 65.71% | 38/38 | remote | 7,719 | 0 |
| `live-fable5-shortcats` | 5 cats / 50 | 62.00% | 21/21 (100%) | 10/29 (34%) | 4,493 | 0 |

`live-fable5-shortcats` (after classify/contracts fixes) — gate precision held
at 100%, zero remote errors. Local 62% understates true accuracy: all 6 code
"failures" are functionally-correct fixes the scorer can't execute, and qa_005/006
are correct-but-terse. Real remote problems this run isolated (next targets):

1. Sentiment aspect **key** mismatch — model emits right polarity, wrong aspect
   keys (`battery life` vs `battery`). aspect_based tasks 004/005/006.
2. Aspect JSON leaks into single-label `classification` sarcasm tasks (007/008),
   which also missed the sarcasm → wrong label.
3. Logic remote answers ramble/truncate instead of terse answer (005/008); the
   answer-only instruction + 40-tok cap isn't binding minimax-m3.
4. qa_010 "how many moons…" misroutes to math (`how many` hint) → math wrapper
   instead of `unanswerable`.
5. qa_007 unanswerable hatch fired on an answerable current-events question.

## Local Model Notes

The baked llama.cpp server is CPU-only by default. Track 1 advertises a
2 vCPU / 4GB RAM grading target and does not promise a compatible Vulkan/CUDA
driver, so a GPU-required image can fail before the router starts. CPU mode is
slower, but it keeps the default Floor-C path portable.

Floor-CL remains optional. The entrypoint only starts llama-server when
`local_candidate.enabled` is true and at least one local category is enabled.
For fast container smoke tests, set `AGENT_FORCE_STUB=1`; for real local
benchmarking, use `scripts/bench_local.py` and override `LLAMA_CTX_SIZE`,
`LLAMA_THREADS`, `LLAMA_STARTUP_WAIT_S`, or `LLAMA_N_GPU_LAYERS` as needed.

## Local Verification

```bash
python agent/local_gate.py
PYTHONPATH=. python -m scripts.acceptance_p25
PYTHONPATH=. python -m scripts.acceptance_p2
python agent/solvers/ner_solve.py
```

Container verification:

```bash
export IMAGE=localhost:5000/floor-cl:test
bash scripts/build_and_size.sh
bash scripts/acceptance_in_container.sh
```

On Windows `cmd.exe`, use Git Bash explicitly:

```cmd
cd /d "C:\Users\jeffe\Desktop\summer26\AMD"
set "IMAGE=localhost:5000/floor-cl:test"
"C:\Program Files\Git\bin\bash.exe" scripts/build_and_size.sh
"C:\Program Files\Git\bin\bash.exe" scripts/acceptance_in_container.sh
```

## Live Fireworks Benchmark

Do not paste API keys into chat or commit them. Set them in the shell:

```bash
export FIREWORKS_API_KEY=...
export FIREWORKS_BASE_URL=...
export ALLOWED_MODELS=...
```

Run a host benchmark:

```bash
python -m scripts.live_benchmark --holdout
```

Run the built container against live Fireworks:

```bash
python -m scripts.live_benchmark --container-image localhost:5000/floor-cl:test --holdout
```

By default, summarization methods that need an LLM judge are marked unscored so
the benchmark does not silently spend extra judge tokens. Add `--score-judge` to
include them.

Offline smoke test:

```bash
python -m scripts.live_benchmark --mock --limit 16
```

## Submission Notes

- The image is built for `linux/amd64`.
- The compressed image size is checked by `scripts/build_and_size.sh`.
- The image writes `/output/results.json` before exit and exits 0 on success.
- The repository does not bundle a benchmark answer cache in the runtime image.
- `dataset.json`, `eval/`, and `scripts/` are evaluation helpers; the Dockerfile
  copies only the runtime agent, dependencies, entrypoint, and baked local model.

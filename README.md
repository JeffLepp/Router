# Hybrid Token-Efficient Router

Track 1 entry for the AMD Developer Hackathon ACT II. The router answers the
fixed task set while minimizing Fireworks tokens; accuracy is always the first
gate. See [AGENTS.md](AGENTS.md) for the competition rules and module map.

## Current status

The latest trusted live artifact is
`benchmark_runs/live-batching-fixes-20260710`: **69/80 strict (86.25%), 72/80
judged (90%), 5,195 Fireworks tokens, and zero remote errors**. The deterministic
and local-queue work after that run has passed local/mock checks but is not
presented as a paid-live result.

After the 42.1% hidden-set result, the submission default is an accuracy-first
recovery profile:

1. Classify the task into one of eight categories.
2. Answer only explicit, recomputable arithmetic and fixed metric conversions
   locally; send everything else through `FIREWORKS_BASE_URL`, using only
   runtime `ALLOWED_MODELS`.
3. Prefer reasoning-capable models for math, logic, and code, request high
   reasoning effort, and use non-starving completion caps.
4. Keep heuristic deterministic gates and batching off until they pass a
   genuinely out-of-distribution precision suite.

The previous token-efficient behavior remains available as
`agent/config.efficiency.yaml` for controlled A/B tests. Accuracy must recover
before any of those optimizations are promoted back into the default.

An optional Floor-CL profile adds a single sequential, summary-only llama.cpp
queue while non-summary remote work runs concurrently. Anything rejected or
timed out falls through to Fireworks.

## Input/output contract

The container reads `/input/tasks.json` and writes `/output/results.json`:

```json
[{"task_id":"t1","prompt":"What is 2 + 2?"}]
```

```json
[{"task_id":"t1","answer":"4"}]
```

Runtime values come from `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and
`ALLOWED_MODELS`. Keys, base URLs, `.env` files, and Fireworks model IDs are not
baked into the image.

## Local setup and validation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run the safe local checks before a live benchmark:

```bash
python -m py_compile agent/contracts.py agent/remote.py agent/gate.py agent/solvers/sentiment_solve.py scripts/live_benchmark.py
python -m eval.score
python -m eval.gate_report --holdout
python -m scripts.acceptance_p2
python -m scripts.acceptance_p25
python -m scripts.acceptance_p1
python -m scripts.live_benchmark --mock --score-judge --out-dir benchmark_runs/mock-agent-check
```

A paid Fireworks run requires explicit approval:

```bash
python -m scripts.live_benchmark --floor floor-c --score-judge --env-file .env.local
```

## Building Floor-C

An ordinary build produces the small CPU-safe Floor-C target:

```bash
docker build --platform linux/amd64 -t router:floor-c .
```

The broader build/size/push/pull smoke is:

```bash
bash scripts/build_and_size.sh
```

On Windows, invoke it through Git Bash:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' scripts/build_and_size.sh
```

## Building a Floor-CL candidate

The `floor-cl` target compiles a static, CPU-only llama.cpp server and requires
model and license URLs plus SHA-256 values. Those arguments intentionally have
no defaults. For example, the measured Apache-2.0 SmolLM3 candidate uses:

```bash
docker build --platform linux/amd64 --target floor-cl \
  --build-arg MODEL_GGUF_URL=https://huggingface.co/ggml-org/SmolLM3-3B-GGUF/resolve/main/SmolLM3-Q4_K_M.gguf \
  --build-arg MODEL_GGUF_SHA256=8334b850b7bd46238c16b0c550df2138f0889bf433809008cc17a8b05761863e \
  --build-arg MODEL_LICENSE_URL=https://www.apache.org/licenses/LICENSE-2.0.txt \
  --build-arg MODEL_LICENSE_SHA256=cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30 \
  --build-arg MODEL_SYSTEM_PROMPT=/no_think \
  -t router:floor-cl-candidate .
```

`agent/config.track2.yaml` enables only summarization: one queue slot, a
35-second per-task deadline, 240-second stage ceiling, and 64/110/150-token
shape caps. Benchmark the baked model under grader-style limits:

```bash
docker run --rm --cpus=2 --memory=4g --entrypoint python3 \
  -v "$PWD:/workspace:ro" -w /workspace router:floor-cl-candidate \
  -m scripts.bench_local --models candidate=/models/model.gguf \
  --server-bin /usr/local/bin/llama-server --require-qualifier
```

Qualification requires 100% accepted precision across the 12 applicable
public+holdout summaries, at least 6/7 unresolved public summaries accepted,
cold readiness under 45 seconds, the summary stage under 240 seconds, and peak
RSS at or below 3.7 GiB.

Neither tested model is promoted. SmolLM3 was resource-safe but accepted only
1/7 unresolved public summaries in the latest full-container run. LFM2 reached 5/7 on the latest
repeated full run, and selection also requires explicit eligibility under the
LFM Open License v1.0 revenue threshold. Floor-C remains the default.

## Submission checklist

- Public, pullable `linux/amd64` image below 10 GB compressed.
- Read `/input/tasks.json`, atomically write valid `/output/results.json`, exit 0.
- No missing answers or remote errors in two approved full-80 validations.
- Runtime model selection remains restricted to `ALLOWED_MODELS`.
- No secrets, benchmark answers, or private artifacts in the image or repo.
- Complete the lablab.ai title, descriptions, tags, cover, video, slides,
  public repository, demo platform, and application URL.

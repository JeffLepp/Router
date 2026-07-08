# AGENTS.md

Purpose: give coding agents the smallest useful map of this repo and the hard Track 1 rules. Keep changes scoped, prove them locally first, and do not spend Fireworks tokens until mock/local checks pass.

## Mission

Track 1: Hybrid Token-Efficient Routing Agent for AMD Developer Hackathon ACT II.

The agent answers a fixed task set while minimizing Fireworks tokens. Passing the accuracy gate comes first; token efficiency only matters after the answer quality threshold is met.

For each remote task, choose the cheapest model from runtime `ALLOWED_MODELS` that can still answer accurately. Deterministic/local answers cost zero Fireworks tokens.

Current default strategy is Floor-C:

1. Classify task.
2. Try deterministic proof solver.
3. Route only unresolved tasks to Fireworks through the judging proxy.

Floor-CL adds a local llama.cpp candidate tier, but it is disabled by default because local startup under the expected 2 vCPU / 4GB style environment is not yet proven inside the readiness window.

## Non-Negotiable Requirements

- Container image must be public, pullable, linux/amd64, and under 10GB compressed.
- On startup read `/input/tasks.json`.
- Before exit write valid JSON to `/output/results.json`.
- Input rows look like `{"task_id":"t1","prompt":"..."}`.
- Output rows must look like `{"task_id":"t1","answer":"..."}`.
- Runtime env comes from the harness: `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS`.
- Never hardcode or bundle keys, base URLs, `.env` files, or model IDs.
- Every Fireworks call must go through `FIREWORKS_BASE_URL`.
- Only models in `ALLOWED_MODELS` may be used.
- Finish within 10 minutes; startup/readiness target is under 60 seconds; remote calls should stay under 30 seconds.
- Exit 0 on success; use nonzero only for true process failure.
- Local model inference is allowed and costs zero Fireworks tokens, but local answers must be verified or safely gated.
- Do not hardcode or cache benchmark answers; hidden prompt variants are used.
- Responses must be English.
- Submissions are rate limited to 10 per hour per team.
- lablab.ai submission also needs title, short/long description, tags, cover image, video, slides, public GitHub repo, demo platform, and application URL.
- Deadline is whatever the lablab.ai Event Schedule tab shows in the participant's local timezone.

Task categories:

1. `actual_qa`
2. `math_reasoning`
3. `sentiment_analysis`
4. `summarization`
5. `named_entity_recognition`
6. `code_debugging`
7. `logic_puzzles`
8. `code_generation`

## Repo Map

Root files:

- `Dockerfile`: builds the submission image. Current path is CPU-safe; do not require Vulkan/GPU for final.
- `docker/entrypoint.sh`: starts optional local server only when enabled, then runs the agent.
- `agent/config.yaml`: runtime knobs: remote policy, local tier, batching, token caps, timeouts.
- `dataset.json`: local public/dev task set only. Do not hardcode its answers.
- `requirements.txt`: small Python dependency set.
- `README.md`: public setup and usage.
- `FABLE5_TODO.md`: lean next-step plan for accuracy recovery.

Core agent:

- `agent/main.py`: orchestrator, input/output contract, atomic snapshots, classification, gate, local tier, remote calls.
- `agent/classify.py`: maps prompts into the 8 canonical categories. Weak cues need corroboration — `how many/much` only routes math with a second signal (standalone number / math verb / ×÷), keeping factual "how many …" on `actual_qa`.
- `agent/gate.py`: dispatches deterministic zero-token solvers.
- `agent/contracts.py`: remote prompt contracts, max-token caps, and final answer assembly.
- `agent/remote.py`: Fireworks client, `ALLOWED_MODELS` parsing, model routing, token ledger.
- `agent/batcher.py`: same-category batching helper; currently disabled in config.
- `agent/local_llm.py`: local llama.cpp client/stub.
- `agent/local_gate.py`: accept-or-defer gate for local candidates.
- `agent/self_judge.py`: lightweight self-check helpers.

Deterministic solvers:

- `agent/solvers/arithmetic.py`: numeric arithmetic.
- `agent/solvers/wordmath.py`: simple word math.
- `agent/solvers/sentiment_solve.py`: high-precision sentiment labels.
- `agent/solvers/ner_solve.py`: safe NER patterns.
- `agent/solvers/logic_solve.py`: constraint and boolean logic patterns.
- `agent/solvers/code_solve.py`: safe Python code fixes/generation.
- `agent/solvers/format_solve.py` and `common.py`: shared formatting helpers.

Verification:

- `agent/verify/math_v.py`: math checks.
- `agent/verify/logic_v.py`: constraint consistency.
- `agent/verify/format_v.py`: answer format checks.
- `agent/verify/code_v.py`: sandboxed inline code verification helpers.

Evaluation and scripts:

- `eval/score.py`: deterministic local scorer for non-judge methods.
- `eval/judge.py`: optional summary judge path.
- `eval/probe_models.py`: model access probe.
- `eval/devset/variants.json`: local holdout variants.
- `scripts/live_benchmark.py`: host/container benchmark runner; use mock before live.
- `scripts/mock_fireworks.py`: local mock Fireworks server.
- `scripts/acceptance_p1.py`: core contract checks.
- `scripts/acceptance_p2.py`: deterministic gate precision/recall checks.
- `scripts/acceptance_p25.py`: local candidate gate checks.
- `scripts/build_and_size.sh`: Docker build, image size, and smoke path.
- `scripts/bench_local.py`: local llama-server smoke benchmark.
- `scripts/dress_rehearsal.py`: broader mock/rehearsal runner.

Ignored/local artifacts:

- `.env.local`: local Fireworks creds only. Never print it and never commit it.
- `benchmark_runs/`: generated benchmark output; do not commit new runs unless explicitly requested.
- `.codex/`, `.agents/`: local agent workspace metadata.

## Current Baseline

Latest trusted live run artifact: `benchmark_runs/live-official-accessible-floor-c-full80-final`.

- Accuracy: `46/70 = 65.71%` on scored local methods.
- Summaries: 10 unscored locally.
- Fireworks: 42 requests, 7,719 tokens, 0 remote errors.
- All 24 scored failures were remote-path failures.
- Main weak spot: remote answer quality/format, not CPU/GPU or local runtime.

Latest mock regression gate (`live_benchmark --mock --score-judge`, Floor-C, both
runs re-scored with the current scorer): strict/judged `38/80 → 40/80` vs Prompt-0
baseline, **zero pass→fail flips**, only `sentiment_analysis` moved (3→5), `qa_010`
now routes `actual_qa`. Mock tokens rose `2,317 → 3,318` (all prompt tokens, from
the hardened contracts) — confirm real net cost on a live key. Gate/transport/token
modules byte-identical to HEAD.

Next accuracy work should start with `FABLE5_TODO.md`.

## Validation

Run these before claiming a router change is safe:

```powershell
python -m py_compile agent\contracts.py agent\remote.py agent\gate.py agent\solvers\sentiment_solve.py scripts\live_benchmark.py
python -m eval.score
python -m scripts.acceptance_p2
python -m scripts.acceptance_p25
python -m scripts.acceptance_p1
python -m scripts.live_benchmark --mock --out-dir .\benchmark_runs\mock-agent-check
```

Docker smoke on Windows:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' scripts/build_and_size.sh
```

Paid/live Fireworks runs require explicit user approval unless the user already asked for a live benchmark in the same turn.

## Editing Rules

- Keep proof solvers conservative: wrong zero-token answers are worse than extra Fireworks calls.
- Remote changes should preserve `ALLOWED_MODELS` runtime selection.
- Do not add new frameworks or broad dependencies.
- Prefer focused fixes plus acceptance tests over broad rewrites.
- Do not delete or rewrite user secrets, benchmark history, or unrelated work.

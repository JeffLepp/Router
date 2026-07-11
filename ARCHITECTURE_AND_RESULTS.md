# Architecture & Benchmark Results

Track 1 — Hybrid Token-Efficient Routing Agent (AMD Hackathon ACT II).
Scored on **Fireworks tokens (ascending)** *after* passing the accuracy gate.
Zero-token local/gate answers are the moat.

Last updated: 2026-07-09.

---

## Architecture

One image, default Floor-C plus opt-in local profiles. Flip `CONFIG_PATH` or
`local_candidate.enabled` at runtime; rebuild only when changing the baked GGUF:

- **Floor-C (default):** classify → deterministic solver → route only unresolved
  tasks to Fireworks. Local llama-server stays **off**, so startup is near-instant.
- **Floor-CL (optional):** adds a local llama.cpp candidate tier between the gate
  and remote. Disabled by default until local startup is proven in the harness env.
- **Track 2 profile (opt-in):** `agent/config.track2.yaml` turns the local tier
  on for short-output, accept-gated categories with k=1 and a 10s cap. Summary
  and code-generation tasks still stay remote.

### Pipeline (Floor-C)

```
/input/tasks.json
   │
   ▼
classify (agent/classify.py)        8 canonical categories; weak cues need corroboration
   │
   ▼
deterministic gate (agent/gate.py)  zero-token solvers → arithmetic, wordmath,
   │  hit → answer                   sentiment, ner, logic, code, format
   │  miss
   ▼
remote (agent/remote.py)            cheapest model from runtime ALLOWED_MODELS,
   │                                 per-category routing, token ledger, retries
   ▼
contracts (agent/contracts.py)      prompt contract + max-token cap + answer assembly
   │
   ▼
/output/results.json                {"task_id": "...", "answer": "..."}
```

### Model routing

- Models come **only** from the runtime `ALLOWED_MODELS` env — never hardcoded.
- `choose_model_for_category` picks from the allowed list via category regex, falls
  back to ranked default; empty list raises. No baked model IDs on the runtime path.
- Every call goes through `FIREWORKS_BASE_URL`.

### Post-merge routing details

- Remote robustness now retries blank completions on a fallback allowed model.
- If a provider rejects `reasoning_effort`, the client caches that per model and
  retries without the param instead of failing the request.
- Model ranking is active-parameter aware for MoE model IDs such as `a4b`; this
  keeps routing cheap when total-parameter and active-parameter counts differ.
- `agent/config.track2.yaml` currently sets `local_slots: 4`,
  `self_consistency_k: 1`, and `latency_cap_s: 10`. Keep Track 2 opt-in until
  the real local model is re-benchmarked inside the grader-style CPU/RAM limit.
- `scripts/dashboard.py` is a dev visualization tool. The runtime Dockerfile
  copies `agent/` and `docker/entrypoint.sh`, not `scripts/`, so the dashboard is
  not part of the submitted container payload unless the Dockerfile changes.

### Key modules

| Module | Role |
|---|---|
| `agent/main.py` | orchestrator, I/O contract, atomic snapshots |
| `agent/classify.py` | prompt → one of 8 categories |
| `agent/gate.py` + `agent/solvers/*` | deterministic zero-token solvers |
| `agent/remote.py` | Fireworks client, `ALLOWED_MODELS` routing, token ledger |
| `agent/contracts.py` | remote prompt contracts + token caps |
| `agent/local_llm.py` + `agent/local_gate.py` | Floor-CL local tier (off by default) |
| `agent/verify/*` | math / logic / format / code checks |

---

## Latest results — `live-master-postmerge-full80-20260709` (2026-07-09)

Full 80 tasks, **live** Fireworks, judge on, code executed. This is the current
post-merge master result after PRs #7-#10.

| Metric | Value |
|---|---|
| **Judged accuracy** | **87.50% (70/80)** |
| Strict accuracy | 83.75% (67/80) |
| Total tokens | 8,840 (prompt 7,575 / completion 1,265) |
| Remote requests | 40 (gate answered the other 40 at 0 tokens) |
| Remote errors | 0 |
| Est. cost | $0.0080 |
| Wall time | 13.90s |

### Per-category (judged run)

| Category | Pass/Total | Gate | Remote |
|---|---:|---:|---:|
| math_reasoning | 10/10 | 10 | 0 |
| named_entity_recognition | 10/10 | 9 | 1 |
| code_debugging | 9/10 | 4 | 6 |
| sentiment_analysis | 10/10 | 5 | 5 |
| actual_qa | 7/10 | 0 | 10 |
| code_generation | 7/10 | 5 | 5 |
| logic_puzzles | 7/10 | 7 | 3 |
| summarization | 7/10 | 0 | 10 |

### Remaining 13 strict losses (10 judged losses)

- **logic (3):** `logic_005/006/008` — scorer wants a bare value; we emit
  `{"answer":...,"valid":...}`. Likely format, not reasoning.
- **QA (3):** `qa_005` verbose ("...was founded first"), `qa_007`, `qa_009`
  unanswerable phrasing.
- **summarization (3):** `summary_003/004/009` content/constraint misses.
- **code_generation (3):** `gen_005/008/010` unit/schema strictness.
- **code_debugging (1):** `debug_010` unit-test strictness.

Format-shaped losses (logic + QA) are the cheapest points — likely fixable with a
zero-token local reformat pass, no extra tokens.

---

## Post-merge validation - master PRs #7-#10 (2026-07-09)

Pulled `origin/master` at `f0c7126` into local `testing` by fast-forward. Local
regression checks passed first, then a full live judged benchmark was run.

| Check | Result |
|---|---|
| `python -m py_compile agent\contracts.py agent\remote.py agent\gate.py agent\solvers\sentiment_solve.py scripts\live_benchmark.py scripts\dashboard.py` | PASS |
| `python -m agent.remote` | PASS, including active-param ranking, blank fallback, and `reasoning_effort` retry self-checks |
| `AgentConfig.from_path('agent/config.track2.yaml')` | PASS: local tier enabled, `local_slots=4`, `self_consistency_k=1`, `latency_cap_s=10`; summary/code-generation stay remote |
| `python -m eval.score` | PASS: gold-back 80/80, corrupted 0/80 |
| `python -m scripts.acceptance_p2` | PASS: gate precision 100%, answered 40, holdout precision 100%, pipeline gate-proven 94/150 |
| `python -m scripts.acceptance_p25` | PASS: local gate precision 100%, stub path reduced remote calls 56 -> 38 and tokens 4,654 -> 2,584 |
| `python -m scripts.acceptance_p1` | PASS: synthetic run, valid JSON on kill, Floor-C default, verifier/module self-checks |
| `python -m scripts.live_benchmark --floor floor-c --score-judge --env-file .env.local --out-dir .\benchmark_runs\live-master-postmerge-full80-20260709` | PASS: judged 87.50% (70/80), strict 83.75% (67/80), 8,840 tokens, 40 gate / 40 remote, 0 remote errors, 13.90s |

The post-merge state preserves the default Track 1 Floor-C behavior while adding
three opt-in/supporting improvements: safer remote fallback, active-parameter
model ranking, and the Track 2 local profile. Track 2 still needs a real
in-container local-model latency run before it should be treated as a scoring
strategy.

---

## Prior runs (context)

| Run | Scope | Accuracy | Tokens | Notes |
|---|---|---:|---:|---|
| `live-master-postmerge-full80-20260709` | 80, judged | **87.50%** | 8,840 | current baseline |
| `live-official-judged-floor-c-full80` | 80, judged | 86.25% | 8,857 | previous baseline |
| `live-official-accessible-floor-c-full80-final` | 80, judge off | 65.71% scored | 7,719 | undercounted (summaries unscored, code not executed) |
| `live-kimi-floor-c-full80` | 80 | — | 7,719 | transport proof |

Lifetime live Fireworks spend across all runs: **~$0.053**.

---

## Headroom (what we can spend for growth)

Scored metric = tokens; accuracy gate must pass. Everything else is slack to
convert into **zero-token local work**.

| Constraint | Limit | Used | Headroom |
|---|---|---|---|
| Total runtime | 600s | ~11s | ~98% free |
| Startup | 60s | ~instant (Floor-C) | ~60s |
| Per remote call | 30s | <5s | ~25s |
| Image (compressed) | 10GB | ~2.1GB | ~7.9GB |

**Levers, both goals at once (tokens down + accuracy up):**

1. Enable **Floor-CL** local tier — offloads remote tasks to zero-token local.
2. **Local verify+reformat** pass — fixes format-only losses at 0 tokens.
3. Bake a **stronger local model** (2.1GB → ~5GB is still legal).
4. **Prompt compression** — 86% of tokens are prompt; pure efficiency win.

Do **not** add self-consistency / multi-sampling: it multiplies the scored metric.

---

## Compliance snapshot

All hard rules in `AGENTS.md` pass (image size, amd64, I/O contract, env-driven
keys/models, ALLOWED_MODELS-only, English, runtime budget). Two submission-time
items remain: **push the image to a public registry** (currently local-only tag),
and re-prove container startup/exit end-to-end via `scripts/build_and_size.sh`.

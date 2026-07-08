# Fable5 Plan: Beat the Frontier Routers

Budget note: keep this pass small. No paid Fireworks runs until local checks and mock benchmark pass. Never print `.env.local`.

## The competition math

Scoring is a two-stage sort: (1) LLM-judge accuracy gate, (2) passers ranked by **ascending Fireworks tokens**. Local tokens are free. Frontier submissions will clear the gate easily; they beat us today because we're failing the gate at 65.71%. But once we clear it, the game flips: they pay full remote tokens per task, we pay ~0 for the 38+ tasks our local gate already solves. **Our token floor (7,719 now, less after fixes) is likely unbeatable by any route-everything-remote design.** So: fix accuracy, don't add tokens.

## Diagnosis (from `live-official-accessible-floor-c-full80-final`)

46/70 scored (65.71%). All 24 failures are remote-path. Local gate: 38/38 correct (math 10/10, NER 9/9, logic 7/7, code gates 9/9). The failures are almost entirely **our own output-shape bugs**, not model capability:

| Bug | Evidence | Tasks at stake |
|---|---|---|
| Markdown fences left on code | debug_003/005/006/010, gen_003/005/008/010 all have ```` ``` ```` in answer; the code inside looks correct | ~8 |
| "unanswerable" leaking into non-QA prompts | sentiment_006–010 remote replies are 2–4 completion tokens = the literal word "unanswerable"; the escape-hatch instruction in `contracts.py:111` is reaching sentiment prompts | ~5 |
| Prose wrapper on logic answers | logic_006/008 wrapped in "The answer is: …" with trailing junk | ~2 |
| Truncation mid-answer | gen_002 cut off at 10 completion tokens ("function sumArray(numbers) {."); qa_010 similar | ~2 |
| Real misses (verbose QA, wrong answers, aspect JSON mismatches) | qa_004/005/006/007, sentiment_004/005, logic_005 | ~7 |

Fixing the first four bug classes alone projects to ~63/70 ≈ 90% with **zero added tokens** (fences/wrappers stripped = fewer completion tokens, even).

## Phase 0 — Output-shape fixes (free accuracy, do first)

1. `agent/contracts.py`: strip markdown fences (```` ```lang … ``` ````) from remote code answers in `code_debugging` and `code_generation` assemble paths.
2. `agent/contracts.py`: scope the "return exactly: unanswerable" escape hatch to `actual_qa` only. Sentiment/logic/code prompts must never contain it.
3. `agent/contracts.py`: `logic_puzzles.assemble()` must not wrap remote output in "The answer is: …"; require and pass through compact JSON `{"answer":"...","valid":true}`.
4. Fix truncation: gen_002 died at 10 completion tokens despite a 300 cap — find where the effective cap gets clamped (batcher `_batch_cap`? per-call override in `main.py:159`?) and ensure code categories actually get their configured cap.
5. `sentiment_analysis` remote prompt: require exact compact aspect JSON; keep JSON unwrapped end to end.

## Phase 1 — Diagnostics (so the next paid run is the last one)

6. `scripts/live_benchmark.py`: write `failures.csv` (task_id, category, route, model, tokens, method, expected, actual, prompt) and add `accuracy_by_route`/`accuracy_by_model`/`tokens_by_category` to `benchmark.json`. No extra live calls. We still can't see `expected` for qa_005 ("Harvard University") / qa_006 ("No") — this closes that gap.

## Phase 2 — Widen the free-token moat (after gate is cleared)

7. Expand deterministic local coverage where precision is obvious: sentiment aspect JSON for common mixed pairs (performance/battery, design/camera, performance/venue); explicit-cue sarcasm only. Every task moved local = fewer tokens AND one less remote-format risk.
8. Check `agent/classify.py` gating on the 3 logic tasks that went remote — `logic_solve.py` already handles those patterns deterministically.
9. Prompt compression on remaining remote calls: trivial QA is spending ~160–200 prompt tokens on instructions for 2-token answers. Compress instruction boilerplate per category.
10. Model routing stays as-is: `kimi-k2p7-code` for code, `minimax-m3` for the rest. No model experiments until diagnostics prove a category-specific failure.

## Phase 3 — Validate, then one focused paid run

Validation order (all free):

```
python -m py_compile agent\contracts.py agent\remote.py agent\solvers\sentiment_solve.py scripts\live_benchmark.py
python -m eval.score
python -m scripts.acceptance_p2
python -m scripts.acceptance_p25
python -m scripts.acceptance_p1
python -m scripts.live_benchmark --mock --out-dir .\benchmark_runs\mock-fable5
```

Then focused live run on failed categories only, then full 80:

```
python -m scripts.live_benchmark --floor floor-c --category actual_qa --category sentiment_analysis --category logic_puzzles --category code_debugging --category code_generation --out-dir .\benchmark_runs\live-fable5-shortcats
python -m scripts.live_benchmark --floor floor-c --out-dir .\benchmark_runs\live-fable5-full80
```

Targets: >=85% scored accuracy (stretch >=90%) at <=7,719 total tokens.

## Endgame — submission strategy

- Keep the dual-floor plan: Sub 1 = Floor-0 (pure local, 0 tokens) to test the A0 ambiguity, Sub 2 = fixed Floor-C, Sub 3 = safe high-budget anchor.
- Frontier routers can't rank below a passing submission with fewer tokens. Once accuracy clears the gate, every remaining point of effort goes into moving tasks local, not into smarter remote prompting.

## Defer

- Same-category batching: only after output parsing is stable (batching multiplies format risk).
- Frugal confirmation mode: after diagnostics log candidates/confidence.
- Local model tuning: not the bottleneck; failures are remote-format, local gate is 38/38.

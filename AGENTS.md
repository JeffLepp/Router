# AGENTS.md

Purpose: give coding agents the smallest current map of the AMD Track 1 router.
The accuracy gate is passed; optimize tokens conservatively and never revive an
old recovery plan without new evidence.

## Mission and current phase

Build a hybrid router that answers the fixed task set accurately while minimizing
Fireworks tokens.

Current official state, reported 2026-07-11:

- Accuracy: **89.5%** (approximately 17/19 tasks).
- Placement: **67th**.
- Phase: token optimization with an accuracy floor above 80%.
- Frozen image: `jeffklin303/amd-router:accuracy-first-20260711`.
- Digest: `sha256:fafd46eef741e6ef1660c05084a1893807572c50b650b85399266d04e82bc0bf`.

On 19 tasks, one extra miss gives 84.2% and two extra misses give 78.9%.
Treat the accuracy budget as one task, not as a comfortable ten-point margin.

## Current runtime

1. Read and deduplicate `/input/tasks.json`.
2. Classify all tasks in one batched Fireworks call into eight categories.
3. Run the strict deterministic proof gate; accept only proven answers.
4. Route unresolved tasks through the judging proxy:
   - Minimax primary: factual QA, math, sentiment.
   - Kimi primary: summarization, NER, debugging, logic, code generation.
   - Other allowed families are transport fallbacks, not unvalidated primaries.
5. Apply the category contract and conservative output normalization.
6. Atomically write `/output/results.json`.

The classifier scored 160/160 on variants2 + variants3. Do not spend the
accuracy budget replacing it unless the candidate reproduces that result.

## Non-negotiable requirements

- Public, pullable `linux/amd64` image under 10 GB compressed.
- Read `/input/tasks.json`; write valid `/output/results.json` before exit.
- Input: `{"task_id":"t1","prompt":"..."}`.
- Output: `{"task_id":"t1","answer":"..."}`.
- Runtime env is supplied by the harness: `FIREWORKS_API_KEY`,
  `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`.
- Never bundle keys, `.env` files, base URLs, or fixed provider model IDs.
- Every Fireworks request must use `FIREWORKS_BASE_URL` and a model present in
  runtime `ALLOWED_MODELS`.
- Finish within 10 minutes; startup target is under 60 seconds.
- Exit 0 on success; nonzero is for true process failure only.
- Do not hardcode or cache benchmark answers; hidden variants are used.
- Responses must be English.
- Local inference is allowed, but candidates must be calibrated and safely gated.
- Submission attempts are rate limited to 10 per hour per team.

Task categories:

1. `actual_qa`
2. `math_reasoning`
3. `sentiment_analysis`
4. `summarization`
5. `named_entity_recognition`
6. `code_debugging`
7. `logic_puzzles`
8. `code_generation`

## Repo map

Core runtime:

- `agent/main.py`: orchestration, deadlines, snapshots, and I/O.
- `agent/classify.py`: local rules plus remote classifier prompt/parser.
- `agent/gate.py`: conservative zero-token proof dispatch.
- `agent/contracts.py`: category prompts, caps, and answer assembly.
- `agent/remote.py`: allowed-model routing, retries, and token ledger.
- `agent/batcher.py`: experimental same-category answer batching; default off.
- `agent/local_llm.py`, `agent/local_gate.py`: optional local candidate path; default off.
- `agent/solvers/`: deterministic proof solvers.
- `agent/verify/`: math, logic, format, and code verification.
- `agent/config.yaml`: frozen accuracy-first defaults.
- `agent/config.efficiency.yaml`: experimental token-saving profile; never promote wholesale.

Evaluation and release:

- `dataset.json`: restored public/dev set; never copy answers into runtime logic.
- `eval/devset/variants2.json`, `variants3.json`: primary OOD regression gates.
- `eval/score.py`: deterministic development scorer; not part of runtime routing.
- `scripts/classifier_remote_audit.py`: classification-only accuracy audit.
- `scripts/live_benchmark.py`: mock/live runner and token ledger.
- `scripts/acceptance_p1.py`: I/O, contract, transport, and routing checks.
- `scripts/acceptance_p2.py`: deterministic gate precision checks.
- `scripts/acceptance_p25.py`: optional local-candidate gate checks.
- `scripts/build_and_size.sh`: build, size, public pull, manifest, and smoke gate.

Documentation:

- `README.md`: public overview and commands.
- `SUBMISSION_READY.md`: one-screen teammate handoff and submission state.
- `ARCHITECTURE_AND_RESULTS.md`: current evidence and experiment roadmap.
- `improvements.txt`: compact current experiment ledger.

## Token-optimization order

Work one lever at a time, in this order:

1. Measure token contribution by stage/category; do not optimize estimates.
2. Replace the remote batch classifier only with a local classifier that remains
   160/160 on variants2 + variants3 and remotely defers low confidence.
3. Add per-category reasoning effort; try lower effort on QA, sentiment, NER,
   and summaries while retaining stronger reasoning for math, logic, and code.
4. Test answer batching only for one low-coupling category at a time.
5. Reduce caps only from observed completion percentiles and truncation checks.
6. Expand deterministic/local answers only at 100% OOD precision.

Do not combine these in one experiment. A combined win cannot identify which
lever caused an accuracy loss on the hidden set.

## Promotion gate

Before claiming a token change is safe:

```powershell
python -m py_compile agent\classify.py agent\contracts.py agent\main.py agent\remote.py scripts\live_benchmark.py
python -m eval.score
python -m scripts.acceptance_p2
python -m scripts.acceptance_p25
python -m scripts.acceptance_p1
python -m scripts.live_benchmark --mock --out-dir .\benchmark_runs\mock-agent-check
```

Then run paired live variants2 and variants3. Promote only when:

- no new remote errors or missing answers;
- no unexplained pass-to-fail changes;
- both judged sets remain at least 90%;
- token savings are measured, not inferred;
- one exact configuration variable changed.

Keep the known-good Docker tag immutable and publish candidates under new tags.

## Editing rules

- Preserve runtime `ALLOWED_MODELS` enforcement.
- Keep proof solvers conservative; a wrong local answer is worse than a remote call.
- Do not add a large local LLM merely because the image cap is 10 GB. Validate
  cold start, 4 GB RAM, and accuracy before size.
- Do not edit or print `.env.local`.
- Do not delete benchmark history or unrelated user work.
- Prefer focused changes plus paired evidence over broad rewrites.

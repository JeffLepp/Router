# AGENTS.md

Purpose: give coding agents the smallest current map of the AMD Track 1 router.
The accuracy gate is passed; optimize tokens conservatively and never revive an
old recovery plan without new evidence.

## Mission and current phase

Build a hybrid router that answers the fixed task set accurately while minimizing
Fireworks tokens.

Current official state, reported 2026-07-11:

- Accuracy: **94.7%** (approximately 18/19 tasks).
- Fireworks tokens: **12,012**.
- Placement: last reported as 67th; the Phase 1 result did not include a new rank.
- Phase: token optimization with an accuracy floor above 80%.
- Frozen image: `jeffklin303/amd-router:phase1-direct-compression-20260711`.
- Digest: `sha256:03dd918dd42bad832842456200c3ddd6678470e242d5b6c469aaf1f59def87fa`.

On 19 tasks, one extra miss gives 89.5%, two extra misses give 84.2%, and
three extra misses give 78.9%. The score now has a two-task floor margin, but
do not spend it casually: preserve paired local answers unless a loss is explained.

## Current runtime

1. Read and deduplicate `/input/tasks.json`.
2. Classify all tasks in one batched Fireworks call into eight categories.
3. Run the strict deterministic proof gate; accept only proven answers.
4. Route unresolved tasks through the judging proxy:
   - Minimax primary: factual QA, math, sentiment.
   - Kimi primary: summarization, NER, debugging, logic, code generation.
   - Other allowed families are transport fallbacks, not unvalidated primaries.
5. Phase 1 compresses direct-output work: QA uses no reasoning; direct NER uses
   no reasoning but event/ambiguity prompts keep high reasoning; QA, sentiment,
   and NER batch by category with per-row validation and individual fallback.
6. Apply the category contract and conservative output normalization.
7. Atomically write `/output/results.json`.

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
- `agent/batcher.py`: Phase 1 same-category batching for QA, sentiment, and NER.
- `agent/local_llm.py`, `agent/local_gate.py`: optional local candidate path; default off.
- `agent/solvers/`: deterministic proof solvers.
- `agent/verify/`: math, logic, format, and code verification.
- `agent/config.yaml`: frozen Phase 1 defaults.
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

1. Phase 0 attribution and Phase 1 direct-output compression are complete.
2. Replace the remote batch classifier only with a local classifier that remains
   160/160 on variants2 + variants3 and remotely defers low confidence.
3. Test the next isolated answer-level lever only after the classifier decision.
4. Reduce caps only from observed completion percentiles and truncation checks.
5. Expand deterministic/local answers only at 100% OOD precision.

Do not combine new levers in one experiment. Phase 1 was a deliberately coupled
direct-output compression policy; future candidates must isolate one variable.

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

# Team Handoff - 94.7% / 12,012-Token Baseline

## Copy/paste this to teammates

```text
AMD Track 1 Router - current state

OFFICIAL RESULT
- Accuracy: 94.7% (about 18/19 tasks)
- Fireworks tokens: 12,012
- Rank: last reported as 67th; no new rank was supplied with this result
- Accuracy gate: PASSED
- New goal: lower Fireworks tokens while remaining above 80%
- Floor margin: two more misses still gives 16/19 = 84.2%; three gives 15/19 = 78.9%.

HOW THE ROUTER WORKS

  /input/tasks.json
          |
          v
  [ONE BATCHED FIREWORKS CLASSIFIER CALL]
          |
          v
  [ONE OF 8 CATEGORY LABELS]
          |
          v
  [STRICT DETERMINISTIC PROOF GATE]
          | proven                 | unresolved
          v                        +--> actual_qa ----------------> Minimax
  zero-token answer ---+           +--> math_reasoning -----------> Minimax
                       |           +--> sentiment_analysis -------> Minimax
                                   +--> summarization ------------> Kimi
                                   +--> named_entity_recognition -> Kimi
                                   +--> code_debugging -----------> Kimi
                                   +--> logic_puzzles ------------> Kimi
                                   +--> code_generation ----------> Kimi
                                                        |
                                                        v
                                             category format contract
                                                        |
                       +--------------------------------+
                       |
                       v
             /output/results.json

WHY THIS VERSION WORKED
- The remote classifier was 160/160 on variants2 + variants3.
- The submitted model roster no longer silently moves NER/sentiment to
  untested Gemma primaries.
- Minimax/Kimi are the validated primary pair; other allowed models are fallback.
- Full prompts, category-specific output schemas, larger safety caps, and
  incomplete-code retries prevent avoidable zero-score answers.

PHASE 1 WIN
- QA uses no hidden reasoning; direct NER uses no reasoning while event/ambiguous NER stays high.
- QA, sentiment, and NER batch with per-row validation and individual fallback.
- Local release gates: 154/160 judged unchanged, 83,977 -> 69,128 tokens (-17.7%).
- Official result: 94.7%, 12,012 tokens (forecast was 11,524).

TOKEN-REDUCTION RULE
- Freeze this image.
- Change one new lever at a time.
- Run mock, variants2, and variants3.
- Reject any new missing answer, remote error, or unexplained pass-to-fail.
- Submit only candidates that remain at least 90% on both local judged sets.

NEXT EXPERIMENTS, IN ORDER
1. Replace the classifier call with a small local ONNX classifier only if it
   stays 160/160; remotely defer low-confidence cases.
2. Test the next answer-level token lever in isolation.
3. Lower output caps only from observed completion percentiles.
4. Expand local answers only after cold-start, 4 GB RAM, and accept-or-defer evidence.

DOCKER IMAGE
jeffklin303/amd-router:phase1-direct-compression-20260711
sha256:03dd918dd42bad832842456200c3ddd6678470e242d5b6c469aaf1f59def87fa
https://hub.docker.com/r/jeffklin303/amd-router
```

## Phase 2 candidate (READY TO SUBMIT, awaiting push)

Image: `jeffklin303/amd-router:phase2-aggressive-7605428`
(manifest list `sha256:df52cb19477c9d6a65e2917922fe107a370c2ee4334062e1fbb865a776b2bff7`)
Source: branch `testing2` @ `7605428` (= prefilter `8ac2699` + math_full gate + reasoning cuts).

Levers stacked on the prefilter candidate (each its own commit for traceback):
1. `8ac2699` zero-token classifier prefilter (~55% coverage, 0/280 wrong, rest defers).
2. `bffcf52` math_full gate profile: generalized zero-token math templates
   (dataset 10/10, variants2 7/10, variants3 5/10 covered, 0 wrong — `scripts/math_audit`).
3. `942db63` + `7605428` reasoning_effort none for qa/ner/sentiment/summarization/math;
   only logic and code keep hidden reasoning.

| Gate | Result |
|---|---:|
| prefilter audit / math audit / self-checks | all PASS, 0 wrong |
| mock routing parity vs prefilter candidate | only delta: 8 math tasks -> local gate |
| variants2 live | **96.25% judged**, 19,741 tokens (-37% vs prefilter run), 0 errors/retries |
| variants3 live | **96.25% judged**, 20,999 tokens (-33% vs prefilter run), 0 errors/retries |
| pass->fail churn | v2: +3 fixed / 1 new (v2_ner_010, empty-entity edge, reasoning-cut casualty); v3: +4 fixed / 1 new (v3_math_002, remote arithmetic slip w/o reasoning) |
| Floor-C keyless smoke | PASS (80/80 answers in results.json) |

Official forecast: ~7,000-7,600 tokens (from 12,012), accuracy risk bounded by the
two explained edge-case fail modes. To ship: `docker push jeffklin303/amd-router:phase2-aggressive-7605428`,
submit, then record digest + result in `submission_history.csv`.

## Release evidence

| Check | Result |
|---|---:|
| Official Phase 1 submission | **94.7%**, **12,012 tokens** |
| Classifier-only audit | **160/160** |
| Phase 1 variants2 full live | **97.50% judged**, 33,959 tokens, 0 errors |
| Phase 1 variants3 full live | **95.00% judged**, 35,169 tokens, 0 errors |
| Final variants3 NER sentinel check | **10/10**, 0 errors |
| Docker size | 76,047,173 bytes compressed (0.08 GB) |
| Docker platform | `linux/amd64` |
| Docker smoke | Floor-C and Floor-CL both passed |

The Phase 1 official run reduced tokens from the prior approximate 14,000 to
**12,012** while improving accuracy from 89.5% to 94.7%.

## Frozen vs experimental

- Frozen source: `master` commits `8dddb0e` and `4fc3ebf`.
- Frozen image: `phase1-direct-compression-20260711` and immutable digest above.
- Experimental token profile: create a new config/image; do not reuse the old
  wholesale efficiency profile.
- Do not overwrite the frozen tag during experiments; use a new candidate tag.

## Fast candidate checklist

```powershell
python -m eval.score
python -m scripts.acceptance_p2
python -m scripts.acceptance_p25
python -m scripts.acceptance_p1
python -m scripts.live_benchmark --mock --out-dir .\benchmark_runs\mock-token-candidate
python -m scripts.live_benchmark --dataset eval\devset\variants2.json --score-judge --remote-timeout 90 --env-file .env.local --out-dir benchmark_runs\live-token-candidate-v2
python -m scripts.live_benchmark --dataset eval\devset\variants3.json --score-judge --remote-timeout 90 --env-file .env.local --out-dir benchmark_runs\live-token-candidate-v3
```

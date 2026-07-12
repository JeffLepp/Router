# Team Handoff - 89.5% Accuracy Baseline

## Copy/paste this to teammates

```text
AMD Track 1 Router - current state

OFFICIAL RESULT
- Accuracy: 89.5% (about 17/19 tasks)
- Place: 67th
- Accuracy gate: PASSED
- New goal: lower Fireworks tokens while remaining above 80%
- Safety margin: one task. 16/19 = 84.2%; 15/19 = 78.9%.

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

TOKEN-REDUCTION RULE
- Freeze this image.
- Change one thing at a time.
- Run mock, variants2, and variants3.
- Reject any new missing answer, remote error, or unexplained pass-to-fail.
- Submit only candidates that remain at least 90% on both local judged sets.

NEXT EXPERIMENTS, IN ORDER
1. Measure tokens by stage and category.
2. Replace the classifier call with a small local ONNX classifier only if it
   stays 160/160; remotely defer low-confidence cases.
3. Lower reasoning effort for QA/sentiment/NER/summary one category at a time.
4. Batch answer calls one category at a time.
5. Lower output caps from observed completion percentiles.
6. Expand zero-token proof solvers only at 100% OOD precision.

DOCKER IMAGE
jeffklin303/amd-router:accuracy-first-20260711
sha256:fafd46eef741e6ef1660c05084a1893807572c50b650b85399266d04e82bc0bf
https://hub.docker.com/r/jeffklin303/amd-router
```

## Release evidence

| Check | Result |
|---|---:|
| Official hidden submission | **89.5%**, 67th |
| Classifier-only audit | **160/160** |
| variants2 full live | **96.25% judged**, 42,025 tokens, 0 errors |
| variants3 full live | **96.25% judged**, 41,952 tokens, 0 errors |
| Final variants2 NER/summary check | **19/20**, 0 errors |
| Docker size | 76,047,173 bytes compressed (0.08 GB) |
| Docker platform | `linux/amd64` |
| Docker smoke | Floor-C and Floor-CL both passed |

The official token count was not included with the reported score, so do not
claim an official token reduction until the next submission reports it.

## Frozen vs experimental

- Frozen accuracy source: current working tree on
  `experiment/remote-batch-classifier`.
- Frozen image: `accuracy-first-20260711` and immutable digest above.
- Experimental token profile: `agent/config.efficiency.yaml`.
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

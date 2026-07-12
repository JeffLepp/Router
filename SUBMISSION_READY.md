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

NEXT EXPERIMENTS, IN ORDER (updated after Phase 3 official result)
1. Change exactly ONE category or ONE lever per submission so the aggregate
   accuracy number is attributable. The hidden set is 19 tasks; one flip is
   5.26 points and the platform gives no per-task feedback.
2. Prefer levers whose token delta is deterministic (prompt trimming, cap
   changes) over levers that change model behavior (reasoning effort).
3. Gate every candidate with the per-task flip diff against the frozen Phase 1
   run, not with aggregate local accuracy. Local judged sets are saturated
   (92.5-97.5%) and have failed to predict hidden accuracy three times.
4. Do not trust local-to-hidden token ratios across configs that change
   batching/fallback behavior; Phase 3 had FEWER local tokens than Phase 1
   (31,891 vs 33,959 on v2) but MORE official tokens (13,005 vs 12,012).

DOCKER IMAGE
jeffklin303/amd-router:phase1-direct-compression-20260711
sha256:03dd918dd42bad832842456200c3ddd6678470e242d5b6c469aaf1f59def87fa
https://hub.docker.com/r/jeffklin303/amd-router
```

## Phase 3 postmortem (official: 89.5%, 13,005 tokens — REJECTED)

Scored 2026-07-12 05:19 PDT, rank 63. Worse than Phase 1 on both axes:
one more miss (17/19 vs 18/19) and +993 tokens. The 10,700-11,300 forecast
missed low by ~2,000 tokens. **Reverted: `agent/` on `testing2` is restored
byte-identical to Phase 1 commit `8dddb0e`. The active submission is the
frozen Phase 1 image below. Do not submit the phase2 or phase3 tags.**

What Phase 3's failure adds to the Phase 2 lessons:

1. **Local sets cannot rank candidates.** All of accuracy-first (89.5),
   phase2 (63.2), phase3 (89.5), and phase1 (94.7) scored 92.5-97.5% on
   variants2/3. Local gates screen for gross breakage only; passing them says
   nothing about which candidate is better on the hidden 19.
2. **Token forecasts don't transfer across batching changes.** Phase 3 used
   fewer local tokens than Phase 1 yet more official tokens — high-reasoning
   NER forced per-row fallback on the hidden set, exactly the risk the
   forecast note flagged, but in the opposite direction of the band.
3. **Multi-lever "safe" bundles still lose.** Phase 3 stacked five changes
   (NER reasoning restore, adaptive sentiment/summary reasoning, sentiment
   final-overall clarification, prefilter, classifier retries). One of them
   cost a task; the aggregate score cannot say which. Every submission that
   changed multiple levers has lost to single-lever Phase 1.

Phase 2 root causes (still valid): no-reasoning math/sentiment/NER caused
answer failures; the prefilter changed deferred-batch composition and a
20-row classifier call silently truncated at 1,024 tokens; noun-trigger
prefilter rules broke on near-miss task shapes.

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
# MANDATORY new gate: per-task flip diff vs the frozen Phase 1 runs.
# Any pass->fail = reject. Any answer change must be individually justified.
python -m scripts.flip_diff benchmark_runs\live-phase1-compressed-direct-v2-final-20260711 benchmark_runs\live-token-candidate-v2
python -m scripts.flip_diff benchmark_runs\live-phase1-compressed-direct-v3-final-20260711 benchmark_runs\live-token-candidate-v3
```

Retroactive proof: this gate rejects Phase 3 (2 pass->fail on variants3 —
`v3_debug_006` broken fix, `v3_ner_009` dropped the dual-role Washington
entity) even though both runs scored an identical 96.25% aggregate.

## Phase 1 token anatomy (v2 run, 33,959 total — where cuts can come from)

| Component | Tokens | Share |
|---|---:|---:|
| classifier (4 calls: 5,099 prompt + 2,216 completion) | 7,315 | 21.5% |
| answer prompts (58 requests) | 11,772 | 34.7% |
| answer completions | 14,872 | 43.8% |
| — of which summarization completions | 5,737 | 16.9% |

Prompt tokens are ~50% of all spend; the classifier alone is a fifth.
Both are cuttable without touching answer behavior.

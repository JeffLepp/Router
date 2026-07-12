# Hybrid Token-Efficient Router

Track 1 entry for the AMD Developer Hackathon ACT II. The router classifies each
task, proves a small safe subset locally, and sends unresolved work to the best
validated model exposed through the judging proxy.

## Current phase

The accuracy gate is passed. The current official result is **94.7%** (about
18/19 tasks) using **12,012 Fireworks tokens**. Development now focuses on
reducing tokens while preserving accuracy above 80%; the previous reported rank
was 67th and the Phase 1 result did not include a new rank.

Because a 19-task score moves in 5.26-point steps, 16/19 is 84.2% while 15/19
is 78.9%. Future token changes ship one at a time behind paired accuracy gates.

Frozen accuracy image:

```text
jeffklin303/amd-router:phase1-direct-compression-20260711
sha256:03dd918dd42bad832842456200c3ddd6678470e242d5b6c469aaf1f59def87fa
```

See [SUBMISSION_READY.md](SUBMISSION_READY.md) for the copy/paste teammate
overview and [ARCHITECTURE_AND_RESULTS.md](ARCHITECTURE_AND_RESULTS.md) for the
full evidence and token-reduction plan.

## Router at a glance

```text
/input/tasks.json
        |
        v
one batched Fireworks classifier -> 8 task categories
        |
        v
strict deterministic proof gate -- proven --> zero-token answer
        |
      unresolved
        v
validated category route
  Minimax: QA, math, sentiment
  Kimi:    summary, NER, debugging, logic, code generation
        |
        v
category contract + output normalization
        |
        v
/output/results.json
```

The official harness supplies `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and
`ALLOWED_MODELS`. The router never bundles credentials or fixed provider model
IDs, and every remote request goes through the supplied base URL.

## Accuracy evidence

| Evidence | Result |
|---|---:|
| Official Phase 1 submission | **94.7%**, **12,012 tokens** |
| Remote classifier audit | **160/160** across variants2 + variants3 |
| Phase 1 variants2 full live gate | **97.50% judged**, 33,959 tokens, 0 errors |
| Phase 1 variants3 full live gate | **95.00% judged**, 35,169 tokens, 0 errors |
| Final affected-category gate | variants3 NER sentinel 10/10, 0 errors |
| Docker acceptance | public linux/amd64, 0.08 GB compressed, both smokes pass |

The local variant scores are regression gates, not claims about the hidden set.
The official 94.7% / 12,012-token result remains the decision anchor.

The Phase 1 direct-output compression candidate combines two related changes:
QA uses no hidden reasoning, NER defers ambiguity/event prompts to high effort,
and QA/sentiment/NER share category batches with per-row validation and individual
fallback. On the paired 160-task release gate it preserved 154/160 judged passes,
reduced tokens from 83,977 to 69,128 (-17.7%), reduced requests from 169 to 123,
and produced zero remote errors. A post-fix NER sentinel gate passed 10/10.

The Phase 1 forecast was 11,524 tokens; the official result was 12,012, only
488 tokens (4.2%) higher. Use this calibration record for the next candidate.
Approaching 1,500 tokens will still require selective local answering;
classifier replacement alone targets only the 17.1% classifier share measured
before Phase 1.

## Input/output contract

The container reads `/input/tasks.json`:

```json
[{"task_id":"t1","prompt":"What is 2 + 2?"}]
```

Before exiting it writes `/output/results.json`:

```json
[{"task_id":"t1","answer":"4"}]
```

## Local validation

Install the small Python dependency set:

```bash
pip install -r requirements.txt
```

Run no-cost gates before a live experiment:

```powershell
python -m py_compile agent\classify.py agent\contracts.py agent\main.py agent\remote.py scripts\live_benchmark.py
python -m eval.score
python -m scripts.acceptance_p2
python -m scripts.acceptance_p25
python -m scripts.acceptance_p1
python -m scripts.live_benchmark --mock --out-dir .\benchmark_runs\mock-agent-check
```

Run paid variant gates only after local checks pass:

```powershell
python -m scripts.live_benchmark --dataset eval\devset\variants2.json --score-judge --remote-timeout 90 --env-file .env.local --out-dir benchmark_runs\live-candidate-v2
python -m scripts.live_benchmark --dataset eval\devset\variants3.json --score-judge --remote-timeout 90 --env-file .env.local --out-dir benchmark_runs\live-candidate-v3
```

Forecast the official token delta from paired control/candidate reports, then
append the real result to `submission_history.csv`:

```powershell
python -m scripts.submission_forecast --baseline <control-v2.json> <control-v3.json> --candidate <candidate-v2.json> <candidate-v3.json>
```

## Docker submission

Build the CPU-safe Floor-C image:

```bash
docker build --platform linux/amd64 -t router:floor-c .
```

The full size, push/pull, manifest, and smoke gate is:

```powershell
$env:IMAGE='your-dockerhub-user/amd-router:candidate'
& 'C:\Program Files\Git\bin\bash.exe' scripts/build_and_size.sh
```

Floor-CL and larger local-model experiments remain optional research paths.
They are not the current submission default because image size is not the main
constraint; cold start, 4 GB RAM, and accuracy calibration are.

## Optimization rule

Freeze the accuracy image, change one lever, and compare paired outputs. Do not
promote a token-saving change if either variant set regresses, a new remote error
appears, or the official score falls below 84.2%.

# Hybrid Token-Efficient LLM Router

**Team Simple Solutions — AMD Developer Hackathon ACT II, Track 1 (General-Purpose AI Agent)**

A production-style LLM router that answers a hidden benchmark of ~19 tasks across
8 categories at the lowest possible API cost. Judged on a **gate-then-rank** rule:
clear 80% accuracy or score nothing — then among the submissions that qualify,
**fewest tokens wins**.

| Final result | |
|---|---|
| **Accuracy** | **84.2%** (gate: 80%) |
| **Tokens** | **10,687** |
| **Rank** | **91st of 143 scored submissions** — a further **249 did not qualify** |
| **Peak accuracy** | **94.7%** on 12,012 tokens (earlier submission) |
| **Field** | 20,727 participants · 4,894 teams · 1,152 submissions |

<p align="center">
  <img src="docs/leaderboard.png" alt="AMD Track 1 leaderboard — Team Simple Solutions" width="840">
</p>

<p align="center">
  <img src="docs/final-stats.png" alt="AMD Hackathon ACT II final stats" width="840">
</p>

---

## What it does

Not every task needs a frontier model, and not every task needs one at all. The router
sorts incoming work by what it actually requires, then spends the minimum to get it right.

```
/input/tasks.json
        │
        ▼
  ONE batched classifier call ──────►  8 task categories
        │
        ▼
  deterministic proof gate ──proven──►  answered locally, ZERO tokens
        │
   unresolved
        ▼
  validated category route
     Minimax ──► QA · math · sentiment
     Kimi    ──► summary · NER · debugging · logic · code generation
        │
        ▼
  category output contract + normalization
        │
        ▼
/output/results.json
```

**One classifier call, not N.** Every task is labeled in a single batched request instead
of one call per task. Validated to **160/160** on paraphrased out-of-distribution sets.

**A proof gate that answers for free.** Tasks the router can *prove* — not guess — are
solved locally at zero API cost. It only accepts proof-backed answers, so it never trades
accuracy for tokens.

**Per-category routing with output contracts.** Each category gets the model that
benchmarked best on it, plus an explicit output schema, normalization, and a cross-model
retry for structurally incomplete code.

---

## How it evolved

Nine graded submissions against a hidden set that returns nothing but an aggregate score.
Every build got its own Docker tag, a predicted token count, and a row in
[`submission_history.csv`](submission_history.csv) — which is how regressions got caught
at all.

| Submission | Accuracy | Tokens | Rank | |
|---|---:|---:|---:|---|
| Early iterations | 42.1% → 87.5% | — | — | Finding the architecture |
| Accuracy-first | 89.5% | ~14,000 | 67th | Cleared the gate |
| **Direct compression** | **94.7%** | **12,012** | — | **Accuracy peak** |
| Aggressive compression | 63.2% | — | — | Failed the gate — reverted |
| Safe recovery | 89.5% | 13,005 | 63rd | Worse on both axes — reverted |
| Classifier compression | 89.5% | 11,673 | — | Saved 339 tokens, cost a task — reverted |
| **Final** | **84.2%** | **10,687** | **91st** | **Shipped** |

The arc is accuracy first, then cost. Once the 80% gate was cleared, ranking is decided
purely on tokens — so the back half of the project is a controlled hunt for spend, one
lever at a time, with a paired local gate and a token forecast before every submission.

Three of those experiments made things *worse* on the hidden set. Each was caught,
attributed, and rolled back to a frozen image rather than shipped.

---

## Stack

Python · Docker (`linux/amd64`, 0.08 GB) · Fireworks AI inference · Minimax · Kimi ·
batched classification · deterministic solvers · LLM-as-judge evaluation harness

---

## Running it

```bash
pip install -r requirements.txt
docker build --platform linux/amd64 -t router:submission .
```

The judging harness supplies `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`.
No credentials or hardcoded model IDs are bundled; every request goes through the supplied
base URL.

Local gates, before any paid run:

```powershell
python -m eval.score                                    # deterministic solver + scoring gate
python -m scripts.live_benchmark --mock                 # full pipeline, no API spend
python -m scripts.submission_forecast --baseline <control> --candidate <candidate>
```

Technical detail, evidence tables, and token attribution: [ARCHITECTURE.md](ARCHITECTURE.md)

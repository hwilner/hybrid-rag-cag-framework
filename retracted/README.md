# Quarantined — do not run, do not cite

The three scripts in this directory produced the fabricated results retracted on **2026-09-29**.
The full account is in [`../docs/INTEGRITY.md`](../docs/INTEGRITY.md).

They are kept here, unmodified, for two reasons:

1. **The retraction record has to be checkable.** A retraction that quietly deletes its own
   evidence is indistinguishable from never having made the claim. The archived outputs in
   `results/` are what the retraction is based on.
2. **So the same failure cannot recur quietly.** Each script now refuses to run without an
   explicit opt-in flag, and when forced writes only to `results/UNSAFE_*.json` — never to a real
   results path.

## What each script does wrong

| File | Problem |
|---|---|
| `option3_full_scale_evaluation.py` | Defines six "systems" — RAG, CAG, Hybrid, FiD, T5-FiD, DPR+FiD. All six are hardcoded dictionary lookups. No checkpoint is loaded anywhere in the file. |
| `expert_evaluation.py` | Contains 26 "human expert" responses as hardcoded Python strings. No human was involved. It also contains the phantom fusion equation `H(q) = α(q)·R(q,D) + (1−α(q))·G(q)`, which exists nowhere in the real system. |
| `train_and_evaluate.py` | Tier driver whose recorded figures trace to the hand-written JSON in `results/`, not to any code path. |

## Running them

They will not run by accident. Note that all three import `torch` at module scope, so in an
environment without torch they die at import — also safe, just less informative. With torch
installed you reach the real guard:

```bash
$ python retracted/expert_evaluation.py
REFUSING TO RUN: the 'human expert' responses in this script are
hardcoded strings in this file. No human study took place.
```

Forcing them requires typing a flag that says what you are doing:

```bash
python option3_full_scale_evaluation.py --i-know-this-is-fake
python expert_evaluation.py --i-know-these-arent-human
```

Output is then written to `results/UNSAFE_*.json`. **That output is fabricated. Do not cite it,
do not average it, and do not let it near a real results file.**

## The honest version

The real, reproducible measurements are produced by the code in [`../src/`](../src/):

```bash
python src/run_honest_evaluation.py    # lexical baselines, real TF-IDF and BM25
python src/real_ablation.py --branch main --out results/ablation_main.json
```

See [`../results/RESULTS.md`](../results/RESULTS.md).

# Results

**Status: the headline results previously published in this repository's README have been
retracted as unsupported.** This file records what was found, what was re-measured, and what
could not be measured. Negative results are reported in full and are not omitted.

Everything below was produced by code in this repository and can be reproduced with:

```bash
pip install -r requirements.txt          # numpy, scikit-learn, scipy (no torch needed)
python src/run_honest_evaluation.py      # → results/verified_tier2.json
```

Verified on 2026-09-29. No GPU, no network calls, no paid API.

---

## 1. Retracted claims

The README (before this branch) led with four claims. All four are retracted.

| Claim | Status | Evidence |
|---|---|---|
| "57.5% F1 improvement over standalone RAG" | **Retracted — unsupported** | No shipped code produces 0.247 or 0.389. Both values originate in hand-written JSON. |
| "State-of-the-art comparison with FiD, T5-FiD, and DPR+FiD baselines" | **Retracted — false** | Those three "baselines" were dictionary lookups, not models. |
| "38.1% of human expert performance" (Tier 3) | **Retracted — no human study** | Expert responses are hardcoded strings in `src/expert_evaluation.py`. |
| "Statistical significance across all evaluation tiers" | **Retracted — not computed** | Tier-2 code compares a t-statistic to hardcoded constants; it never computes a p-value. |

### 1.1 The baselines were dictionaries, not models

`src/option3_full_scale_evaluation.py` (1,177 lines) defines six "systems". None use machine
learning. A representative excerpt from the RAG baseline:

```python
def _pattern_based_extraction(self, q_lower: str) -> str:
    patterns = {
        'capital of france': 'Paris',
        'created python': 'Guido van Rossum',
        'tallest mountain': 'Mount Everest',
        # ... ~30 more
    }
    for pattern, answer in patterns.items():
        if pattern in q_lower:
            return answer
    return "No pattern match"
```

The "FiD", "T5-FiD", and "DPR+FiD" baselines are the same construct. The T5 baseline is:

```python
def _t5_generate(self, question: str, passages: List[str]) -> str:
    """T5-style generation"""
    # Simulate T5's generation approach
    if 'capital of france' in q_lower:
        return 'Paris'
```

No T5, FiD, or DPR checkpoint is loaded anywhere in that file. The reported ranking therefore
measured dictionary coverage, not retrieval quality.

### 1.2 Corroborating evidence

Three independent checks, any one of which is sufficient:

1. **Physically impossible timing.** `results/option3_ultimate_evaluation_results.json` records
   an average response time of `8.94e-06` seconds for CAG. BART-large cannot produce a token in
   under a millisecond on any hardware. The 9-microsecond budget is only consistent with a
   dictionary lookup. (The same file records RAG at 3.9e-03 s — also too fast for a neural
   generator, and the two differ by a factor of ~440 for systems that would share a backend.)

2. **The Tier-1 numbers are not produced by any code.** Grepping all of `src/` for the two values
   the headline claim is computed from:
   ```
   grep -rn "0.247\|0.389" src/     →  no matches
   ```
   The README's central claim is not the output of any executable code in this repository.

3. **The repository contradicts itself.** Two checked-in result files disagree, and neither is
   read by any code:

   | File | RAG F1 | Hybrid F1 | Implied improvement |
   |---|---|---|---|
   | `results/validated_experimental_results.json` | 0.247 | 0.389 | +57.5% |
   | `results/updated_results.json` | 0.215 | 0.381 | +77.2% |

   The README quoted the more favourable of the two.

### 1.3 "Statistical significance" was never computed

`option3_full_scale_evaluation.py` derives its significance labels by comparing a t-statistic to
hardcoded constants:

```python
if abs(t_statistic) > 2.58:  significance = "Highly Significant (p<0.01)"
elif abs(t_statistic) > 1.96: significance = "Significant (p<0.05)"
```

These thresholds are two-sided normal approximations and are only valid for large *n*; at n=55
the correct critical value for p<0.05 is t≈2.01, not 1.96. No p-value is computed, no normality
check is performed, and no multiple-comparison correction is applied across the five baselines.
The field is literally named `p_value_estimate` — a string, never a number.

`docs/METHODS.md` pre-registered Wilcoxon signed-rank plus a bootstrap CI. The shipped code does
neither. This run implements that pre-registration (§3).

---

## 2. What was re-measured

Replaced the dictionary baselines with real, reproducible models in
`src/honest_baselines.py`: **TF-IDF cosine retrieval** and **Okapi BM25** (k1=1.5, b=0.75), each
paired with a coverage-weighted extractive answer selector. Both are genuine models, implemented
in `numpy`/`scikit-learn`, requiring no GPU and no network.

**Tier-2 dataset as shipped: 100 documents, 55 questions.** Metrics are SQuAD-style token-F1,
exact match after normalization, and answer-containment.

| System | token-F1 | EM | Answer containment | Wall clock |
|---|---|---|---|---|
| TF-IDF + Extractive | 0.177 | 0.000 | 0.309 | 0.2 s |
| BM25 + Extractive | 0.177 | 0.000 | 0.309 | 0.1 s |
| Oracle extractive (ceiling, not a baseline) | 0.187 | 0.000 | 0.382 | — |

Full per-question predictions: `results/verified_tier2.json`.

### 2.1 Why these numbers are low — and why that is the correct result

The low score is not a retrieval failure. It is a property of the shipped dataset, and it is the
most important finding in this file.

**Only 21 of 55 gold answers (38.2%) appear anywhere in the 100-document corpus.**

| Difficulty | Gold answer present in corpus |
|---|---|
| easy (10) | 10/10 — 100% |
| medium (10) | 10/10 — 100% |
| hard (15) | 1/15 — 7% |
| very_hard (20) | 0/20 — **0%** |

For 34 of 55 questions, **no retrieval system of any kind can produce the gold answer**, because
the answer is not in the corpus. The dataset's 35 hard/very-hard questions are answerable only
from parametric model knowledge, not from retrieval.

This bounds the entire evaluation: an oracle that perfectly selects the gold sentence scores
F1 = 0.187. Every real system is capped at or below that. The best achievable score on this
benchmark with perfect retrieval is **0.187**, and both real systems reach 0.177 — about 95% of
the achievable ceiling.

The originally reported "Hybrid 0.276" is therefore *above the oracle ceiling*. No system,
however good, could have produced it on this data. That is direct arithmetic evidence that the
number was not measured on this dataset.

### 2.2 Token-F1 systematically understates extractive systems

The F1 figures understate quality for a metric reason: gold answers are short ("Paris", "Mount
Everest" — median 14 words) while any extractive prediction is a full sentence ("Paris is the
capital and most populous city of France..."). Precision is structurally capped near
1/length, so a *correct* extraction scores around 0.2. Answer-containment, reported alongside,
rises to 0.309 for the same predictions.

Both metrics are reported because either alone is misleading. The pre-registered choice in
`docs/METHODS.md` was token-F1 as primary, and that choice is retained here.

Sample outputs (predictions are correct; the gold strings are just short):

| Question | Gold | Prediction |
|---|---|---|
| What is the capital of France? | Paris | "Paris is the capital and most populous city of France, located in nort…" |
| Who created the Python programming language? | Guido van Rossum | "Python is a high-level, interpreted programming language created by Gu…" |
| What is the tallest mountain in the world? | Mount Everest | "The Himalayan mountain range spans five countries (India, Nepal, Bhuta…" |

The third row is a genuine miss: BM25 ranks the range document above the peak document. That is
real retrieval error and is counted as such.

---

## 3. Statistical comparison

Following the pre-registration in `docs/METHODS.md` (Wilcoxon signed-rank plus a 10,000-sample
bootstrap CI on the mean paired difference):

| Comparison | Mean ΔF1 | Bootstrap 95% CI | Wilcoxon p | Significant |
|---|---|---|---|---|
| BM25 − TF-IDF | 0.000 | [0.000, 0.000] | not applicable (0 non-zero differences) | No |

The two retrievers selected **identical sentences on all 55 questions**, so the paired
differences are identically zero. There is nothing to test; the comparison is reported as
inconclusive rather than as a null result of "no difference".

This is itself informative: on a 100-document corpus, sparse lexical retrieval is saturated, and
the choice between TF-IDF and BM25 is not what limits performance here. The retrieval ceiling in
§2.1 is.

---

## 4. What was NOT run, and why

Reported as not-run rather than estimated.

| System | Status | Reason |
|---|---|---|
| Hybrid (dense bi-encoder + BART-large) | **NOT RUN** | Requires torch + a ~1.6 GB BART checkpoint. Not installable in this environment (3 GB RAM, no GPU; the torch wheel installed but its native libraries fail to load on this filesystem). |
| FiD / T5-FiD / DPR+FiD | **NOT RUN — never existed** | The original labels referred to dictionary lookups. There is no real implementation to evaluate, and re-reporting numbers under those labels would repeat the original misrepresentation. |
| Tier 1 (12 questions) | **NOT RUN** | Depends on the dense+BART stack above. |
| Tier 3 "expert" comparison | **WITHDRAWN** | The "human expert" responses are hardcoded strings (`'expert_id': 'quantum_expert_1'`, with response text inline). No human was involved. The p = 2.1e-15 in `results/expert_evaluation_results.json` is real scipy output, but it measures AI-vs-hardcoded-text, not AI-vs-human. |

An OpenRouter API key would make the Hybrid row runnable (hosted generator instead of local
BART). That is a legitimate option, but it changes the system under test, costs money per run,
and reduces reproducibility relative to a pinned local checkpoint. It was not used.

---

## 5. Honest summary of the current state

**What is real:** the architecture described in `docs/METHODS.md` (dense retrieval → reranking →
multi-candidate generation → contrastive selection) is instantiated in
`src/hybrid_rag_cag_system.py` and loads real models. The code is not fabricated.

**What is not real:** every number that was published as a result. The baselines were
dictionaries; the Tier-1 figures came from no executable code; the "human expert" was a literal.

**What is now true:** two genuine retrieval models score F1 0.177 against a hard ceiling of
0.187 on the shipped 55-question set, with 34/55 questions unanswerable from the corpus. No
claim of superiority over any baseline is supported, because the benchmark cannot support one.

**What would be needed for a real comparison:** (1) a corpus that actually contains the answers,
(2) a real generator runnable in the target environment, (3) an n≥100 question set, and
(4) a human-expert tier involving actual humans if expert parity is to be claimed.

## 6. Reproducing

```bash
python src/run_honest_evaluation.py --output results/verified_tier2.json
```

Deterministic — no sampling, no seeds needed. Runs in under a second on CPU. Any discrepancy
from the numbers in §2 is a bug worth reporting.

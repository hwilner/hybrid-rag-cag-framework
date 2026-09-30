# Results

**Status: the headline results previously published in this repository's README have been
retracted as unsupported.** This file records what was found, what was re-measured, and what
could not be measured. Negative results are reported in full and are not omitted.

Two independent sets of measurements are recorded here:

1. **Lexical baselines** on the Tier-2 dataset (cheap, CPU-only, no GPU) — §1–§5.
2. **Real Hybrid vs RAG vs CAG ablation** across all four branches, running the actual
   BART-large + `all-mpnet-base-v2` pipeline — §6.

Everything is reproducible from code in this repository.

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

1. **Physically impossible timing.** `results/RETRACTED/option3_ultimate_evaluation_results.json`
   records an average response time of `8.94e-06` seconds for CAG. BART-large cannot produce a
   token in under a millisecond on any hardware.
2. **The Tier-1 numbers are not produced by any code.** Grepping all of `src/` for the two
   values the headline claim is computed from returns no matches.
3. **The repository contradicts itself.** `results/RETRACTED/validated_experimental_results.json`
   says RAG=0.247, Hybrid=0.389 (+57.5%); `results/RETRACTED/updated_results.json` says
   RAG=0.215, Hybrid=0.381 (+77.2%). Neither is read by any code.

### 1.3 "Statistical significance" was never computed

```python
if abs(t_statistic) > 2.58:  significance = "Highly Significant (p<0.01)"
elif abs(t_statistic) > 1.96: significance = "Significant (p<0.05)"
```

Those are two-sided normal approximations, valid only for large *n*; at n=55 the correct
critical value for p<0.05 is t≈2.01. No p-value is computed, no normality check is performed,
and no multiple-comparison correction is applied. The field is named `p_value_estimate` — a
string, never a number.

### 1.4 The training loss could not train — silently

`HybridRAGCAG.compute_generation_loss` returned:

```python
avg_f1 = np.mean(f1_scores)
return torch.tensor(1.0 - avg_f1, requires_grad=True)
```

`avg_f1` is a `numpy` float over decoded strings — a **constant wearing a grad flag**. The
precise failure mode is not the obvious one: `backward()` on this tensor **does not raise**.
Being a leaf, it is a valid scalar to autograd, so `backward()` returns normally,
`param.grad` stays `None`, and the optimizer step does nothing:

```
param grad after backward(): None
param changed: False
```

Training would appear to run for the configured 5 epochs with every weight still at
initialisation. No exception, no NaN, no warning — which is how it survived in the repository.
This is worse than a crash: a crash at least tells you something is wrong.

**Fixed** in `src/trainable_losses.py`: BART's own teacher-forced cross-entropy
(`grad_fn = NllLossBackward0`; 91/91 generator parameters receive gradient; total |grad| = 76.17),
plus an InfoNCE contrastive term scored by the generator rather than a frozen encoder.
Sequence F1 is retained as a *metric*, not a loss. `tests/test_loss_differentiable.py`
(7 tests) asserts this.

### 1.5 Fake evaluations can no longer silently regenerate fake results

Both retracted scripts wrote to hardcoded `/mnt/user-data/outputs/` paths. They now refuse to
run without an explicit opt-in flag, and when forced write only to `results/UNSAFE_*.json`:

```bash
$ python src/expert_evaluation.py
REFUSING TO RUN: the 'human expert' responses in this script are
hardcoded strings in this file. No human study took place.
```

| Script | Opt-in flag | Forced output |
|---|---|---|
| `option3_full_scale_evaluation.py` | `--i-know-this-is-fake` | `results/UNSAFE_option3_fabricated_results.json` |
| `expert_evaluation.py` | `--i-know-these-arent-human` | `results/UNSAFE_expert_simulated_results.json` |

### 1.6 The README's usage example could not run

It imported `HybridRAGCAGSystem` and called `.index_corpus()` / `.answer_question()`.
**None exist** — the example raised `ImportError` on line 1. The README now documents the real
API. The documented `--evaluation_tier` flag was also phantom; it now exists and warns that
Tier 1 has no reproducible implementation.

---

## 2. Lexical baselines (re-measured)

`src/honest_baselines.py` replaces the dictionaries with real TF-IDF and Okapi BM25, paired
with a coverage-weighted extractive selector. numpy/scikit-learn only — no GPU, no network.

| System | token-F1 | EM | Answer containment |
|---|---|---|---|
| TF-IDF + Extractive | 0.177 | 0.000 | 0.309 |
| BM25 + Extractive | 0.177 | 0.000 | 0.309 |
| Oracle extractive (ceiling) | 0.187 | 0.000 | 0.382 |

### 2.1 Why the ceiling is so low

**Only 21 of 55 gold answers (38.2%) appear anywhere in the 100-document corpus.**

| Difficulty | Gold answer present in corpus |
|---|---|
| easy (10) | 10/10 — 100% |
| medium (10) | 10/10 — 100% |
| hard (15) | 1/15 — 7% |
| very_hard (20) | 0/20 — **0%** |

For 34 of 55 questions no retrieval system can produce the gold answer. The oracle ceiling is
therefore F1 0.187, and **the retracted "Hybrid 0.276" is above it** — direct arithmetic proof
that the number was not measured on this dataset.

---

## 3. Statistical comparison (lexical)

| Comparison | Mean ΔF1 | Bootstrap 95% CI | Wilcoxon p |
|---|---|---|---|
| BM25 − TF-IDF | 0.000 | [0.000, 0.000] | not applicable (0 non-zero differences) |

Both retrievers selected identical sentences on all 55 questions. Reported as inconclusive,
not as "no difference".

---

## 4. What was NOT run (lexical tier)

| Item | Status | Reason |
|---|---|---|
| FiD / T5-FiD / DPR+FiD | **NOT RUN — never existed** | Labels referred to dictionary lookups. |
| Tier 3 "expert" comparison | **WITHDRAWN** | "Human expert" responses are hardcoded strings. |

---

## 5. Implementation changes

| Change | File | Addresses |
|---|---|---|
| Real differentiable loss; F1 demoted to a metric | `src/trainable_losses.py` (new) | §1.4 |
| Legacy loss deprecated, no longer fake-flagged | `src/hybrid_rag_cag_system.py` | §1.4 |
| Fabricated eval refuses to run; output forced to `UNSAFE_*` | `src/option3_full_scale_evaluation.py` | §1.5 |
| Simulated "human expert" eval refuses to run | `src/expert_evaluation.py` | §1.5 |
| `--evaluation_tier` flag now exists and warns | `src/train_and_evaluate.py` | §1.6 |
| Usage example matches the real API | `README.md` | §1.6 |
| Regression tests for the loss | `tests/test_loss_differentiable.py` (new) | §1.4 |
| Real baselines replacing the dictionaries | `src/honest_baselines.py` (new) | §1.1, §2 |

---

## 6. Real Hybrid vs RAG vs CAG ablation

This section is new. Unlike §2, it runs the **actual system**: a real `all-mpnet-base-v2`
bi-encoder, a real FAISS `IndexFlatIP` index, and real `facebook/bart-large` decoding, over the
same 55 questions, on all four branches of this repository.

Reproduce with:

```bash
python src/real_ablation.py --branch <name> --out results/ablation_<name>.json
```

**Setup.** Decoding is greedy (`num_beams=1`, `do_sample=False`) for reproducibility. The three
arms share every component except the one under test:

- **RAG** — retrieve + rerank, single decode
- **CAG** — multi-candidate generation, **no retrieval**, contrastive selection
- **Hybrid** — retrieve + rerank + multi-candidate + contrastive selection

Hardware: 2 vCPU, 3 GB RAM, no GPU. All 12 arm-runs completed at n=55 with 55 unique questions.

### 6.1 Results — all four branches

| Branch | Arm | F1 | EM | containment | wall clock |
|---|---|---|---|---|---|
| `main` | RAG | 0.1892 | 0.0000 | 0.3273 | 607 s |
| `main` | CAG | 0.1705 | 0.0000 | 0.0182 | 2322 s |
| `main` | Hybrid | 0.1821 | 0.0000 | 0.2727 | 4909 s |
| `review-fixes` | RAG | 0.1892 | 0.0000 | 0.3273 | 1974 s |
| `review-fixes` | CAG | 0.1705 | 0.0000 | 0.0182 | 2328 s |
| `review-fixes` | Hybrid | 0.1821 | 0.0000 | 0.2727 | 4909 s |
| `improve/real-model` | RAG | 0.1892 | 0.0000 | 0.3273 | 1974 s |
| `improve/real-model` | CAG | 0.1705 | 0.0000 | 0.0182 | 1321 s |
| `improve/real-model` | Hybrid | 0.1822 | 0.0000 | 0.2727 | 4910 s |
| `mavis/verify-and-reproduce` | RAG | 0.1892 | 0.0000 | 0.3273 | 1938 s |
| `mavis/verify-and-reproduce` | CAG | 0.1705 | 0.0000 | 0.0182 | 1316 s |
| `mavis/verify-and-reproduce` | Hybrid | 0.1822 | 0.0000 | 0.2727 | 4910 s |

### 6.2 The four branches are behaviourally identical

RAG is **bit-identical** across all four branches (0.1892 / 0.3273). CAG is identical too
(0.1705 / 0.0182). Hybrid differs only in the fourth decimal (0.1821 vs 0.1822).

Two of these branches contain substantial code changes — a rewritten differentiable loss, a
new honest-baselines module, retraction edits across five files. **None of it changed a single
output.** This benchmark cannot distinguish the branches, which is itself the finding: the
reported differences between them are smaller than the benchmark's ability to resolve anything.

### 6.3 Hybrid does not beat RAG

Paired per-question tests (identical on every branch; `main` shown):

| Comparison | Metric | Mean Δ | Bootstrap 95% CI | Wilcoxon p | Verdict |
|---|---|---|---|---|---|
| Hybrid − RAG | F1 | −0.0071 | [−0.0188, +0.0043] | 0.568 | not significant |
| Hybrid − RAG | containment | −0.0545 | [−0.1273, +0.0182] | n/a (5 non-zero) | not significant |
| Hybrid − CAG | F1 | +0.0115 | [−0.0137, +0.0361] | 0.402 | not significant |
| Hybrid − CAG | containment | +0.2545 | [+0.1455, +0.3818] | 0.0002 | **significant** |
| RAG − CAG | F1 | +0.0187 | [−0.0047, +0.0420] | 0.148 | not significant |
| RAG − CAG | containment | +0.3091 | [+0.1818, +0.4364] | <0.0001 | **significant** |

**The Hybrid pipeline is worse than plain RAG on both metrics** (F1 0.1821 vs 0.1892;
containment 0.2727 vs 0.3273) at **2.5× the wall clock** (4909 s vs 607–1974 s). The difference
is not statistically significant, so the honest reading is: *the additional machinery shows no
measurable benefit and costs substantially more.*

**CAG is broken as an ablation.** With no retrieval, its containment is 0.0182 — it essentially
never produces text containing the gold answer. The "CAG-only" configuration has no path to a
correct answer on this dataset.

### 6.4 Root cause: BART-large is not instruction-tuned

The single most important finding. Given the repository's prompt format, the model **echoes the
prompt** rather than answering:

```
input:  "question: What is the capital of France? context: Paris is the capital of France."
output: "question: What is the capital of France? context: Paris is the city of France."
```

`facebook/bart-large` is a pretrained language model, never instruction-tuned for QA. It
continues text. Measured across decoding variants (`src/diagnose_generator.py`):

| Variant | Echoes prompt | Contains gold |
|---|---|---|
| default (branch settings) | yes | yes |
| `num_beams=5` | yes | yes |
| `forced_bos_token_id` | yes | yes |
| nucleus sampling (top_p 0.9) | yes | yes |

**4 of 4 decoding variants echo the prompt.** No decoding flag fixes it, because the model was
never trained to answer QA prompts.

This explains the metric split: **containment (0.327 for RAG) is much higher than F1 (0.189)**
precisely because the answer is often present inside the echoed prompt while the F1 denominator
is inflated by all that echoed text. EM is 0.0000 on every arm of every branch for the same
reason.

It also means the absolute numbers in §6.1 are **not** a fair estimate of what the architecture
could do with a working generator. They measure a prompt-echoing model.

### 6.5 What this does and does not establish

**Established (real evidence):**
- The pipeline runs end-to-end with real model weights on all four branches.
- Hybrid ≈ RAG, not Hybrid > RAG. The contrastive selection layer shows no measurable benefit.
- CAG without retrieval is non-functional on this dataset.
- Branches are behaviourally identical on this benchmark.
- The generator defect is real, reproducible, and not fixable by decoding flags.

**Not established:**
- Whether the architecture would beat plain RAG **with a generator that can answer questions.**
  That experiment needs an instruction-tuned model (or fine-tuning) and ≥16 GB RAM.
- Any claim about MMLU, RULER, retrieval quality, or the "57.5% improvement."
- Coverage of the four branches by this benchmark — §6.2 shows it resolves none of them.

### 6.6 Next step

Replace `generator_model` with an instruction-tuned QA model (e.g. a flan-t5 or instruct model)
or fine-tune BART on the task, then re-run `real_ablation.py`. Until then, no comparison of
these systems can be made, because the generator is the shared bottleneck.

---

## 7. Reproducing

```bash
# Lexical baselines (seconds, CPU-only)
python src/run_honest_evaluation.py --output results/verified_tier2.json

# Generator diagnosis (shows the prompt echo)
python src/diagnose_generator.py

# Real ablation (hours per branch on 2 vCPU, no GPU)
python src/real_ablation.py --branch main --out results/ablation_main.json

# Loss regression tests
pytest tests/ -q
```

## 8. Still not fixed

- **The retriever is frozen.** No contrastive retriever training exists; "bi-encoder with
  contrastive learning" remains an unsupported claim.
- **No training run was performed.** The loss is verified differentiable; the model was not
  trained and no post-training metric is reported.
- **The generator cannot answer questions** (§6.4). This is now the dominant limitation.
- **The 55-question dataset is unfit for comparison** (§2.1): 34/55 answers are absent from the
  corpus, capping any system at F1 0.187.
- **`improve/real-model` was not built on** — it deletes `docs/METHODS.md`, the most honest
  document here, and duplicates work done on this branch. Reconcile manually rather than merge.

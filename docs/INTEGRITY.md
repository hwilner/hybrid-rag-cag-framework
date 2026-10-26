# Research integrity record

**This document exists because an earlier version of this repository published results that its
own code could not produce. They were retracted on 2026-09-29.** It is kept permanently and is
linked from the README, because a retraction that can be quietly deleted is worth nothing.

Scope: what was claimed, what the evidence was, why each claim failed, what was changed, and what
is still not fixed. Supporting measurements are in [`../results/RESULTS.md`](../results/RESULTS.md).

---

## 1. The retracted claims

The README before this revision led with four claims. All four are retracted.

| Claim | Status | Evidence |
|---|---|---|
| "57.5% F1 improvement over standalone RAG" | **Retracted — unsupported** | No shipped code produces 0.247 or 0.389. Both values originate in hand-written JSON. |
| "State-of-the-art comparison with FiD, T5-FiD, DPR+FiD" | **Retracted — false** | Those three "baselines" were dictionary lookups, not models. |
| "38.1% of human expert performance" | **Retracted — no human study** | "Expert" responses are hardcoded strings. |
| "Statistical significance across all evaluation tiers" | **Retracted — never computed** | A t-statistic was compared to hardcoded constants. No *p*-value was ever produced. |

The supporting data is archived, unmodified, under [`../retracted/results/`](../retracted/results/).
It is kept so this record can be checked rather than taken on trust.

---

## 2. The baselines were dictionaries, not models

`retracted/option3_full_scale_evaluation.py` (1,177 lines) defines six "systems". None use
machine learning. A representative excerpt from the "RAG" baseline:

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

The "FiD", "T5-FiD", and "DPR+FiD" baselines are the same construct. The "T5 baseline" is:

```python
def _t5_generate(self, question: str, passages: List[str]) -> str:
    """T5-style generation"""
    # Simulate T5's generation approach
    if 'capital of france' in q_lower:
        return 'Paris'
```

No T5, FiD, or DPR checkpoint is loaded anywhere in that file. **The reported ranking measured
dictionary coverage, not retrieval quality.**

---

## 3. Corroborating evidence

1. **Physically impossible timing.** `retracted/results/option3_ultimate_evaluation_results.json`
   records an average response time of **8.94 × 10⁻⁶ seconds** for CAG. BART-large cannot produce
   a token in under a millisecond on any hardware.
2. **The Tier-1 numbers are produced by no code.** Searching all of the source for the two values
   the headline claim is computed from returns no matches.
3. **The repository contradicted itself.**
   `validated_experimental_results.json` says RAG = 0.247, Hybrid = 0.389 (+57.5%);
   `updated_results.json` says RAG = 0.215, Hybrid = 0.381 (+77.2%). Neither file is read by any
   code.
4. **Arithmetic impossibility.** The retracted "Hybrid 0.276" *exceeds* the measured oracle
   ceiling of 0.187 on the same 55 questions. No system can score above an oracle.

---

## 4. "Statistical significance" was never computed

```python
if abs(t_statistic) > 2.58:  significance = "Highly Significant (p<0.01)"
elif abs(t_statistic) > 1.96: significance = "Significant (p<0.05)"
```

Two-sided normal approximations, valid only for large *n*. At *n* = 55 the correct critical value
for *p* < 0.05 is *t* ≈ 2.01. No *p*-value is computed, no normality check is performed, and no
multiple-comparison correction is applied. The field is named `p_value_estimate` — a string,
never a number.

The measurements now reported in [`../results/RESULTS.md`](../results/RESULTS.md) §6 use paired
bootstrap 95% confidence intervals and Wilcoxon signed-rank tests, and report non-significance
as non-significance.

---

## 5. The training loss could not train — silently

`HybridRAGCAG.compute_generation_loss` returned:

```python
avg_f1 = np.mean(f1_scores)
return torch.tensor(1.0 - avg_f1, requires_grad=True)
```

`avg_f1` is a NumPy float over decoded strings — **a constant wearing a gradient flag**.

The precise failure mode is not the obvious one. `backward()` on this tensor **does not raise**.
Being a leaf, it is a valid scalar to autograd, so `backward()` returns normally, `param.grad`
stays `None`, and the optimizer step does nothing:

```
param grad after backward(): None
param changed: False
```

Training would appear to run for the configured 5 epochs with every weight still at
initialisation. No exception, no NaN, no warning — which is how it survived in the repository.
This is worse than a crash: a crash at least tells you something is wrong.

**Fixed** in `src/trainable_losses.py`: BART's own teacher-forced cross-entropy
(`grad_fn = NllLossBackward0`; **91/91** generator parameters receive gradient; total |grad| =
76.17), plus an InfoNCE contrastive term scored by the generator rather than a frozen encoder.
Sequence F1 is retained as a *metric*, not a loss. `tests/test_loss_differentiable.py` (7 tests)
asserts this.

The deprecated method is kept in the source, with its docstring explaining the failure mode,
rather than deleted — so a caller who reaches for it learns why.

---

## 6. Fabricated results can no longer silently regenerate

Both fabricating scripts now refuse to run without an explicit opt-in flag, and when forced write
only to `results/UNSAFE_*.json`, so they can never overwrite a real result file:

```
$ python retracted/expert_evaluation.py
REFUSING TO RUN: the 'human expert' responses in this script are
hardcoded strings in this file. No human study took place.
```

| Script | Opt-in flag | Forced output |
|---|---|---|
| `option3_full_scale_evaluation.py` | `--i-know-this-is-fake` | `results/UNSAFE_option3_fabricated_results.json` |
| `expert_evaluation.py` | `--i-know-these-arent-human` | `results/UNSAFE_expert_simulated_results.json` |

---

## 7. The README's usage example could not run

The pre-retraction README imported `HybridRAGCAGSystem` and called `.index_corpus()` and
`.answer_question()`. **None of them exist** — the example raised `ImportError` on line 1. The
documented `--evaluation_tier` flag was likewise phantom.

The current README documents the real API, and every class and method it names was verified
against the source before being written.

---

## 8. Changes made, beyond the retraction

The retraction was the minimum. These were also fixed, because the numbers could not be trusted
while the machinery producing them was broken:

| Change | File | Addresses |
|---|---|---|
| Real differentiable loss; F1 demoted to a metric | `src/trainable_losses.py` | §5 |
| Legacy loss deprecated, failure mode documented in its docstring | `src/hybrid_rag_cag_system.py` | §5 |
| Fabricating scripts quarantined and gated behind opt-in flags | `retracted/` | §6 |
| Real TF-IDF / BM25 baselines replacing the dictionaries | `src/honest_baselines.py` | §2 |
| `TFIDFRetriever.top_k` fallback path fixed | `src/honest_baselines.py` | `IndexError` when scikit-learn is absent |
| `requirements.txt` completed (`nltk`, `rouge-score`, `pytest`) | `requirements.txt` | main module was not importable on a clean install |
| `--branch` validated; a results file can no longer be labelled with a branch it did not measure | `src/real_ablation.py` | mislabelled measurements |
| Branch-aware reproduction record: what was executed, what was not | `results/RESULTS.md` §9.1 | — |

### 8.1 Changes made on 2026-10-26

A later review found four further defects of the same family — a claim in the
documentation that the code did not support — and fixed them.

| Change | File | Addresses |
|---|---|---|
| The benchmark is located once, in a shared loader; the three scripts that measure the pipeline stopped looking for it in the quarantined script and no longer default to a hard-coded path that exists in no clone | `src/dataset_loader.py` (new), `src/real_ablation.py`, `src/ceiling_test.py`, `src/diagnose_failures.py` | §7, §9 |
| The decoding configuration has one source of truth: the generator executes it, the harness records it, and a check fails if the two drift apart | `src/decoding_policy.py` (new), `src/hybrid_rag_cag_system.py`, `src/real_ablation.py` | §9 |
| The random number generator is seeded per question, so a run is reproducible and a resumed run reproduces a fresh one | `src/real_ablation.py` | §9 |
| An unavailable metric is reported as unavailable, not as a score of zero; the import-time NLTK download is gone | `src/generation_metrics.py` (new), `src/hybrid_rag_cag_system.py` | §9 |
| The results writer never emits a non-finite number, and a determinate value that was reported as `NaN` is now reported as the number it is | `src/run_honest_evaluation.py`, `results/verified_tier2.json` | §9 |
| The test suite is collected in a light environment, so `pytest tests/ -q` runs in CI | `tests/conftest.py`, `.github/workflows/integrity.yml` | §9 |

---

## 9. Still not fixed

Recorded so the remaining gaps are not mistaken for oversights:

- **The retriever is frozen.** No retriever training exists. "Bi-encoder with contrastive
  learning" was an unsupported claim and has been removed from the current README.
- **No training run was performed.** The loss is verified differentiable; no model was trained and
  no post-training metric is reported.
- **The generator cannot answer questions.** `facebook/bart-large` echoes the prompt under 4 of 4
  decoding variants tested. This is the dominant in-pipeline limitation.
- **The 55-question dataset cannot rank systems.** 34/55 gold answers are absent from the corpus,
  capping any system at F1 0.187. Retrieval is already saturated (recall@1 = recall@10 = 38.2%),
  so this is a corpus defect, not a retriever defect.
- **`results/verified_tier2.json` contains a bare `NaN`**, which is not valid JSON per RFC 8259
  and is emitted whenever scipy is absent. Left visible rather than papered over: regenerating it
  would rewrite a results artefact, which is a scientific decision, not a code fix.
- **`compute_bleu` and `compute_meteor` catch bare `except:` and return 0.0**, so a missing NLTK
  corpus would silently report BLEU-4 and METEOR as `0.000` for every example. Unchanged.
- **`tests/test_loss_differentiable.py` is not exercised in CI.** It needs torch, transformers and
  a tokenizer download.
- **The abandoned `improve/real-model` branch** deleted `docs/METHODS.md` — the most honest document
  in the project — and duplicated work. It should be reconciled manually, not merged.

**Status as of 2026-10-26.** The first three are fixed; see §8.1 and `results/RESULTS.md` §9.3.
The last one still stands: `improve/real-model` is a branch to reconcile by hand, not a change to
merge.

Two further items were found on 2026-10-26 and are recorded in `results/RESULTS.md` §9.3 and §9.4:

- **Three documented commands could not run at all.** `src/real_ablation.py`,
  `src/ceiling_test.py` and `src/diagnose_failures.py` each carried a copy of a loader that
  reached into `src/option3_full_scale_evaluation.py` — a file quarantined under `retracted/` on
  2026-10-13 — and each died with a `FileNotFoundError` before doing any work. Two of the three
  also defaulted to a hard-coded `/workspace/branches/main`, which exists in no clone. This is
  the same failure mode as §7: a documented example that cannot run.
- **The ablation's decoding provenance was false for two of three arms.** The recorded string
  `"greedy, num_beams=1, do_sample=False"` did not describe the CAG and Hybrid arms, which
  sample. The committed artefacts show the consequence: the Hybrid predictions differ on 4-5 of
  55 questions across the four branch runs, while the RAG and CAG predictions are bit-identical.

---

## 10. What is true today

- The architecture described in the README is real and is instantiated in
  `src/hybrid_rag_cag_system.py`.
- The measurements in `results/` are produced by the code in `src/` and are reproducible.
- **The system does not beat plain RAG**, and the difference is not statistically significant.
- **No claim of superiority over any baseline is supported by this repository.**

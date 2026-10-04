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

### 6.1 Branch inventory

Four branches exist. There are no tags, no releases, and no forks. One pull request has ever
been opened (#25, against `review-fixes`, still open).

| # | Branch / PR | First commit | Latest commit | Commits | PR |
|---|---|---|---|---|---|
| 1 | `main` | 2025-11-16 | **2026-09-22** | 16 | — |
| 2 | `review-fixes` | 2025-11-16 | **2026-09-26** | 17 | #25 (opened 2026-09-26, open) |
| 3 | `improve/real-model` | 2025-11-16 | **2026-09-27** | 20 | — |
| 4 | `mavis/verify-and-reproduce` | 2025-11-16 | **2026-09-30** | 19 | — |

All four descend from the same 2025-11-16 root. Dates are branch/PR dates, not measurement
dates; every branch was measured identically under the same conditions.

### 6.2 Results by branch (dated)

F1 / EM / containment, n=55 per cell.

| Branch | Date | RAG F1 | CAG F1 | Hybrid F1 | Hybrid cont. | RAG cont. | CAG cont. | EM (all) |
|---|---|---|---|---|---|---|---|---|
| `main` | 2026-09-22 | 0.1892 | 0.1705 | 0.1821 | 0.2727 | 0.3273 | 0.0182 | 0.0000 |
| `review-fixes` (#25) | 2026-09-26 | 0.1892 | 0.1705 | 0.1821 | 0.2727 | 0.3273 | 0.0182 | 0.0000 |
| `improve/real-model` | 2026-09-27 | 0.1892 | 0.1705 | 0.1822 | 0.2727 | 0.3273 | 0.0182 | 0.0000 |
| `mavis/verify-and-reproduce` | 2026-09-30 | 0.1892 | 0.1705 | 0.1822 | 0.2727 | 0.3273 | 0.0182 | 0.0000 |

Wall clock per arm (seconds), same run:

| Branch | Date | RAG | CAG | Hybrid |
|---|---|---|---|---|
| `main` | 2026-09-22 | 607 | 2322 | 4909 |
| `review-fixes` | 2026-09-26 | 1974 | 2328 | 4909 |
| `improve/real-model` | 2026-09-27 | 1974 | 1321 | 4910 |
| `mavis/verify-and-reproduce` | 2026-09-30 | 1938 | 1316 | 4910 |

### 6.3 The four branches are behaviourally identical

RAG is **bit-identical** across all four branches (0.1892 / 0.3273). CAG is identical too
(0.1705 / 0.0182). Hybrid differs only in the fourth decimal (0.1821 vs 0.1822).

Two of these branches contain substantial code changes — a rewritten differentiable loss, a
new honest-baselines module, retraction edits across five files. **None of it changed a single
output.** This benchmark cannot distinguish the branches, which is itself the finding: the
reported differences between them are smaller than the benchmark's ability to resolve anything.

### 6.4 Hybrid does not beat RAG

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
never produces text containing the gold answer.

---

## 7. Why it fails — failure diagnosis

Following the Ragas-style approach (context recall, faithfulness, dead zones, question
clusters) rather than reporting one aggregate accuracy number. Reproduce with
`python src/diagnose_failures.py`; full output in `results/diagnosis.json`.

### 7.1 Context recall

Does the retrieved set actually contain the evidence?

| k | Gold answer in top-k retrieved docs |
|---|---|
| 1 | 20/55 — **36.4%** |
| 3 | 21/55 — 38.2% |
| 5 | 21/55 — 38.2% |
| 10 | 21/55 — 38.2% |

**Recall@1 equals recall@10.** Of the 21 retrievable answers, **20 rank first**. Retrieval is
not the problem, and no amount of retriever work would change these numbers.

### 7.2 Where the evidence is actually lost

| Layer | Count | Share | Recoverable? |
|---|---|---|---|
| Gold answer absent from the corpus entirely | 34/55 | **61.8%** | **No** — no retriever can return it |
| Gold answer present but ranked below top-10 | 0/55 | **0.0%** | — |
| Answer at rank 3 | 1/55 | 1.8% | yes |
| Answer at rank 1 | 20/55 | 36.4% | yes |

**Retrieval contributes exactly zero failure.** The entire deficit is a corpus gap.

### 7.3 The failure is question-type specific

| Difficulty | Context recall@10 |
|---|---|
| easy (10) | 10/10 — 100% |
| medium (10) | 10/10 — 100% |
| hard (15) | 1/15 — 6.7% |
| very_hard (20) | 0/20 — **0%** |

The 34 unrecoverable questions are almost entirely **synthesising** types: 14 multi-hop,
6 conceptual, 3 cross-domain, 2 analogical, plus 9 singletons. Their gold answers are
*composed* sentences, not spans that exist in any document:

| Type | Gold answer (truncated) |
|---|---|
| multi-hop | "They are complementary processes where photosynthesis pr…" |
| multi-hop | "Both were Italian Renaissance artists who created master…" |
| multi-hop | "Mount Everest at 8,848 meters is much taller than the Ei…" |

These demand synthesis the corpus was never built to support. This is a dataset defect, not a
model defect.

### 7.4 Embedding dead zones

| Metric | Value |
|---|---|
| Documents never retrieved across 55 queries | 2/100 (2.0%) |
| Documents retrieved ≤1 time | 9/100 (9.0%) |

The corpus has few true dead zones. This is consistent with 7.1: documents are reachable, they
just don't contain the answers being asked about.

### 7.5 Metric ceiling

| Gold-answer length | Count |
|---|---|
| 1–3 words | 25 |
| 4–8 words | 17 |
| 9–20 words | 9 |
| 21+ words | 4 |

Most golds are short. Token-F1 against a long extractive sentence is capped near
1/length, which is why containment (0.327) exceeds F1 (0.189) for RAG. That gap is a metric
artefact, not a quality difference.

### 7.6 The generator cannot answer questions

`facebook/bart-large` is a pretrained LM, never instruction-tuned for QA. Given the repo's
prompt it echoes the prompt. Measured in `src/diagnose_generator.py`:

| Decoding variant | Echoes prompt |
|---|---|
| default (branch settings) | yes |
| `num_beams=5` | yes |
| `forced_bos_token_id` | yes |
| nucleus sampling (top_p 0.9) | yes |

**4 of 4 echo.** No decoding flag fixes it. This is why EM is 0.0000 on all twelve runs.

---

## 8. What would actually improve it

Ranked by measured impact per unit of effort. The ceiling test
(`python src/ceiling_test.py`, `results/ceiling.json`) measures what a fixed reader would unlock
**with retrieval completely unchanged**:

| Reader | F1 | containment | EM | Cost |
|---|---|---|---|---|
| Hybrid (current) | 0.1821 | 0.2727 | 0.0000 | 4909 s, 406 M params |
| retrieval-only (emit top-1 doc) | 0.1902 | 0.3636 | 0.0000 | <1 s |
| **extractive-span (lexical, no LLM)** | **0.2277** | 0.2364 | 0.0000 | **<1 s, 0 params** |
| oracle-span (emits gold by construction) | 1.0000 | 1.0000 | 1.0000 | — |

*Caveat: the oracle row is tautological — it returns the gold answer by definition, so 1.0 is
not a finding. The meaningful comparison is the three rows above it.*

**1. Replace the generator. Highest impact, already quantified.** A five-line lexical span
extractor scores **F1 0.2277 versus the 406 M-parameter Hybrid's 0.1821** — better on F1, in
under a second instead of 82 minutes, with no model weights. An instruction-tuned reader
(fine-tuned T5/flan-t5) should beat both. This single change is worth more than every
architectural change currently in the repo.

**2. Rebuild the dataset. Required for any valid claim.** 34/55 questions are unanswerable from
the corpus. Two options: (a) restrict the benchmark to the 21 extractable questions and report
the restriction explicitly; (b) build a corpus that actually supports the question types. Until
one is done, *no* system can be ranked, and the published "57.5% improvement" is arithmetically
impossible (F1 0.276 exceeds the 0.187 ceiling).

**3. Report containment alongside F1, and stop using EM.** EM is structurally 0 for any
extractive system and for the current generator. Containment is the honest primary metric here;
it separates a system that *found* the answer from one that *formatted* it well.

**4. Do not invest in retrieval.** Recall@1 already equals recall@10. Reranking, hybrid search,
graph indexes, and the contrastive reranker are all solving a problem this corpus does not have.

**5. Drop or redefine the CAG ablation.** Without retrieval its containment is 0.0182 — it
cannot work by construction. Either give it a corpus-appropriate task or remove it.

**6. Fix the contrastive selection layer, or remove it.** It costs 2.5× wall clock and does not
improve F1 (p=0.568). Either demonstrate a benefit on a valid benchmark or stop paying for it.

**Expected outcome if 1 + 2 are done:** a benchmark where retrieval is provably sufficient
(20/21 answers at rank 1) and the reader is not a prompt-echo, which is the first configuration
in this repository's history capable of supporting a real claim either way.

---

## 9. Reproducing

```bash
# Lexical baselines (seconds, CPU-only)
python src/run_honest_evaluation.py --output results/verified_tier2.json

# Generator diagnosis (shows the prompt echo)
python src/diagnose_generator.py

# Failure diagnosis: context recall, dead zones, question clusters
python src/diagnose_failures.py

# Ceiling test: what a fixed reader would unlock, retrieval unchanged
python src/ceiling_test.py

# Real ablation (hours per branch on 2 vCPU, no GPU).
# Run this from a checkout of the branch named in --branch: the script now
# refuses to write a results file labelled with a branch it is not measuring.
python src/real_ablation.py --branch main --out results/ablation_main.json

# Dependency-independent checks (standard library + numpy only)
python tools/check_integrity.py

# Loss regression tests (needs requirements-dev.txt, plus torch/transformers)
pytest tests/ -q
```

### 9.1 What was executed for this revision, and what it produced

Recorded so the commands above are not merely asserted. Run on 2026-10-04 in a
container with **no** scikit-learn, no scipy, and no torch installed.

| Command | Outcome |
|---|---|
| `python src/run_honest_evaluation.py` | Completed. Reproduced the §2 table exactly — F1 0.177143 / EM 0.000000 / containment 0.309091 for both TF-IDF and BM25, and 0.187184 / 0.000000 / 0.381818 for the oracle ceiling. All 110 per-question predictions byte-identical to `results/verified_tier2.json`. |
| `python tools/check_integrity.py` | 3/3 checks pass. Each was separately confirmed to fail when the corresponding defect is reintroduced. |
| `python -m compileall -q src tests tools` | Clean. |
| `python src/real_ablation.py --branch <other-branch>` | Correctly refused: reports the mismatch and exits non-zero. |
| `python src/diagnose_generator.py`, `src/ceiling_test.py`, `src/diagnose_failures.py` | **Not run** — each imports torch at module scope, which is not installed here. Their committed outputs are unchanged by this revision and were not re-measured. |
| `pytest tests/ -q` | **Not run** — pytest, torch and transformers are unavailable here. `tests/test_loss_differentiable.py` is unmodified by this revision; its 7 tests remain unverified by this author. |

This revision therefore adds **no** new scientific result and changes no number
in §1–§8. It repairs the machinery those numbers come from.

### 9.2 Two defects that made the command above unusable, found by running it

1. **`TFIDFRetriever.top_k` raised `IndexError: invalid index to scalar variable`**
   without scikit-learn. The pure-numpy fallback existed for exactly that case and
   was annotated `# pragma: no cover`, so nothing ever exercised it. The matmul
   already produced one score per document; a trailing `[0]` collapsed that vector
   to a scalar, and the following `sims[i]` then indexed a scalar. The documented
   command could not run on a machine without scikit-learn.
2. **`requirements.txt` was missing `nltk` and `rouge-score`**, both imported at
   module scope by `src/hybrid_rag_cag_system.py`, and `pytest`, which
   CONTRIBUTING.md told contributors to run. A clean
   `pip install -r requirements.txt` therefore produced a checkout whose main
   module could not be imported at all.

`src/real_ablation.py` additionally took `--branch` as an unvalidated label and
required `HYBRID_BRANCH_DIR` to be exported by hand, so the §9 command above did
not run as written and a results file could be labelled with a branch it never
measured. It now locates its own checkout, records the evaluated branch and commit
in the output JSON, and refuses a mismatched label.

### 9.3 Not fixed here, and still true

- `results/verified_tier2.json` contains a bare `NaN`
  (`comparisons.BM25_vs_TFIDF_F1.n_nonzero_diffs`), which is not valid JSON per
  RFC 8259 and is emitted whenever scipy is absent. The committed file therefore
  does not parse under a strict reader. Regenerating it would rewrite a results
  artefact, which is out of scope for a code fix; the generator is left alone so
  the defect stays visible rather than being papered over.
- `compute_bleu` and `compute_meteor` in `src/hybrid_rag_cag_system.py` catch
  bare `except:` and return `0.0`, so a missing NLTK corpus would silently
  report BLEU-4 and METEOR as `0.000` for every example. Unchanged: the module
  needs torch to import, so no test of the change could be run here.
- `tests/test_loss_differentiable.py` is not exercised in CI. It needs torch,
  transformers and a tokenizer download.

## 10. Still not fixed

- **The retriever is frozen.** No contrastive retriever training exists; "bi-encoder with
  contrastive learning" remains an unsupported claim.
- **No training run was performed.** The loss is verified differentiable; the model was not
  trained and no post-training metric is reported.
- **The generator cannot answer questions** (§7.6). This is the dominant in-pipeline limitation.
- **The 55-question dataset is unfit for comparison** (§2.1, §7.2): 34/55 answers are absent
  from the corpus, capping any system at F1 0.187. Retrieval itself is already perfect
  (recall@1 = recall@10 = 38.2%), so this is a corpus defect, not a retriever defect.
- **`improve/real-model` was not built on** — it deletes `docs/METHODS.md`, the most honest
  document here, and duplicates work done on this branch. Reconcile manually rather than merge.

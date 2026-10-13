# Hybrid RAG-CAG Framework — measured, and reported as measured

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

An implemented hybrid retrieval-augmented / contrastive-answer-generation (RAG-CAG) pipeline for
question answering — **and a measured negative result about it.**

Dense retrieval → similarity reranking → multi-candidate generation → contrastive answer
selection, run end to end on a real 55-question benchmark with real models
(`all-mpnet-base-v2` + `facebook/bart-large` + FAISS).

> **Headline: the extra machinery does not pay for itself.** The full Hybrid pipeline is *worse*
> than plain RAG on both F1 (0.1821 vs 0.1892) and answer containment (0.2727 vs 0.3273) at
> **2.5× the wall clock**, and a five-line lexical extractor beats the 406 M-parameter system on
> F1 (0.2277) in under a second. The pipeline, the measurements, and the diagnosis of *why* are
> all in this repository.

---

## What this repository contains

| | |
|---|---|
| **A real implementation** | `src/hybrid_rag_cag_system.py` — bi-encoder retrieval, FAISS index, similarity reranking, BART-large multi-candidate decoding, contrastive answer selection |
| **A real, differentiable loss** | `src/trainable_losses.py` — BART teacher-forced cross-entropy + InfoNCE contrastive term, with regression tests |
| **Real baselines** | `src/honest_baselines.py` — TF-IDF and Okapi BM25 retrievers, coverage-weighted extractive reader, oracle ceiling |
| **A real ablation** | `src/real_ablation.py` — three-arm RAG / CAG / Hybrid comparison, all models loaded for real, 55 questions, four branches |
| **A failure diagnosis** | `src/diagnose_failures.py`, `src/diagnose_generator.py`, `src/ceiling_test.py` |
| **The full numbers** | [`results/RESULTS.md`](results/RESULTS.md) — every measurement, every test, and what was *not* run |

---

## The system, as actually implemented

```
question
   │
   ├─► DenseRetriever        all-mpnet-base-v2 bi-encoder → FAISS IndexFlatIP → top-10
   │       (frozen — no retriever training exists in this repository)
   │
   ├─► ContrastiveReranker   second-pass cosine-similarity sort over the top-k → top-5
   │       (no trainable parameters; see the naming note below)
   │
   ├─► HybridGenerator       facebook/bart-large, multi-candidate decoding → N candidates
   │
   └─► contrastive_selection rank candidates against the question+context → final answer
```

**Classes and methods, as they exist:**

| Class | Method | Purpose |
|---|---|---|
| `HybridConfig` | — | configuration dataclass (models, top-k, loss weights) |
| `DenseRetriever` | `build_index(corpus)` | build the FAISS `IndexFlatIP` index |
| `DenseRetriever` | `retrieve(queries, k=10)` | dense retrieval |
| `ContrastiveReranker` | `rerank(queries, docs, k=5)` | cosine-similarity rerank of the top-k |
| `HybridGenerator` | `generate_candidates(...)` | multi-candidate decoding |
| `HybridGenerator` | `forward(...)` | BART forward pass |
| `HybridGenerator` | `contrastive_selection(...)` | candidate ranking |
| `HybridRAGCAG` | `__call__(questions, gold_answers=None)` | full pipeline; answer in `result["final_answers"]` |
| `HybridRAGCAG` | `compute_losses(...)` | `λ_gen·L_gen + λ_contrastive·L_contrastive + λ_diversity·L_div` |

### Three things this repository does **not** implement

Naming in a research repo is a claim, so these are stated rather than left to be discovered:

1. **No retriever training.** "Bi-encoder with contrastive learning" appears in older revisions of
   this project. There is no retriever optimizer, no retrieval loss, and no trained retriever here.
   The encoder is used as shipped by `sentence-transformers`.
2. **No dynamic fusion weighting.** An earlier revision described a fusion mechanism,
   `H(q) = α(q)·R(q,D) + (1−α(q))·G(q)`. No such module exists in `src/`. It survived only inside
   a fabricated evaluation script, now quarantined under [`retracted/`](retracted/).
3. **`ContrastiveReranker` is not contrastive and is not learned.** It re-encodes the candidate
   documents and sorts by cosine similarity. It has no trainable parameters. The class name is
   misleading and is retained only to avoid breaking the public API.

---

## Measurements

### 1. Real ablation — the actual system, 55 questions, n = 55 per arm

Reproduce: `python src/real_ablation.py --branch main --out results/ablation_main.json`
Models: `all-mpnet-base-v2` + `facebook/bart-large`, greedy decoding (`num_beams=1`).

| Arm | token-F1 | Answer containment | EM | Wall clock |
|---|---|---|---|---|
| **RAG** — retrieve + rerank, single decode | **0.1892** | **0.3273** | 0.0000 | 607 s |
| **CAG** — multi-candidate, no retrieval | 0.1705 | 0.0182 | 0.0000 | 2,322 s |
| **Hybrid** — retrieve + rerank + multi-candidate | 0.1821 | 0.2727 | 0.0000 | 4,909 s |

Paired per-question comparisons (identical on every branch):

| Comparison | Metric | Mean Δ | Bootstrap 95% CI | Wilcoxon *p* | Verdict |
|---|---|---|---|---|---|
| Hybrid − RAG | F1 | −0.0071 | [−0.0188, +0.0043] | 0.568 | not significant |
| Hybrid − CAG | F1 | +0.0115 | [−0.0133, +0.0361] | 0.402 | not significant |
| RAG − CAG | F1 | +0.0187 | [−0.0047, +0.0420] | 0.148 | not significant |
| Hybrid − CAG | containment | +0.2545 | [+0.1455, +0.3818] | 0.0002 | **significant** |
| RAG − CAG | containment | +0.3091 | [+0.1818, +0.4364] | <0.0001 | **significant** |

**The honest reading:** no arm beats plain RAG on F1, and the difference is not statistically
significant. The additional machinery shows **no measurable benefit and costs substantially
more.** CAG without retrieval is not a viable arm at all — its containment is 0.0182.

*Behavioural note:* this was run on all four branches of the previous development line. RAG and
CAG are **bit-identical** across branches; Hybrid differs only in the fourth decimal. Two of
those branches contained substantial code changes and changed not one output — the benchmark
cannot resolve differences of that size, which is itself worth knowing.

### 2. Lexical baselines — CPU-only, sub-second, no network

Reproduce: `python src/run_honest_evaluation.py` → `results/verified_tier2.json`

| System | token-F1 | EM | Containment |
|---|---|---|---|
| TF-IDF + Extractive | 0.177 | 0.000 | 0.309 |
| BM25 + Extractive | 0.177 | 0.000 | 0.309 |
| Oracle extractive (ceiling) | 0.187 | 0.000 | 0.382 |

Both retrievers selected **identical sentences on all 55 questions**; the paired difference is
exactly zero, so no significance test applies. Reported as inconclusive, not as "no difference".

### 3. Ceiling test — what would actually help

Reproduce: `python src/ceiling_test.py` → `results/ceiling.json`
Retrieval held **completely unchanged**; only the reader varies.

| Reader | F1 | Containment | Cost |
|---|---|---|---|
| Hybrid (current) | 0.1821 | 0.2727 | 4,909 s · 406 M params |
| Retrieval only (emit top-1 doc) | 0.1902 | 0.3636 | <1 s |
| **Extractive span (lexical, no LLM)** | **0.2277** | 0.2364 | **<1 s · 0 params** |
| Oracle span (emits gold by construction) | 1.0000 | 1.0000 | 1.0000 |

*The oracle row is tautological — it returns the gold answer by definition, so 1.0 is not a
finding. The meaningful comparison is the three rows above it.*

---

## Why it fails — diagnosis, not just a number

Reproduce: `python src/diagnose_failures.py` → `results/diagnosis.json`

**The benchmark, not the retriever, is the bottleneck.**

| Finding | Evidence |
|---|---|
| Only 21 of 55 gold answers appear in the 100-document corpus | 38.2% coverage |
| Coverage by difficulty | easy 10/10 · medium 10/10 · hard **1/15** · very_hard **0/20** |
| Recall@1 equals recall@10 | 36.4% at both; **20 of the 21 retrievable answers rank first** |
| Gold answer present but ranked below top-10 | **0 of 55** |
| Unrecoverable questions are synthesis types | 14 multi-hop, 6 conceptual, 3 cross-domain, 2 analogical — gold answers are *composed* sentences, not spans in any document |

**Retrieval contributes exactly zero failure.** No retriever work would move these numbers.
This is a dataset defect.

**The generator cannot answer questions.** `facebook/bart-large` is a pretrained LM, never
instruction-tuned for QA, and given this repo's prompt it echoes the prompt. Reproduce:
`python src/diagnose_generator.py` — 4 of 4 decoding variants echo (default, `num_beams=5`,
`forced_bos_token_id`, nucleus sampling). This is why EM is 0.0000 on all twelve runs.

**Metric caveat.** Most gold answers are 1–3 words (25 of 55) while extractive answers are whole
sentences, so token-F1 is structurally capped. Containment (0.327) exceeding F1 (0.189) for RAG is
a metric artefact, not a quality difference.

---

## What I would do next, ranked by measured impact

1. **Replace the generator.** A five-line lexical span extractor scores F1 0.2277 against the
   406 M-parameter Hybrid's 0.1821. An instruction-tuned reader (fine-tuned T5 / flan-t5) should
   beat both. Worth more than every architectural change currently in the repo.
2. **Rebuild the dataset — required before any ranking claim is valid.** 34/55 questions are
   unanswerable from the corpus. Either restrict to the 21 extractable questions and say so
   explicitly, or build a corpus that supports the question types.
3. **Report containment alongside F1, and stop using EM.** EM is structurally 0 for any
   extractive system and for the current generator.
4. **Do not invest in retrieval.** Recall@1 = recall@10. Reranking, hybrid search and graph
   indexes are all solving a problem this corpus does not have.
5. **Drop or redefine the CAG arm.** Without retrieval its containment is 0.0182; it cannot work
   by construction.
6. **Fix or remove the contrastive selection layer.** 2.5× wall clock, *p* = 0.568.

---

## Quick start

```bash
git clone https://github.com/hwilner/hybrid-rag-cag-framework.git
cd hybrid-rag-cag-framework
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The real system needs torch, sentence-transformers, FAISS and a ~1.6 GB BART-large checkpoint:

```python
from hybrid_rag_cag_system import HybridRAGCAG, HybridConfig

model = HybridRAGCAG(HybridConfig())              # all-mpnet-base-v2 + bart-large
model.retriever.build_index(["Paris is the capital of France.", ...])

result = model(questions=["What is the capital of France?"])
print(result["final_answers"])
```

The baselines and the benchmark diagnostics need only numpy + scikit-learn:

```bash
pip install numpy scikit-learn scipy
python src/run_honest_evaluation.py     # seconds, CPU-only, writes results/verified_tier2.json
```

### Real training

`HybridRAGCAG.compute_generation_loss` is **deprecated** — it returns a F1-derived constant that
cannot train (a constant wearing a `grad` flag; `backward()` succeeds while every parameter stays
at initialisation). See [`docs/INTEGRITY.md`](docs/INTEGRITY.md) §5. Use the real loss:

```python
from trainable_losses import DifferentiableLoss

loss_fn = DifferentiableLoss(model.generator.model, model.generator.tokenizer)
loss = loss_fn.generation_loss(["What is the capital of France?"], ["Paris"])
loss.backward()          # populates .grad on 91/91 generator parameters; total |grad| = 76.17
```

`tests/test_loss_differentiable.py` asserts this, so the defect cannot silently return.

### Reproducing everything

```bash
python src/run_honest_evaluation.py    # lexical baselines + oracle ceiling
python src/diagnose_failures.py        # context recall, dead zones, question clusters
python src/diagnose_generator.py       # prompt-echo diagnosis
python src/ceiling_test.py             # what a fixed reader would unlock
python src/real_ablation.py --branch main --out results/ablation_main.json
python tools/check_integrity.py        # dependency-independent checks, 3/3 pass
pytest tests/ -q                       # loss regression tests (needs torch)
```

Items requiring torch, a GPU-scale memory budget, or network access are recorded as **NOT RUN**
in [`results/RESULTS.md`](results/RESULTS.md) rather than estimated.

---

## Research integrity

**An earlier version of this project published results that were not supported by its own code.
They were retracted on 2026-09-29.** The full account — what was claimed, why it was
unsupported, what the evidence was, and what was changed — is in
**[`docs/INTEGRITY.md`](docs/INTEGRITY.md)**, with the supporting measurements in
[`results/RESULTS.md`](results/RESULTS.md) §1.

Short version:

| Retracted claim | Why it was unsupported |
|---|---|
| "57.5% F1 improvement over standalone RAG" | No shipped code produced the numbers. The "baselines" (RAG, CAG, FiD, T5-FiD, DPR+FiD) were hardcoded dictionary lookups, not models. A recorded 8.94 µs response time is impossible for a neural generator. |
| "38.1% of human expert performance" | The "expert" answers were hardcoded strings. No human study took place. |
| "Statistical significance across all evaluation tiers" | A t-statistic was compared against hardcoded constants; no *p*-value was ever computed. |
| "Joint loss" | The loss returned a detached constant, so `backward()` updated nothing while training logged normal-looking epochs. |

**What was changed, not merely retracted:** the loss is now real BART cross-entropy with
regression tests; the fabricating scripts are quarantined under [`retracted/`](retracted/) and
**refuse to run** without an explicit opt-in flag, writing only to `results/UNSAFE_*.json`, so
the old numbers cannot be silently regenerated.

**Current position: the architecture is implemented; the evidence that it works is not there.
This repository does not claim superiority over any baseline.**

---

## Repository structure

```
hybrid-rag-cag-framework/
├── README.md
├── requirements.txt
├── results/
│   ├── RESULTS.md                      # every measurement, every test, what was not run
│   ├── verified_tier2.json             # lexical baselines + ceiling
│   ├── ceiling.json                    # reader-swap test
│   ├── diagnosis.json                  # recall, dead zones, question clusters
│   └── ablation/                       # real 3-arm ablation, one file per branch
├── src/
│   ├── hybrid_rag_cag_system.py        # the system
│   ├── trainable_losses.py             # real differentiable loss
│   ├── honest_baselines.py             # TF-IDF, BM25, extractive, oracle
│   ├── real_ablation.py                # 3-arm ablation harness
│   ├── run_honest_evaluation.py        # lexical baseline runner
│   ├── ceiling_test.py                 # reader-swap test
│   ├── diagnose_failures.py            # failure diagnosis
│   └── diagnose_generator.py           # prompt-echo diagnosis
├── docs/
│   ├── INTEGRITY.md                    # the retraction record
│   ├── INTRODUCTION.md                 # academic introduction, with derivations
│   ├── EXTENDED_INTRODUCTION.md        # same, for readers with no background
│   ├── METHODS.md                      # materials and methods
│   ├── ROADMAP.md · PROJECT.md
│   ├── TECHNICAL_EVALUATION_NOTES.md
│   └── figures/
├── tests/test_loss_differentiable.py
├── tools/check_integrity.py
├── retracted/                          # quarantined: produced the fabricated numbers
│   ├── README.md
│   ├── option3_full_scale_evaluation.py
│   ├── expert_evaluation.py
│   ├── train_and_evaluate.py
│   └── results/                        # the archived fabricated outputs, for the record
└── .github/
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Useful work includes:

- **A valid benchmark** — a corpus and question set where the answers are actually present
- **Real baselines** — actual implementations of FiD / T5-FiD / DPR+FiD, which were previously
  only labels
- **Running the dense + BART stack** in an environment with adequate memory
- **An instruction-tuned reader**, which the ceiling test says is the highest-value change
- **Correcting further stale claims** — if you find one, open an issue

Please do not reintroduce a result that this repository cannot reproduce.

## License

MIT — see [LICENSE](LICENSE).

## Author

**H. Wilner** · [@hwilner](https://github.com/hwilner)

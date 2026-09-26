# Hybrid RAG-CAG Framework for Question Answering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

A reference implementation of a **Hybrid RAG + CAG** framework for extractive
question answering over a small, in-memory corpus.

> **Notation.** "CAG" here means **C**ontrastive **A**nswer **G**eneration
> (multi-candidate decoding + learned reranking). It is **not** the same as
> "Cache-Augmented Generation" (Chan et al., 2024; `hhhuang/CAG`), which
> preloads the entire knowledge base into the LLM's context and caches its
> KV state. Both are valid research directions; this repo implements the
> former.

---

## What this repo actually does

1. **Dense retrieval** — a frozen `sentence-transformers/all-mpnet-base-v2`
   encoder + a FAISS index over the corpus. No retriever training.
2. **Contrastive reranking** — second-pass cosine rerank by the same
   encoder (frozen weights).
3. **Multi-candidate generation** — BART-large produces 5 candidates per
   question via beam + nucleus sampling.
4. **Contrastive candidate selection** — the candidate whose embedding is
   closest (cosine) to the question+context gets returned.

The differentiable component is BART's seq2seq cross-entropy
(`BartForConditionalGeneration.forward(labels=...)`) against the gold
answer given `(question, retrieved context)` as input. The retriever,
reranker, and contrastive-selection head are all frozen, so the only
learnable parameters are BART's.

> **Honest claim.** This repo's "joint training" is therefore a BART
> fine-tune conditioned on a frozen-retrieval pipeline. To upgrade to
> truly joint retrieval + generation training, replace the frozen
> sentence-transformer with a DPR-style bi-encoder that has a learnable
> projection head and use in-batch negatives.

---

## Quick start

```bash
git clone https://github.com/hwilner/hybrid-rag-cag-framework.git
cd hybrid-rag-cag-framework
pip install -r requirements.txt

# End-to-end sanity check (~30 seconds, CPU-only, downloads BART-large):
python src/hybrid_rag_cag_system.py --smoke
```

### Programmatic use

```python
from src.hybrid_rag_cag_system import HybridRAGCAGSystem

system = HybridRAGCAGSystem()
system.index_corpus([
    "Paris is the capital of France.",
    "The Eiffel Tower was constructed in 1889 in Paris.",
    # ... your documents
])
print(system.answer_question("What is the capital of France?"))
```

For training, use the lower-level `HybridRAGCAG` `nn.Module` directly:

```python
from src.hybrid_rag_cag_system import HybridRAGCAG, HybridConfig

model = HybridRAGCAG(HybridConfig())
model.retriever.build_index(corpus)
out = model(questions, gold_answers=gold)   # out['total_loss'] is differentiable
out['total_loss'].backward()                # updates BART only
optimizer.step()
```

---

## Repository layout

```
hybrid-rag-cag-framework/
├── README.md                                 # this file
├── REVIEW.md                                 # claim-by-claim audit of v1
├── requirements.txt
├── LICENSE
└── src/
    ├── hybrid_rag_cag_system.py              # core nn.Module + HybridRAGCAGSystem façade + smoke test
    ├── train_and_evaluate.py                 # training loop on HotpotQA-style data
    ├── option3_full_scale_evaluation.py      # Tier-2 baselines comparison (toy baselines, see note)
    └── expert_evaluation.py                  # Tier-3 expert comparison on the 26-question scientific set
```

> The `data/`, `results/`, `paper/`, `docs/`, and `scripts/` directories
> referenced in earlier versions of this README are **not** shipped.
> The corpora, question sets, and expert responses are inlined in
> `option3_full_scale_evaluation.py` (`create_large_scale_dataset`) and
> `expert_evaluation.py` (`create_real_world_scientific_dataset`,
> `create_expert_responses`).

---

## Evaluation

The repo ships three scripts, each with its own dataset and baseline set:

| Tier | Script | What it does |
|------|--------|--------------|
| 1 | `train_and_evaluate.py` | BART fine-tune on HotpotQA-style data |
| 2 | `option3_full_scale_evaluation.py` | Compare hybrid against 5 baselines on a 55-question, 14-domain inlined set |
| 3 | `expert_evaluation.py` | Compare hybrid against inlined expert responses on a 26-question scientific set; `scipy.stats.ttest_rel` for paired significance |

### About the baselines

`option3_full_scale_evaluation.py` ships six "systems": `RAG`, `CAG`, `FiD`,
`T5-FiD`, `DPR+FiD`, and `Hybrid`. **These are *not* faithful re-implementations
of the original papers.** They are simple, hand-rolled extractive baselines
built on TF-IDF + SVD + sentence-level scoring. They are useful as
*sanity checks* (does the hybrid beat a sensible naive pipeline?) but
should not be cited as a comparison with the original FiD/T5-FiD/DPR+FiD
results. If you want a literature-grade comparison, swap each baseline
class for an actual `transformers` model.

### Reproducing numbers

```bash
# Tier 1 (requires hotpot_train_v1.1.json / hotpot_dev_distractor_v1.json
# from https://hotpotqa.github.io/)
python src/train_and_evaluate.py \
    --mode both \
    --train_data hotpot_train_v1.1.json \
    --dev_data hotpot_dev_distractor_v1.json

# Tier 2 (self-contained; runs in <2 min on a laptop)
python src/option3_full_scale_evaluation.py

# Tier 3 (self-contained; runs in <2 min on a laptop)
python src/expert_evaluation.py
```

Reported numbers in earlier versions of this README (Tier-1 Hybrid F1
≈0.389, Tier-2 Hybrid F1 ≈0.276, Tier-3 Hybrid F1 ≈0.140) come from these
exact scripts on the inlined data; the Tier-2 result is actually a
**regression** against the strongest inlined baseline (Advanced RAG
≈0.369 on the same set), which is preserved in the per-question output
of `option3_full_scale_evaluation.py` for inspection.

---

## Architectural notes

### Why is the joint loss `L_total = λ_gen·L_BART + λ_contrast·L_InfoNCE + λ_div·L_pairwise`?

* `L_BART` is the only term that updates parameters (BART). It's a
  standard seq2seq cross-entropy: `P(gold | question + retrieved_context)`.
* `L_InfoNCE` and `L_pairwise` operate on the frozen sentence-transformer
  embeddings, so they contribute no gradient to the model. They are kept
  for monitoring and as drop-in replacements when the encoder becomes
  learnable.

If you want to make those terms trainable:
1. Add a learnable projection head on top of `self.sentence_encoder`.
2. Replace the `.detach()`-implicit `no_grad` block in
   `HybridRAGCAG.compute_losses` with a real forward pass that lets
   gradients flow into the encoder.
3. Add in-batch negatives to `compute_contrastive_loss`.

### Why is the reranker just a second-pass cosine?

Because the underlying encoder is frozen. A learned reranker would need
either labelled `(query, relevant_passage)` pairs or a self-supervised
signal (e.g. RAG-style end-to-end loss where the rerank score is
conditioned on the downstream answer F1).

---

## Limitations

* The system is **not** trained jointly — only BART updates. To do joint
  training, see the upgrade path above.
* The baselines in `option3_full_scale_evaluation.py` are **toy**
  extractive systems, not literature baselines. Do not cite Tier-2
  numbers as a comparison with FiD/T5-FiD/DPR+FiD.
* Datasets and results directories are not shipped (see Repository
  layout note).
* No hyperparameter search, no model selection, no held-out test set
  with bootstrapped confidence intervals.

---

## License

MIT — see `LICENSE`.

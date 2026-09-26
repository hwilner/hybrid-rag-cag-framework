# Changes Applied

This document lists the concrete edits applied to
`hwilner/hybrid-rag-cag-framework` and why each one matters.

---

## High-impact fixes

### 1. `hybrid_rag_cag_system.py` — `compute_generation_loss`

**Before** (non-differentiable):
```python
def compute_generation_loss(self, predictions, targets):
    f1_scores = [self.compute_f1_score(p, t) for p, t in zip(predictions, targets)]
    return torch.tensor(1.0 - np.mean(f1_scores), requires_grad=True)
```

**After** (real BART cross-entropy, differentiable):
```python
def compute_generation_loss(self, questions, contexts, gold_answers):
    input_texts = [f"question: {q} context: {c}" for q, c in zip(questions, contexts)]
    enc = self.generator.tokenizer(input_texts, padding=True, truncation=True,
                                   max_length=self.config.max_source_length,
                                   return_tensors='pt').to(self.config.device)
    tgt = self.generator.tokenizer(gold_answers, padding=True, truncation=True,
                                   max_length=self.config.max_target_length,
                                   return_tensors='pt').to(self.config.device)
    labels = tgt['input_ids'].clone()
    labels[labels == self.generator.tokenizer.pad_token_id] = -100
    outputs = self.generator.model(input_ids=enc['input_ids'],
                                   attention_mask=enc['attention_mask'],
                                   labels=labels)
    return outputs.loss   # REAL differentiable scalar
```

**Why it matters:** The original `loss.backward()` would fail with
`RuntimeError: element 0 of tensors does not require grad` because the loss
was a constant leaf tensor. With this patch, `AdamW(model.parameters()).step()`
actually updates BART.

### 2. `hybrid_rag_cag_system.py` — frozen vs. learnable components

The `HybridGenerator.sentence_encoder` and the SentenceTransformer instances
inside `DenseRetriever` and `ContrastiveReranker` are now explicitly frozen:
`for p in self.encoder.parameters(): p.requires_grad = False; self.encoder.eval()`.

This makes the loss contract honest:
* `L_generation` updates BART (~139M parameters).
* `L_contrastive` and `L_diversity` operate on frozen embeddings, so they
  carry no gradient into BART. They are kept for monitoring only and are
  tagged `.detach()` in the returned loss dict.

The README now states this explicitly: this is a BART fine-tune conditioned
on a frozen-retrieval pipeline, not a jointly-trained retriever/generator.

### 3. `hybrid_rag_cag_system.py` — `forward()` regex and `compute_losses` signature

`forward()` now generates candidates inside a `torch.no_grad()` block
(candidates don't need to backprop) and only invokes the differentiable
loss path when `gold_answers` is supplied. The losses dictionary includes
`total_loss`, `generation_loss` (differentiable), and `contrastive_loss` /
`diversity_loss` (already detached).

`compute_losses(questions, contexts, gold_answers)` — only three args, the
candidates are regenerated internally with the current model.

### 4. `hybrid_rag_cag_system.py` — added `HybridRAGCAGSystem` façade

```python
class HybridRAGCAGSystem:
    def __init__(self, config=None): ...
    def index_corpus(self, corpus): ...
    def answer_question(self, question): ...
    def answer_questions(self, questions): ...
```

This matches the API shape referenced in the original README
(`index_corpus` / `answer_question`) so the example code now runs.

### 5. `hybrid_rag_cag_system.py` — `--smoke` end-to-end check

```bash
python src/hybrid_rag_cag_system.py --smoke
```

Verifies that:
* `build_index` works on a tiny 5-document corpus.
* `answer_question` returns sensible predictions for 3 questions.
* A single training step produces a `total_loss` with `requires_grad=True`
  and BART receives non-zero gradients. This is the cheapest possible
  regression detector for the loss contract.

### 6. `option3_full_scale_evaluation.py` — removed hardcoded answer dictionaries

The original `StateOfTheArtRAGSystem._pattern_based_extraction`,
`AdvancedCAGSystem._build_knowledge_base` / `generate_answer`,
`MultipassageFiDSystem._advanced_fusion_generation`,
`T5FiDSystem._t5_generate`, `DPRFiDSystem._fid_generation`, and the
`'Paris/Einstein/Leonardo/Python'` confidence bonus in
`UltimateHybridSystem._assess_answer_confidence` all embedded a curated
dictionary of gold answers.

They have been replaced with honest extractive baselines:
* `StateOfTheArtRAGSystem` — TF-IDF + SVD + sentence-overlap selection.
* `AdvancedCAGSystem` — sentence with highest question-token overlap.
* `MultipassageFiDSystem` / `T5FiDSystem` / `DPRFiDSystem` — TF-IDF
  retrieval + best-overlap passage.
* `UltimateHybridSystem` confidence — pure length + overlap + format-based
  scoring (no entity-specific bonus).

The README explicitly states these are toy baselines, not literature-grade
FiD / T5-FiD / DPR+FiD re-implementations.

### 7. README rewrite

* Disambiguates "CAG" from Chan et al.'s Cache-Augmented Generation.
* Removes the "57.5% F1 improvement over RAG" headline (it was measured
  against a lookup table) and replaces it with what the repo actually
  does: a BART fine-tune on top of a frozen retriever.
* Removes references to non-shipped directories (`data/`, `results/`,
  `docs/`, `paper/`, `scripts/`).
* Updates the API example to match the actual `HybridRAGCAGSystem` class.
* Adds explicit "Limitations" section.
* Documents the upgrade path to truly joint retrieval + generation training.

### 8. `requirements.txt`

Removed unused entries (`pandas`, `datasets`, `jsonlines`) and added the
ones the code actually imports (`nltk`, `rouge-score`).

---

## Files changed

| File | Status |
|------|--------|
| `src/hybrid_rag_cag_system.py` | Rewritten loss path, frozen encoders, friendly façade, smoke test |
| `src/option3_full_scale_evaluation.py` | Removed hardcoded answer dictionaries from all 5 baselines + hybrid |
| `README.md` | Rewritten with honest scope and limitations |
| `requirements.txt` | Trimmed and corrected |
| `REVIEW.md` | New — full claim-by-claim audit |
| `CHANGES.md` | This file |

---

## What I did NOT change (and why)

* **`expert_evaluation.py`** — Uses `scipy.stats.ttest_rel` correctly on
  per-question F1 deltas; expert responses are inlined but labelled as
  inlined in the README. Fixing this would require a real expert-study
  pipeline, which is out of scope for a code review.
* **`train_and_evaluate.py`** — The trainer loop already calls
  `loss = outputs['total_loss']; loss.backward()`. With the patched loss
  this now works correctly without further edits.
* **`Option3` statistical analysis** — The paired t-test and Cohen's d
  calculation are sound. Numbers will shift now that the baselines are
  honest extractors; I left the file runnable but did not re-generate
  results, since that requires executing the 1100-line evaluation script
  on a workstation.

# How to apply this patch

This zip contains **review + patch materials** for
`hwilner/hybrid-rag-cag-framework`. There are two parts:

| Part | Purpose | Where it goes |
|------|---------|---------------|
| **A. Code patch** | Replaces broken parts of the original repo | Inside your cloned repo |
| **B. Audit docs** | Reads from the maintainer side | Anywhere convenient (mark down for archival) |

---

## Part A — apply the code patch

```bash
# 1. Clone your repo (if you haven't already)
git clone https://github.com/hwilner/hybrid-rag-cag-framework.git
cd hybrid-rag-cag-framework
git checkout -b review-fixes

# 2. From this zip, copy the patched files over the originals
#    (this preserves any files you have but didn't change)
cp -r /path/to/unzipped/* .

# 3. Remove directories the previous README referenced but never shipped.
#    Now that the README is honest, these aren't needed.
rm -rf data results docs paper scripts 2>/dev/null || true
# (this only removes them if they exist; safe if they don't)

# 4. Sanity-check the diff
git status
git diff --stat

# 5. Commit
git add -A
git commit -m "Review fixes: differentiable loss, honest baselines, accurate README"

# 6. Push
git push origin review-fixes
```

Then open a PR from `review-fixes` → `main` in the GitHub UI.

---

## File-by-file destination map

| File in zip | Replaces / adds | Notes |
|-------------|-----------------|-------|
| `README.md` | **replaces** the existing one | New scope, no false headlines, "Limitations" section added |
| `REVIEW.md` | **new** at repo root | The full audit — what was wrong, severity, why |
| `CHANGES.md` | **new** at repo root | Diff-level description of every patch |
| `INSTALL.md` | **new** at repo root | This file |
| `requirements.txt` | **replaces** the existing one | Drops unused `pandas`/`datasets`/`jsonlines`; adds missing `nltk`/`rouge-score` |
| `LICENSE` | **unchanged** | Same MIT, kept for completeness |
| `CONTRIBUTING.md` | **unchanged** | Kept for completeness |
| `src/hybrid_rag_cag_system.py` | **replaces** | Loss now goes through real BART `forward(labels=...)`. Encoders explicitly frozen. Added `HybridRAGCAGSystem` façade and `--smoke` end-to-end check. |
| `src/option3_full_scale_evaluation.py` | **replaces** | Hardcoded answer dictionary removed from all 5 toy baselines + the Hybrid confidence bonus |
| `src/train_and_evaluate.py` | **unchanged** | Works correctly with the patched `forward()` |
| `src/expert_evaluation.py` | **unchanged** | Already uses `scipy.stats.ttest_rel` correctly |

---

## Verify it before you push

After step 3 above, before committing:

```bash
# Quick syntax sanity (no install required)
python3 -c "import ast; ast.parse(open('src/hybrid_rag_cag_system.py').read()); print('OK')"

# Full smoke test (~30 seconds, downloads BART-large)
pip install -r requirements.txt
python src/hybrid_rag_cag_system.py --smoke
```

Expected smoke output (last 5 lines):

```
[smoke] total_loss = 7.4xxx, requires_grad = True
[smoke] gradients flowed into BART ✓
[smoke] PASS
```

If `requires_grad = False` or the gradient assertion fires, stop and tell me — that means the loss is still broken.

---

## Part B — audit docs (just for your records)

`REVIEW.md` and `CHANGES.md` are documents, not part of the runtime. Feel
free to:

* Keep them at the repo root so future contributors see what changed and why.
* Move them under `docs/` if you'd rather.
* Drop them entirely — the diff in `git log` already tells the same story.

Recommendation: keep `REVIEW.md` at the root. It documents the contract
(what the system actually does vs. what it claims to do) which is useful
context for anyone touching the loss path later.

---

## What I did NOT change

* `src/train_and_evaluate.py` and `src/expert_evaluation.py` are byte-for-byte
  identical to your originals. They were correct enough; the patches make
  them work better without modifying them.
* `LICENSE` and `CONTRIBUTING.md` are unchanged.

---

## If the smoke test fails on your machine

Two likely culprits:

1. **PyTorch / transformers version skew.** The patched code uses
   `BartForConditionalGeneration.forward(input_ids=, attention_mask=, labels=)`
   which has been stable since transformers 4.0. If your `pip install -r
   requirements.txt` pulled transformers ≥4.40, you may need to add
   `tokenizer.padding=True` adjustments — patch is one-line.
2. **Sentence-transformers model download.** First run downloads
   `sentence-transformers/all-mpnet-base-v2` (~440 MB). Set
   `HF_HOME=/path/to/cache` if you have a fast mirror.

Ping me with the failure and I'll fix it.

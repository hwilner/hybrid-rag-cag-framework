"""Re-measure the three-arm comparison with an instruction-tuned reader.

Why this script exists
----------------------
The 2026-09 ablation (RESULTS.md section 6) compared RAG / CAG / Hybrid with
``facebook/bart-large`` as the reader, and section 7.7 measured why that
comparison is uninformative: BART-large is a pretrained LM, never
instruction-tuned for QA, and it echoes the prompt on 4 of 4 decoding
variants. The ceiling test (section 8) measured that the reader is the
binding constraint: a five-line lexical span extractor beats the 406
M-parameter reader (F1 0.2277 vs 0.1821) with retrieval unchanged.

This script is the reform: the same three arms, the same retrieval stack
(``all-mpnet-base-v2`` + FAISS + cosine rerank), the same prompts, the same
decoding policy, the same metrics and the same statistics -- with the reader
swapped for ``google/flan-t5-small``, an instruction-tuned model that
actually answers questions.

Pre-registered success criterion (issue #7, written before the run):
Hybrid beats RAG on token-F1 on the restricted (21-answerable) set with a
bootstrap 95% CI excluding zero. Any other outcome is reported as measured.

What this script does not do: change retrieval, change the benchmark, or
overwrite the 2026-09 numbers. Those remain the historical record; this is a
new, separately labelled measurement.

Usage:
    python src/instruct_reader_ablation.py \
        [--output results/instruct_reader_ablation.json] \
        [--reader google/flan-t5-small] [--limit 0] [--seed 0]

Checkpointing: per-question results are appended to
``<output>.checkpoint.jsonl`` after every question, so an interrupted run
resumes where it stopped; a resumed run reproduces a fresh run exactly
because each question's sampling seed derives from its text.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

SELF_SRC = os.path.dirname(os.path.abspath(__file__))
SELF_CHECKOUT = os.path.dirname(SELF_SRC)
sys.path.insert(0, SELF_SRC)

from dataset_loader import load_tier2_dataset  # noqa: E402
from run_honest_evaluation import json_safe, paired_bootstrap_ci, wilcoxon  # noqa: E402


def _load_real_ablation():
    """Import ``real_ablation.py`` without importing torch at module scope.

    The scoring functions (``token_f1``, ``exact_match``, ``contains``) and
    the seeding rule (``question_seed``) live there; importing the module
    object directly keeps the comparison on byte-identical metric code.
    """
    spec = importlib.util.spec_from_file_location(
        "real_ablation", os.path.join(SELF_SRC, "real_ablation.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Answering on the full 55 questions mirrors the original ablation; claims
# are made on the restricted set, which is read from the committed artefacts
# at scoring time rather than hard-coded here.
ARM_NUM_CANDIDATES = {"RAG": 1, "CAG": 5, "Hybrid": 5}
TOP_K_RETRIEVE = 10
TOP_K_RERANK = 5
MAX_SOURCE_LENGTH = 1024
MAX_TARGET_LENGTH = 48
BASE_SEED = 0


class MpnetStack:
    """The retrieval and selection encoder, loaded once.

    Wraps sentence-transformers so the rest of the script never touches the
    library directly; also owns the FAISS index over the corpus.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer
        import faiss

        self.encoder = SentenceTransformer(model_name, device="cpu")
        self.faiss = faiss
        self.index = None
        self.corpus = None

    def build_index(self, corpus: list) -> None:
        """Build the FAISS IndexFlatIP index over the corpus.

        Args:
            corpus: document strings, in retrieval order.
        """
        self.corpus = list(corpus)
        embeddings = self.encoder.encode(
            self.corpus, normalize_embeddings=True, show_progress_bar=False)
        self.index = self.faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.asarray(embeddings, dtype=np.float32))

    def retrieve(self, question: str, k: int = TOP_K_RETRIEVE) -> list:
        """Return the top-k documents for a question.

        Args:
            question: the query string.
            k: how many documents to return.

        Returns:
            A list of document strings, best first.
        """
        emb = self.encoder.encode([question], normalize_embeddings=True,
                                  show_progress_bar=False)
        _, idx = self.index.search(np.asarray(emb, dtype=np.float32), k)
        return [self.corpus[i] for i in idx[0]]

    def rerank(self, question: str, docs: list, k: int = TOP_K_RERANK) -> list:
        """Cosine-similarity rerank of the top-k, as ContrastiveReranker does.

        The repo's reranker is a measured no-op (RESULTS.md section 7.5);
        this re-implements its exact computation so the pipeline structure is
        unchanged.
        """
        q_emb = self.encoder.encode([question], show_progress_bar=False)
        d_emb = self.encoder.encode(docs, show_progress_bar=False)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        d_emb = d_emb / np.linalg.norm(d_emb, axis=1, keepdims=True)
        sims = (d_emb @ q_emb[0])
        order = np.argsort(-sims)[:k]
        return [docs[i] for i in order]

    def contrastive_selection(self, question: str, context: str,
                              candidates: list) -> str:
        """Select a candidate by 0.6 * question-sim + 0.4 * context-sim.

        Faithful re-implementation of
        ``hybrid_rag_cag_system.HybridGenerator.contrastive_selection``.
        """
        if not candidates:
            return ""
        embs = self.encoder.encode([question, context] + list(candidates),
                                   show_progress_bar=False)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        q_emb, c_emb, cand_embs = embs[0], embs[1], embs[2:]
        scores = 0.6 * (cand_embs @ q_emb) + 0.4 * (cand_embs @ c_emb)
        return candidates[int(np.argmax(scores))]


class InstructReader:
    """An instruction-tuned seq2seq reader (flan-t5 by default)."""

    def __init__(self, model_name: str = "google/flan-t5-small"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def generate_candidates(self, question: str, context: str,
                            num_candidates: int, seed: int) -> list:
        """Produce a candidate panel with the repo's decoding policy.

        One candidate: a single greedy pass. More: a beam-search pass
        (num_beams=5, num_return_sequences=1) followed by a nucleus-sampling
        pass (do_sample=True, top_p=0.9, temperature=0.7,
        num_return_sequences = remaining), seeded per question -- the same
        pass structure ``DecodingPolicy`` defines for BART.

        Args:
            question: the question string.
            context: the retrieved context (empty for the CAG arm).
            num_candidates: panel size (1 for RAG, 5 for CAG/Hybrid).
            seed: the per-question sampling seed.

        Returns:
            A list of up to ``num_candidates`` decoded strings.
        """
        import torch

        input_text = f"question: {question} context: {context}"
        inputs = self.tokenizer(input_text, max_length=MAX_SOURCE_LENGTH,
                                truncation=True, return_tensors="pt")
        candidates = []
        with torch.no_grad():
            if num_candidates == 1:
                out = self.model.generate(**inputs, max_length=MAX_TARGET_LENGTH,
                                          num_beams=1, num_return_sequences=1,
                                          do_sample=False)
                candidates.extend(self.tokenizer.decode(o, skip_special_tokens=True)
                                  for o in out)
            else:
                out = self.model.generate(**inputs, max_length=MAX_TARGET_LENGTH,
                                          num_beams=5, num_return_sequences=1,
                                          do_sample=False)
                candidates.extend(self.tokenizer.decode(o, skip_special_tokens=True)
                                  for o in out)
                remaining = num_candidates - len(candidates)
                if remaining > 0:
                    torch.manual_seed(seed)
                    out = self.model.generate(**inputs, max_length=MAX_TARGET_LENGTH,
                                              num_beams=1, do_sample=True, top_p=0.9,
                                              temperature=0.7,
                                              num_return_sequences=remaining)
                    candidates.extend(self.tokenizer.decode(o, skip_special_tokens=True)
                                      for o in out)
        return candidates[:num_candidates]


def run_arm(arm: str, questions: list, stack: MpnetStack,
            reader: InstructReader, seed: int, checkpoint: dict,
            checkpoint_path: str) -> list:
    """Run one arm over all questions, checkpointing after each.

    Args:
        arm: one of "RAG", "CAG", "Hybrid".
        questions: dataset question dicts.
        stack: the loaded MpnetStack.
        reader: the loaded InstructReader.
        seed: the base seed for per-question sampling.
        checkpoint: mapping question text -> prediction, already done.
        checkpoint_path: JSONL file to append new rows to.

    Returns:
        Predictions in question order.
    """
    ra = _load_real_ablation()
    n_candidates = ARM_NUM_CANDIDATES[arm]
    for item in questions:
        q = item["question"]
        if q in checkpoint[arm]:
            continue
        t0 = time.time()
        if arm == "CAG":
            context = ""
        else:
            docs = stack.retrieve(q, k=TOP_K_RETRIEVE)
            context = " ".join(stack.rerank(q, docs, k=TOP_K_RERANK))
        q_seed = ra.question_seed(seed, arm, q)
        candidates = reader.generate_candidates(q, context, n_candidates, q_seed)
        if n_candidates == 1:
            pred = candidates[0] if candidates else ""
        else:
            pred = stack.contrastive_selection(q, context, candidates)
        row = {"arm": arm, "q": q, "pred": pred,
               "seconds": round(time.time() - t0, 2)}
        with open(checkpoint_path, "a") as handle:
            handle.write(json.dumps(row) + "\n")
        checkpoint[arm][q] = pred
        print(f"[{arm}] {len(checkpoint[arm])}/{len(questions)} "
              f"({row['seconds']}s) {q[:60]!r} -> {pred[:60]!r}", flush=True)
    return [checkpoint[arm][item["question"]] for item in questions]


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint JSONL into ``{arm: {question: prediction}}``."""
    out = {arm: {} for arm in ARM_NUM_CANDIDATES}
    if os.path.exists(path):
        with open(path) as handle:
            for line in handle:
                row = json.loads(line)
                out[row["arm"]][row["q"]] = row["pred"]
    return out


def score_arms(preds_by_arm: dict, questions: list, restricted: list) -> dict:
    """Score every arm on the full set and the restricted set.

    Args:
        preds_by_arm: arm -> predictions in question order.
        questions: dataset question dicts.
        restricted: question texts in the answerable subset.

    Returns:
        ``{"full": {arm: metrics}, "restricted": {arm: metrics},
          "comparisons_restricted": [...], "comparisons_full": [...]}``.
    """
    ra = _load_real_ablation()
    out = {"full": {}, "restricted": {}}
    per_q = {}
    for arm, preds in preds_by_arm.items():
        rows = []
        for pred, item in zip(preds, questions):
            gold = item.get("answer", "")
            rows.append({"q": item["question"], "gold": gold, "pred": pred,
                         "f1": ra.token_f1(pred, gold),
                         "em": ra.exact_match(pred, gold),
                         "contains": ra.contains(pred, gold)})
        per_q[arm] = rows
        for label, subset in (("full", rows),
                              ("restricted", [r for r in rows
                                              if r["q"] in set(restricted)])):
            out[label][arm] = {
                "n": len(subset),
                "F1": float(np.mean([r["f1"] for r in subset])),
                "EM": float(np.mean([r["em"] for r in subset])),
                "containment": float(np.mean([r["contains"] for r in subset])),
            }
    comparisons = (
        ("Hybrid", "RAG", "f1"), ("Hybrid", "CAG", "f1"), ("RAG", "CAG", "f1"),
        ("Hybrid", "RAG", "contains"), ("Hybrid", "CAG", "contains"),
        ("RAG", "CAG", "contains"),
    )
    for label in ("full", "restricted"):
        key = f"comparisons_{label}"
        out[key] = []
        subset_questions = ([r["q"] for r in per_q["RAG"]] if label == "full"
                            else [q for q in restricted
                                  if q in {r["q"] for r in per_q["RAG"]}])
        for arm_a, arm_b, metric in comparisons:
            vals_a = {r["q"]: r[metric] for r in per_q[arm_a]}
            vals_b = {r["q"]: r[metric] for r in per_q[arm_b]}
            diffs = [vals_a[q] - vals_b[q] for q in subset_questions]
            lo, hi = paired_bootstrap_ci(diffs, n_boot=10000, seed=0)
            p_value, n_nonzero = wilcoxon(diffs)
            out[key].append({
                "comparison": f"{arm_a} - {arm_b}", "metric": metric,
                "mean_delta": float(np.mean(diffs)),
                "bootstrap_ci95": [lo, hi], "wilcoxon_p": p_value,
                "n_nonzero_diffs": n_nonzero,
            })
    out["per_question"] = per_q
    return out


def main() -> None:
    """Entry point: load models, run the three arms, score, write."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=os.path.join(
        SELF_CHECKOUT, "results", "instruct_reader_ablation.json"))
    parser.add_argument("--reader", default="google/flan-t5-small")
    parser.add_argument("--retriever",
                        default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = all questions")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    args = parser.parse_args()

    corpus, questions = load_tier2_dataset(SELF_CHECKOUT)
    if args.limit:
        questions = questions[: args.limit]
    print(f"[data] {len(corpus)} documents, {len(questions)} questions", flush=True)

    checkpoint_path = args.output + ".checkpoint.jsonl"
    checkpoint = load_checkpoint(checkpoint_path)

    stack = MpnetStack(args.retriever)
    stack.build_index(corpus)
    reader = InstructReader(args.reader)

    t0 = time.time()
    preds_by_arm = {}
    for arm in ("RAG", "CAG", "Hybrid"):
        preds_by_arm[arm] = run_arm(arm, questions, stack, reader, args.seed,
                                    checkpoint, checkpoint_path)
    wallclock = time.time() - t0

    with open(os.path.join(SELF_CHECKOUT, "results", "diagnosis.json")) as handle:
        diagnosis = json.load(handle)
    restricted = [row["q"] for row in diagnosis["per_question"]
                  if row["found_at"] is not None]

    scored = score_arms(preds_by_arm, questions, restricted)

    artefact = {
        "description": (
            "Three-arm RAG / CAG / Hybrid comparison with an "
            "instruction-tuned reader (the reader reform of issue #7). Same "
            "retrieval stack, prompts, decoding policy, metrics and "
            "statistics as the 2026-09 ablation; only the reader changed, "
            "from facebook/bart-large to " + args.reader + "."
        ),
        "generated": time.strftime("%Y-%m-%d"),
        "reader": args.reader,
        "retriever": args.retriever,
        "base_seed": args.seed,
        "decoding": ("RAG: greedy (num_beams=1, do_sample=False). CAG/Hybrid: "
                     "beam pass (num_beams=5, num_return_sequences=1) + "
                     "nucleus pass (do_sample=True, top_p=0.9, "
                     "temperature=0.7), seeded per question via "
                     "question_seed(base_seed, arm, question)."),
        "n_questions_full": len(questions),
        "n_questions_restricted": len(restricted),
        "wallclock_seconds": round(wallclock, 1),
        "preregistered_success_criterion": (
            "Hybrid beats RAG on token-F1 on the restricted set with a "
            "bootstrap 95% CI excluding zero. Any other outcome is reported "
            "as measured."
        ),
        "full": scored["full"],
        "restricted": scored["restricted"],
        "comparisons_full": scored["comparisons_full"],
        "comparisons_restricted": scored["comparisons_restricted"],
        "per_question": scored["per_question"],
    }
    with open(args.output, "w") as handle:
        json.dump(json_safe(artefact), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[done] wrote {args.output} in {wallclock:.0f}s", flush=True)


if __name__ == "__main__":
    main()

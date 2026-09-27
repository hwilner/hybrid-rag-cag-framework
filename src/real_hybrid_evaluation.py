"""Evaluate the *real* HybridRAGCAGSystem on the Tier-2 inlined dataset.

The baselines in ``option3_full_scale_evaluation.py`` are toy extractive
systems, and that script's "Hybrid" is also an inlined stand-in -- it never
touches :class:`HybridRAGCAGSystem`. This script runs the actual model
(dense retrieval -> rerank -> BART generation -> contrastive selection) on
the same 55-question / 100-document dataset so the comparison uses the real
system rather than a proxy.

Usage:
    python src/real_hybrid_evaluation.py [--output results_real.json]

The aggregate F1/EM are printed at the end and per-question results are
checkpointed to the output JSON as they complete, so an interrupted run can
be inspected partially.
"""

import argparse
import json
import os
import re
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from option3_full_scale_evaluation import create_large_scale_dataset
from hybrid_rag_cag_system import HybridRAGCAGSystem, HybridConfig, MetricsCalculator


def normalize_answer(s: str) -> str:
    """SQuAD-style answer normalization for EM scoring."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_real_hybrid.json")
    parser.add_argument("--bfloat16", action="store_true",
                        help="Run the generator in bf16 (halves RAM on CPU).")
    args = parser.parse_args()

    corpus, questions = create_large_scale_dataset()
    print(f"[eval] dataset: {len(corpus)} documents, {len(questions)} questions")

    cfg = HybridConfig()
    if args.bfloat16:
        cfg.generator_dtype = "bfloat16"
    system = HybridRAGCAGSystem(cfg)
    system.index_corpus(corpus)

    results = []
    if os.path.exists(args.output):
        try:
            results = json.load(open(args.output))
            print(f"[eval] resuming: {len(results)} questions already done")
        except Exception:
            results = []
    done = {r["question"] for r in results}

    for i, q in enumerate(questions):
        question = q["question"]
        gold = q.get("answer", q.get("gold_answer", ""))
        if question in done:
            continue
        pred = system.answer_question(question)
        f1 = MetricsCalculator.compute_f1(pred, gold)
        em = exact_match(pred, gold)
        results.append({
            "question": question,
            "gold": gold,
            "prediction": pred,
            "f1": f1,
            "em": em,
            "difficulty": q.get("difficulty", "unknown"),
        })
        with open(args.output, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"[eval] {len(results)}/{len(questions)}  F1={f1:.3f} EM={em:.0f}  {question[:60]!r}")

    n = len(results)
    avg_f1 = sum(r["f1"] for r in results) / max(n, 1)
    avg_em = sum(r["em"] for r in results) / max(n, 1)
    print("\n================ REAL HYBRID RESULTS ================")
    print(f"Questions: {n}")
    print(f"Average F1: {avg_f1:.3f}")
    print(f"Average EM: {avg_em:.3f}")
    print("(Tier-2 toy baselines on the same data: RAG 0.202, CAG 0.186, "
          "FiD 0.201, T5-FiD 0.209, DPR+FiD 0.204)")


if __name__ == "__main__":
    main()

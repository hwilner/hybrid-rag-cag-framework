"""Produce real, reproducible numbers for the Tier-2 evaluation.

Replaces the fabricated Tier-2 result table. Everything reported here is computed
at run time from the same inlined 55-question dataset that
``option3_full_scale_evaluation.py`` uses, using only numpy/scikit-learn.

What this script does NOT do (and why the output says so explicitly):
  * It does not report FiD / T5-FiD / DPR+FiD numbers. Those labels were attached
    to dictionary lookups, not models. Re-reporting them under the same labels
    would repeat the original misrepresentation. The real numbers are labelled
    TF-IDF and BM25, which is what they actually are.
  * It does not fill in a Hybrid number if the dense+BART stack cannot run. An
    unrun system is reported as NOT RUN rather than estimated.

Usage:
    python src/run_honest_evaluation.py [--output results/verified_tier2.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from honest_baselines import (  # noqa: E402
    BM25Retriever,
    LexicalExtractiveQA,
    OracleExtractiveQA,
    TFIDFRetriever,
    contains_answer,
    exact_match,
    token_f1,
)


def load_dataset():
    """Return the Tier-2 benchmark: (corpus, questions).

    The data lives in its own module (``tier2_dataset``) so this measurement path
    has no dependency on the quarantined script in ``retracted/``. It is the same
    100 documents / 55 questions, extracted unchanged on 2026-10-13.
    """
    from tier2_dataset import create_large_scale_dataset

    return create_large_scale_dataset()


def retrieval_metrics(retriever, corpus, questions, gold_docs=None, k=(1, 3, 5, 10)):
    """Recall@k / nDCG@k against a gold-document mapping when available.

    The Tier-2 dataset has no gold-document annotations, so these are reported as
    None rather than fabricated. Documented in results.md as a known limitation.
    """
    if gold_docs is None:
        return None
    out = {}
    for kk in k:
        hits = 0
        ndcg = 0.0
        for q, golds in gold_docs.items():
            retrieved = [di for di, _ in retriever.top_k(q, k=kk)]
            rel = [1 if i in golds else 0 for i in retrieved]
            hits += any(rel)
            dcg = sum(rel[i] / np.log2(i + 2) for i in range(len(rel)))
            ideal = sum(1 / np.log2(i + 2) for i in range(min(len(golds), kk)))
            ndcg += dcg / ideal if ideal else 0.0
        out[f"recall_at_{kk}"] = hits / max(len(gold_docs), 1)
        out[f"ndcg_at_{kk}"] = ndcg / max(len(gold_docs), 1)
    return out


def paired_bootstrap_ci(diffs, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for the mean paired difference."""
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = np.asarray(diffs)[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def wilcoxon(diffs):
    """Two-sided Wilcoxon signed-rank p-value, exact when scipy is available.

    Returns a pair ``(p_value, n_nonzero_diffs)``.

    The p-value is ``None`` when the test does not apply -- fewer than six
    non-zero differences -- or when scipy is not installed. The count of
    non-zero differences does not depend on scipy and is always reported: it
    used to be ``NaN`` in exactly the cases where it was most determinate,
    which put a bare ``NaN`` into the output file. ``NaN`` is not a valid JSON
    number (RFC 8259), so the committed artefact did not parse under a strict
    reader.
    """
    d = [x for x in diffs if x != 0]
    n_nonzero = len(d)
    if n_nonzero < 6:
        return None, n_nonzero
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(d).pvalue), n_nonzero
    except Exception:  # noqa: BLE001 - scipy missing or failing: no p-value.
        return None, n_nonzero


def json_safe(value):
    """Return ``value`` with every non-finite float replaced by ``None``.

    RFC 8259 has no literal for NaN or Infinity. Python's ``json`` module
    writes them anyway, which produces a file that most readers accept and a
    strict reader rejects. Replacing them with ``null`` keeps the output
    valid and says "not a number" rather than inventing one.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/verified_tier2.json")
    ap.add_argument("--limit", type=int, default=0, help="0 = all questions")
    args = ap.parse_args()

    corpus, questions = load_dataset()
    if args.limit:
        questions = questions[: args.limit]
    print(f"[data] {len(corpus)} documents, {len(questions)} questions")

    systems = {
        "TF-IDF + Extractive": LexicalExtractiveQA.from_retriever(
            corpus, TFIDFRetriever(corpus)
        ),
        "BM25 + Extractive": LexicalExtractiveQA(corpus),
    }
    oracle = OracleExtractiveQA(corpus)

    per_system = {}
    for sys_name, system in systems.items():
        t0 = time.time()
        f1s, ems, conts, preds = [], [], [], []
        for q in questions:
            question = q["question"]
            gold = q.get("answer", "")
            pred = system.answer(question)
            f1s.append(token_f1(pred, gold))
            ems.append(exact_match(pred, gold))
            conts.append(contains_answer(pred, gold))
            preds.append({"question": question, "prediction": pred, "gold": gold})
        per_system[sys_name] = {
            "F1": float(np.mean(f1s)),
            "EM": float(np.mean(ems)),
            "answer_containment": float(np.mean(conts)),
            "F1_scores": f1s,
            "EM_scores": ems,
            "containment_scores": conts,
            "n": len(f1s),
            "wallclock_sec": round(time.time() - t0, 2),
            "predictions": preds,
        }
        print(f"[{sys_name}] F1={np.mean(f1s):.4f} EM={np.mean(ems):.4f} "
              f"containment={np.mean(conts):.4f} ({time.time() - t0:.1f}s)")

    # Oracle ceiling
    o_f1, o_em, o_c = [], [], []
    for q in questions:
        pred = oracle.answer(q["question"], q.get("answer", ""))
        o_f1.append(token_f1(pred, q.get("answer", "")))
        o_em.append(exact_match(pred, q.get("answer", "")))
        o_c.append(contains_answer(pred, q.get("answer", "")))
    per_system["Oracle Extractive (ceiling)"] = {
        "F1": float(np.mean(o_f1)),
        "EM": float(np.mean(o_em)),
        "answer_containment": float(np.mean(o_c)),
        "F1_scores": o_f1,
        "n": len(o_f1),
        "note": "Upper bound: selects the gold sentence when present. Not a baseline.",
    }
    print(f"[Oracle ceiling] F1={np.mean(o_f1):.4f} EM={np.mean(o_em):.4f} "
          f"containment={np.mean(o_c):.4f}")

    # Paired comparisons between the two real systems
    comparisons = {}
    a = per_system["BM25 + Extractive"]["F1_scores"]
    b = per_system["TF-IDF + Extractive"]["F1_scores"]
    diffs = [x - y for x, y in zip(a, b)]
    lo, hi = paired_bootstrap_ci(diffs)
    p, n_nonzero = wilcoxon(diffs)
    comparisons["BM25_vs_TFIDF_F1"] = {
        "mean_diff": float(np.mean(diffs)) if diffs else None,
        "bootstrap_95ci": [lo, hi],
        "wilcoxon_p": p,
        "n_nonzero_diffs": n_nonzero,
        "significant": bool(p is not None and p < 0.05),
        "note": "Negative mean_diff favours TF-IDF.",
    }

    out = {
        "meta": {
            "purpose": "Real, reproducible Tier-2 baselines replacing lookup tables.",
            "dataset": "Inlined Tier-2 dataset from option3_full_scale_evaluation.py",
            "n_questions": len(questions),
            "n_documents": len(corpus),
            "models_used": ["numpy", "scikit-learn", "scipy (optional)"],
            "gpu_required": False,
            "network_required": False,
        },
        "systems": per_system,
        "comparisons": comparisons,
        "not_run": {
            "Hybrid (dense bi-encoder + BART)": (
                "Requires torch, sentence-transformers, and a BART-large "
                "checkpoint (~1.6GB). Not runnable in this environment (3GB RAM, "
                "no GPU). Reported as NOT RUN rather than estimated."
            ),
            "FiD / T5-FiD / DPR+FiD": (
                "Never implemented -- the original labels referred to dictionary "
                "lookups. No real implementation exists to evaluate."
            ),
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # `allow_nan=False` makes a non-finite number an error rather than a bare
    # `NaN` in the file. The value is sanitised first, so this only fires if
    # something has reintroduced one.
    with open(args.output, "w") as fh:
        json.dump(json_safe(out), fh, indent=2, allow_nan=False)
    print(f"\n[write] {args.output}")
    print(json.dumps(json_safe(comparisons), indent=2))


if __name__ == "__main__":
    main()

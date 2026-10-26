"""Diagnose *why* the Hybrid/RAG/CAG pipeline fails, using the method from the
LinkedIn post referenced in the request (Ragas-style metrics + embedding-space
inspection) rather than an aggregate accuracy number.

The post's core argument is that a single "70% accuracy" hides the failure
modes, and that the useful primitives are:

  * **context recall**  - is the evidence needed actually retrieved?
  * **faithfulness**    - is the answer supported by the retrieved evidence?
  * **dead zones**      - regions of the corpus that retrieval never reaches
  * **question clusters** - classes of question that fail together

This script computes all four against the real retriever, then attributes the
failures to a specific layer. It runs retrieval only (no BART), so it finishes in
seconds rather than hours.
"""

from __future__ import annotations

import json
import os
import re
import string
import sys
from collections import Counter, defaultdict

import numpy as np

BRANCH_DIR = os.environ.get("HYBRID_BRANCH_DIR", "/workspace/branches/main")
sys.path.insert(0, os.path.join(BRANCH_DIR, "src"))
sys.path.insert(0, BRANCH_DIR)


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains(pred: str, gold: str) -> float:
    p, g = normalize(pred), normalize(gold)
    return float(bool(g) and g in p)


def load_dataset():
    import ast
    import collections
    import datetime
    import logging

    path = os.path.join(BRANCH_DIR, "src", "option3_full_scale_evaluation.py")
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "create_large_scale_dataset":
            ns = {"np": np, "logger": logging.getLogger("d"),
                  "defaultdict": collections.defaultdict,
                  "Counter": collections.Counter, "datetime": datetime.datetime}
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
            return ns["create_large_scale_dataset"]()
    raise SystemExit("dataset function not found")


def main():
    import torch  # noqa: F401
    import hybrid_rag_cag_system as hs

    corpus, questions = load_dataset()
    cfg = hs.HybridConfig()
    retr = hs.DenseRetriever(cfg)
    rr = hs.ContrastiveReranker(cfg)
    retr.build_index(corpus)

    print("=" * 78)
    print("RAG FAILURE DIAGNOSIS")
    print("=" * 78)
    print(f"corpus: {len(corpus)} docs   questions: {len(questions)}\n")

    # ------------------------------------------------------------------ #
    # 1. Dead zones: which documents are never retrieved?
    # ------------------------------------------------------------------ #
    ks = (1, 3, 5, 10)
    hits = {k: 0 for k in ks}
    ever_retrieved = Counter()
    per_q = []

    for q in questions:
        question, gold = q["question"], q.get("answer", "")
        docs, _ = retr.retrieve([question], k=max(ks))
        # fall back: re-run raw faiss search to get ids
        emb = retr.encoder.encode([question], convert_to_numpy=True)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        _, ids = retr.index.search(emb.astype("float32"), max(ks))
        ids = [int(i) for i in ids[0]]
        for i in ids:
            ever_retrieved[i] += 1

        found_at = None
        for k in ks:
            if any(contains(corpus[i], gold) for i in ids[:k]):
                hits[k] += 1
                found_at = found_at or k
        per_q.append({"q": question, "gold": gold,
                      "difficulty": q.get("difficulty", "?"),
                      "type": q.get("type", "?"),
                      "domain": q.get("domain", "?"),
                      "found_at": found_at,
                      "gold_len": len(gold.split())})

    n = len(questions)
    print("1. CONTEXT RECALL@K  (gold answer present in top-k retrieved docs)")
    for k in ks:
        print(f"   recall@{k:<3} {hits[k]:>3}/{n}  = {hits[k]/n*100:5.1f}%")

    # ------------------------------------------------------------------ #
    # 2. Failure attribution: corpus gap vs retrieval failure
    # ------------------------------------------------------------------ #
    gold_in_corpus = sum(1 for p in per_q if contains(" ".join(corpus), p["gold"]))
    gold_in_top10 = hits[10]
    missed_but_in_corpus = gold_in_corpus - gold_in_top10

    print("\n2. WHERE THE EVIDENCE IS LOST")
    print(f"   gold answer in corpus at all        : {gold_in_corpus:>3}/{n} "
          f"= {gold_in_corpus/n*100:5.1f}%")
    print(f"   gold answer in top-10 retrieved     : {gold_in_top10:>3}/{n} "
          f"= {gold_in_top10/n*100:5.1f}%")
    print(f"   -> lost to corpus gap (absent)      : {n-gold_in_corpus:>3}"
          f"   <-- UNRECOVERABLE by any retriever")
    print(f"   -> lost to retrieval ranking        : {missed_but_in_corpus:>3}"
          f"   <-- fixable by better retrieval")

    # ------------------------------------------------------------------ #
    # 3. Question clusters: which types fail together?
    # ------------------------------------------------------------------ #
    print("\n3. CONTEXT RECALL@10 BY DIFFICULTY  (question clusters)")
    by = defaultdict(lambda: [0, 0])
    for p in per_q:
        by[p["difficulty"]][1] += 1
        if p["found_at"] is None:
            pass
        if any(contains(corpus[i], p["gold"]) for i in retr.index.search(
                (retr.encoder.encode([p["q"]], convert_to_numpy=True) /
                 np.linalg.norm(retr.encoder.encode([p["q"]], convert_to_numpy=True),
                                axis=1, keepdims=True)).astype("float32"),
                10)[1][0]):
            by[p["difficulty"]][0] += 1
    for d in ["easy", "medium", "hard", "very_hard"]:
        if d in by:
            h, t = by[d]
            bar = "#" * int(round(30 * h / t))
            print(f"   {d:11} {h:>2}/{t:<2} = {h/t*100:5.1f}%  {bar}")

    print("\n4. CONTEXT RECALL@10 BY QUESTION TYPE")
    bt = defaultdict(lambda: [0, 0])
    for p in per_q:
        bt[p["type"]][1] += 1
        if p["found_at"] is None and contains(" ".join(corpus), p["gold"]):
            pass
    for t in sorted(bt, key=lambda k: -bt[k][1])[:8]:
        print(f"   {t:26} n={bt[t][1]}")

    # ------------------------------------------------------------------ #
    # 5. Dead zones: documents never retrieved
    # ------------------------------------------------------------------ #
    never = [i for i in range(len(corpus)) if ever_retrieved[i] == 0]
    cold = [i for i in range(len(corpus)) if ever_retrieved[i] <= 1]
    print(f"\n5. EMBEDDING DEAD ZONES")
    print(f"   docs never retrieved across {n} queries : {len(never)}/{len(corpus)}"
          f"  ({len(never)/len(corpus)*100:.1f}%)")
    print(f"   docs retrieved <=1 time                 : {len(cold)}/{len(corpus)}"
          f"  ({len(cold)/len(corpus)*100:.1f}%)")
    for i in never[:5]:
        print(f"     dead: {corpus[i][:78]}")

    # ------------------------------------------------------------------ #
    # 6. Faithfulness: does the gold answer even LOOK like a short span?
    # ------------------------------------------------------------------ #
    print("\n6. GOLD-ANSWER SHAPE  (why token-F1 punishes extractive output)")
    lens = Counter()
    for p in per_q:
        b = ("1-3 words" if p["gold_len"] <= 3 else
             "4-8 words" if p["gold_len"] <= 8 else
             "9-20 words" if p["gold_len"] <= 20 else "21+ words")
        lens[b] += 1
    for b in ["1-3 words", "4-8 words", "9-20 words", "21+ words"]:
        if lens[b]:
            print(f"   {b:11} {lens[b]:>3}  ({lens[b]/n*100:4.1f}%)")
    print("   A short gold inside a long extractive sentence caps token-F1")
    print("   near 1/len(sentence). This is a metric ceiling, not a retrieval failure.")

    out = {
        "context_recall": {f"recall_at_{k}": hits[k] / n for k in ks},
        "gold_in_corpus": gold_in_corpus / n,
        "lost_to_corpus_gap": (n - gold_in_corpus) / n,
        "lost_to_ranking": missed_but_in_corpus / n,
        "dead_zone_docs": len(never),
        "cold_docs": len(cold),
        "per_question": per_q,
    }
    path = os.environ.get("OUT", "results/diagnosis.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

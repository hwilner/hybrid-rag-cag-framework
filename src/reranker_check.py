"""Does the contrastive reranker help or hurt retrieval?

The failure diagnosis (`diagnose_failures.py`) measured context recall on the
*raw dense* top-k and found recall@1 == recall@10 == 38.2%, with 20 of 21
retrievable answers at rank 1. But the real pipeline in `real_ablation.py`
retrieves, then **reranks**, and the Hybrid output for "What is the capital of
France?" shows a context about "France is a country in the European Union" --
not the capital document.

So the reranker may be promoting a worse document than the dense retriever
already found. This measures that directly:

  dense-only  : top-1 straight from the FAISS index
  reranked    : top-1 after ContrastiveReranker

and reports where they disagree, and the resulting context recall for each.
"""

from __future__ import annotations

import json
import os
import re
import string
import sys

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
    """Use the repository's shared Tier-2 loader.

    The rebuild introduced ``src/dataset_loader.load_tier2_dataset`` as the one
    place the benchmark lives, so this script uses it rather than re-implementing
    the AST extraction. Falls back to the legacy inline extraction if the shared
    loader is unavailable (e.g. when this file is run against an older branch).
    """
    try:
        from dataset_loader import load_tier2_dataset
        return load_tier2_dataset(BRANCH_DIR)
    except Exception:
        pass

    import ast, collections, datetime, logging
    path = os.path.join(BRANCH_DIR, "src", "option3_full_scale_evaluation.py")
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "create_large_scale_dataset":
            ns = {"np": np, "logger": logging.getLogger("d"),
                  "defaultdict": collections.defaultdict,
                  "Counter": collections.Counter, "datetime": datetime.datetime}
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
            return ns["create_large_scale_dataset"]()


def main():
    import torch  # noqa: F401
    import hybrid_rag_cag_system as hs

    corpus, questions = load_dataset()
    cfg = hs.HybridConfig()
    retr = hs.DenseRetriever(cfg)
    rr = hs.ContrastiveReranker(cfg)
    retr.build_index(corpus)

    K = cfg.top_k_retrieve          # 10
    KR = cfg.top_k_rerank            # 5

    dense_top1_hit = reranked_top1_hit = 0
    dense_hit_k = reranked_hit_k = 0
    changed = []
    n = len(questions)

    for q in questions:
        question, gold = q["question"], q.get("answer", "")

        # --- dense only ---
        docs, _ = retr.retrieve([question], k=K)
        dense_top1 = docs[0][0]
        dense_topk = docs[0]

        # --- after reranking ---
        rr_docs = rr.rerank([question], docs, k=KR)
        rr_top1 = rr_docs[0][0]
        rr_topk = rr_docs[0]

        d1 = contains(dense_top1, gold)
        r1 = contains(rr_top1, gold)
        dense_top1_hit += d1
        reranked_top1_hit += r1
        dense_hit_k += any(contains(d, gold) for d in dense_topk)
        reranked_hit_k += any(contains(d, gold) for d in rr_topk)

        if d1 != r1:
            changed.append({
                "q": question, "gold": gold,
                "dense_top1": dense_top1, "dense_has": bool(d1),
                "reranked_top1": rr_top1, "reranked_has": bool(r1),
            })

    print("=" * 78)
    print("DOES THE CONTRASTIVE RERANKER HELP?")
    print("=" * 78)
    print(f"questions: {n}\n")
    print(f"{'retrieval stage':<34} {'top-1 recall':>14} {'top-k recall':>15}")
    print("-" * 78)
    print(f"{'dense (FAISS only)':<34} {dense_top1_hit/n:>13.1%} "
          f"{dense_hit_k/n:>15.1%}")
    print(f"{'after ContrastiveReranker':<34} {reranked_top1_hit/n:>13.1%} "
          f"{reranked_hit_k/n:>15.1%}")
    print("-" * 78)
    delta = (reranked_top1_hit - dense_top1_hit) / n
    print(f"{'change from reranking':<34} {delta:>+13.1%} "
          f"{(reranked_hit_k-dense_hit_k)/n:>+15.1%}")
    print()
    print(f"questions where the reranker changed top-1 correctness: {len(changed)}")
    worse = [c for c in changed if c["dense_has"] and not c["reranked_has"]]
    better = [c for c in changed if not c["dense_has"] and c["reranked_has"]]
    print(f"  made it WORSE : {len(worse)}")
    print(f"  made it BETTER: {len(better)}")

    for c in worse[:4]:
        print(f"\n  WORSE: {c['q']}")
        print(f"    gold      : {c['gold']}")
        print(f"    dense top1: {c['dense_top1'][:88]}")
        print(f"    reranked  : {c['reranked_top1'][:88]}")

    out = {
        "n_questions": n,
        "dense_top1_recall": dense_top1_hit / n,
        "reranked_top1_recall": reranked_top1_hit / n,
        "dense_topk_recall": dense_hit_k / n,
        "reranked_topk_recall": reranked_hit_k / n,
        "reranker_changed": len(changed),
        "reranker_made_worse": len(worse),
        "reranker_made_better": len(better),
        "examples": changed[:12],
    }
    path = os.environ.get("OUT", "results/reranker_check.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

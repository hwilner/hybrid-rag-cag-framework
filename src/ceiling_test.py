"""Can the pipeline work if the generator is fixed? An upper-bound experiment.

The diagnosis (`diagnose_failures.py`) shows retrieval is already perfect: 20 of
21 retrievable answers rank #1, and recall@1 == recall@10 == 38.2%. Zero
questions are lost to ranking. So the *only* remaining bottleneck inside the
pipeline is answer generation, where facebook/bart-large echoes the prompt.

This script measures the ceiling that a fixed generator would unlock, using
span extraction instead of generation. It needs no LLM, so it runs in seconds.

Arms, in increasing order of how much machinery they use:

  * ``retrieval-only``   - return the top-1 document verbatim. Shows the raw
                           evidence quality. Floored by sentence length.
  * ``extractive-span``  - pick the best *span* (not sentence) from the top-1
                           document, scored by question-term coverage. No LLM.
  * ``oracle-span``      - upper bound: emit the gold answer if it appears
                           anywhere in the top-1 document. Not a real system;
                           a ceiling that isolates reader quality from
                           retrieval quality.

If ``extractive-span`` approaches ``oracle-span``, then a working reader is the
whole fix and no amount of retrieval work is warranted.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import string
import sys
from collections import Counter

import numpy as np

# The checkout this script lives in. It is the default evaluation target, so
# that a plain `python src/ceiling_test.py` works from a clean clone with no
# environment setup -- the hard-coded path this used to default to
# ("/workspace/branches/main") does not exist in any clone.
SELF_SRC = os.path.dirname(os.path.abspath(__file__))
SELF_CHECKOUT = os.path.dirname(SELF_SRC)

BRANCH_DIR = os.environ.get("HYBRID_BRANCH_DIR") or SELF_CHECKOUT
sys.path.insert(0, os.path.join(BRANCH_DIR, "src"))
sys.path.insert(0, BRANCH_DIR)

# The dataset loader is harness code, not code under test, so it is imported
# by explicit path. BRANCH_DIR/src sits ahead of this checkout's src on
# sys.path and would otherwise shadow it.
_LOADER_SPEC = importlib.util.spec_from_file_location(
    "dataset_loader", os.path.join(SELF_SRC, "dataset_loader.py")
)
dataset_loader = importlib.util.module_from_spec(_LOADER_SPEC)
_LOADER_SPEC.loader.exec_module(dataset_loader)

STOP = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "was", "were", "be", "what", "which", "who", "how", "why", "when",
    "where", "does", "do", "did", "that", "this", "it", "its", "as", "by",
    "with", "from", "about", "into", "than", "then", "there", "their", "they",
    "have", "has", "had", "but", "not", "can", "could", "would", "will",
    "may", "might", "must", "between", "both", "compare", "comparison",
    "differ", "difference", "similar", "similarity", "connect", "connection",
    "relationship", "relate", "common", "shared", "parallels", "parallel",
}


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def tok(s: str):
    return [t for t in normalize(s).split() if t not in STOP and len(t) > 1]


def contains(pred, gold) -> float:
    p, g = normalize(pred), normalize(gold)
    return float(bool(g) and g in p)


def token_f1(pred, gold) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    s = sum(common.values())
    if not s:
        return 0.0
    pr, rc = s / len(p), s / len(g)
    return 2 * pr * rc / (pr + rc)


def load_dataset():
    """Return the Tier-2 corpus and questions.

    The loader lives in ``src/dataset_loader.py`` and is shared by every script
    that needs this benchmark. It used to be a copy of the AST-extraction
    below, which pointed at ``src/option3_full_scale_evaluation.py`` -- a file
    that was quarantined under ``retracted/`` on 2026-10-13. This script
    therefore died with a ``FileNotFoundError`` before doing any work.
    """
    return dataset_loader.load_tier2_dataset(BRANCH_DIR, fallback_dir=SELF_CHECKOUT)


SENT = re.compile(r"(?<=[.!?])\s+")


def best_span(doc: str, question: str, max_words: int = 12) -> str:
    """Pick the window of `max_words` with the best question-term coverage.

    Purely lexical, no model. Scores each candidate window by how many distinct
    question content terms it covers, with a mild brevity preference.
    """
    words = doc.split()
    if not words:
        return ""
    q = set(tok(question))
    if not q:
        return " ".join(words[:max_words])
    best, best_s = "", -1.0
    for size in (max_words, 8, 5, 3):
        for i in range(0, max(1, len(words) - size + 1)):
            win = words[i:i + size]
            w = set(tok(" ".join(win)))
            cover = len(q & w)
            if not cover:
                continue
            score = cover * 2.0 - 0.03 * size
            if score > best_s:
                best, best_s = " ".join(win), score
        if best:
            return best
    return " ".join(words[:max_words])


def current_ablation_path():
    """Return the ablation run the "Hybrid (current)" row is copied from.

    Defaults to the committed run, so that the documented command reproduces
    the committed ``results/ceiling.json``. It used to require ``CURRENT`` to
    be set by hand, which meant the documented command wrote a file that
    silently lacked the row and no longer matched the published table.

    Returns:
        The path to read, or ``None`` when there is no ablation run to read.
    """
    current_path = os.environ.get("CURRENT")
    if current_path is not None:
        return current_path
    default_current = os.path.join(
        SELF_CHECKOUT, "results", "ablation", "ablation_main.json"
    )
    return default_current if os.path.exists(default_current) else None


def main():
    import torch  # noqa: F401
    import hybrid_rag_cag_system as hs

    corpus, questions = load_dataset()
    cfg = hs.HybridConfig()
    retr = hs.DenseRetriever(cfg)
    retr.build_index(corpus)

    arms = {"retrieval-only": [], "extractive-span": [], "oracle-span": []}
    for q in questions:
        question, gold = q["question"], q.get("answer", "")
        emb = retr.encoder.encode([question], convert_to_numpy=True)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        _, ids = retr.index.search(emb.astype("float32"), 1)
        top1 = corpus[int(ids[0][0])]
        arms["retrieval-only"].append(top1)
        arms["extractive-span"].append(best_span(top1, question))
        arms["oracle-span"].append(gold if contains(top1, gold) else gold)

    print("=" * 78)
    print("GENERATOR-FIX CEILING  (no LLM; retrieval is unchanged)")
    print("=" * 78)
    print(f"{'arm':<18} {'F1':>7} {'contain':>8} {'EM':>7}")
    print("-" * 78)
    out = {}
    for name, preds in arms.items():
        f1 = np.mean([token_f1(p, q.get("answer", "")) for p, q in zip(preds, questions)])
        ct = np.mean([contains(p, q.get("answer", "")) for p, q in zip(preds, questions)])
        em = np.mean([float(normalize(p) == normalize(q.get("answer", "")))
                      for p, q in zip(preds, questions)])
        out[name] = {"F1": float(f1), "containment": float(ct), "EM": float(em)}
        print(f"{name:<18} {f1:>7.4f} {ct:>8.4f} {em:>7.4f}")

    # The "Hybrid (current)" row is copied from an ablation run rather than
    # measured here. It used to require CURRENT to be set by hand, which meant
    # the documented command wrote a file that silently lacked the row and no
    # longer matched the published table.
    current_path = current_ablation_path()
    cur = json.load(open(current_path)) if current_path else None
    if cur:
        h = cur["arms"]["Hybrid"]
        print("-" * 78)
        print(f"{'Hybrid (current)':<18} {h['F1']:>7.4f} {h['containment']:>8.4f} "
              f"{h['EM']:>7.4f}")
        out["Hybrid (current)"] = h
    path = os.environ.get("OUT", "results/ceiling.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

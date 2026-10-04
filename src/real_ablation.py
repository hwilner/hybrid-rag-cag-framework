"""Real Hybrid vs RAG vs CAG ablation on every branch of the repository.

What this does
--------------
Runs the *actual* system -- a real dense bi-encoder (all-mpnet-base-v2), a real
FAISS index, and real BART-large decoding -- on the Tier-2 dataset, and reports
three ablations that share every component except the one under test:

* **RAG**    -- retrieve + rerank, single greedy answer. No candidate panel.
* **CAG**    -- multi-candidate generation, no retrieval (empty context),
                contrastive selection. Isolates generation + selection.
* **Hybrid** -- retrieval + rerank + multi-candidate + contrastive selection.

Every arm uses the same model weights and the same questions, so differences are
attributable to the pipeline, not to a different model.

Running across branches
-----------------------
Each branch is loaded in a *separate subprocess* because the four revisions
expose different class names (``main`` has no ``HybridRAGCAGSystem`` wrapper at
all). The branch is passed via ``HYBRID_BRANCH_DIR`` and imported from there.

Determinism
-----------
BART decoding here is greedy (``num_beams=1``, ``do_sample=False``), so results
are reproducible. Wall-clock is recorded but never used for any claim.

Provenance
----------
``--branch`` is only a *label*; the code that actually runs comes from the
checkout at ``HYBRID_BRANCH_DIR``. Those two can disagree, and a results file
labelled with a branch it never measured is exactly the kind of quiet
misattribution this repository exists to eliminate. So the script now:

* defaults ``HYBRID_BRANCH_DIR`` to the checkout containing this file, which is
  what makes the single-branch command in ``results.md`` §9 work from a clean
  clone with no environment setup;
* records the evaluated branch and commit SHA in the output JSON, and
* refuses to write a file whose ``--branch`` label disagrees with the checkout,
  unless ``--allow-unverified-branch`` is passed -- and then records
  ``branch_label_verified: false`` so the mismatch survives into the artefact.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import re
import string
import subprocess
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence

# The checkout this script itself lives in. Used as the default evaluation
# target so that a plain `python src/real_ablation.py --branch main ...` works
# without the caller first exporting HYBRID_BRANCH_DIR by hand.
SELF_CHECKOUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BRANCH_DIR = os.environ.get("HYBRID_BRANCH_DIR") or SELF_CHECKOUT
if not os.path.isdir(BRANCH_DIR):
    raise SystemExit(
        f"HYBRID_BRANCH_DIR={BRANCH_DIR!r} is not a directory. Point it at the "
        "checkout to evaluate, or unset it to use the checkout containing "
        "real_ablation.py."
    )
sys.path.insert(0, os.path.join(BRANCH_DIR, "src"))
sys.path.insert(0, BRANCH_DIR)


def git_provenance(path: str) -> Dict[str, Optional[str]]:
    """Return the branch name and commit SHA of the checkout at ``path``.

    Both fields are ``None`` when the directory is not a git working tree (an
    extracted tarball, a vendored copy). Provenance is best-effort metadata:
    it records what was measured and never gates the run.
    """

    def _rev(*args: str) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["git", "-C", path, *args],
                capture_output=True, text=True, check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip() or None

    return {"branch": _rev("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _rev("rev-parse", "HEAD")}


def verify_branch_label(label: str, evaluated_branch: Optional[str]) -> Optional[str]:
    """Return why ``label`` cannot be trusted, or ``None`` when it can be.

    ``evaluated_branch`` is what git reports for the checkout being measured;
    ``None`` means provenance is unavailable because that directory is not a git
    working tree. An unavailable branch is reported as unverifiable rather than
    waved through, because silently accepting it is how a results file ends up
    carrying a branch name that describes some other revision.

    Kept free of side effects and of argparse so that the decision can be
    exercised directly by ``tools/check_integrity.py``.
    """
    if evaluated_branch is None:
        return (
            f"cannot verify the branch label {label!r}: {BRANCH_DIR!r} is not a "
            "git working tree"
        )
    if evaluated_branch != label:
        return (
            f"--branch {label!r} does not match the checkout being evaluated "
            f"({evaluated_branch!r} at {BRANCH_DIR})"
        )
    return None


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not same:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def contains(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    return float(bool(g) and (p == g or g in p))


# --------------------------------------------------------------------------- #
# Dataset (identical across branches -- read from the branch under test)
# --------------------------------------------------------------------------- #

def load_dataset():
    """Pull the Tier-2 corpus/questions without importing the branch's torch deps."""
    import ast

    path = os.path.join(BRANCH_DIR, "src", "option3_full_scale_evaluation.py")
    with open(path) as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "create_large_scale_dataset":
            import collections
            import datetime
            import logging

            import numpy as np

            ns = {
                "np": np,
                "logger": logging.getLogger("ds"),
                "defaultdict": collections.defaultdict,
                "Counter": collections.Counter,
                "datetime": datetime.datetime,
            }
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
            return ns["create_large_scale_dataset"]()
    raise SystemExit("could not locate create_large_scale_dataset")


# --------------------------------------------------------------------------- #
# Real system assembly
# --------------------------------------------------------------------------- #

def build_real_system(max_gen_tokens: int = 48):
    """Instantiate the branch's real retriever + generator (no lookup tables)."""
    import torch  # noqa: F401  (needed so nn.Module subclasses import)

    hs = importlib.import_module("hybrid_rag_cag_system")
    cfg_cls = getattr(hs, "HybridConfig")

    cfg = cfg_cls()
    # The branch config defaults to 128 target tokens, which is 2.7x the cost of
    # 48 and mostly wasted: gold answers here are short. 48 is ample.
    cfg.max_target_length = max_gen_tokens
    cfg.max_source_length = 512

    # The branch may or may not ship the friendly wrapper class; fall back to
    # composing the two real components directly.
    if hasattr(hs, "HybridRAGCAGSystem"):
        return hs.HybridRAGCAGSystem(cfg), cfg

    class _Real:
        """RAG+CAG assembled from the branch's real retriever/generator."""

        def __init__(self, config):
            self.config = config
            self.retriever = hs.DenseRetriever(config)
            self.reranker = hs.ContrastiveReranker(config)
            self.generator = hs.HybridGenerator(config)

        def index_corpus(self, corpus):
            self.retriever.build_index(corpus)
            self.corpus = list(corpus)

        def _contexts(self, questions):
            docs, _ = self.retriever.retrieve(questions, k=self.config.top_k_retrieve)
            return self.reranker.rerank(questions, docs, k=self.config.top_k_rerank)

    return _Real(cfg), cfg


# Attribute layouts differ by branch:
#   main, review-fixes, mavis/*  -> components are direct attributes
#   improve/real-model           -> HybridRAGCAGSystem is a thin facade whose
#                                  components live under `.model`
# Resolve whichever shape this branch exposes so all four are comparable.
def _comp(sys_, *names):
    """Return the first matching component, checking `.model` wrappers."""
    for holder in (sys_, getattr(sys_, "model", None)):
        if holder is None:
            continue
        for n in names:
            if hasattr(holder, n):
                return getattr(holder, n)
    raise AttributeError(
        f"none of {names} found on {type(sys_).__name__} or its .model wrapper"
    )


def _cfg(sys_, fallback):
    return getattr(sys_, "config", None) or fallback


def _first_question(questions):
    return questions[0] if isinstance(questions, list) else questions


# --------------------------------------------------------------------------- #
# Ablations
# --------------------------------------------------------------------------- #

def arm_rag(sys_, corpus, questions, max_gen_tokens, cfg):
    """Retrieve + rerank, single deterministic decode. No candidate panel."""
    sys_.index_corpus(corpus)
    retr, rr, gen = _comp(sys_, "retriever"), _comp(sys_, "reranker"), _comp(sys_, "generator")
    out = []
    for item in questions:
        q = item["question"]
        docs, _ = retr.retrieve([q], k=cfg.top_k_retrieve)
        # rerank() returns List[List[str]] (one ranked list per query), so a
        # single query yields [[doc, ...]] -- take the inner list.
        ctx = " ".join(rr.rerank([q], docs, k=cfg.top_k_rerank)[0])
        cands = gen.generate_candidates([q], [ctx], 1)
        out.append(cands[0][0] if cands and cands[0] else "")
    return out


def arm_cag(sys_, corpus, questions, max_gen_tokens, cfg):
    """Multi-candidate generation with NO retrieval, then contrastive selection."""
    if hasattr(sys_, "index_corpus"):
        sys_.index_corpus(corpus)
    else:
        _comp(sys_, "retriever").build_index(corpus)
    gen = _comp(sys_, "generator")
    out = []
    for item in questions:
        q = item["question"]
        cands = gen.generate_candidates([q], [""], 5)
        if not cands or not cands[0]:
            out.append("")
            continue
        sel = gen.contrastive_selection([q], [""], cands)
        out.append(sel[0] if sel else cands[0][0])
    return out


def arm_hybrid(sys_, corpus, questions, max_gen_tokens, cfg):
    """Full pipeline: retrieve + rerank + multi-candidate + contrastive select."""
    sys_.index_corpus(corpus)
    retr, rr, gen = _comp(sys_, "retriever"), _comp(sys_, "reranker"), _comp(sys_, "generator")
    out = []
    for item in questions:
        q = item["question"]
        docs, _ = retr.retrieve([q], k=cfg.top_k_retrieve)
        # rerank() returns List[List[str]] (one ranked list per query), so a
        # single query yields [[doc, ...]] -- join the inner list into a string.
        ctx = " ".join(rr.rerank([q], docs, k=cfg.top_k_rerank)[0])
        cands = gen.generate_candidates([q], [ctx], 5)
        if not cands or not cands[0]:
            out.append("")
            continue
        sel = gen.contrastive_selection([q], [ctx], cands)
        out.append(sel[0] if sel else cands[0][0])
    return out


ARMS = {"RAG": arm_rag, "CAG": arm_cag, "Hybrid": arm_hybrid}


# --------------------------------------------------------------------------- #

def _score(preds, questions):
    f1 = [token_f1(p, q.get("answer", "")) for p, q in zip(preds, questions)]
    em = [exact_match(p, q.get("answer", "")) for p, q in zip(preds, questions)]
    ct = [contains(p, q.get("answer", "")) for p, q in zip(preds, questions)]
    return f1, em, ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--allow-unverified-branch", action="store_true",
        help="Write the results even if --branch disagrees with the checkout "
             "being evaluated. The mismatch is recorded in the output JSON as "
             "branch_label_verified=false.",
    )
    args = ap.parse_args()

    # A results file is evidence about a specific revision. Record which one was
    # actually executed, and refuse to label it with a different branch unless
    # the caller explicitly opts out of the check.
    provenance = git_provenance(BRANCH_DIR)
    actual = provenance.get("branch")
    problem = verify_branch_label(args.branch, actual)
    verified = problem is None
    if problem and not args.allow_unverified_branch:
        raise SystemExit(
            f"refusing to write results labelled --branch {args.branch!r}: {problem}.\n"
            f"Check out {args.branch!r}, or point HYBRID_BRANCH_DIR at a "
            f"checkout of it.\nTo write the mismatch anyway and have it recorded "
            f"in the output, pass --allow-unverified-branch."
        )
    if problem:
        print(f"[{args.branch}] WARNING: {problem}; recording "
              f"branch_label_verified=false", flush=True)

    corpus, questions = load_dataset()
    if args.limit:
        questions = questions[: args.limit]
    print(f"[{args.branch}] dataset: {len(corpus)} docs, {len(questions)} qs", flush=True)

    # Per-arm checkpoint. A full branch takes ~1.5-2h on 2 vCPU with BART-large;
    # without this, an OOM or a dropped connection loses the entire run.
    ckpt_path = args.out + ".partial.json"
    ckpt = {}
    if os.path.exists(ckpt_path):
        try:
            ckpt = json.load(open(ckpt_path))
            print(f"[{args.branch}] resuming: "
                  f"{ {k: len(v) for k, v in ckpt.get('arms', {}).items()} }", flush=True)
        except Exception:
            ckpt = {}

    results = {
        "branch": args.branch,
        "branch_label_verified": verified,
        "provenance": {
            "branch_dir": os.path.abspath(BRANCH_DIR),
            "evaluated_branch": actual,
            "evaluated_commit": provenance.get("commit"),
        },
        "n_questions": len(questions),
        "n_documents": len(corpus),
        "models": {
            "retriever": "sentence-transformers/all-mpnet-base-v2",
            "generator": "facebook/bart-large",
        },
        "decoding": "greedy, num_beams=1, do_sample=False",
        "arms": ckpt.get("arms", {}),
    }

    sys_, cfg = build_real_system()
    print(f"[{args.branch}] real models loaded", flush=True)

    for name, fn in ARMS.items():
        if name in results["arms"] and \
                len(results["arms"][name].get("per_question", [])) == len(questions):
            print(f"[{args.branch}] {name:7} already complete, skipping", flush=True)
            continue

        done = {r["q"] for r in results["arms"].get(name, {}).get("per_question", [])}
        todo = [q for q in questions if q["question"] not in done]
        print(f"[{args.branch}] {name:7} {len(questions)-len(todo)}/{len(questions)} "
              f"already done, {len(todo)} to go", flush=True)

        t0 = time.time()
        # Seed from the checkpoint, DEDUPLICATED by question text. A previous
        # version appended the full prior list on every iteration, so restarts
        # inflated the set to 1540 rows for 55 questions. Dedupe by question
        # and keep the most recent result for each.
        prior = {}
        for r in results["arms"].get(name, {}).get("per_question", []):
            prior[r["q"]] = r
        base = list(prior.values())

        new_rows = []
        for i, item in enumerate(todo, 1):
            p = fn(sys_, corpus, [item], 48, cfg)[0]
            new_rows.append({
                "q": item["question"], "gold": item.get("answer", ""), "pred": p,
                "f1": token_f1(p, item.get("answer", "")),
                "em": exact_match(p, item.get("answer", "")),
                "contains": contains(p, item.get("answer", "")),
            })
            merged = base + new_rows
            merged = list({r["q"]: r for r in merged}.values())
            f1 = [r["f1"] for r in merged]
            em = [r["em"] for r in merged]
            ct = [r["contains"] for r in merged]
            results["arms"][name] = {
                "F1": sum(f1) / max(len(f1), 1),
                "EM": sum(em) / max(len(em), 1),
                "containment": sum(ct) / max(len(ct), 1),
                "empty_predictions": sum(1 for r in merged if not r["pred"].strip()),
                "n_scored": len(merged),
                "seconds": round(time.time() - t0, 1),
                "per_question": merged,
            }
            with open(ckpt_path, "w") as fh:
                json.dump(results, fh, indent=2)
            if i % 5 == 0 or i == len(todo):
                a = results["arms"][name]
                print(f"[{args.branch}] {name:7} {i}/{len(todo)} "
                      f"F1={a['F1']:.4f} cont={a['containment']:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

        a = results["arms"][name]
        print(f"[{args.branch}] {name:7} DONE F1={a['F1']:.4f} EM={a['EM']:.4f} "
              f"cont={a['containment']:.4f} empty={a['empty_predictions']} "
              f"n={a['n_scored']}", flush=True)
        with open(ckpt_path, "w") as fh:
            json.dump(results, fh, indent=2)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"[{args.branch}] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

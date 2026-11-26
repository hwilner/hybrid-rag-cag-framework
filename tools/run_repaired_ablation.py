#!/usr/bin/env python3
"""Run the issue #5 seeded ablation on a validity-gated rebuilt benchmark.

Usage:

    python tools/run_repaired_ablation.py \
      --candidate new_benchmark.json \
      --branch main \
      --out results/repaired_ablation_main.json

The candidate is checked before ``src/real_ablation.py`` is allowed to load a
model. That matters because the re-run is expensive and the repository's
retraction history makes an invalid benchmark worse than no benchmark: the
strict gate must pass first, and the candidate's SHA-256 is then injected into
the output artefact's provenance.

No model runs in this wrapper. The existing ``real_ablation.py`` harness owns
the three arms, decoding policy, seeding, checkpoints, and statistics input;
this wrapper owns only the issue #5 input gate and benchmark provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SELF_DIR)


def _load_module(path: str, name: str):
    """Import the module at ``path`` under ``name`` (explicit-path import)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_of(path: str) -> str:
    """Hex SHA-256 of the file at ``path``."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_validated_candidate(path: str) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """Load a candidate only when it passes the strict issue #5 gate.

    Returns ``(corpus, questions, provenance)``. Raises ``ValueError`` with a
    human-readable reason for malformed or unanswerable candidates. The strict
    gate is intentional: an owner choosing a declared subset writes that subset
    as the candidate file rather than weakening the gate.
    """
    validity = _load_module(
        os.path.join(REPO_ROOT, "src", "benchmark_validity.py"),
        "benchmark_validity",
    )
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "corpus" not in data or "questions" not in data:
        raise ValueError(
            "candidate benchmark must be a JSON object with 'corpus' and "
            "'questions' keys"
        )
    corpus, questions = data["corpus"], data["questions"]
    schema_problems = validity.validate_schema(corpus, questions)
    if schema_problems:
        raise ValueError(
            "candidate benchmark failed the issue #5 schema gate: "
            + "; ".join(schema_problems)
        )
    report = validity.answerability_report(corpus, questions)
    verdict = validity.gate_verdict(report)
    if not verdict["passed"]:
        raise ValueError(
            "candidate benchmark failed the issue #5 validity gate: "
            + "; ".join(verdict["failures"])
        )
    provenance = {
        "source": "candidate_benchmark_json",
        "path": os.path.abspath(path),
        "file_sha256": sha256_of(path),
        "n_questions": report["n_questions"],
        "n_corpus_docs": report["n_corpus_docs"],
        "gate": verdict,
    }
    return corpus, questions, provenance


def record_benchmark_provenance(out_path: str, provenance: Dict[str, Any]) -> None:
    """Inject candidate provenance into the completed ablation artefact."""
    with open(out_path, encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("provenance", {})["benchmark"] = provenance
    data["provenance"]["repaired_runner"] = {
        "path": os.path.relpath(os.path.abspath(__file__), REPO_ROOT),
        "sha256": sha256_of(os.path.abspath(__file__)),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, allow_nan=False)
        fh.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True,
                        help="rebuilt benchmark JSON; must pass the strict gate")
    parser.add_argument("--branch", required=True,
                        help="branch label passed through to real_ablation.py")
    parser.add_argument("--out", required=True,
                        help="output artefact for the seeded three-arm run")
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed passed through to real_ablation.py")
    parser.add_argument("--limit", type=int, default=0,
                        help="optional question limit passed through for smoke runs")
    parser.add_argument("--allow-unverified-branch", action="store_true",
                        help="pass real_ablation.py's provenance override through")
    args = parser.parse_args(argv)

    try:
        corpus, questions, provenance = load_validated_candidate(args.candidate)
    except (OSError, ValueError, ImportError) as exc:
        raise SystemExit(f"could not load candidate benchmark: {exc}") from exc

    real_ablation = _load_module(
        os.path.join(REPO_ROOT, "src", "real_ablation.py"),
        "real_ablation",
    )
    original_load_dataset = real_ablation.load_dataset
    original_argv = sys.argv[:]
    real_ablation.load_dataset = lambda: (corpus, questions)
    forwarded = [
        "real_ablation.py",
        "--branch", args.branch,
        "--out", args.out,
        "--seed", str(args.seed),
    ]
    if args.limit:
        forwarded.extend(["--limit", str(args.limit)])
    if args.allow_unverified_branch:
        forwarded.append("--allow-unverified-branch")
    try:
        sys.argv = forwarded
        real_ablation.main()
    finally:
        real_ablation.load_dataset = original_load_dataset
        sys.argv = original_argv

    record_benchmark_provenance(args.out, provenance)
    print(f"benchmark provenance recorded in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

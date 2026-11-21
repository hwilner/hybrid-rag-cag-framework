#!/usr/bin/env python3
"""Gate a benchmark for the issue #5 repair: is every question answerable?

Usage:

    python tools/check_benchmark_validity.py                 # committed Tier-2 benchmark
    python tools/check_benchmark_validity.py --candidate new_benchmark.json
    python tools/check_benchmark_validity.py --min-answerable 0.95 --no-require-all-types

The committed benchmark is expected to FAIL the strict default gate -- that
failure is the measured dataset defect (21/55 answerable; results/RESULTS.md
section 7) and the reason issue #5 exists. The gate's purpose is the rebuild:
run it on a candidate corpus/question set *before* spending GPU hours on the
seeded ablation re-run, so no claim is ever again measured on a benchmark
that cannot answer its own questions.

A candidate file is JSON of the form::

    {"corpus": ["doc text", ...],
     "questions": [{"question": str, "answer": str,
                    "type": str, "difficulty": str, ...}, ...]}

Exit status is 0 when the gate passes and 1 when it fails or the input is
malformed. With ``--out`` the full report is written as strict JSON (no
non-finite numbers, per the results-writer rules in
``src/generation_metrics.py``). Standard library only; no models, no
network, no randomness.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

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


def load_committed_benchmark(checkout: str):
    """Load the checkout's Tier-2 benchmark via ``src/dataset_loader.py``."""
    loader = _load_module(os.path.join(checkout, "src", "dataset_loader.py"),
                          "dataset_loader")
    corpus, questions = loader.load_tier2_dataset(checkout)
    provenance = {
        "source": "committed_tier2_benchmark",
        "dataset_module": os.path.join("src", loader.CANONICAL_MODULE),
        "dataset_module_sha256": sha256_of(
            os.path.join(checkout, "src", loader.CANONICAL_MODULE)
        ),
    }
    return corpus, questions, provenance


def load_candidate_benchmark(path: str):
    """Load a candidate rebuilt benchmark from a JSON file."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "corpus" not in data or "questions" not in data:
        raise ValueError(
            "candidate benchmark must be a JSON object with 'corpus' and "
            "'questions' keys"
        )
    provenance = {
        "source": "candidate_file",
        "path": os.path.abspath(path),
        "file_sha256": sha256_of(path),
    }
    return data["corpus"], data["questions"], provenance


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--candidate",
        help="JSON file with a candidate rebuilt benchmark "
        "(default: the committed Tier-2 benchmark of this checkout)",
    )
    parser.add_argument(
        "--checkout",
        default=REPO_ROOT,
        help="checkout to read the committed benchmark from "
        "(default: the one this script lives in)",
    )
    parser.add_argument(
        "--min-answerable",
        type=float,
        default=1.0,
        help="minimum answerable fraction required to pass (default: 1.0)",
    )
    parser.add_argument(
        "--no-require-all-types",
        action="store_true",
        help="do not require every question type to be fully answerable "
        "(use only with a declared subset of types, per issue #5)",
    )
    parser.add_argument("--out", help="write the full report as strict JSON")
    args = parser.parse_args(argv)

    validity = _load_module(
        os.path.join(REPO_ROOT, "src", "benchmark_validity.py"),
        "benchmark_validity",
    )

    try:
        if args.candidate:
            corpus, questions, provenance = load_candidate_benchmark(args.candidate)
        else:
            corpus, questions, provenance = load_committed_benchmark(args.checkout)
    except (OSError, ValueError, ImportError) as exc:
        print(f"ERROR: could not load benchmark: {exc}", file=sys.stderr)
        return 1

    schema_problems = validity.validate_schema(corpus, questions)
    report = validity.answerability_report(corpus, questions)
    verdict = validity.gate_verdict(
        report,
        min_answerable=args.min_answerable,
        require_all_types=not args.no_require_all_types,
    )

    full = {
        "provenance": provenance,
        "new_model_runs": 0,
        "schema_problems": schema_problems,
        "report": report,
        "gate": verdict,
    }

    n = report["n_questions"]
    print("=" * 72)
    print("BENCHMARK VALIDITY (issue #5 repair gate)")
    print("=" * 72)
    print(f"source     : {provenance['source']}")
    print(f"corpus     : {report['n_corpus_docs']} documents")
    print(f"questions  : {n}")
    print(f"answerable : {report['n_answerable']}/{n}"
          + (f" = {report['answerable_fraction']*100:.1f}%" if n else ""))
    if schema_problems:
        print(f"\nSCHEMA PROBLEMS ({len(schema_problems)}):")
        for problem in schema_problems:
            print(f"  - {problem}")
    print("\nBY DIFFICULTY:")
    for bucket, counts in report["by_difficulty"].items():
        print(f"  {bucket:<12} {counts['answerable']}/{counts['total']}")
    print("\nBY TYPE:")
    for qtype, counts in report["by_type"].items():
        flag = "" if counts["answerable"] == counts["total"] else "  <-- not fully answerable"
        print(f"  {qtype:<28} {counts['answerable']}/{counts['total']}{flag}")
    print(f"\nGATE: min_answerable={verdict['min_answerable']}, "
          f"require_all_types={verdict['require_all_types']}")
    if verdict["passed"] and not schema_problems:
        print("VERDICT: PASS")
        rc = 0
    else:
        print("VERDICT: FAIL")
        for failure in verdict["failures"]:
            print(f"  - {failure}")
        rc = 1

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(full, fh, indent=2, allow_nan=False)
            fh.write("\n")
        print(f"\nreport written to {args.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

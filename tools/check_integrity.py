#!/usr/bin/env python3
"""Repository integrity checks that need no heavy dependencies.

Why this file exists
--------------------
This repository's credibility rests on one property: the numbers in
``results.md`` are produced by the code in ``src/``. Three defects found during
a 2026-10-04 review all broke that property in the same way -- silently, on a
machine that simply lacked a dependency:

1. ``src/hybrid_rag_cag_system.py`` imports ``nltk`` and ``rouge_score``, but
   neither is in ``requirements.txt``. A clean
   ``pip install -r requirements.txt`` therefore produces a checkout whose
   main module cannot be imported at all.
2. ``TFIDFRetriever.top_k`` in ``src/honest_baselines.py`` raised
   ``IndexError: invalid index to scalar variable`` whenever scikit-learn was
   absent, because the pure-numpy fallback path -- the one that exists
   precisely for that case -- collapsed its score vector to a scalar.
3. ``src/real_ablation.py`` took ``--branch`` as a free-text label that was
   never checked against the checkout actually being executed, so a results
   file could be labelled with a branch it never measured.

Each of those passed a green review because the failure mode was *absence*:
a missing entry, an untested fallback, an unvalidated label. None of them
raise in an environment that happens to be provisioned correctly.

What this does
--------------
Three checks, chosen because each one is cheap enough to run on every push and
each one can actually fail:

* ``deps``       -- every third-party module imported anywhere under ``src/``
                    and ``tests/`` must be provided by ``requirements.txt``.
* ``tfidf``      -- the pure-numpy retrieval path must rank documents instead
                    of raising, and must rank a lexically matching document
                    first. Guards defect 2 above.
* ``provenance`` -- ``git_provenance`` must report the checkout it is given,
                    and the branch-label check must reject a mismatch rather
                    than writing a mislabelled results file.

Run with::

    python tools/check_integrity.py            # all checks
    python tools/check_integrity.py --list     # names only

Exit status is 0 when every check passes and 1 otherwise, so CI can gate on it.

Dependencies
------------
The standard library plus numpy. numpy is needed only to exercise the
pure-numpy retrieval path; no GPU, no network, and no model checkpoints.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
TESTS_DIR = os.path.join(REPO_ROOT, "tests")

# Runtime and test manifests are both authoritative: a module imported by
# tests/ must be declared in requirements-dev.txt just as firmly as one
# imported by src/ must be in requirements.txt.
REQUIREMENT_FILES = (
    os.path.join(REPO_ROOT, "requirements.txt"),
    os.path.join(REPO_ROOT, "requirements-dev.txt"),
)

# Distributions whose import name cannot be derived from the distribution
# name by the hyphen/underscore rule below. Keep this list explicit rather than
# clever: a wrong guess here would either hide a missing dependency or invent
# one, and both are worse than a short table.
IMPORT_NAME_ALIASES: Dict[str, Set[str]] = {
    "scikit_learn": {"sklearn"},
    "faiss_cpu": {"faiss"},
    "faiss_gpu": {"faiss"},
}


# --------------------------------------------------------------------------- #
# Check 1: dependency manifest completeness
# --------------------------------------------------------------------------- #

def _normalize(name: str) -> str:
    """Fold a distribution or module name to its import-name form."""
    return name.strip().lower().replace("-", "_").replace(".", "_")


def parse_requirements(paths: Sequence[str]) -> Dict[str, Set[str]]:
    """Map each normalised distribution name to the module names it provides.

    Accepts several manifests so that runtime and test requirements can live
    in separate files. Only the distribution name is used; version specifiers
    are stripped because the check asks "is this importable after installing
    the file", not "is this version pinned".
    """
    provided: Dict[str, Set[str]] = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                line = raw.split("#", 1)[0].strip()
                if not line or line.startswith("-"):
                    continue
                dist = _normalize(line.split(";")[0].strip())
                for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
                    dist = dist.split(separator)[0]
                if not dist:
                    continue
                provided[dist] = set(IMPORT_NAME_ALIASES.get(dist, {dist}))
    return provided


def check_deps() -> List[str]:
    """Every third-party import must be provided by a requirements manifest."""
    provided = parse_requirements(REQUIREMENT_FILES)
    if not provided:
        return [f"no requirements manifest found at {REQUIREMENT_FILES!r}"]
    importable: Set[str] = set()
    for modules in provided.values():
        importable |= modules

    first_party = _first_party_modules() | set(sys.stdlib_module_names)
    missing: Dict[str, List[str]] = {}

    for directory in (SRC_DIR, TESTS_DIR):
        if not os.path.isdir(directory):
            continue
        for entry in sorted(os.listdir(directory)):
            if not entry.endswith(".py"):
                continue
            path = os.path.join(directory, entry)
            for module in sorted(collect_imports(path)):
                if module in first_party or module in importable:
                    continue
                missing.setdefault(module, []).append(f"{directory}/{entry}".replace(REPO_ROOT + os.sep, ""))

    if not missing:
        return []

    report = [
        f"the requirements manifests do not provide {len(missing)} module(s) "
        "imported by this repository:",
    ]
    for module, sources in sorted(missing.items()):
        report.append(f"  - {module}  (imported by {', '.join(sources)})")
    report.append(
        "Add a distribution to requirements.txt (runtime) or requirements-dev.txt "
        "(tests), or drop the import."
    )
    return report


def _first_party_modules() -> Set[str]:
    """Module names that this repository itself defines."""
    names: Set[str] = {"src", "tools", "tests"}
    for directory in (SRC_DIR, TESTS_DIR, os.path.join(REPO_ROOT, "tools")):
        if not os.path.isdir(directory):
            continue
        for entry in os.listdir(directory):
            if entry.endswith(".py"):
                names.add(_normalize(entry[: -len(".py")]))
    return names


def collect_imports(path: str) -> Set[str]:
    """Every top-level module imported by ``path``, at any nesting depth.

    Imports inside functions are included deliberately: this repository defers
    its heaviest imports into function bodies on purpose, so a top-level-only
    scan would miss exactly the imports most likely to be forgotten in
    ``requirements.txt``.
    """
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)

    modules: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(_normalize(alias.name.split(".")[0]))
        elif isinstance(node, ast.ImportFrom):
            # A relative import (level > 0) is always first-party.
            if node.level == 0 and node.module:
                modules.add(_normalize(node.module.split(".")[0]))
    return modules


# --------------------------------------------------------------------------- #
# Check 2: pure-numpy retrieval path
# --------------------------------------------------------------------------- #

def check_tfidf() -> List[str]:
    """The no-scikit-learn retrieval path must rank, not raise.

    The fallback branch is forced on regardless of what is installed. Without
    that, a CI runner with scikit-learn would take the other branch and this
    check would pass while testing nothing about the path that was broken --
    the same silent-passing failure mode as the original defect.

    Two assertions, because either alone is too weak: calling ``top_k`` proves
    the code runs, and the ranking assertions prove the vector it sorts over is
    a per-document score vector rather than a single number that happens not
    to raise.
    """
    sys.path.insert(0, SRC_DIR)
    try:
        import honest_baselines
        from honest_baselines import TFIDFRetriever
    except ImportError as exc:  # pragma: no cover - reported as a failure
        return [f"could not import honest_baselines: {exc}"]

    corpus = [
        "Paris is the capital and most populous city of France.",
        "Python was created by Guido van Rossum and first released in 1991.",
        "Mount Everest is the highest mountain above mean sea level.",
    ]

    # Force the pure-numpy path, then put the module back the way it was.
    original = honest_baselines._HAS_SKLEARN
    honest_baselines._HAS_SKLEARN = False
    try:
        retriever = TFIDFRetriever(corpus)
        try:
            ranked = retriever.top_k("What is the capital of France?", k=3)
        except Exception as exc:  # noqa: BLE001 - the failure text is the message
            return [f"TFIDFRetriever.top_k raised {type(exc).__name__}: {exc}"]
    finally:
        honest_baselines._HAS_SKLEARN = original

    failures: List[str] = []
    if len(ranked) != 3:
        failures.append(f"top_k(k=3) returned {len(ranked)} results, expected 3")
    if not ranked:
        return failures + ["top_k returned no results; nothing was ranked"]

    indices = [index for index, _ in ranked]
    scores = [score for _, score in ranked]
    if len(set(indices)) != len(indices):
        failures.append(f"top_k returned duplicate document indices: {indices}")
    if scores != sorted(scores, reverse=True):
        failures.append(f"top_k results are not sorted by descending score: {scores}")
    if indices[0] != 0:
        failures.append(
            f"top_k ranked document {indices[0]} first for 'What is the capital "
            f"of France?'; expected the Paris document (index 0). Scores: {scores}"
        )
    if scores[0] <= 0.0:
        failures.append(
            f"the best-matching document scored {scores[0]}; a non-positive "
            "score means the query and corpus vectors were never compared"
        )
    return failures


# --------------------------------------------------------------------------- #
# Check 3: results provenance
# --------------------------------------------------------------------------- #

def check_provenance() -> List[str]:
    """A results file must not be labelled with a branch it never measured."""
    sys.path.insert(0, SRC_DIR)
    try:
        import real_ablation
    except SystemExit as exc:  # pragma: no cover - reported as a failure
        return [f"importing real_ablation refused to start: {exc}"]

    failures: List[str] = []

    # The script must locate its own checkout, so the command documented in
    # results.md works from a clean clone with no environment setup.
    if not os.path.isdir(os.path.join(real_ablation.SELF_CHECKOUT, "src")):
        failures.append(
            f"SELF_CHECKOUT={real_ablation.SELF_CHECKOUT!r} does not look like a "
            "checkout: expected a 'src' directory beside real_ablation.py"
        )

    # git_provenance must resolve the checkout it is handed, so a results file
    # can record which revision produced it.
    provenance = real_ablation.git_provenance(REPO_ROOT)
    if not provenance.get("commit"):
        failures.append(
            f"git_provenance() returned no commit SHA for {REPO_ROOT!r} "
            f"(branch={provenance.get('branch')!r}); results would not record "
            "which revision was measured"
        )

    # The decision that prevents a mislabelled results file. Each case asserts
    # the specific outcome, so a check that always returns None -- the exact
    # shape of the original defect -- fails here.
    verify = real_ablation.verify_branch_label
    cases: List[Tuple[str, Optional[str], bool]] = [
        # (label, branch reported by the checkout, should_be_rejected)
        ("main", "main", False),                    # honest label is allowed
        ("main", "review-fixes", True),             # a branch it never measured
        ("main", None, True),                       # unverifiable provenance
    ]
    for label, actual, should_reject in cases:
        try:
            verdict = verify(label, actual)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                f"verify_branch_label({label!r}, {actual!r}) raised "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        rejected = verdict is not None
        if rejected != should_reject:
            failures.append(
                f"verify_branch_label({label!r}, {actual!r}) "
                f"{'rejected' if rejected else 'accepted'} the label, but this "
                f"case must be {'rejected' if should_reject else 'accepted'}"
                + (f"; verdict was {verdict}" if verdict else "; verdict was None")
            )
    return failures


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

CHECKS = {
    "deps": ("requirements.txt covers every third-party import", check_deps),
    "tfidf": ("pure-numpy retrieval path ranks instead of raising", check_tfidf),
    "provenance": ("ablation results carry verifiable branch provenance", check_provenance),
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true", help="list check names and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name in CHECKS:
            print(name)
        return 0

    failed = 0
    for name, (summary, func) in CHECKS.items():
        try:
            failures = func()
        except Exception as exc:  # noqa: BLE001 - a crashing check is a failed check
            failures = [f"check raised unexpectedly: {type(exc).__name__}: {exc}"]

        if failures:
            failed += 1
            print(f"FAIL  {name}: {summary}")
            for line in failures:
                print(f"        {line}")
        else:
            print(f"ok    {name}: {summary}")

    if failed:
        print(f"\n{failed} of {len(CHECKS)} checks failed.")
        return 1
    print(f"\nall {len(CHECKS)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

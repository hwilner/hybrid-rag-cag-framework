"""Restricted-set re-analysis of the committed ablation: the 21 answerable questions.

The benchmark defect (README "What I would do next", item 2) is that 34 of the
55 questions are unanswerable from the 100-document corpus: their gold answers
appear in no document, so no retrieval or reader work could ever answer them.
``results/RESULTS.md`` section 7 measures this directly. Until the corpus is
rebuilt, the only defensible reading of the pipeline comparison is the
**restricted set**: the 21 questions whose gold answer appears in the corpus.

This script performs that repair as a *re-analysis of committed measured
data*, not a new experiment:

  * per-question F1 / EM / containment for the three arms come from
    ``results/ablation/ablation_main.json`` (the 2026-09 real runs);
  * the answerable set comes from ``results/diagnosis.json`` (the measured
    retrieval diagnostic: ``found_at`` is the rank of the first retrieved
    document containing the gold answer, ``null`` when no document does).

No model is loaded and no question is re-run. Every number written to the
output artefact is an arithmetic re-aggregation of those committed
per-question measurements.

Integrity guard: before restricting, the script recomputes the **full-set**
per-arm aggregates from the per-question records and requires them to match
the committed arm-level aggregates exactly. If the join or the arithmetic
were wrong, this check would fail and no artefact would be written.

Usage:
    python src/restricted_answerable_eval.py
        [--ablation results/ablation/ablation_main.json]
        [--diagnosis results/diagnosis.json]
        [--output results/restricted_answerable.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date

import numpy as np

SELF_SRC = os.path.dirname(os.path.abspath(__file__))
SELF_CHECKOUT = os.path.dirname(SELF_SRC)
sys.path.insert(0, SELF_SRC)

from run_honest_evaluation import json_safe, paired_bootstrap_ci, wilcoxon  # noqa: E402

METRICS = ("f1", "em", "contains")
ARMS = ("RAG", "CAG", "Hybrid")

# The README's paired-comparison table on the full set, replicated on the
# restricted set: same comparisons, same statistics.
COMPARISONS = (
    ("Hybrid", "RAG", "f1"),
    ("Hybrid", "CAG", "f1"),
    ("RAG", "CAG", "f1"),
    ("Hybrid", "CAG", "contains"),
    ("RAG", "CAG", "contains"),
)


def sha256_of(path: str) -> str:
    """Return the hex SHA-256 of a file, for artefact provenance.

    Args:
        path: filesystem path to hash.

    Returns:
        Lowercase hex digest string.
    """
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def load_inputs(ablation_path: str, diagnosis_path: str) -> tuple:
    """Load the two committed artefacts and join them on question text.

    Args:
        ablation_path: path to the committed three-arm ablation artefact.
        diagnosis_path: path to the committed failure-diagnosis artefact.

    Returns:
        A pair ``(ablation, answerable)`` where ``ablation`` is the parsed
        ablation JSON and ``answerable`` is the list of question strings
        whose gold answer was found in the retrieved documents.

    Raises:
        ValueError: if the two artefacts do not cover identical question
            sets, or contain duplicate question texts (in which case a join
            on question text would be ambiguous).
    """
    with open(ablation_path) as handle:
        ablation = json.load(handle)
    with open(diagnosis_path) as handle:
        diagnosis = json.load(handle)

    diag_questions = [row["q"] for row in diagnosis["per_question"]]
    abl_questions = [row["q"] for row in ablation["arms"][ARMS[0]]["per_question"]]
    if len(set(diag_questions)) != len(diag_questions):
        raise ValueError("diagnosis artefact contains duplicate question texts")
    if set(diag_questions) != set(abl_questions):
        raise ValueError("ablation and diagnosis artefacts cover different questions")

    answerable = [row["q"] for row in diagnosis["per_question"]
                  if row["found_at"] is not None]
    return ablation, answerable


def verify_full_set_aggregates(ablation: dict) -> None:
    """Recompute full-set arm aggregates and require an exact committed match.

    This is the integrity guard: the restricted re-analysis is only
    trustworthy if the per-question records it draws on are exactly the
    records behind the published arm-level numbers.

    Args:
        ablation: parsed ablation artefact.

    Raises:
        ValueError: if any recomputed full-set aggregate differs from the
            committed arm-level value.
    """
    metric_key = {"f1": "F1", "em": "EM", "contains": "containment"}
    for arm in ARMS:
        rows = ablation["arms"][arm]["per_question"]
        for metric in METRICS:
            recomputed = float(np.mean([row[metric] for row in rows]))
            committed = float(ablation["arms"][arm][metric_key[metric]])
            if abs(recomputed - committed) > 1e-12:
                raise ValueError(
                    f"full-set integrity check failed: {arm}.{metric} "
                    f"recomputed {recomputed!r} != committed {committed!r}"
                )


def restricted_scores(ablation: dict, answerable: list) -> dict:
    """Recompute per-arm aggregates on the answerable subset.

    Args:
        ablation: parsed ablation artefact with per-question records.
        answerable: question texts in the restricted set.

    Returns:
        Mapping ``arm -> {"n": int, "F1": float, "EM": float,
        "containment": float, "per_question": list}`` restricted to the
        answerable questions, in ablation order.
    """
    out = {}
    metric_key = {"f1": "F1", "em": "EM", "contains": "containment"}
    for arm in ARMS:
        rows = [row for row in ablation["arms"][arm]["per_question"]
                if row["q"] in set(answerable)]
        out[arm] = {
            "n": len(rows),
            "per_question": rows,
        }
        for metric in METRICS:
            out[arm][metric_key[metric]] = float(np.mean([row[metric] for row in rows]))
    return out


def paired_comparisons(restricted: dict) -> list:
    """Paired statistics on the restricted set, README table replicated.

    Args:
        restricted: output of :func:`restricted_scores`.

    Returns:
        One record per entry in :data:`COMPARISONS`, with the mean paired
        difference, a percentile bootstrap 95% CI (seeded, deterministic),
        and the two-sided Wilcoxon p-value (``None`` when the test does not
        apply).
    """
    records = []
    for arm_a, arm_b, metric in COMPARISONS:
        rows_a = {row["q"]: row[metric] for row in restricted[arm_a]["per_question"]}
        rows_b = {row["q"]: row[metric] for row in restricted[arm_b]["per_question"]}
        diffs = [rows_a[q] - rows_b[q]
                 for q in [row["q"] for row in restricted[arm_a]["per_question"]]]
        lo, hi = paired_bootstrap_ci(diffs, n_boot=10000, seed=0)
        p_value, n_nonzero = wilcoxon(diffs)
        records.append({
            "comparison": f"{arm_a} - {arm_b}",
            "metric": metric,
            "mean_delta": float(np.mean(diffs)),
            "bootstrap_ci95": [lo, hi],
            "wilcoxon_p": p_value,
            "n_nonzero_diffs": n_nonzero,
        })
    return records


def main() -> None:
    """Entry point: load, verify, restrict, compare, write the artefact."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ablation",
                        default=os.path.join(SELF_CHECKOUT, "results", "ablation",
                                             "ablation_main.json"))
    parser.add_argument("--diagnosis",
                        default=os.path.join(SELF_CHECKOUT, "results", "diagnosis.json"))
    parser.add_argument("--output",
                        default=os.path.join(SELF_CHECKOUT, "results",
                                             "restricted_answerable.json"))
    args = parser.parse_args()

    ablation, answerable = load_inputs(args.ablation, args.diagnosis)
    verify_full_set_aggregates(ablation)
    restricted = restricted_scores(ablation, answerable)
    comparisons = paired_comparisons(restricted)

    artefact = {
        "description": (
            "Restricted-set re-analysis of the committed three-arm ablation, "
            "limited to the questions whose gold answer appears in the "
            "corpus (the benchmark-repair option A from the README). "
            "No model was loaded and no question was re-run: every number "
            "here is an arithmetic re-aggregation of the committed "
            "per-question measurements."
        ),
        "generated": date.today().isoformat(),
        "method": "post-hoc re-aggregation of committed per-question measurements",
        "new_model_runs": 0,
        "source_artefacts": {
            "ablation": {"path": os.path.relpath(args.ablation, SELF_CHECKOUT),
                          "sha256": sha256_of(args.ablation)},
            "diagnosis": {"path": os.path.relpath(args.diagnosis, SELF_CHECKOUT),
                           "sha256": sha256_of(args.diagnosis)},
        },
        "full_set_integrity_check": (
            "recomputed full-set per-arm aggregates match the committed "
            "arm-level aggregates exactly (tolerance 1e-12)"
        ),
        "n_questions_full": ablation["n_questions"],
        "n_questions_restricted": len(answerable),
        "answerable_questions": answerable,
        "arms_restricted": {
            arm: {key: value for key, value in restricted[arm].items()
                  if key != "per_question"}
            for arm in ARMS
        },
        "comparisons_restricted": comparisons,
    }

    with open(args.output, "w") as handle:
        json.dump(json_safe(artefact), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[restricted] {len(answerable)}/{ablation['n_questions']} answerable; "
          f"wrote {args.output}")


if __name__ == "__main__":
    main()

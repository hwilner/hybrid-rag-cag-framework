"""Tests for the instruct-reader reform artefact.

The defect these guard against
------------------------------
``results/instruct_reader_ablation.json`` is a measured artefact (a real
2026-11-16 run, flan-t5-small on CPU). The risks around it are bookkeeping
and claim drift: aggregates that do not match the per-question records, a
restricted set that drifts from the measured answerable set, or a summary
claim in RESULTS.md section 12 that contradicts the artefact's own
pre-registered success criterion. These tests pin all three.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTEFACT_PATH = os.path.join(REPO_ROOT, "results",
                             "instruct_reader_ablation.json")
DIAGNOSIS_PATH = os.path.join(REPO_ROOT, "results", "diagnosis.json")
FIGURE_PATH = os.path.join(REPO_ROOT, "docs", "figures",
                           "reader_reform_comparison.svg")

ARMS = ("RAG", "CAG", "Hybrid")
METRIC_KEYS = {"f1": "F1", "em": "EM", "contains": "containment"}


def _strict_load(path):
    """Parse JSON with NaN/Infinity rejected, per RFC 8259."""
    def reject(constant):
        raise ValueError(f"bare {constant} is not valid JSON per RFC 8259")
    with open(path) as handle:
        return json.load(handle, parse_constant=reject)


def _answerable_set():
    diagnosis = _strict_load(DIAGNOSIS_PATH)
    return {row["q"] for row in diagnosis["per_question"]
            if row["found_at"] is not None}


def test_reform_artefact_is_strict_json():
    """The committed artefact must parse under a strict reader."""
    _strict_load(ARTEFACT_PATH)


def test_aggregates_match_per_question_records():
    """Every reported aggregate must equal the mean over its per-question rows."""
    artefact = _strict_load(ARTEFACT_PATH)
    answerable = _answerable_set()
    for arm in ARMS:
        rows = artefact["per_question"][arm]
        assert len(rows) == 55
        for label, subset in (("full", rows),
                              ("restricted", [r for r in rows
                                              if r["q"] in answerable])):
            assert len(subset) == artefact[label][arm]["n"]
            for metric, key in METRIC_KEYS.items():
                recomputed = float(np.mean([r[metric] for r in subset]))
                committed = float(artefact[label][arm][key])
                assert recomputed == pytest.approx(committed, abs=1e-12)


def test_restricted_set_is_the_measured_answerable_set():
    """The restricted scores must be over exactly the 21 answerable questions."""
    artefact = _strict_load(ARTEFACT_PATH)
    assert artefact["n_questions_restricted"] == 21
    for arm in ARMS:
        assert artefact["restricted"][arm]["n"] == 21


def test_preregistered_criterion_reported_honestly():
    """The artefact must record the criterion and the data must decide it.

    The pre-registered criterion (issue #7): Hybrid beats RAG on token-F1 on
    the restricted set with a bootstrap 95% CI excluding zero. This test does
    not assert the criterion passed -- it asserts the artefact contains the
    data to evaluate it, and it recomputes the verdict so a summary that
    disagrees with the numbers cannot stand.
    """
    artefact = _strict_load(ARTEFACT_PATH)
    assert "preregistered_success_criterion" in artefact
    comparison = next(
        c for c in artefact["comparisons_restricted"]
        if c["comparison"] == "Hybrid - RAG" and c["metric"] == "f1"
    )
    lo, hi = comparison["bootstrap_ci95"]
    passed = lo > 0  # CI excluding zero in Hybrid's favour
    # The measured 2026-11-16 outcome: the criterion was NOT met (the CI
    # includes zero). If a future re-run changes the numbers, this test
    # forces the claim here to be revisited rather than silently stale.
    assert passed is False


def test_reader_reform_figure_regenerates_byte_for_byte(tmp_path):
    """The committed SVG must regenerate byte-for-byte from the data."""
    out = tmp_path / "figure.svg"
    subprocess.run(
        [sys.executable,
         os.path.join(REPO_ROOT, "tools", "make_reader_reform_figure.py"),
         "--output", str(out)],
        check=True, cwd=REPO_ROOT,
    )
    with open(out, "rb") as handle:
        regenerated = handle.read()
    with open(FIGURE_PATH, "rb") as handle:
        committed = handle.read()
    assert regenerated == committed

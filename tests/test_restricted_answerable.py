"""Tests for the restricted-set benchmark-repair re-analysis.

The defects these guard against
-------------------------------
The restricted set (the 21 questions whose gold answer appears in the
corpus) is the only subset of this benchmark on which any system comparison
is meaningful. The re-analysis in ``results/restricted_answerable.json`` is
a re-aggregation of committed per-question measurements, so the risks are
not modelling risks but bookkeeping ones: a wrong join between the ablation
and diagnosis artefacts, a silent drift between the committed artefact and
the script that produced it, or a figure that no longer matches the data.
These tests pin all three.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

ABLATION_PATH = os.path.join(REPO_ROOT, "results", "ablation",
                             "ablation_main.json")
DIAGNOSIS_PATH = os.path.join(REPO_ROOT, "results", "diagnosis.json")
RESTRICTED_PATH = os.path.join(REPO_ROOT, "results",
                               "restricted_answerable.json")
FIGURE_PATH = os.path.join(REPO_ROOT, "docs", "figures",
                           "restricted_answerable_comparison.svg")

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


def test_restricted_artefact_is_strict_json():
    """The committed artefact must parse under a strict reader."""
    _strict_load(RESTRICTED_PATH)


def test_restricted_set_matches_measured_answerable_set():
    """The restricted set must be exactly the 21 measured-answerable questions."""
    artefact = _strict_load(RESTRICTED_PATH)
    answerable = _answerable_set()
    assert len(answerable) == 21
    assert set(artefact["answerable_questions"]) == answerable
    assert artefact["n_questions_restricted"] == 21
    assert artefact["n_questions_full"] == 55


def test_no_new_model_runs_claimed():
    """The artefact must state that it re-aggregates, not re-measures."""
    artefact = _strict_load(RESTRICTED_PATH)
    assert artefact["new_model_runs"] == 0
    assert "re-aggregation" in artefact["method"]


def test_full_set_aggregates_match_committed_ablation():
    """Recomputed full-set aggregates must match the committed arm numbers.

    Independent re-implementation of the script's own integrity guard, so a
    defect in the script cannot silently validate itself.
    """
    ablation = _strict_load(ABLATION_PATH)
    for arm in ARMS:
        rows = ablation["arms"][arm]["per_question"]
        assert len(rows) == 55
        for metric, key in METRIC_KEYS.items():
            recomputed = float(np.mean([row[metric] for row in rows]))
            committed = float(ablation["arms"][arm][key])
            assert recomputed == pytest.approx(committed, abs=1e-12)


def test_restricted_aggregates_match_per_question_records():
    """Restricted arm aggregates must equal the mean over the subset."""
    artefact = _strict_load(RESTRICTED_PATH)
    ablation = _strict_load(ABLATION_PATH)
    answerable = _answerable_set()
    for arm in ARMS:
        rows = [row for row in ablation["arms"][arm]["per_question"]
                if row["q"] in answerable]
        assert len(rows) == 21
        for metric, key in METRIC_KEYS.items():
            recomputed = float(np.mean([row[metric] for row in rows]))
            committed = float(artefact["arms_restricted"][arm][key])
            assert recomputed == pytest.approx(committed, abs=1e-12)


def test_script_reproduces_committed_artefact(tmp_path):
    """Re-running the script must reproduce the committed artefact.

    Compared as parsed JSON with the generation date excluded: every number
    must be identical, while the date is allowed to reflect the day the test
    runs.
    """
    out = tmp_path / "restricted.json"
    subprocess.run(
        [sys.executable, os.path.join(SRC_DIR, "restricted_answerable_eval.py"),
         "--output", str(out)],
        check=True, cwd=REPO_ROOT,
    )
    regenerated = _strict_load(str(out))
    committed = _strict_load(RESTRICTED_PATH)
    regenerated.pop("generated")
    committed.pop("generated")
    assert regenerated == committed


def test_figure_script_reproduces_committed_svg(tmp_path):
    """The committed SVG must regenerate byte-for-byte from the data."""
    out = tmp_path / "figure.svg"
    subprocess.run(
        [sys.executable,
         os.path.join(REPO_ROOT, "tools", "make_restricted_figure.py"),
         "--output", str(out)],
        check=True, cwd=REPO_ROOT,
    )
    with open(out, "rb") as handle:
        regenerated = handle.read()
    with open(FIGURE_PATH, "rb") as handle:
        committed = handle.read()
    assert regenerated == committed

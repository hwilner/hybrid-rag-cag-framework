"""Tests for ``src/benchmark_validity.py`` and the committed validity report.

The committed Tier-2 benchmark is expected to FAIL the strict gate -- that
failure is the measured dataset defect that issue #5 tracks, and these tests
pin the measured counts so the report cannot silently change.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import benchmark_validity as bv  # noqa: E402
from dataset_loader import load_tier2_dataset  # noqa: E402

VALIDITY_JSON = os.path.join(REPO_ROOT, "results", "benchmark_validity.json")
TOOL = os.path.join(REPO_ROOT, "tools", "check_benchmark_validity.py")


# ---------------------------------------------------------------------------
# Equivalence with the diagnosis definition of answerability
# ---------------------------------------------------------------------------

def test_normalize_matches_diagnose_failures():
    """The validator's normalisation must agree with the diagnosis script.

    ``diagnose_failures.normalize`` defined the 21/55 answerability count in
    ``results/diagnosis.json``; if the two ever drifted, the gate would
    measure a different quantity than the diagnosis it gates.
    """
    diagnose = pytest.importorskip("diagnose_failures")
    samples = [
        "Paris",
        "The Quick, Brown Fox!",
        "  multiple   spaces\tand\nnewlines ",
        "AN UPPER-CASE answer.",
        "",
        "a an the",
        "Numbers 3.14 and 42, with punctuation: yes.",
    ]
    for s in samples:
        assert bv.normalize(s) == diagnose.normalize(s), s
    pairs = [
        ("The capital of France is Paris.", "Paris"),
        ("No answer here.", "Paris"),
        ("", "Paris"),
        ("Paris", ""),
    ]
    for pred, gold in pairs:
        assert bv.contains(pred, gold) == bool(diagnose.contains(pred, gold))


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_schema_accepts_well_formed():
    corpus = ["Paris is the capital of France."]
    questions = [{"question": "Capital of France?", "answer": "Paris",
                  "type": "factual", "difficulty": "easy"}]
    assert bv.validate_schema(corpus, questions) == []


def test_schema_rejects_duplicate_questions():
    q = {"question": "Same?", "answer": "x", "type": "t", "difficulty": "d"}
    problems = bv.validate_schema(["x is here."], [q, dict(q)])
    assert any("duplicates" in p for p in problems)


def test_schema_rejects_answer_that_normalises_to_empty():
    corpus = ["Some document."]
    questions = [{"question": "Q?", "answer": "the a an",
                  "type": "t", "difficulty": "d"}]
    problems = bv.validate_schema(corpus, questions)
    assert any("normalises to empty" in p for p in problems)


def test_schema_rejects_empty_corpus_and_missing_fields():
    problems = bv.validate_schema([], [{"question": "Q?"}])
    assert any("corpus is empty" in p for p in problems)
    assert any("missing" in p for p in problems)


# ---------------------------------------------------------------------------
# Answerability and the gate on synthetic benchmarks
# ---------------------------------------------------------------------------

def test_fully_answerable_benchmark_passes():
    corpus = ["Paris is the capital of France.",
              "The mitochondrion is the powerhouse of the cell."]
    questions = [
        {"question": "Capital of France?", "answer": "Paris",
         "type": "factual", "difficulty": "easy"},
        {"question": "Powerhouse of the cell?", "answer": "mitochondrion",
         "type": "factual", "difficulty": "easy"},
    ]
    report = bv.answerability_report(corpus, questions)
    assert report["n_answerable"] == 2
    assert report["answerable_fraction"] == 1.0
    verdict = bv.gate_verdict(report)
    assert verdict["passed"], verdict["failures"]


def test_partially_answerable_benchmark_fails_strict_gate():
    corpus = ["Paris is the capital of France."]
    questions = [
        {"question": "Capital of France?", "answer": "Paris",
         "type": "factual", "difficulty": "easy"},
        {"question": "Composed answer?", "answer": "a sentence in no document",
         "type": "multi-hop", "difficulty": "hard"},
    ]
    report = bv.answerability_report(corpus, questions)
    assert report["n_answerable"] == 1
    verdict = bv.gate_verdict(report)
    assert not verdict["passed"]
    assert any("multi-hop" in f for f in verdict["failures"])
    relaxed = bv.gate_verdict(report, min_answerable=0.5,
                             require_all_types=False)
    assert relaxed["passed"]


def test_gate_uses_integer_comparison_at_boundaries():
    """A float boundary must not flip the verdict: 2/3 vs min 2/3."""
    corpus = ["alpha beta", "gamma delta"]
    questions = [
        {"question": "a?", "answer": "alpha", "type": "t", "difficulty": "d"},
        {"question": "g?", "answer": "gamma", "type": "t", "difficulty": "d"},
        {"question": "x?", "answer": "absent", "type": "t", "difficulty": "d"},
    ]
    report = bv.answerability_report(corpus, questions)
    verdict = bv.gate_verdict(report, min_answerable=2.0 / 3.0,
                              require_all_types=False)
    assert verdict["passed"]


# ---------------------------------------------------------------------------
# The committed benchmark and its committed report
# ---------------------------------------------------------------------------

def test_committed_benchmark_counts_are_the_measured_ones():
    """Pin the diagnosis numbers: 21/55 answerable, per-difficulty split."""
    corpus, questions = load_tier2_dataset(REPO_ROOT)
    assert len(corpus) == 100
    assert len(questions) == 55
    report = bv.answerability_report(corpus, questions)
    assert report["n_answerable"] == 21
    assert report["by_difficulty"] == {
        "easy": {"answerable": 10, "total": 10, "fraction": 1.0},
        "medium": {"answerable": 10, "total": 10, "fraction": 1.0},
        "hard": {"answerable": 1, "total": 15, "fraction": 1 / 15},
        "very_hard": {"answerable": 0, "total": 20, "fraction": 0.0},
    }
    assert report["by_type"]["factual"]["answerable"] == 10
    assert report["by_type"]["inference"]["answerable"] == 10
    assert report["by_type"]["multi-hop"]["answerable"] == 1
    verdict = bv.gate_verdict(report)
    assert not verdict["passed"]


def test_committed_report_matches_recomputation():
    """results/benchmark_validity.json must equal a fresh measurement.

    The committed file is a report of a real run of the tool; if the
    benchmark or the tool changes, this test forces the report to be
    regenerated rather than silently going stale.
    """
    corpus, questions = load_tier2_dataset(REPO_ROOT)
    fresh = bv.answerability_report(corpus, questions)
    with open(VALIDITY_JSON, encoding="utf-8") as fh:
        committed = json.load(fh)
    assert committed["report"] == fresh
    assert committed["new_model_runs"] == 0
    assert committed["gate"]["passed"] is False
    assert committed["schema_problems"] == []


def test_tool_cli_fails_committed_benchmark_and_passes_candidate(tmp_path):
    """End to end: exit 1 on the committed benchmark, 0 on a valid one."""
    bad = subprocess.run([sys.executable, TOOL], capture_output=True, text=True)
    assert bad.returncode == 1
    assert "21/55" in bad.stdout
    assert "VERDICT: FAIL" in bad.stdout

    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({
        "corpus": ["Paris is the capital of France."],
        "questions": [{"question": "Capital of France?", "answer": "Paris",
                       "type": "factual", "difficulty": "easy"}],
    }))
    good = subprocess.run(
        [sys.executable, TOOL, "--candidate", str(candidate)],
        capture_output=True, text=True)
    assert good.returncode == 0
    assert "VERDICT: PASS" in good.stdout


def test_tool_report_is_strict_json(tmp_path):
    out = tmp_path / "report.json"
    subprocess.run([sys.executable, TOOL, "--out", str(out)],
                   capture_output=True, text=True)
    text = out.read_text(encoding="utf-8")
    # A strict reader (RFC 8259: no NaN/Infinity) must parse it.
    parsed = json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(
        ValueError(f"non-strict JSON constant: {c}")))
    assert parsed["report"]["n_questions"] == 55

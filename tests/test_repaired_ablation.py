"""Tests for ``tools/run_repaired_ablation.py``.

The wrapper's data-gate path is exercised without torch, checkpoints, or any
model run. The seeded three-arm run itself remains deferred on GPU-scale
compute and the owner's corpus-construction decision.
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO_ROOT, "tools", "run_repaired_ablation.py")


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_repaired_ablation", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(path, answer="Paris"):
    path.write_text(json.dumps({
        "corpus": ["Paris is the capital of France."],
        "questions": [{
            "question": "What is the capital of France?",
            "answer": answer,
            "type": "factual",
            "difficulty": "easy",
        }],
    }), encoding="utf-8")
    return path


def test_valid_candidate_loads_with_gate_provenance(tmp_path):
    runner = _load_runner()
    candidate = _candidate(tmp_path / "candidate.json")
    corpus, questions, provenance = runner.load_validated_candidate(str(candidate))
    assert corpus == ["Paris is the capital of France."]
    assert len(questions) == 1
    assert provenance["source"] == "candidate_benchmark_json"
    assert provenance["gate"]["passed"] is True
    assert provenance["n_questions"] == 1
    assert len(provenance["file_sha256"]) == 64


def test_unanswerable_candidate_is_refused(tmp_path):
    runner = _load_runner()
    candidate = _candidate(tmp_path / "bad.json", answer="A sentence in no document")
    with pytest.raises(ValueError, match="validity gate"):
        runner.load_validated_candidate(str(candidate))


def test_schema_invalid_candidate_is_refused_cleanly(tmp_path):
    runner = _load_runner()
    candidate = tmp_path / "schema_bad.json"
    candidate.write_text(json.dumps({
        "corpus": ["Paris is here."],
        "questions": [{"question": "Q?"}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="schema gate"):
        runner.load_validated_candidate(str(candidate))


def test_main_rejects_invalid_candidate_before_loading_harness(tmp_path):
    runner = _load_runner()
    candidate = _candidate(tmp_path / "bad.json", answer="Not in the corpus")
    out = tmp_path / "should_not_exist.json"
    with pytest.raises(SystemExit, match="validity gate"):
        runner.main([
            "--candidate", str(candidate),
            "--branch", "main",
            "--out", str(out),
            "--allow-unverified-branch",
        ])
    assert not out.exists()


def test_record_benchmark_provenance_updates_completed_artefact(tmp_path):
    runner = _load_runner()
    out = tmp_path / "ablation.json"
    out.write_text(json.dumps({"provenance": {"evaluated_branch": "main"},
                               "arms": {}}), encoding="utf-8")
    provenance = {"source": "candidate_benchmark_json", "file_sha256": "abc123"}
    runner.record_benchmark_provenance(str(out), provenance)
    updated = json.loads(out.read_text(encoding="utf-8"))
    assert updated["provenance"]["benchmark"] == provenance
    assert updated["provenance"]["repaired_runner"]["sha256"]
    assert updated["arms"] == {}

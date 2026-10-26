"""Tests for the dataset loader.

The defect these guard against
------------------------------
``src/real_ablation.py``, ``src/ceiling_test.py`` and
``src/diagnose_failures.py`` each carried their own copy of a loader that
reached into ``src/option3_full_scale_evaluation.py`` and ``exec``-ed the
dataset function out of it. That file was quarantined under ``retracted/`` on
2026-10-13, so all three scripts died with a ``FileNotFoundError`` before
doing any work -- including the three commands the README documents under
"Reproducing everything".

These tests also pin the second half of the same failure: two of the three
scripts defaulted ``BRANCH_DIR`` to a hard-coded ``/workspace/branches/main``,
which does not exist in any clone.
"""

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset_loader import load_tier2_dataset  # noqa: E402


def test_loader_finds_the_canonical_module():
    """The loader must find the benchmark in a clean checkout."""
    corpus, questions = load_tier2_dataset(REPO_ROOT)

    assert len(corpus) == 100
    assert len(questions) == 55


def test_loaded_dataset_matches_the_module_it_prefers():
    """The loader must return exactly what ``tier2_dataset`` produces."""
    import tier2_dataset

    expected_corpus, expected_questions = tier2_dataset.create_large_scale_dataset()
    corpus, questions = load_tier2_dataset(REPO_ROOT)

    assert corpus == expected_corpus
    assert questions == expected_questions


def test_loader_reports_every_path_it_tried():
    """A missing dataset must name the paths, not just fail.

    The failure this replaces was a bare ``FileNotFoundError`` on a path the
    reader would not expect.
    """
    with pytest.raises(FileNotFoundError) as excinfo:
        load_tier2_dataset("/nonexistent/branch", fallback_dir="/nonexistent/fallback")

    message = str(excinfo.value)
    assert "tier2_dataset.py" in message
    assert "option3_full_scale_evaluation.py" in message
    assert "/nonexistent/branch" in message
    assert "/nonexistent/fallback" in message


def test_legacy_layout_is_still_supported(tmp_path):
    """A checkout that predates the extraction must still load.

    The harness is built to evaluate other branches, and older branches have
    the dataset in the pre-extraction file.
    """
    legacy_dir = tmp_path / "branch"
    (legacy_dir / "src").mkdir(parents=True)
    (legacy_dir / "src" / "option3_full_scale_evaluation.py").write_text(
        "def create_large_scale_dataset():\n"
        "    return ['a legacy document'], [{'question': 'q', 'answer': 'a'}]\n"
    )

    corpus, questions = load_tier2_dataset(str(legacy_dir))

    assert corpus == ["a legacy document"]
    assert questions == [{"question": "q", "answer": "a"}]


@pytest.mark.parametrize(
    "module_name",
    ["real_ablation", "ceiling_test", "diagnose_failures"],
)
def test_the_harness_scripts_can_load_the_dataset(module_name):
    """Every script that needs the benchmark must be able to load it.

    This is the direct regression guard: before the fix, each of these
    raised ``FileNotFoundError`` on the quarantined file's path.
    """
    module = __import__(module_name)

    corpus, questions = module.load_dataset()

    assert len(corpus) == 100
    assert len(questions) == 55


@pytest.mark.parametrize(
    "module_name",
    ["real_ablation", "ceiling_test", "diagnose_failures"],
)
def test_the_harness_scripts_default_to_their_own_checkout(module_name):
    """The default branch must be the checkout the script lives in.

    Two of the three scripts used to default to a hard-coded
    ``/workspace/branches/main``, which exists in no clone.
    """
    module = __import__(module_name)

    assert module.BRANCH_DIR == module.SELF_CHECKOUT


def test_the_ceiling_test_defaults_to_the_committed_ablation_run():
    """The documented command must reproduce the committed ceiling artefact.

    The "Hybrid (current)" row is copied from an ablation run. It used to
    require ``CURRENT`` to be set by hand, so the documented command wrote a
    file that silently lacked the row and no longer matched the published
    table.
    """
    import ceiling_test

    path = ceiling_test.current_ablation_path()

    assert path is not None
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as handle:
        ablation = json.load(handle)
    assert "Hybrid" in ablation["arms"]


def test_the_ceiling_test_honours_an_explicit_current(monkeypatch):
    """An explicit CURRENT must still win over the default."""
    import ceiling_test

    monkeypatch.setenv("CURRENT", "/tmp/some-ablation.json")

    assert ceiling_test.current_ablation_path() == "/tmp/some-ablation.json"

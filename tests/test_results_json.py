"""Tests for the results artefacts and the writer that produces them.

The defect these guard against
------------------------------
``src/run_honest_evaluation.py`` wrote a bare ``NaN`` into
``results/verified_tier2.json`` for ``comparisons.BM25_vs_TFIDF_F1
.n_nonzero_diffs``. ``NaN`` is not a valid JSON number (RFC 8259), so the
committed artefact did not parse under a strict reader.

The value was also wrong in a way that mattered: the count of non-zero paired
differences does not depend on scipy, and was exactly determinate. It was
reported as ``NaN`` precisely in the cases where the paired differences were
all zero.
"""

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import run_honest_evaluation  # noqa: E402


def _reject_constant(constant):
    raise ValueError(f"bare {constant} is not valid JSON per RFC 8259")


def test_committed_results_are_valid_json():
    """Every artefact under ``results/`` must parse under a strict reader."""
    results_dir = os.path.join(REPO_ROOT, "results")
    checked = 0

    for directory, _, filenames in os.walk(results_dir):
        for filename in sorted(filenames):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(directory, filename)
            relative = os.path.relpath(path, REPO_ROOT)
            with open(path, encoding="utf-8") as handle:
                json.load(handle, parse_constant=_reject_constant)
            checked += 1

    assert checked, "no results artefacts were found to check"


def test_json_safe_replaces_non_finite_numbers():
    """Non-finite numbers become null, which is valid JSON."""
    assert run_honest_evaluation.json_safe(float("nan")) is None
    assert run_honest_evaluation.json_safe(float("inf")) is None
    assert run_honest_evaluation.json_safe(float("-inf")) is None
    assert run_honest_evaluation.json_safe(1.5) == 1.5


def test_json_safe_walks_nested_structures():
    """The sanitiser has to reach into the nested comparison payload."""
    payload = {
        "a": float("nan"),
        "b": [1.0, float("inf")],
        "c": {"d": float("-inf"), "e": 2.0},
    }

    assert run_honest_evaluation.json_safe(payload) == {
        "a": None,
        "b": [1.0, None],
        "c": {"d": None, "e": 2.0},
    }


def test_sanitised_payload_round_trips_through_a_strict_reader():
    """The point of the sanitiser: the result must survive a strict parse."""
    payload = {"value": float("nan"), "nested": [float("inf")]}

    text = json.dumps(run_honest_evaluation.json_safe(payload), allow_nan=False)
    parsed = json.loads(text, parse_constant=_reject_constant)

    assert parsed == {"value": None, "nested": [None]}


def test_wilcoxon_reports_the_true_count_of_non_zero_differences():
    """The count is determinate and does not depend on scipy."""
    _, n_nonzero = run_honest_evaluation.wilcoxon([0.0] * 55)

    assert n_nonzero == 0


def test_wilcoxon_reports_the_count_when_the_test_does_apply():
    _, n_nonzero = run_honest_evaluation.wilcoxon([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    assert n_nonzero == 7


def test_wilcoxon_p_value_is_none_when_the_test_does_not_apply():
    """Fewer than six non-zero differences: no p-value, but still a count."""
    p_value, n_nonzero = run_honest_evaluation.wilcoxon([0.0, 1.0, 2.0])

    assert p_value is None
    assert n_nonzero == 2


def test_wilcoxon_p_value_is_a_number_or_none_never_nan():
    """The p-value may be missing, but it must never be NaN."""
    p_value, _ = run_honest_evaluation.wilcoxon([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    assert p_value is None or isinstance(p_value, float)
    if p_value is not None:
        assert p_value == p_value  # not NaN
        assert 0.0 <= p_value <= 1.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_writing_a_non_finite_number_is_an_error(value):
    """The writer must fail loudly rather than write an invalid file.

    ``json.dump`` writes ``NaN`` unless it is told not to, and the committed
    artefact was exactly that: a well-formed file to a lenient reader and a
    broken one to a strict reader.
    """
    with pytest.raises(ValueError):
        json.dump({"value": value}, sys.stdout, allow_nan=False)


def test_the_documented_command_writes_strict_json(tmp_path):
    """End to end: the command the README documents must produce a valid file.

    This runs the real script, into a scratch path so that a results artefact
    in the working tree is never touched.
    """
    output = tmp_path / "verified_tier2.json"

    subprocess.run(
        [
            sys.executable,
            os.path.join(REPO_ROOT, "src", "run_honest_evaluation.py"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )

    with open(output, encoding="utf-8") as handle:
        parsed = json.load(handle, parse_constant=_reject_constant)

    comparison = parsed["comparisons"]["BM25_vs_TFIDF_F1"]
    # The count of non-zero paired differences is determinate: both retrievers
    # select the same sentence on every question, so every difference is zero.
    assert comparison["n_nonzero_diffs"] == 0
    assert comparison["wilcoxon_p"] is None

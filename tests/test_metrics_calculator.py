"""Tests for how MetricsCalculator handles a metric that cannot be computed.

The defect these guard against
------------------------------
``MetricsCalculator.compute_bleu`` and ``MetricsCalculator.compute_meteor`` in
``src/hybrid_rag_cag_system.py`` caught a bare ``except:`` and returned
``0.0``. A missing NLTK corpus therefore made both metrics report ``0.000``
for every example -- a number indistinguishable from a genuine score of zero,
and one that nothing in the output flagged as "not computed".

The module under test imports torch, transformers, faiss and
sentence-transformers, so this file is skipped when they are not installed.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")
pytest.importorskip("nltk")
pytest.importorskip("rouge_score")

import generation_metrics  # noqa: E402
from generation_metrics import MetricUnavailable  # noqa: E402
from hybrid_rag_cag_system import MetricsCalculator  # noqa: E402


@pytest.fixture
def no_nltk_data(monkeypatch):
    """Simulate a machine where the NLTK data is missing and cannot be fetched."""
    import nltk

    def missing(*args, **kwargs):
        raise LookupError("NLTK data missing")

    monkeypatch.setattr(nltk, "word_tokenize", missing)
    monkeypatch.setattr(generation_metrics, "_nltk_data_present", lambda name: False)
    monkeypatch.setattr(generation_metrics, "_download_nltk_data", lambda name: False)


@pytest.fixture
def working_nltk(monkeypatch):
    """Simulate a machine where the tokenizer works but no corpus was installed.

    ``nltk.data.find`` is never consulted on this path, so the metrics
    compute without any NLTK data being present.
    """
    import nltk

    monkeypatch.setattr(nltk, "word_tokenize", lambda text: text.lower().split())


def test_compute_bleu_raises_instead_of_returning_zero(no_nltk_data):
    """The defect, at the level of the public helper."""
    with pytest.raises(MetricUnavailable):
        MetricsCalculator.compute_bleu("a b c", "a b c")


def test_compute_meteor_raises_instead_of_returning_zero(no_nltk_data):
    """The defect, at the level of the public helper."""
    with pytest.raises(MetricUnavailable):
        MetricsCalculator.compute_meteor("a b c", "a b c")


def test_unavailable_metric_is_reported_as_none_not_zero(no_nltk_data):
    """The aggregate must say "not computed", not "scored zero".

    This is the whole point: a zero is a measurement, and an unavailable
    metric is not one.
    """
    calculator = MetricsCalculator()

    metrics = calculator.calculate_all_metrics(["a b c"], ["a b c"])

    assert metrics["bleu4"] is None
    assert metrics["meteor"] is None
    assert "bleu4" in metrics["unavailable_metrics"]
    assert "meteor" in metrics["unavailable_metrics"]
    # The metrics that do work are still reported.
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["exact_match"] == pytest.approx(1.0)


def test_available_metric_is_reported_as_a_number(working_nltk):
    """The fix must not turn every metric into an error."""
    calculator = MetricsCalculator()

    metrics = calculator.calculate_all_metrics(["a b c"], ["a b c"])

    assert metrics["bleu4"] is not None
    assert metrics["bleu4"] > 0.0
    assert "unavailable_metrics" not in metrics


def test_mismatched_lengths_are_rejected():
    """Scoring the shorter list silently would report a mean over fewer
    examples than the caller supplied, with nothing in the output to say so.
    """
    calculator = MetricsCalculator()

    with pytest.raises(ValueError):
        calculator.calculate_all_metrics(["a b", "c d"], ["a b"])

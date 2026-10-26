"""Tests for the generation metrics.

The defect these guard against
------------------------------
``MetricsCalculator.compute_bleu`` and ``MetricsCalculator.compute_meteor`` in
``src/hybrid_rag_cag_system.py`` caught a bare ``except:`` and returned
``0.0``. A missing NLTK corpus therefore made BLEU-4 and METEOR report
``0.000`` for every example -- a number indistinguishable from a genuine
score of zero, and one that nothing in the output flagged as "not computed".

The fix moves both metrics into ``src/generation_metrics.py``, where an
unavailable metric raises ``MetricUnavailable`` instead of returning a number.
"""

import pytest

pytest.importorskip("nltk")

import generation_metrics  # noqa: E402
from generation_metrics import (  # noqa: E402
    METEOR_DATA,
    TOKENIZER_DATA,
    MetricUnavailable,
    compute_bleu,
    compute_meteor,
)


@pytest.fixture
def no_nltk_data(monkeypatch):
    """Simulate a machine where the NLTK data is missing and cannot be fetched.

    This is the state the old code silently turned into a score of zero.
    """
    import nltk

    def missing(*args, **kwargs):
        raise LookupError("NLTK data missing")

    monkeypatch.setattr(nltk, "word_tokenize", missing)
    monkeypatch.setattr(generation_metrics, "_nltk_data_present", lambda name: False)
    monkeypatch.setattr(generation_metrics, "_download_nltk_data", lambda name: False)


def test_bleu_is_unavailable_rather_than_zero(no_nltk_data):
    """The defect: a missing corpus used to produce 0.0.

    A score of zero and a metric that could not run are different facts, and
    only one of them is a measurement.
    """
    with pytest.raises(MetricUnavailable):
        compute_bleu("the cat sat on the mat", "the cat sat on the mat")


def test_meteor_is_unavailable_rather_than_zero(no_nltk_data):
    """The same must hold for METEOR."""
    with pytest.raises(MetricUnavailable):
        compute_meteor("the cat sat on the mat", "the cat sat on the mat")


def test_unavailable_message_names_the_data_and_how_to_install_it(no_nltk_data):
    """The error has to be actionable: the usual cause is an offline machine."""
    with pytest.raises(MetricUnavailable) as excinfo:
        compute_bleu("a b", "a b")

    message = str(excinfo.value)
    assert "nltk.downloader" in message
    for name in TOKENIZER_DATA:
        assert name in message


def test_bleu_computes_a_real_value_when_the_data_is_available():
    """The metric must still work; the fix is not "always raise"."""
    score = compute_bleu("the cat sat on the mat", "the cat sat on the mat",
                          tokenize=str.split)

    assert 0.0 <= score <= 1.0
    assert score > 0.0


def test_bleu_of_a_disjoint_answer_is_zero():
    """A genuine zero has to stay a zero, and be distinguishable from unavailable."""
    score = compute_bleu("alpha beta", "gamma delta", tokenize=str.split)

    assert score == 0.0


def test_meteor_asks_for_wordnet_when_it_is_missing(monkeypatch):
    """METEOR needs WordNet for its synonym lookup, and must say so."""
    import nltk

    def missing(*args, **kwargs):
        raise LookupError("NLTK data missing")

    monkeypatch.setattr(nltk, "word_tokenize", missing)
    monkeypatch.setattr(generation_metrics, "_nltk_data_present", lambda name: False)
    monkeypatch.setattr(generation_metrics, "_download_nltk_data", lambda name: False)

    with pytest.raises(MetricUnavailable) as excinfo:
        compute_meteor("a b", "a b")

    assert "METEOR" in str(excinfo.value)


def test_ensure_nltk_data_reports_the_alternatives():
    """The helper must name the data it was looking for."""
    import nltk

    original_present = generation_metrics._nltk_data_present
    original_download = generation_metrics._download_nltk_data

    try:
        generation_metrics._nltk_data_present = lambda name: False
        generation_metrics._download_nltk_data = lambda name: False

        with pytest.raises(MetricUnavailable) as excinfo:
            generation_metrics.ensure_nltk_data(METEOR_DATA, "METEOR")
    finally:
        generation_metrics._nltk_data_present = original_present
        generation_metrics._download_nltk_data = original_download

    message = str(excinfo.value)
    for name in METEOR_DATA:
        assert name in message
    assert "METEOR" in message
    assert nltk is not None

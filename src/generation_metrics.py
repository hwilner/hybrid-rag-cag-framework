"""BLEU and METEOR, with "unavailable" kept distinct from "zero".

Why this module exists
----------------------
``MetricsCalculator.compute_bleu`` and ``MetricsCalculator.compute_meteor`` in
``src/hybrid_rag_cag_system.py`` used to look like this::

    try:
        ...
        return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
    except:
        return 0.0

The bare ``except:`` swallowed every failure, including the one that actually
happens in practice: NLTK's tokenizer data not being installed. The result was
that BLEU-4 and METEOR reported ``0.000`` for every example -- a number that
is indistinguishable from a genuine score of zero, and indistinguishable from
a system that is simply bad. Nothing in the output said the metric had not
been computed at all.

The download that was supposed to prevent this made it worse. The module did::

    nltk.download('punkt', quiet=True)

at import time. ``quiet=True`` hides the failure when there is no network,
and since NLTK 3.8.2 ``nltk.word_tokenize`` needs the ``punkt_tab`` dataset
rather than ``punkt`` -- so on a current NLTK the download did not even supply
what the tokenizer asked for. The import-time download also made importing the
module a network operation, which is not something a library should do.

This module computes the two metrics, and raises :class:`MetricUnavailable`
when the NLTK data they need is not installed and cannot be fetched. An
unavailable metric is then reported as unavailable, which is a different thing
from a score of zero, and the difference survives into the output.

Dependencies
------------
The standard library plus NLTK, which is imported lazily so that the module
can be imported -- and its failure modes tested -- without the NLTK data
being installed.
"""

from __future__ import annotations

from typing import List, Sequence

#: Where each NLTK dataset lives, for ``nltk.data.find``.
_FIND_PATHS = {
    "punkt_tab": "tokenizers/punkt_tab",
    "punkt": "tokenizers/punkt",
    "wordnet": "corpora/wordnet",
}

#: The tokenizer data, newest name first. NLTK 3.8.2 moved the tokenizer to
#: ``punkt_tab``; older releases use ``punkt``. Either one is enough.
TOKENIZER_DATA = ("punkt_tab", "punkt")

#: The data METEOR needs for its synonym lookup.
METEOR_DATA = ("wordnet",)


class MetricUnavailable(RuntimeError):
    """Raised when a metric cannot be computed.

    This is deliberately not the same thing as a score of zero. A caller that
    catches it can report "not computed" instead of a number that looks like a
    measurement.
    """


def _nltk_data_present(name: str) -> bool:
    """Return whether the NLTK dataset ``name`` is installed."""
    import nltk

    try:
        nltk.data.find(_FIND_PATHS[name])
    except LookupError:
        return False
    return True


def _download_nltk_data(name: str) -> bool:
    """Try to download the NLTK dataset ``name``. Return whether it worked."""
    import nltk

    try:
        return bool(nltk.download(name, quiet=True))
    except Exception:  # noqa: BLE001 - a failed download is reported, not raised.
        return False


def ensure_nltk_data(alternatives: Sequence[str], metric: str) -> None:
    """Make sure at least one of the alternative NLTK datasets is installed.

    Args:
        alternatives: dataset names, in order of preference. The first one
            that is present is used.
        metric: the name of the metric that needs the data, used in the error
            message.

    Raises:
        MetricUnavailable: if none of the alternatives is installed and none
            could be downloaded. The message names the command that installs
            them, because the usual cause is an offline machine.
    """
    for name in alternatives:
        if _nltk_data_present(name):
            return

    for name in alternatives:
        if _download_nltk_data(name) and _nltk_data_present(name):
            return

    raise MetricUnavailable(
        f"{metric} is unavailable: the NLTK data "
        f"{'/'.join(alternatives)} is not installed and could not be "
        f"downloaded. Install it with "
        f"`python -m nltk.downloader {' '.join(alternatives)}` and re-run."
    )


def _tokenize(text: str, metric: str, tokenize=None) -> List[str]:
    """Tokenize ``text``, fetching the NLTK tokenizer data if it is missing.

    Args:
        text: the string to tokenize.
        metric: the name of the metric that needs the tokens, used in the error
            message.
        tokenize: an optional tokenizer. Supplying one bypasses NLTK
            entirely, which is how the metric can be exercised without the
            NLTK data being installed.

    Raises:
        MetricUnavailable: if the tokenizer data cannot be obtained.
    """
    if tokenize is not None:
        return list(tokenize(text))

    import nltk

    try:
        return nltk.word_tokenize(text.lower())
    except LookupError:
        # The tokenizer data is missing. Fetch it once and try again.
        pass

    ensure_nltk_data(TOKENIZER_DATA, metric)
    try:
        return nltk.word_tokenize(text.lower())
    except LookupError as exc:
        raise MetricUnavailable(
            f"{metric} is unavailable: NLTK's tokenizer still cannot load "
            "after its data was fetched."
        ) from exc


def compute_bleu(prediction: str, reference: str, tokenize=None) -> float:
    """Return the sentence-level BLEU-4 score of ``prediction``.

    Args:
        prediction: the candidate answer.
        reference: the gold answer.
        tokenize: an optional tokenizer, used instead of NLTK's. It exists so
            that the metric can be tested without the NLTK data installed.

    Returns:
        The smoothed BLEU-4 score, in [0, 1].

    Raises:
        MetricUnavailable: if NLTK's tokenizer data is not available.
    """
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    pred_tokens = _tokenize(prediction, "BLEU-4", tokenize)
    ref_tokens = _tokenize(reference, "BLEU-4", tokenize)
    smoothing = SmoothingFunction().method1
    return float(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing))


def compute_meteor(prediction: str, reference: str, tokenize=None) -> float:
    """Return the METEOR score of ``prediction``.

    Args:
        prediction: the candidate answer.
        reference: the gold answer.
        tokenize: an optional tokenizer, used instead of NLTK's. It exists so
            that the metric can be tested without the NLTK data installed.

    Returns:
        The METEOR score, in [0, 1].

    Raises:
        MetricUnavailable: if NLTK's tokenizer or WordNet data is not
            available.
    """
    from nltk.translate.meteor_score import meteor_score

    pred_tokens = _tokenize(prediction, "METEOR", tokenize)
    ref_tokens = _tokenize(reference, "METEOR", tokenize)
    try:
        return float(meteor_score([ref_tokens], pred_tokens))
    except LookupError:
        # METEOR needs WordNet for its synonym lookup.
        pass

    ensure_nltk_data(METEOR_DATA, "METEOR")
    try:
        return float(meteor_score([ref_tokens], pred_tokens))
    except LookupError as exc:
        raise MetricUnavailable(
            "METEOR is unavailable: NLTK's meteor scorer still cannot load "
            "after its data was fetched."
        ) from exc

"""Honest, real baselines for the Tier-2 evaluation.

Why this module exists
----------------------
`option3_full_scale_evaluation.py` ships six "systems" (RAG, CAG, FiD, T5-FiD,
DPR+FiD, Hybrid). None of them are machine-learning systems. They are ``if``-chains
over a hardcoded dictionary, e.g.::

    if 'capital of france' in q_lower: return 'Paris'

The reported Tier-2 ranking ("Hybrid 0.276 vs Advanced RAG 0.369") is therefore a
measurement of dictionary coverage, not of retrieval quality. The giveaway is CAG's
average response time of 8.9e-06 s -- BART-large cannot emit a token in under a
millisecond, so no model inference ever took place.

This module replaces those with real, reproducible baselines that use only
numpy/scikit-learn:

* ``TFIDFRetriever``      - TF-IDF cosine retrieval (a real sparse lexical model)
* ``BM25Retriever``       - Okapi BM25 ranking (a real probabilistic model)
* ``LexicalExtractiveQA`` - BM25 sentence selection, the honest extractive control
* ``OracleExtractiveQA``  - upper bound if the gold sentence were perfectly selected

None of these use torch, a GPU, or a network call, so every number below is
reproducible on any machine. They are *baselines*, not FiD/T5-FiD/DPR+FiD, and are
labelled as such in the output.

The gold Hybrid (dense bi-encoder + BART) is evaluated separately by
``run_honest_evaluation.py`` if the model dependencies are available; when they are
not, the comparison is reported as not-run rather than filled in.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:  # sklearn is the only hard requirement
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _HAS_SKLEARN = True
except Exception:  # degraded pure-numpy path; covered by tools/check_integrity.py
    _HAS_SKLEARN = False

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "being", "what", "which", "who", "whom",
    "how", "why", "when", "where", "does", "do", "did", "that", "this", "these",
    "those", "it", "its", "as", "by", "with", "from", "about", "into", "than",
    "then", "there", "their", "they", "have", "has", "had", "but", "not", "can",
    "could", "would", "should", "will", "shall", "may", "might", "must",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, drop stopwords."""
    text = text.lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    return [t for t in _TOKEN_RE.findall(text) if t not in STOPWORDS and len(t) > 1]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def normalize_answer(s: str) -> str:
    """SQuAD-style normalization used for exact match."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    """SQuAD token-level F1."""
    p_toks, g_toks = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(p_toks)
    recall = same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def contains_answer(pred: str, gold: str) -> float:
    """Answer-containment: does the prediction contain the gold answer string?

    Token-F1 is the metric the original repo used, but it structurally punishes
    extractive systems: the Tier-2 gold answers are short ("Paris", "Mount
    Everest") while any extractive prediction is a whole sentence ("Paris is the
    capital and most populous city of France..."). Precision is capped near
    1/len(sentence), so a *correct* extraction scores ~0.2. Containment is the
    standard mitigation and is reported alongside F1 so the two can be read
    against each other.
    """
    p, g = normalize_answer(pred), normalize_answer(gold)
    if not g:
        return 0.0
    if p == g:
        return 1.0
    return float(g in p)


# --------------------------------------------------------------------------- #
# Retrievers
# --------------------------------------------------------------------------- #

class TFIDFRetriever:
    """TF-IDF (sublinear tf, L2 norm) cosine retrieval. A real lexical model."""

    name = "TF-IDF + Extractive"

    def __init__(self, corpus: Sequence[str]):
        self.corpus = list(corpus)
        if _HAS_SKLEARN:
            self.vec = TfidfVectorizer(
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",
                sublinear_tf=True,
                stop_words="english",
            )
            self.matrix = self.vec.fit_transform(self.corpus)
        else:  # minimal fallback so the module still runs
            self._fit_fallback()

    def _fit_fallback(self) -> None:
        self.vocab: Dict[str, int] = {}
        docs = []
        for doc in self.corpus:
            counts = Counter(tokenize(doc))
            for tok in counts:
                self.vocab.setdefault(tok, len(self.vocab))
            docs.append(counts)
        self.docs = docs
        n = max(len(docs), 1)
        self.idf = {t: math.log(1 + n / (1 + sum(1 for d in docs if t in d)))
                    for t in self.vocab}

    def _vecs(self, texts: Sequence[str]):
        if _HAS_SKLEARN:
            return self.vec.transform(list(texts))
        out = []
        for t in texts:
            counts = Counter(tokenize(t))
            row = np.zeros(len(self.vocab), dtype=np.float32)
            for tok, c in counts.items():
                idx = self.vocab.get(tok)
                if idx is not None:
                    row[idx] = (1 + math.log(c)) * self.idf[tok]
            norm = np.linalg.norm(row) or 1.0
            out.append(row / norm)
        return np.array(out)

    def top_k(self, question: str, k: int = 5) -> List[Tuple[int, float]]:
        qv = self._vecs([question])
        if _HAS_SKLEARN:
            sims = cosine_similarity(qv, self.matrix)[0]
        else:
            # Shapes here are qv=(1, V) and corpus vectors=(N, V), so the
            # matmul is already a 1-D score per document. The trailing `[0]`
            # that used to close this expression collapsed that length-N
            # vector to a scalar, so the `sims[i]` below raised
            # "IndexError: invalid index to scalar variable" on any machine
            # without scikit-learn -- the very case this fallback exists for.
            sims = np.asarray(qv)[0] @ np.asarray(self._vecs(self.corpus)).T
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]


class BM25Retriever:
    """Okapi BM25 (k1=1.5, b=0.75). A real probabilistic ranking model."""

    name = "BM25 + Extractive"

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.corpus = list(corpus)
        self.k1, self.b = k1, b
        self.doc_toks = [tokenize(d) for d in self.corpus]
        self.doc_len = np.array([len(t) for t in self.doc_toks], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 1.0
        self.N = len(self.doc_toks)
        df: Counter = Counter()
        for toks in self.doc_toks:
            df.update(set(toks))
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }
        self.tf = [Counter(t) for t in self.doc_toks]

    def scores(self, question: str) -> np.ndarray:
        q_toks = tokenize(question)
        out = np.zeros(self.N, dtype=np.float32)
        for i, tf in enumerate(self.tf):
            dl = self.doc_len[i] or 1.0
            s = 0.0
            for tok in q_toks:
                if tok not in tf:
                    continue
                f = tf[tok]
                s += self.idf.get(tok, 0.0) * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            out[i] = s
        return out

    def top_k(self, question: str, k: int = 5) -> List[Tuple[int, float]]:
        s = self.scores(question)
        order = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in order]


# --------------------------------------------------------------------------- #
# Answering
# --------------------------------------------------------------------------- #

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or [text.strip() or ""]


class LexicalExtractiveQA:
    """Answer a question by extracting the best sentence from top-ranked docs.

    The selection score is BM25(question, sentence), optionally reweighted by a
    coverage factor for how much of the question's content the sentence covers.
    This is the honest extractive control the toy baselines should have had.
    """

    name = "BM25 + Extractive"

    def __init__(self, corpus: Sequence[str], retriever=None):
        self.corpus = list(corpus)
        self.retriever = retriever if retriever is not None else BM25Retriever(corpus)
        self.bm25 = self.retriever
        # pre-split sentences per doc
        self.doc_sents = [_split_sentences(d) for d in self.corpus]

    @classmethod
    def from_retriever(cls, corpus, retriever):
        """Build an extractive system over an arbitrary retriever (TF-IDF, BM25, ...)."""
        obj = cls.__new__(cls)
        obj.corpus = list(corpus)
        obj.retriever = retriever
        obj.bm25 = retriever
        obj.doc_sents = [_split_sentences(d) for d in obj.corpus]
        return obj

    def answer(self, question: str, top_k: int = 5) -> str:
        ranked = self.retriever.top_k(question, k=top_k)
        order = [di for di, _ in ranked]
        if not order:
            return ""
        q_content = set(tokenize(question))
        best, best_score = "", -1.0
        for di in order:
            for sent in self.doc_sents[int(di)]:
                s_toks = tokenize(sent)
                if not s_toks:
                    continue
                s_set = set(s_toks)
                overlap = len(q_content & s_set)
                if overlap == 0:
                    continue
                # coverage-weighted scoring: prefer sentences that hit more of
                # the question's content terms, with a mild brevity preference
                freq = sum(1 for t in q_content if t in s_set)
                score = (
                    freq * 2.0
                    + overlap
                    + 0.5 * overlap / max(len(s_set), 1)
                )
                score -= 0.02 * len(s_toks)
                if score > best_score:
                    best, best_score = sent, score
        if best:
            return best
        return self.corpus[int(order[0])][:200]


class OracleExtractiveQA(LexicalExtractiveQA):
    """Upper bound: pick the gold sentence if it exists anywhere in the corpus.

    Not a baseline -- a ceiling. Reported separately so the gap between it and the
    real extractive baseline is visible (it isolates retrieval failure from
    answer-formatting failure).
    """

    name = "Oracle Extractive (ceiling, not a baseline)"

    def answer(self, question: str, gold: str = "", top_k: int = 5) -> str:
        g = normalize_answer(gold)
        if not g:
            return super().answer(question, top_k)
        for di in range(len(self.corpus)):
            for sent in _split_sentences(self.corpus[di]):
                if g and g in normalize_answer(sent):
                    return sent
        return super().answer(question, top_k)

"""Benchmark-validity gate: is a corpus/question benchmark answerable at all?

Why this module exists
----------------------
The failure diagnosis (``results/diagnosis.json``, ``results/RESULTS.md``
section 7) measured that only 21 of the 55 Tier-2 gold answers appear
anywhere in the 100-document corpus. The other 34 questions are
unrecoverable by *any* retriever or reader: their gold answers are composed
sentences that exist in no document. No ranking claim measured on such a
benchmark is valid, which is why issue #5 (full benchmark repair) requires a
corpus rebuild before the seeded ablation re-run.

A rebuild is only worth the GPU hours if the rebuilt benchmark is known to be
answerable *before* the models are run. This module is that gate. It is a
pure function of the corpus and question set -- no models, no network, no
randomness -- so it runs in seconds on CPU and its verdict is deterministic.

Answerability is defined exactly as the diagnosis defines it: the normalised
gold answer string must appear as a substring of the normalised corpus. The
normalisation here is a copy of ``src/diagnose_failures.py``'s
``normalize``/``contains`` (lowercase, punctuation to spaces, articles
dropped, whitespace collapsed); ``tests/test_benchmark_validity.py`` pins
the two implementations to agree, so they cannot silently drift apart.

The module validates two kinds of input:

* the committed Tier-2 benchmark of a checkout (via
  ``src/dataset_loader.py``), for an honest report of the status quo;
* a *candidate* rebuilt benchmark as a JSON file of the form
  ``{"corpus": [str, ...], "questions": [{...}, ...]}``, which additionally
  gets schema checks (required fields, unique question texts, non-empty
  answers and documents).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, List, Sequence

QUESTION_REQUIRED_FIELDS = ("question", "answer", "type", "difficulty")


def normalize(s: str) -> str:
    """Normalise text for answer containment, mirroring the diagnosis.

    This is deliberately identical to ``diagnose_failures.normalize``;
    ``tests/test_benchmark_validity.py`` asserts the equivalence.
    """
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains(pred: str, gold: str) -> bool:
    """True iff the normalised ``gold`` appears in normalised ``pred``."""
    p, g = normalize(pred), normalize(gold)
    return bool(g) and g in p


def validate_schema(corpus: Sequence[Any], questions: Sequence[Any]) -> List[str]:
    """Check the structural requirements of a benchmark. Returns problems.

    An empty list means the benchmark is well-formed. Every problem is
    reported (not just the first) so a candidate rebuild can be fixed in one
    pass. These checks catch the failure modes that would corrupt a
    measurement silently: duplicate question texts break per-question joins,
    and an empty gold answer would be scored "answerable" by a naive
    substring test, so it is rejected outright.
    """
    problems: List[str] = []
    if not corpus:
        problems.append("corpus is empty")
    for i, doc in enumerate(corpus):
        if not isinstance(doc, str) or not doc.strip():
            problems.append(f"corpus[{i}] is not a non-empty string")
    if not questions:
        problems.append("question set is empty")
    seen = set()
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            problems.append(f"questions[{i}] is not an object")
            continue
        for field in QUESTION_REQUIRED_FIELDS:
            if field not in q or not str(q[field]).strip():
                problems.append(f"questions[{i}] is missing a non-empty '{field}'")
        text = str(q.get("question", ""))
        if text in seen:
            problems.append(f"questions[{i}] duplicates question text: {text!r}")
        seen.add(text)
        answer = q.get("answer")
        if isinstance(answer, str) and not normalize(answer):
            problems.append(
                f"questions[{i}] gold answer normalises to empty: {answer!r}"
            )
    return problems


def answerability_report(corpus: Sequence[str], questions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Measure which questions are answerable from the corpus.

    Returns a JSON-serialisable report: overall counts, per-difficulty and
    per-type breakdowns (each as ``{"answerable": a, "total": t}``), and a
    per-question list. No floats that could be non-finite: fractions are
    computed by the caller-facing helpers only when denominators are known
    non-zero, and the gate verdict uses integer comparison internally.
    """
    joined = normalize(" ".join(corpus))
    per_question: List[Dict[str, Any]] = []
    by_difficulty: Dict[str, Counter] = {}
    by_type: Dict[str, Counter] = {}
    n_answerable = 0
    for q in questions:
        ok = bool(normalize(q["answer"])) and normalize(q["answer"]) in joined
        n_answerable += int(ok)
        for key, table in (("difficulty", by_difficulty), ("type", by_type)):
            bucket = str(q.get(key, "?"))
            table.setdefault(bucket, Counter())
            table[bucket]["total"] += 1
            table[bucket]["answerable"] += int(ok)
        per_question.append(
            {
                "question": q["question"],
                "answer": q["answer"],
                "type": q.get("type", "?"),
                "difficulty": q.get("difficulty", "?"),
                "answerable": ok,
            }
        )

    def _fractions(table: Dict[str, Counter]) -> Dict[str, Dict[str, Any]]:
        return {
            bucket: {
                "answerable": counts["answerable"],
                "total": counts["total"],
                "fraction": counts["answerable"] / counts["total"],
            }
            for bucket, counts in sorted(table.items())
        }

    n = len(questions)
    return {
        "n_questions": n,
        "n_corpus_docs": len(corpus),
        "n_answerable": n_answerable,
        "answerable_fraction": (n_answerable / n) if n else None,
        "by_difficulty": _fractions(by_difficulty),
        "by_type": _fractions(by_type),
        "per_question": per_question,
    }


def gate_verdict(
    report: Dict[str, Any],
    min_answerable: float = 1.0,
    require_all_types: bool = True,
) -> Dict[str, Any]:
    """Decide whether a benchmark passes the repair gate for issue #5.

    A benchmark passes iff its answerable fraction meets ``min_answerable``
    and -- when ``require_all_types`` is true -- every question type present
    in the set is answerable for every question of that type. The defaults
    are the strict reading of issue #5 (\"all ... question types are
    answerable\"); a declared subset of types can be accommodated by passing
    ``require_all_types=False`` and an explicit ``min_answerable``.

    The comparison is done on integer counts (``n_answerable >=
    ceil(min_answerable * n)``) so a borderline float cannot flip the verdict.
    """
    import math

    n = report["n_questions"]
    failures: List[str] = []
    if n == 0:
        failures.append("no questions")
    else:
        needed = math.ceil(min_answerable * n)
        if report["n_answerable"] < needed:
            failures.append(
                f"answerable {report['n_answerable']}/{n} "
                f"({report['answerable_fraction']:.4f}) below required "
                f"{needed}/{n} (min_answerable={min_answerable})"
            )
    if require_all_types:
        for qtype, counts in report["by_type"].items():
            if counts["answerable"] < counts["total"]:
                failures.append(
                    f"type {qtype!r} not fully answerable: "
                    f"{counts['answerable']}/{counts['total']}"
                )
    return {
        "passed": not failures,
        "min_answerable": min_answerable,
        "require_all_types": require_all_types,
        "failures": failures,
    }

"""The decoding configuration a panel of candidate answers is produced with.

Why this module exists
----------------------
``src/real_ablation.py`` used to record the decoding configuration of every
ablation arm as a hand-written string::

    "decoding": "greedy, num_beams=1, do_sample=False"

That string is a claim about the code, and it was wrong for two of the three
arms. ``HybridGenerator.generate_candidates`` produces a panel of
``num_candidates`` answers in up to two passes: a beam-search pass, and -- when
the beam pass did not already return enough candidates -- a nucleus-sampling
pass with ``do_sample=True``, ``top_p=0.9`` and ``temperature=0.7``. The
ablation harness asks for five candidates in the CAG and Hybrid arms and one
in the RAG arm, so the sampling pass runs for CAG and Hybrid and not for RAG.
Nothing seeded the sampling, so those two arms were not reproducible.

The committed artefacts show the consequence. Across the four branch runs, the
RAG and CAG predictions are bit-identical on all 55 questions, while the Hybrid
predictions differ on 4-5 of them. Under the recorded "greedy" description all
three arms would have been bit-identical.

This module makes the decoding configuration a single inspectable object. The
generator *executes* it and the harness *records* it, so the description in a
results file cannot drift away from the code that produced the numbers.

Dependencies
------------
The standard library only. It has to be importable without torch, because the
integrity checks and the unit tests read it in an environment that has no
model dependencies installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DecodingPass:
    """One call to ``transformers``' ``model.generate(...)``.

    Attributes:
        num_beams: the beam count passed to ``generate``.
        num_return_sequences: how many sequences the call returns.
        do_sample: whether sampling is used.
        early_stopping: beam-search early stopping, or ``None`` to leave it at
            the library default.
        top_p: nucleus-sampling mass, or ``None`` when not sampling.
        temperature: sampling temperature, or ``None`` when not sampling.
    """

    num_beams: int
    num_return_sequences: int
    do_sample: bool
    early_stopping: Optional[bool] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None

    def generate_kwargs(self) -> Dict[str, object]:
        """Return the keyword arguments for ``model.generate(...)``.

        Only the arguments this pass actually uses are included, so that the
        call is identical to the one it replaces: ``early_stopping`` is
        omitted rather than set to ``False``, and the sampling parameters are
        omitted entirely for a deterministic pass.
        """
        kwargs: Dict[str, object] = {
            "num_beams": self.num_beams,
            "num_return_sequences": self.num_return_sequences,
            "do_sample": self.do_sample,
        }
        if self.early_stopping is not None:
            kwargs["early_stopping"] = self.early_stopping
        if self.do_sample:
            kwargs["top_p"] = self.top_p
            kwargs["temperature"] = self.temperature
        return kwargs

    def describe(self) -> str:
        """Return a human-readable description of this pass."""
        parts = [
            f"num_beams={self.num_beams}",
            f"num_return_sequences={self.num_return_sequences}",
            f"do_sample={self.do_sample}",
        ]
        if self.early_stopping is not None:
            parts.append(f"early_stopping={self.early_stopping}")
        if self.do_sample:
            parts.append(f"top_p={self.top_p}")
            parts.append(f"temperature={self.temperature}")
        label = "nucleus sampling" if self.do_sample else "beam search"
        return f"{label} ({', '.join(parts)})"


@dataclass(frozen=True)
class DecodingPolicy:
    """How a panel of ``num_candidates`` answers is decoded.

    Attributes:
        num_candidates: the size of the candidate panel.
        top_p: nucleus-sampling mass used by the sampling pass.
        temperature: sampling temperature used by the sampling pass.
    """

    num_candidates: int
    top_p: float = 0.9
    temperature: float = 0.7

    def __post_init__(self) -> None:
        if self.num_candidates < 1:
            raise ValueError(
                f"num_candidates must be at least 1, got {self.num_candidates}"
            )

    def passes(self) -> List[DecodingPass]:
        """Return the decoding passes, in the order they are executed.

        The first pass is a beam search that returns up to three sequences. A
        second, sampling pass runs only when the first did not already fill the
        panel -- which is the case for the five-candidate panels the ablation
        harness uses, and not for the single-candidate RAG panel.
        """
        beam = DecodingPass(
            num_beams=self.num_candidates,
            num_return_sequences=min(self.num_candidates, 3),
            do_sample=False,
            early_stopping=True,
        )
        if beam.num_return_sequences >= self.num_candidates:
            return [beam]

        sample = DecodingPass(
            num_beams=1,
            num_return_sequences=self.num_candidates - beam.num_return_sequences,
            do_sample=True,
            top_p=self.top_p,
            temperature=self.temperature,
        )
        return [beam, sample]

    def is_deterministic(self) -> bool:
        """Return whether every pass is deterministic.

        A policy that samples is not reproducible on its own: the sampled
        candidates depend on the state of the random number generator. A
        caller can make it reproducible by seeding before the call.
        """
        return all(not decoding_pass.do_sample for decoding_pass in self.passes())

    def describe(self) -> str:
        """Return a human-readable description of the whole policy."""
        described = " + ".join(
            decoding_pass.describe() for decoding_pass in self.passes()
        )
        if self.is_deterministic():
            return f"{described} (deterministic)"
        return f"{described} (stochastic: the sampling pass needs a seed)"

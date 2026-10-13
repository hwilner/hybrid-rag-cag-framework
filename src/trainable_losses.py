"""Differentiable training for the Hybrid RAG-CAG generator.

Why this module exists
----------------------
`HybridRAGCAG.compute_generation_loss` returned::

    return torch.tensor(1.0 - avg_f1, requires_grad=True)

where `avg_f1` is a ``numpy`` float over decoded strings. That is a **constant
with a decorative grad flag**: it has no ``grad_fn`` and no computational graph.

The precise failure mode is worth stating, because it is not the obvious one:
``backward()`` on this tensor does **not** raise. Being a leaf, it is a
perfectly valid scalar as far as autograd is concerned, so ``backward()``
returns normally, ``param.grad`` stays ``None``, and the optimizer step does
nothing at all. Training would appear to run for the configured 5 epochs while
every weight remained at initialisation. The defect is therefore *silent* -- no
exception, no NaN, no warning -- which is how it survived in the repository.

The contrastive and diversity terms have the same defect in a subtler form: they
wrap `SentenceTransformer.encode()` output in ``torch.tensor(...)``, which
detaches it. The retriever encoders are frozen anyway, so those terms could not
reach the generator even if they were wired correctly.

This module replaces that with a loss that actually trains:

* **Generation loss** -- real teacher-forced cross-entropy from BART's own
  ``forward(labels=...)``. This is the term that produces gradients for the
  generator parameters.
* **Contrastive loss** -- InfoNCE over *trainable* candidate representations,
  obtained by scoring candidates with the generator's own logits rather than a
  frozen sentence encoder, so gradients flow.
* **Sequence-level F1** -- retained as a *metric*, explicitly not as a loss.

The frozen-encoder limitation is documented rather than papered over: the
sentence-transformer bi-encoder is not trained by this module. Training a real
DPR-style retriever requires in-batch negatives over a trainable encoder and is
out of scope here; see ``docs/METHODS.md`` for what that would require.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


def sequence_f1(prediction: str, gold: str) -> float:
    """Token-level F1. Used as a *metric* only -- see module docstring."""
    p = set(prediction.lower().split())
    g = set(gold.lower().split())
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


class DifferentiableLoss:
    """Loss functions that produce real gradients for the BART generator.

    Parameters
    ----------
    model:
        A ``BartForConditionalGeneration`` (or any seq2seq LM exposing
        ``(input_ids, attention_mask, labels) -> .loss``).
    tokenizer:
        Matching tokenizer, used to encode source/target pairs.
    temperature:
        Softmax temperature for the contrastive term.
    """

    def __init__(self, model, tokenizer, temperature: float = 0.07):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature

    # ------------------------------------------------------------------ #
    # Generation: the only term that trains the generator
    # ------------------------------------------------------------------ #
    def generation_loss(
        self,
        sources: Sequence[str],
        targets: Sequence[str],
        max_source_length: int = 1024,
        max_target_length: int = 128,
    ) -> torch.Tensor:
        """Teacher-forced cross-entropy. Differentiable w.r.t. model weights."""
        if not sources:
            raise ValueError("generation_loss requires at least one source")

        enc = self.tokenizer(
            list(sources),
            max_length=max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            list(targets),
            max_length=max_target_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).input_ids

        # Mask pad tokens in labels so they are not trained as a target.
        labels = labels.masked_fill(
            self.tokenizer.pad_token_id is not None
            and labels == self.tokenizer.pad_token_id,
            -100,
        )

        out = self.model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            labels=labels,
        )
        return out.loss

    # ------------------------------------------------------------------ #
    # Contrastive: score candidates with the generator, not a frozen encoder
    # ------------------------------------------------------------------ #
    def candidate_logprob(
        self, source: str, candidate: str,
        max_source_length: int = 1024, max_target_length: int = 128,
    ) -> torch.Tensor:
        """Mean token log-probability of ``candidate`` given ``source``.

        Differentiable: the returned scalar carries a grad_fn into the model.
        """
        enc = self.tokenizer(
            [source], max_length=max_source_length,
            truncation=True, return_tensors="pt",
        )
        lab = self.tokenizer(
            [candidate], max_length=max_target_length,
            truncation=True, return_tensors="pt",
        ).input_ids
        if lab.shape[1] == 0:
            return self.model(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=lab
            ).loss * 0.0

        out = self.model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            labels=lab,
        )
        # BART returns summed CE over unmasked tokens; normalise by token count.
        n_tok = int((lab != self.tokenizer.pad_token_id).sum().item()) or 1
        return out.loss / n_tok

    def contrastive_loss(
        self,
        source: str,
        candidates: Sequence[str],
        gold: str,
        min_f1: float = 0.1,
    ) -> Optional[torch.Tensor]:
        """InfoNCE over candidate log-probs, positive = best-F1 candidate.

        Returns ``None`` when no candidate is good enough to act as a positive;
        the caller should then skip the term for this example.
        """
        cands = [c for c in candidates if c and c.strip()]
        if len(cands) < 2:
            return None

        f1s = [sequence_f1(c, gold) for c in cands]
        best = max(range(len(cands)), key=lambda i: f1s[i])
        if f1s[best] < min_f1:
            return None

        logits = torch.stack([
            self.candidate_logprob(source, c) for c in cands
        ]) / self.temperature

        return F.cross_entropy(
            logits.unsqueeze(0), torch.tensor([best], device=logits.device)
        )

    # ------------------------------------------------------------------ #
    # Composite
    # ------------------------------------------------------------------ #
    def combined(
        self,
        sources: Sequence[str],
        targets: Sequence[str],
        candidates_batch: Optional[Sequence[Sequence[str]]] = None,
        lambda_gen: float = 1.0,
        lambda_contrastive: float = 0.5,
        temperature: Optional[float] = None,
        **kw,
    ) -> Dict[str, torch.Tensor]:
        """L = lambda_gen * L_gen + lambda_contrastive * L_contrastive.

        ``L_gen`` is always present and always differentiable. The contrastive
        term is included only for examples where a valid positive exists, and is
        reported as ``0.0`` (a real zero, not a missing key) otherwise.
        """
        if temperature is not None:
            self.temperature = temperature

        gen = self.generation_loss(sources, targets, **kw)

        total = lambda_gen * gen
        out: Dict[str, torch.Tensor] = {"generation_loss": gen}

        if candidates_batch and lambda_contrastive:
            terms = []
            for src, cands, tgt in zip(sources, candidates_batch, targets):
                c = self.contrastive_loss(src, cands, tgt)
                if c is not None:
                    terms.append(c)
            if terms:
                con = torch.stack(terms).mean()
                out["contrastive_loss"] = con
                total = total + lambda_contrastive * con
            else:
                out["contrastive_loss"] = torch.zeros(
                    (), device=gen.device, dtype=gen.dtype
                )

        out["total_loss"] = total
        return out


def assert_differentiable(loss: torch.Tensor) -> None:
    """Raise if ``loss`` cannot be backpropagated.

    Used by the test suite to keep this defect from coming back.
    """
    if not loss.requires_grad:
        raise AssertionError(
            "loss.requires_grad is False -- the loss is detached from the graph"
        )
    if loss.grad_fn is None:
        raise AssertionError(
            "loss.grad_fn is None -- requires_grad was set on a constant tensor"
        )

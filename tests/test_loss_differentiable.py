"""Tests proving the training loss is actually differentiable.

These exist because the original loss shipped a silent defect: it returned
``torch.tensor(1.0 - np.mean(f1), requires_grad=True)`` -- a numpy constant with
a decorative grad flag. Nothing failed loudly; ``loss.backward()`` simply had
nothing to differentiate.

A tiny randomly-initialised BART is used so the suite runs in seconds on CPU
with no checkpoint download.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trainable_losses import (  # noqa: E402
    DifferentiableLoss,
    assert_differentiable,
    sequence_f1,
)


@pytest.fixture(scope="module")
def tiny():
    """A tiny randomly-initialised BART + its tokenizer. No checkpoint download."""
    from transformers import BartConfig, BartForConditionalGeneration, BartTokenizerFast

    try:
        tok = BartTokenizerFast.from_pretrained("facebook/bart-base")
    except Exception:
        pytest.skip("tokenizer unavailable (no network); loss logic untested here")

    # vocab_size must be >= the tokenizer's vocabulary, or token ids index
    # outside the embedding matrix.
    torch.manual_seed(0)
    cfg = BartConfig(
        vocab_size=max(1000, len(tok)), d_model=32, encoder_layers=2, decoder_layers=2,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=64, decoder_ffn_dim=64, max_position_embeddings=128,
        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id, decoder_start_token_id=tok.eos_token_id,
    )
    model = BartForConditionalGeneration(cfg)
    return model, tok, DifferentiableLoss(model, tok)


def test_generation_loss_is_differentiable(tiny):
    model, _, loss_fn = tiny
    loss = loss_fn.generation_loss(
        ["Paris is the capital of France"], ["Paris"]
    )
    assert_differentiable(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_backward_populates_generator_gradients(tiny):
    """The defect this test guards: a constant produces no gradients at all."""
    model, _, loss_fn = tiny
    model.zero_grad()
    loss = loss_fn.generation_loss(
        ["The Nile is a river in Africa", "Python was created by Guido"],
        ["The Nile", "Guido van Rossum"],
    )
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "no parameter received a gradient"
    total = sum(float(g.abs().sum()) for g in grads if g is not None)
    assert total > 0, "gradients are all exactly zero"


def test_contrastive_term_carries_a_graph(tiny):
    _, _, loss_fn = tiny
    loss = loss_fn.contrastive_loss(
        "What is the capital of France?",
        ["Paris is the capital of France", "The weather is nice today"],
        gold="Paris",
    )
    assert loss is not None
    assert_differentiable(loss)
    loss.backward()


def test_contrastive_returns_none_without_valid_positive(tiny):
    _, _, loss_fn = tiny
    assert loss_fn.contrastive_loss(
        "What is the capital of France?",
        ["Completely unrelated text", "Another unrelated string"],
        gold="Paris",
    ) is None


def test_combined_produces_single_trainable_total(tiny):
    model, _, loss_fn = tiny
    model.zero_grad()
    out = loss_fn.combined(
        sources=["What is the capital of France?"],
        targets=["Paris"],
        candidates_batch=[["Paris is the capital of France", "It is sunny"]],
        lambda_gen=1.0,
        lambda_contrastive=0.5,
    )
    assert {"generation_loss", "total_loss"} <= set(out)
    assert_differentiable(out["total_loss"])
    out["total_loss"].backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in model.parameters() if p.requires_grad
    )


def test_combined_handles_empty_contrastive_gracefully(tiny):
    _, _, loss_fn = tiny
    out = loss_fn.combined(
        sources=["q"], targets=["a"],
        candidates_batch=[["unrelated", "more unrelated"]],
    )
    assert "contrastive_loss" in out
    assert float(out["contrastive_loss"]) == 0.0
    assert_differentiable(out["total_loss"])


def test_sequence_f1_is_symmetric_and_bounded():
    assert sequence_f1("a b", "a b") == 1.0
    assert sequence_f1("a b", "c d") == 0.0
    assert abs(sequence_f1("a b", "b c") - sequence_f1("b c", "a b")) < 1e-9
    assert 0.0 <= sequence_f1("a b c", "a") <= 1.0


def test_legacy_loss_is_flagged_as_untrainable():
    """The old HybridRAGCAG.compute_generation_loss must not silently return a
    grad-flagged constant. If the class is importable, it must warn."""
    pytest.importorskip("faiss")
    try:
        from src.hybrid_rag_cag_system import HybridRAGCAG
    except Exception as exc:  # torch/faiss missing in this env
        pytest.skip(f"legacy module unimportable: {exc}")

    obj = HybridRAGCAG.__new__(HybridRAGCAG)
    with pytest.warns(DeprecationWarning):
        out = obj.compute_generation_loss(["a b"], ["a b"])
    assert out.grad_fn is None, "legacy loss must not pretend to be trainable"

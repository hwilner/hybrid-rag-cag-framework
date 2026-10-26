"""Tests for the decoding policy.

The defect these guard against
------------------------------
``src/real_ablation.py`` used to record the decoding configuration of every
arm as the hard-coded string ``"greedy, num_beams=1, do_sample=False"``. That
was false for two of the three arms: the CAG and Hybrid arms ask for five
candidates, which the generator produces with a beam search *and* a
nucleus-sampling pass. The recorded description therefore did not describe the
code that produced the numbers.

The fix puts the decoding configuration in one place
(``src/decoding_policy.py``), which the generator executes and the harness
records. These tests pin that policy's behaviour, and pin the values it
produces to the ones the generator used before, so that moving the parameters
into the policy cannot quietly change the measurement.
"""

import ast
import os

import pytest

from decoding_policy import DecodingPass, DecodingPolicy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATOR_PATH = os.path.join(REPO_ROOT, "src", "hybrid_rag_cag_system.py")

#: The decoding parameters that must come from the policy, not from a literal
#: in the generator.
DECODING_KEYWORDS = frozenset(
    {
        "num_beams",
        "num_return_sequences",
        "do_sample",
        "early_stopping",
        "top_p",
        "temperature",
    }
)


def test_single_candidate_panel_is_greedy():
    """The RAG arm asks for one candidate, and decodes it greedily."""
    policy = DecodingPolicy(num_candidates=1)

    assert policy.is_deterministic()
    passes = policy.passes()
    assert len(passes) == 1
    assert passes[0].num_beams == 1
    assert passes[0].num_return_sequences == 1
    assert passes[0].do_sample is False


def test_five_candidate_panel_samples():
    """The CAG and Hybrid arms ask for five, which is beam search plus sampling."""
    policy = DecodingPolicy(num_candidates=5)

    assert not policy.is_deterministic()
    passes = policy.passes()
    assert len(passes) == 2

    beam, sample = passes
    assert beam.do_sample is False
    assert sample.do_sample is True
    assert sample.top_p == 0.9
    assert sample.temperature == 0.7


def test_policy_preserves_the_parameters_the_generator_used():
    """Pin the values to the ones the generator passed before the refactor.

    These are the numbers the committed ablation artefacts were produced with,
    so a change here is a change to the measurement, not a refactor.
    """
    beam, sample = DecodingPolicy(num_candidates=5).passes()

    assert beam.generate_kwargs() == {
        "num_beams": 5,
        "num_return_sequences": 3,
        "do_sample": False,
        "early_stopping": True,
    }
    assert sample.generate_kwargs() == {
        "num_beams": 1,
        "num_return_sequences": 2,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.7,
    }


def test_sampling_parameters_are_omitted_for_a_deterministic_pass():
    """A deterministic pass must not carry sampling parameters.

    Passing ``top_p`` with ``do_sample=False`` is accepted by transformers
    but ignored; omitting it keeps the call identical to the one it replaced.
    """
    kwargs = DecodingPolicy(num_candidates=1).passes()[0].generate_kwargs()

    assert "top_p" not in kwargs
    assert "temperature" not in kwargs


@pytest.mark.parametrize("num_candidates", [1, 2, 3, 4, 5, 6, 8])
def test_passes_fill_the_panel_exactly(num_candidates):
    """The passes together must return exactly the requested panel size.

    The generator truncates the candidate list to ``num_candidates``, so a
    policy that returned fewer would silently shrink the panel and change what
    the contrastive selection chooses between.
    """
    policy = DecodingPolicy(num_candidates=num_candidates)

    produced = sum(p.num_return_sequences for p in policy.passes())
    assert produced == num_candidates


def test_description_says_that_the_panel_samples():
    """The description is what ends up in the results file, so it must not lie."""
    described = DecodingPolicy(num_candidates=5).describe()

    assert "do_sample=True" in described
    assert "do_sample=False" in described
    assert "stochastic" in described


def test_description_of_a_greedy_panel_says_it_is_deterministic():
    described = DecodingPolicy(num_candidates=1).describe()

    assert "do_sample=False" in described
    assert "deterministic" in described
    assert "stochastic" not in described


def test_zero_candidate_panel_is_rejected():
    """A panel of zero candidates is not a thing, and must not be accepted."""
    with pytest.raises(ValueError):
        DecodingPolicy(num_candidates=0)


def test_pass_describe_lists_the_parameters_it_uses():
    """A pass must describe itself in terms of what it actually passes."""
    described = DecodingPass(
        num_beams=1, num_return_sequences=2, do_sample=True,
        top_p=0.9, temperature=0.7,
    ).describe()

    assert "nucleus sampling" in described
    assert "top_p=0.9" in described
    assert "temperature=0.7" in described
    assert "early_stopping" not in described


def test_generator_takes_its_decoding_parameters_from_the_policy():
    """The generator must not pass a decoding parameter as a literal.

    This is the link between the policy and the code that runs it. Without it
    the two could drift apart again, which is the defect this whole change is
    about: the results file described a decoding configuration that the
    generator did not use.
    """
    with open(GENERATOR_PATH, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=GENERATOR_PATH)

    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "generate_candidates"
    ]
    assert functions, "generate_candidates is missing from the generator"

    for function in functions:
        generate_calls = [
            call for call in ast.walk(function)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "generate"
        ]
        assert generate_calls, "generate_candidates makes no call to model.generate()"

        for call in generate_calls:
            literals = sorted(
                keyword.arg
                for keyword in call.keywords
                if keyword.arg in DECODING_KEYWORDS
            )
            assert not literals, (
                f"generate_candidates passes {literals} as a literal; the "
                "decoding parameters must come from DecodingPolicy"
            )

        uses_policy = any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "passes"
            for call in ast.walk(function)
        )
        assert uses_policy, "generate_candidates never consults DecodingPolicy.passes()"

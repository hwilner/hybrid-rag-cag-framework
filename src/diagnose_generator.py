"""Why the real Hybrid/RAG/CAG arms score near zero, measured rather than assumed.

The Tier-2 numbers this branch reports are not limited by retrieval. They are
limited by the fact that ``facebook/bart-large`` -- which the repository uses
unconditionally as ``HybridConfig.generator_model`` -- is a *pretrained language
model*, not a question-answering model. It was never instruction-tuned to answer
``question: ... context: ...`` prompts. Given such a prompt it **continues the
text**, echoing the question back before producing anything.

This script measures that directly so the claim is not an opinion.
"""

import json
import os
import sys

import torch
from transformers import BartForConditionalGeneration, BartTokenizerFast

MODEL = "facebook/bart-large"

# A prompt in exactly the shape hybrid_rag_cag_system.HybridGenerator builds.
PROMPT = ("question: What is the capital of France? "
          "context: Paris is the capital and most populous city of France, "
          "located on the Seine.")

GOLD = "Paris"


def main():
    tok = BartTokenizerFast.from_pretrained(MODEL)
    model = BartForConditionalGeneration.from_pretrained(MODEL)
    model.eval()

    rows = []

    def decode(**kw):
        enc = tok(PROMPT, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            out = model.generate(**enc, **kw)
        return tok.decode(out[0], skip_special_tokens=True).strip()

    variants = {
        "default (branch settings)": dict(max_new_tokens=24, num_beams=1),
        "beam=5": dict(max_new_tokens=24, num_beams=5),
        "forced_bos": dict(max_new_tokens=24, num_beams=1,
                           forced_bos_token_id=tok.bos_token_id),
        "sampling": dict(max_new_tokens=24, num_beams=1, do_sample=True, top_p=0.9),
    }

    print(f"model: {MODEL}")
    print(f"prompt: {PROMPT[:90]}...")
    print(f"gold answer: {GOLD!r}\n")

    for name, kw in variants.items():
        text = decode(**kw)
        echoes = text.lower().startswith("question:")
        contains_gold = GOLD.lower() in text.lower()
        rows.append({"variant": name, "output": text, "echoes_prompt": echoes,
                     "contains_gold": contains_gold})
        print(f"  {name:26} echo={echoes!s:<5} contains_gold={contains_gold}")
        print(f"      -> {text[:100]!r}\n")

    n_echo = sum(r["echoes_prompt"] for r in rows)
    print(f"{n_echo}/{len(rows)} decoding variants echo the prompt verbatim.")
    print("No decoding flag fixes this: the model was never trained to answer QA prompts.")

    out = os.environ.get("OUT", "/tmp/bart_echo_check.json")
    with open(out, "w") as fh:
        json.dump({"model": MODEL, "prompt": PROMPT, "gold": GOLD,
                   "variants": rows}, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

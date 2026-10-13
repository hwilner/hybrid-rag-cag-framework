# Technical Evaluation Notes

## Purpose

This repository is an **experimental implementation** of a Hybrid RAG-CAG
question-answering system. The material here is intended to document the
implementation, illustrative comparisons, and observed limitations. It does
not establish external suitability for any consequential use.

## Evaluation scope

The repository contains three small exploratory evaluation tiers: a foundational
question set, a broader baseline comparison, and an expert-level scientific-domain
comparison using simulated expert responses. Reported values are implementation
outputs, not independently validated benchmarks.

## Interpretation

The included runs show that performance varies substantially with question
difficulty and domain. In the expert-level comparison, the Hybrid configuration
remains below the simulated expert reference. Results should be reproduced with
held-out datasets, clearly specified data provenance, independently reviewed
baselines, and confidence intervals before they inform external claims.

## Known limitations

- The bundled datasets are small and are not substitutes for established public benchmarks.
- Expert responses are simulated in the evaluation code and must not be presented as an external human study.
- Statistical outputs are exploratory and do not establish generalization or causal effects.
- The system exhibits substantial errors on multi-hop, abstract, and specialized questions.
- No deployment, clinical, educational, or scientific-research suitability is established by this repository.

## Reproducibility

Use the scripts in `src/` to inspect the implementation and regenerate local
outputs. Record dependencies, input data, hardware, random seeds, and any code
changes before comparing results across runs.

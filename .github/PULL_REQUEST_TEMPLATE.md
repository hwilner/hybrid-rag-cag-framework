## Summary

> **⚠️ Retracted (2026-09-29).** The Tier-1 figures previously stated here (+57.5% F1,
> Hybrid 0.389 vs RAG 0.247) are **unsupported**: no code in this repository produces
> them, and the evaluation baselines were hardcoded dictionary lookups rather than models.
> See [`results.md`](../results.md) for the evidence and the reproducible replacement numbers.

<!-- What does this PR do, in 2–3 sentences? -->

## Linked issue

Closes #

## Test evidence

<!-- Commands run and a summary of output, e.g.:
     `pytest` — 24 passed
     <!-- NOTE: the Tier-1 0.389 baseline is RETRACTED; see results.md. Report
          only numbers produced by scripts in this repo. -->

## Checklist

- [ ] Tests pass locally (`pytest`)
- [ ] Docs updated for any behavior change
- [ ] No scope creep — diff stays within the issue's Boundary
- [ ] Metrics/tests pre-registered in the issue before benchmarking (if an evaluation claim is made)
- [ ] Negative or neutral results reported honestly, not dropped
- [ ] CAG contrastive-selection core untouched, or change explicitly justified

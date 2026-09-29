# Retracted result files

Every JSON in this directory produced the numbers retracted in the top-level `results.md`.
They are retained **only** so the retraction can be audited against the original values.

**Do not cite these files.** They contain hardcoded measurements that no code in this
repository computes, and the two "validated" files disagree with each other:

| File | RAG F1 | Hybrid F1 | Implied improvement |
|---|---|---|---|
| `validated_experimental_results.json` | 0.247 | 0.389 | +57.5% |
| `updated_results.json` | 0.215 | 0.381 | +77.2% |

`enhanced_evaluation_results.json` is 0 bytes.

See `../../results.md` for the retraction evidence and for the real, reproducible numbers
(`../verified_tier2.json`).

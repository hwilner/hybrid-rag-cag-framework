# Pre-registration: issue #5 rebuilt-benchmark ablation

**Status:** scaffold only. No rebuilt corpus has been chosen, no repaired
benchmark exists, and no repaired-benchmark result has been measured. This
document pre-registers the analysis that must run *after* the scientific owner
makes the corpus-construction decision and *after* the candidate benchmark
passes the validity gate.

This file exists to keep the deferred repair from becoming post hoc. The
repository has a retraction history: no cell in the tables below may be filled
by hand, and no number may be copied into them unless it comes from a committed
artefact produced by the commands named here.

## Research question

On a benchmark whose questions are all answerable from its corpus, does the
Hybrid RAG-CAG pipeline beat plain RAG?

- **RAG** — retrieve + rerank, single greedy decode.
- **CAG** — multi-candidate generation, no retrieval, contrastive selection.
- **Hybrid** — retrieve + rerank + multi-candidate generation + contrastive
  selection.

## Corpus-construction decision (owner gate)

The scientific owner must choose one construction before compute is spent:

1. a rebuilt corpus/question set covering the declared question types; or
2. a declared subset of question types, written as the candidate benchmark.

Automation does not make that decision. Whichever option is chosen, the
candidate is a single JSON file:

```json
{"corpus": ["document text", "..."],
 "questions": [{"question": "...", "answer": "...",
                "type": "...", "difficulty": "..."}]}
```

Acceptance gate, before any model is loaded:

```bash
python tools/check_benchmark_validity.py --candidate new_benchmark.json
```

The strict default must pass: every question answerable from the corpus, with
no schema problems. The seeded harness enforces the same gate itself:

```bash
python tools/run_repaired_ablation.py \
  --candidate new_benchmark.json \
  --branch main \
  --out results/repaired_ablation_main.json
```

`tools/run_repaired_ablation.py` refuses an invalid candidate before
`src/real_ablation.py` imports torch or loads a checkpoint. The output artefact
records the candidate file path and SHA-256 under `provenance.benchmark`.

## Pre-registered measurements

1. **Primary metric:** token-F1, computed per question.
2. **Secondary metrics:** exact match and answer containment, reported
   alongside F1. EM is expected to remain structurally low if systems emit
   extractive sentences; it is not the primary endpoint.
3. **Primary comparison:** paired Hybrid − RAG per-question F1 difference on
   the same questions, same retrieval stack, same decoding policy, same seed.
4. **Controls:** the CAG no-retrieval arm is the architecture control; the RAG
   single-decode arm is the pipeline control; the validity gate is the dataset
   control. No ranking claim is valid unless all three are present in the same
   committed artefact.
5. **Statistics:** bootstrap 95% confidence interval on the mean paired F1
   difference (10,000 resamples, seed 0) and two-sided Wilcoxon signed-rank
   when at least six paired differences are non-zero. Report the number of
   non-zero differences whenever Wilcoxon is not applicable.
6. **Success criterion:** Hybrid is claimed to beat RAG only if the Hybrid −
   RAG F1 bootstrap CI excludes zero in the positive direction. A CI including
   zero is reported as *no measurable benefit*, not as evidence of equality.
7. **Null result:** a null or negative Hybrid − RAG result is a publishable
   outcome and must be committed with the same prominence as a positive one.

## Result tables and figure placeholders

These captions are pre-registered. The table cells are intentionally empty;
they are not results.

**Table P1 — Rebuilt-benchmark validity.** Corpus documents, questions,
answerable count, per-type and per-difficulty answerability, gate verdict, and
candidate SHA-256. Source: `tools/check_benchmark_validity.py --candidate ...`.

| Corpus docs | Questions | Answerable | Gate verdict | Candidate SHA-256 |
|---|---:|---:|---|---|
| — | — | — | — | — |

**Table P2 — Three-arm ablation on the repaired benchmark.** Per-arm token-F1,
EM, containment, empty predictions, and wall clock. Source:
`results/repaired_ablation_main.json`, produced by `src/real_ablation.py`.

| Arm | F1 | EM | Containment | Empty predictions | Wall clock |
|---|---:|---:|---:|---:|---:|
| RAG | — | — | — | — | — |
| CAG | — | — | — | — | — |
| Hybrid | — | — | — | — | — |

**Table P3 — Paired controls on the repaired benchmark.** Paired per-question
differences, bootstrap 95% CI, Wilcoxon *p* where applicable, and the number
of non-zero differences. Source: the committed analysis of
`results/repaired_ablation_main.json`.

| Comparison | Metric | Mean Δ | Bootstrap 95% CI | Wilcoxon *p* | n non-zero | Verdict |
|---|---|---:|---|---:|---:|---|
| Hybrid − RAG | F1 | — | — | — | — | — |
| Hybrid − CAG | F1 | — | — | — | — | — |
| RAG − CAG | F1 | — | — | — | — | — |
| Hybrid − RAG | containment | — | — | — | — | — |
| Hybrid − CAG | containment | — | — | — | — | — |
| RAG − CAG | containment | — | — | — | — | — |

**Figure P1 — Per-arm metrics on the repaired benchmark.** Grouped bars for
F1, EM, and containment by arm, with the figure caption naming the candidate
benchmark SHA-256 and the output artefact. Placeholder only; the figure script
and measured values do not exist yet.

## Integrity rules

- No repaired-benchmark claim appears in README, docs, or results until Table
  P1 is filled from a passing gate report and Tables P2–P3 are filled from a
  committed ablation artefact.
- The artefact must record `new_model_runs` honestly; re-aggregation of
  existing per-question rows is not a new model run, but running any arm is.
- If the candidate changes after a run starts, the old artefact is kept and
  the new candidate gets a new output file and a new SHA-256 row.
- The current defective benchmark remains the historical record. It is not
  edited to look repaired.

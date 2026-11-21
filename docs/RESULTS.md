# Results

This is the paper-style results section: the findings, stated with their
measured numbers, their controls, and their captions. The complete record —
every measurement, every command, every environment note, and everything that
was *not* run — lives in [`../results/RESULTS.md`](../results/RESULTS.md);
sections below reference it as §n. Nothing is reported here that is not
measured there. The integrity history (the 2026-09-29 retraction) is in
[`INTEGRITY.md`](INTEGRITY.md).

> **Benchmark-validity caveat that gates every ranking claim.** The Tier-2
> benchmark is measurably defective: only 21 of 55 gold answers appear
> anywhere in the 100-document corpus (38.2%; by difficulty: easy 10/10,
> medium 10/10, hard 1/15, very_hard 0/20). Measured by
> `tools/check_benchmark_validity.py` → `../results/benchmark_validity.json`;
> the 34 unanswerable questions are synthesis types whose gold answers are
> composed sentences present in no document (§7). The restricted-set
> re-analysis (§11) and the reader reform (§12) therefore report on the
> 21-answerable subset, and the full corpus rebuild + seeded re-run remains
> open as issue #5.

**Research question.** Does the Hybrid RAG-CAG pipeline — dense retrieval,
similarity reranking, multi-candidate generation, contrastive answer
selection — beat plain RAG on question answering? Answer, measured twice
with two different readers: **no.**

## R1. The Hybrid machinery shows no measurable benefit over plain RAG (2026-09 runs)

Three arms (RAG / CAG / Hybrid), 55 questions, `all-mpnet-base-v2` retrieval,
`facebook/bart-large` reader (§6, Table 6.1):

| Arm | token-F1 | Containment | EM | Wall clock |
|---|---|---|---|---|
| RAG | **0.1892** | **0.3273** | 0.0000 | 607 s |
| CAG | 0.1705 | 0.0182 | 0.0000 | 2,322 s |
| Hybrid | 0.1821 | 0.2727 | 0.0000 | 4,909 s |

*Table R1: per-arm metrics, full benchmark (n = 55). CAG (no retrieval) is
not a viable arm — containment 0.0182. The Hybrid arm costs 8× the RAG wall
clock for less F1.*

Paired controls (bootstrap 95% CI, Wilcoxon; §6 Table 6.2): Hybrid − RAG F1
Δ = −0.0071, CI [−0.0188, +0.0043], *p* = 0.568 — not significant. No arm
beats RAG on F1.

**Controls:** lexical baselines (TF-IDF / BM25 + extractive reader) reach
F1 0.177 in under a second on CPU (§2); an oracle-extractive ceiling reaches
0.187 (§2). A five-line lexical extractor beats the 406 M-parameter system.

## R2. The failure is located: benchmark first, reader second (diagnosis)

Failure attribution (§7): recall@1 = recall@10 = 36.4%, and 20 of the 21
retrievable answers already rank first — **retrieval contributes exactly
zero failures**; the corpus gap accounts for everything else. The reranker
is a measured no-op: it recomputes the retriever's own scores, and the
reranked order equals the dense order on 55/55 questions (§7.5). The
BART-large reader cannot answer questions at all — it echoes the prompt
under all four decoding variants tested (§7.7), which is why EM is 0.0000
on every 2026-09 run.

## R3. The negative result is robust to the benchmark restriction (2026-11-16)

Re-aggregating the committed per-question measurements onto the 21
answerable questions — no new model runs (§11, Tables 11.1–11.2,
Figure 11.1): Hybrid still does not beat RAG on F1 (*p* = 0.195), and RAG
now beats CAG on F1 (*p* = 0.016). The conclusion of R1 is not an artefact
of the unanswerable questions.

## R4. Fixing the reader fixes answering — and the machinery still does not pay (2026-11-16)

Pre-registered reader reform (issue #7): identical arms, retrieval, prompts,
decoding policy, metrics and statistics; only the reader swapped to
`google/flan-t5-small` (§12, Tables 12.1–12.2, Figure 12.1). On the
restricted set:

| Arm | F1 (BART) | F1 (flan) | EM (BART) | EM (flan) |
|---|---|---|---|---|
| RAG | 0.1095 | **0.8122** | 0.0000 | 0.7143 |
| CAG | 0.0571 | 0.2095 | 0.0000 | 0.0952 |
| Hybrid | 0.0882 | 0.7896 | 0.0000 | 0.7143 |

*Table R2: restricted-set (n = 21) metrics before and after the reader
swap. The instruction-tuned reader lifts every arm; the RAG–Hybrid gap
stays within noise in both eras.*

**The pre-registered success criterion was not met:** Hybrid − RAG F1
Δ = −0.0226, bootstrap 95% CI [−0.1327, +0.0490] includes zero, *p* n/a
(3 non-zero differences). This is the second independent measurement —
different reader, restricted set — finding that the extra machinery does
not pay for itself.

**New measured failure mode:** context dilution — on "What is the capital
of France?" the flan reader answers "Paris" from the rank-1 document alone
but "Tokyo" from the joined top-5 context (§12).

## What was not run

Larger instruction-tuned readers (flan-t5-base and up); the corpus rebuild
and seeded ablation re-run (issue #5 — requires GPU-scale compute and an
owner decision on corpus construction). These are recorded as NOT RUN in
§12 and §13 rather than estimated. No ranking claim about the repaired
benchmark is made anywhere in this repository.

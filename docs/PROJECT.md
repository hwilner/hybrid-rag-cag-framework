# Project Board — Hybrid RAG-CAG Framework

In-repo mirror of the [live GitHub contribution board](https://github.com/users/hwilner/projects/10).
**Implementation, tests, data utilities, benchmarks, documentation, and scoped extensions to the
original project are welcome from all contributors.** Every backlog
issue is mapped to a phase, status, and its dependencies so newcomers can see at a
glance what is ready to pick up.

**How to read this:** pick any card marked **Ready** — it has no unfinished
prerequisites. Cards marked **Blocked** list what must merge first. Epics (`[L]`)
are tracking umbrellas; the work happens in their children.

## Status: Ready to start (no unfinished prerequisites)

| Issue | Title | Size | Module |
|---|---|---|---|
| #6 | Add data/ directory loader utilities + manifest with hashes | XS | repo hygiene |
| #7 | Add pyproject.toml + pytest layout | XS | repo hygiene |
| #9 | Write docs/API_REFERENCE.md for HybridRAGCAGSystem | S | docs |
| #10 | Implement src/graph_index.py (LightRAG-style, synthetic tests) | S | graph index |
| #11 | Implement src/grader.py (CRAG-style corrective grader) | S | corrective |
| #12 | Implement src/fusion.py (multi-query + reciprocal rank fusion) | S | fusion |
| #13 | Implement src/agent.py (ReAct-style controller) | S | agentic |
| #8 | Implement src/cache_path.py (cache-augmented path) | S | cache |
| #14 | Implement src/critique.py (Self-RAG-style self-critique) | S | critique |

## Phase 1 — Query fusion and corrective grading (Epic #5)

| Issue | Title | Size | Blocked by |
|---|---|---|---|
| #12 | Implement src/fusion.py | S | — Ready |
| #11 | Implement src/grader.py | S | — Ready |
| #16 | Fusion + grader integration test on Tier-1 harness | XS | #12, #11 |
| #19 | Benchmark: Tier-1/Tier-2 re-run, honest report | S | #16 |

## Phase 2 — Graph-structured index (Epic #3)

| Issue | Title | Size | Blocked by |
|---|---|---|---|
| #10 | Implement src/graph_index.py | S | — Ready |
| #17 | Router v1 in src/router.py | S | #10 |
| #22 | Benchmark relational/global slice: graph vs vector vs both | S | #17 |

## Phase 3 — Agentic controller + active retrieval (Epic #2)

| Issue | Title | Size | Blocked by |
|---|---|---|---|
| #13 | Implement src/agent.py | S | — Ready |
| #15 | FLARE-style active re-retrieval | S | #13, #11 |
| #21 | Multi-hop benchmark slice + failure taxonomy | S | #15 |

## Phase 4 — Cache path and cost router (Epic #4)

| Issue | Title | Size | Blocked by |
|---|---|---|---|
| #8 | Implement src/cache_path.py | S | — Ready |
| #20 | Self-Route-style cost router | S | #8, #17 |
| #23 | Latency/cost benchmark table | XS | #20 |

## Phase 5 — Self-critique and answer-verification polish (Epic #1)

| Issue | Title | Size | Blocked by |
|---|---|---|---|
| #14 | Implement src/critique.py | S | — Ready |
| #18 | SuRe-style candidate-conditioned summaries | S | #14 |
| #24 | Final three-tier re-run + ablation table + limitations update | S | #18, #19, #22, #21, #23 |

## Suggested contribution paths

- **First contribution (no ML background needed):** #6, #7, or #9.
- **Core module, self-contained with synthetic tests:** #12, #11, #10, #13, #8, #14.
- **Benchmark/analysis work (needs modules merged):** #19, #22, #21, #23, #24.

## Rules

- Every code card must ship with tests (`python -m pytest -q` passes) and works on
  synthetic data first — no card requires external datasets or API keys to start.
- Report honest negative results; never silently drop a failing configuration.
- Pre-registered metrics before any benchmark re-run (see docs/METHODS.md).

---

**Synchronization rule:** the live board mirrors the operational backlog and current
workflow, while issue bodies and native parent/sub-issue relationships remain authoritative
for scope, acceptance criteria, boundaries, and dependencies. Keep this file synchronized
when an issue is added, closed, split, or materially reclassified; do not delete or replace
an assigned issue during backlog maintenance.

# Concept Figure

The canonical concept figure for this repository is
[`concept_figure.svg`](concept_figure.svg) — a flat-design, NeurIPS/FigForge-style
diagram (white background, pastel modules, thin strokes) showing the Hybrid RAG-CAG pipeline — query, dense retrieval (DPR + FAISS), contrastive reranking, multi-candidate generation, CAG contrastive selection, final answer.
It is embedded near the top of [../INTRODUCTION.md](../INTRODUCTION.md) and
[../EXTENDED_INTRODUCTION.md](../EXTENDED_INTRODUCTION.md).

> Note: an AI-generated PNG rendering of the same figure (1536x1024) also
> exists but is kept out of git (binary assets are not committed via the
> project tooling). The SVG above is the source of truth.

If your viewer cannot render SVG, here is a faithful Mermaid sketch of the
same structure:

```mermaid
flowchart LR
    Q[Query] --> RET[Dense retriever<br/>DPR + FAISS, top-10]
    RET --> RR[Contrastive reranker<br/>re-score, keep top-5]
    RR --> GEN[Multi-candidate generator<br/>beam search + sampling]
    GEN --> C1[candidate 1] --> SEL
    GEN --> C2[candidate 2] --> SEL
    GEN --> C3[candidate 3] --> SEL
    SEL[CAG contrastive selection<br/>0.6 question + 0.4 context] --> A[Final answer]
    L[Training losses:<br/>generation + InfoNCE + diversity] -.-> SEL
```

# Extended Introduction: RAG and Hybrid RAG-CAG, Explained From Zero

This guide assumes **no background in machine learning, math beyond
arithmetic, or NLP**. Every technical idea is introduced the same way:
first a tiny example with real numbers you can check by hand, then the
intuition, and only then the notation — as shorthand for the procedure you
just saw. (A denser, technical version is in
[INTRODUCTION.md](INTRODUCTION.md).)

## The problem in plain terms

A chatbot like the large language models (LLMs) behind modern AI assistants
has read enormous amounts of text and, in a loose sense, "remembers" it. But
it has two embarrassing habits:

1. **It makes things up.** When it doesn't know an answer, it often invents a
   plausible-sounding one with full confidence. This is called
   *hallucination*.
2. **Its memory is frozen.** It knows nothing about anything that happened
   after its training data was collected, and it never learned your private
   documents at all.

The standard fix is called **Retrieval-Augmented Generation (RAG)** [1], and
the best everyday analogy is an **open-book exam**. Instead of forcing the
student (the LLM) to answer from memory, you hand them a library card. For
each question, a **librarian** (the *retriever*) runs to the shelves, fetches
the most relevant pages, and lays them on the desk. The student then writes
the answer *while looking at those pages*.

This helps a lot — but only if the librarian brings the right books, and only
if the student actually reads them carefully. Naive RAG fails on both counts
surprisingly often.

## How the librarian actually finds pages

Computers can't search by meaning directly; they search by numbers. Every
document is converted into a short list of numbers such that texts with
similar meanings get similar lists. The question gets the same treatment, and
the librarian returns the documents whose numbers are "closest" to the
question's numbers.

**A tiny example with real numbers.** Suppose our whole library has three
documents, and (in a toy two-number version) meaning is captured by just two
numbers: "how much about animals" and "how much about cooking."

- Question "my cat won't eat": `(0.9, 0.1)` — very animal-y, slightly food-y.
- Doc A, "feline nutrition guide": `(0.8, 0.2)`
- Doc B, "the history of pasta": `(0.0, 1.0)`
- Doc C, "training a puppy": `(0.7, 0.0)`

Which document should the librarian fetch? Compare the question to each
document by multiplying matching entries and adding the results (this is
called a *dot product*):

- Question·A = 0.9×0.8 + 0.1×0.2 = 0.72 + 0.02 = **0.74**
- Question·B = 0.9×0.0 + 0.1×1.0 = 0.00 + 0.10 = **0.10**
- Question·C = 0.9×0.7 + 0.1×0.0 = 0.63 + 0.00 = **0.63**

Doc A wins — exactly what common sense says. That whole comparison is what
people mean by **cosine similarity**: treat each list as an arrow, and the
dot product (after dividing by arrow lengths) measures how much the two
arrows point the same way. The lists themselves are called **embeddings**.
This retrieval approach, **Dense Passage Retrieval (DPR)** [2], beats
old-fashioned keyword search by a wide margin, because "feline" and "cat"
match by meaning even though the words differ.

Real systems use lists of hundreds of numbers instead of two, and fast search
over millions of them is done with a library called FAISS — think of it as an
extremely well-organized card catalog. The procedure is exactly the one you
just did by hand, only bigger.

## What this repository adds: don't trust the first draft

Here is the key insight of the **Hybrid RAG-CAG** system in
`src/hybrid_rag_cag_system.py`. A naive system does: *search once → read once
→ answer once*. Each "once" is a chance to go wrong. This system adds two
extra layers of skepticism:

**1. A second librarian's opinion (reranking).** The first search returns ten
candidate passages; a second, more careful pass re-scores them (same dot
product trick as above) and keeps only the best few. Cheap, and it filters
out a lot of noise before the student ever sees it.

**2. A panel of candidate answers (the CAG layer).** Instead of writing one
answer, the student drafts *several* different answers (some careful and
systematic — "beam search" — and some more varied — "sampling"). Then a judge
scores every draft on two questions: *Does it actually answer the question?*
and *Is it supported by the pages on the desk?* The draft with the best
combined score wins. This "generate a panel, then contrast and select" idea
is closely related to published work like SuRe [5] (candidate answers checked
against summaries of the evidence) and Adaptive Contrastive Decoding [6]
(contrasting outputs to cope with noisy retrieved context).

**A tiny example of the judging math.** Say the panel has three drafts, and
the judge's combined score (0.6 × similarity to the question + 0.4 ×
similarity to the evidence) comes out as:

- Draft 1: 2.0, Draft 2: 1.0, Draft 3: 0.0

At test time the system simply picks the biggest number — Draft 1. During
*training*, the scores are converted into probabilities so the model can be
nudged: exponentiate each score (e^2.0 ≈ 7.39, e^1.0 ≈ 2.72, e^0.0 = 1.00,
total ≈ 11.11) and divide:

- Draft 1: 7.39 / 11.11 ≈ **0.67**, Draft 2: ≈ **0.24**, Draft 3: ≈ **0.09**

That recipe — exponentiate, then divide by the total — is all that **softmax**
is. Training pushes the good draft's probability up and the others down (this
specific push is the **InfoNCE contrastive loss**; 0.6/0.4 weighting and a
"temperature" of 0.07 are just dials on the same procedure). The training
procedure simultaneously teaches the system to (a) write answers that match
correct ones, (b) rank the best candidate above the rest, and (c) keep the
candidates diverse, so the panel doesn't collapse into five copies of the
same guess.

## The whole pipeline at a glance

![Concept figure: the Hybrid RAG-CAG pipeline — a query flows through dense retrieval, contrastive reranking, multi-candidate generation, and CAG contrastive selection to produce the final answer, with training losses attached to the selection stage.](figures/concept_figure.svg)

```mermaid
flowchart LR
    Q[Question] --> RET[Dense retriever<br/>FAISS vector search, top-10]
    RET --> RR[Contrastive reranker<br/>re-score, keep top-5]
    RR --> CTX[Context:<br/>top passages joined]
    CTX --> GEN[Candidate generator<br/>beam search + sampling<br/>~5 draft answers]
    Q --> GEN
    GEN --> SEL[Contrastive selection<br/>score vs question + context]
    SEL --> A[Final answer]
```

## Naive RAG vs. this hybrid

```mermaid
flowchart TB
    subgraph Naive["Naive RAG"]
        N1[Question] --> N2[Search once] --> N3[Generate once] --> N4[Answer<br/>no second chances]
    end
    subgraph Hybrid["Hybrid RAG-CAG (this repo)"]
        H1[Question] --> H2[Search] --> H3[Rerank<br/>skeptical pass]
        H3 --> H4[Generate several<br/>candidate answers]
        H4 --> H5[Contrast & select best] --> H6[Answer]
    end
```

## Does it work? Honest numbers

Answers are scored by comparing their words against a reference answer.
**Tiny example first:** if the system's answer is `"the black cat"` and the
reference is `"the black dog"`, the answers share 2 words. *Precision* = of
the 3 words the system produced, 2 were right (2/3 ≈ 0.67). *Recall* = of
the 3 words that should appear, 2 were found (2/3 ≈ 0.67). **F1** is just a
single number that punishes you if *either* of those is low — technically the
harmonic mean, `2 × p × r / (p + r)` — here F1 = 0.67. A sloppy answer (low
precision) or an incomplete one (low recall) both drag F1 down.

The system was tested on three tiers of increasing difficulty (details in
[TECHNICAL_EVALUATION_NOTES.md](TECHNICAL_EVALUATION_NOTES.md)):

- **Tier 1 (12 basic questions):** F1 **0.389** — a **57.5% improvement**
  over the same system with the extra layers switched off (plain RAG, 0.247).
  The layered design clearly helps.
- **Tier 2 (55 questions, compared with 5 other published-style systems):**
  F1 **0.276** — competitive, but *not* the best; a stronger "Advanced RAG"
  baseline scored 0.369.
- **Tier 3 (26 PhD-level science questions):** F1 **0.140**, which is only
  **38.1% of the expert reference** (0.368). On hard physics questions the
  system essentially failed.

**Caveats that matter:** the datasets are small; the Tier-3 "expert" answers
are simulated inside the evaluation code, not collected from real humans; and
the statistics are exploratory. This is a research prototype, not a product.
The honest takeaway: the hybrid design beats its own simpler self, matches
roughly the middle of the current field, and is nowhere near expert-level
scientific reasoning.

## Where the field is going — and this repo's roadmap

RAG has evolved from "naive" pipelines into *modular* systems [15] where each
stage can be swapped and upgraded. The planned upgrades (full details in
[ROADMAP.md](ROADMAP.md)) each borrow a published idea:

- **RAG-Fusion** [14]: ask the librarian the same question phrased several
  different ways, then merge the result lists. **Tiny example:** a document
  ranked 1st in one list and 3rd in another gets merged score
  1/(1 + 60) + 1/(3 + 60) ≈ 0.0164 + 0.0159 = 0.0323, beating a document that
  is 1st in only one list (0.0164). Documents near the top of *several* lists
  win. That recipe — sum 1/(rank + a constant like 60) across lists — is all
  that **reciprocal rank fusion** is.
- **CRAG** [8]: a grader that judges whether the fetched pages are any good,
  and triggers a fallback (like a web search) when they're not.
- **GraphRAG / LightRAG** [12, 13]: besides the page-by-page catalog, build a
  *map of who-is-related-to-what* (a knowledge graph), which answers big
  "how does X connect to Y" questions that page search misses.
- **Agentic RAG / ReAct / FLARE** [9, 10, 11, 16]: let the system *plan*
  multi-step searches and fetch more pages mid-answer when it notices it's
  uncertain — like a student allowed to return to the library mid-exam.
- **Cache-augmented generation + Self-Route** [4, 17]: if the whole bookshelf
  is small, just put it on the desk (preload into the model's working
  memory), and route each question to the cheapest method that can handle it.
- **Self-RAG** [7]: teach the system to critique its own draft before
  showing it to you.

The **one rule of the roadmap**: the CAG "panel of candidates + contrastive
selection" core never changes; every new idea plugs in *around* it.

```mermaid
flowchart TB
    Q[Question] --> ROUTER{Router<br/>easy? relational? global?}
    ROUTER -->|small stable corpus| CACHE[Cache path<br/>preloaded context]
    ROUTER -->|factoid| VEC[Vector search]
    ROUTER -->|relational/global| GRAPH[Graph index]
    VEC --> FUSE[Query expansion +<br/>rank fusion]
    GRAPH --> FUSE
    FUSE --> GRADER{Corrective grader<br/>retrieval good enough?}
    GRADER -->|no| ESC[Filter / web fallback] --> FUSE
    GRADER -->|yes| AGENT[Agent controller<br/>multi-step + mid-answer re-retrieval]
    CACHE --> CAG
    AGENT --> CAG[CAG core: candidate panel<br/>+ contrastive selection]
    CAG --> CRIT[Self-critique pass] --> A[Final answer]
```

## Where to go next

- Technical version: [INTRODUCTION.md](INTRODUCTION.md)
- What is built vs. planned: [METHODS.md](METHODS.md)
- Improvement plan: [ROADMAP.md](ROADMAP.md)
- Evaluation caveats: [TECHNICAL_EVALUATION_NOTES.md](TECHNICAL_EVALUATION_NOTES.md)

## References

1. Lewis et al. 2020, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020. arXiv:2005.11401
2. Karpukhin et al. 2020, "Dense Passage Retrieval for Open-Domain Question Answering" (DPR), EMNLP 2020. arXiv:2004.04906
3. Izacard & Grave 2021, "Leveraging Passage Retrieval with Generative Models for Open Domain QA" (FiD), EACL 2021. arXiv:2007.01282
4. Chan et al. 2024, "Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks". arXiv:2412.15605
5. Kim et al. 2024, "SuRe: Summarizing Retrievals using Answer Candidates for Open-domain QA of LLMs", ICLR 2024. arXiv:2404.13081
6. Kim et al. 2024, "Adaptive Contrastive Decoding in Retrieval-Augmented Generation for Noisy Contexts" (ACD). arXiv:2408.01084
7. Asai et al. 2023, "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", ICLR 2024. arXiv:2310.11511
8. Yan et al. 2024, "Corrective Retrieval Augmented Generation" (CRAG). arXiv:2401.15884
9. Yao et al. 2022, "ReAct: Synergizing Reasoning and Acting in Language Models", ICLR 2023. arXiv:2210.03629
10. Schick et al. 2023, "Toolformer: Language Models Can Teach Themselves to Use Tools", NeurIPS 2023. arXiv:2302.04761
11. Singh/Ehtesham et al. 2025, "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG". arXiv:2501.09136
12. Edge et al. 2024, "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (GraphRAG). arXiv:2404.16130
13. Guo et al. 2024, "LightRAG: Simple and Fast Retrieval-Augmented Generation", Findings of EMNLP 2025. arXiv:2410.05779
14. Rackauckas 2024, "RAG-Fusion: a New Take on Retrieval-Augmented Generation", IJNLC 13(1). arXiv:2402.03367
15. Gao et al. 2023, "Retrieval-Augmented Generation for Large Language Models: A Survey". arXiv:2312.10997
16. Jiang et al. 2023, "Active Retrieval Augmented Generation" (FLARE), EMNLP 2023. arXiv:2305.06983
17. Li, Xu et al. 2024, "Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach" (Self-Route). arXiv:2407.16833

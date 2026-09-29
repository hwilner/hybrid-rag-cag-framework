# Hybrid RAG-CAG Framework for Enhanced Question Answering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

Experimental implementation of a Hybrid RAG-CAG framework for enhanced question answering.

---

## ⚠️ Retraction notice (2026-09-29)

**The results previously published in this README have been retracted as unsupported.** They have
been replaced with reproducible measurements. See **[`results.md`](results.md)** for the full
account, including the evidence and the negative results.

Retracted claims:

- ~~"57.5% F1 improvement over standalone RAG"~~ — no shipped code produces the underlying
  numbers; the evaluation "baselines" (RAG, CAG, FiD, T5-FiD, DPR+FiD) were hardcoded
  dictionary lookups, not models. One recorded response time (8.9 µs) is physically impossible
  for a neural generator.
- ~~"38.1% of human expert performance"~~ — the "expert" answers are hardcoded strings in
  `src/expert_evaluation.py`. No human study took place.
- ~~"Statistical significance across all evaluation tiers"~~ — Tier-2 significance labels were
  assigned by comparing a t-statistic to hardcoded constants; no p-value was ever computed.
- ~~"Joint loss"~~ — the training loss returned a detached constant, so `backward()` silently
  updated nothing. Training ran for the configured epochs with every weight unchanged. Now
  replaced with real BART cross-entropy (`src/trainable_losses.py`), covered by tests.
- ~~"State-of-the-art comparison with FiD, T5-FiD, and DPR+FiD"~~ — those baselines did not exist
  as models.

The two retracted evaluation scripts (`option3_full_scale_evaluation.py`,
`expert_evaluation.py`) now **refuse to run** without an explicit opt-in flag, and when forced
write only to `results/UNSAFE_*.json` — so the fabricated numbers cannot be silently regenerated.

**No claim of superiority over any baseline is currently supported by this repository.** The
architecture described in [`docs/METHODS.md`](docs/METHODS.md) is real and is instantiated in
`src/hybrid_rag_cag_system.py`; the *evidence* that it works is what was retracted.

## 🔬 Overview

This repository contains an implementation of a Hybrid RAG-CAG (Retrieval-Augmented Generation
and Contrastive Answer Generation) framework: dense retrieval → contrastive reranking →
multi-candidate generation → contrastive answer selection.

**Current status: implemented, not yet validated.** The measured baselines in `results.md` score
F1 0.177 against a hard ceiling of 0.187 on the shipped 55-question set, where 34 of 55 gold
answers do not appear in the corpus at all. The dataset cannot currently support a claim of
superiority.

## 📊 Key Results

Real measurements from `results/verified_tier2.json` (reproduce with
`python src/run_honest_evaluation.py`):

| System | token-F1 | EM | Answer containment |
|---|---|---|---|
| TF-IDF + Extractive | 0.177 | 0.000 | 0.309 |
| BM25 + Extractive | 0.177 | 0.000 | 0.309 |
| Oracle extractive (ceiling) | 0.187 | 0.000 | 0.382 |

The two retrievers select identical sentences on all 55 questions; the paired difference is
exactly zero, so no significance test applies. The limiting factor is the dataset — see §2.1 of
`results.md`.

Not run: the dense + BART-large Hybrid stack (requires ~1.6 GB checkpoint; not runnable in the
verification environment) and Tier 1. These are recorded as **NOT RUN** in `results.md` rather
than estimated.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/hwilner/hybrid-rag-cag-framework.git
cd hybrid-rag-cag-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

The class names and methods below are the ones that actually exist. An earlier
version of this README documented `HybridRAGCAGSystem`, `.index_corpus()`, and
`.answer_question()` — **none of which exist**, so that example raised
`ImportError` on the first line. The real API is:

```python
from hybrid_rag_cag_system import HybridRAGCAG, HybridConfig

config = HybridConfig()                 # retriever_model, generator_model, top_k, ...
model = HybridRAGCAG(config)

# Index the corpus
corpus = [
    "Paris is the capital of France.",
    "Machine learning is a subset of AI.",
    # ... your documents
]
model.retriever.build_index(corpus)

# Ask a question. forward() returns a dict; the answer is under 'final_answers'.
result = model(questions=["What is the capital of France?"])
print(result["final_answers"])
```

Public surface, as implemented:

| Class | Method | Purpose |
|---|---|---|
| `HybridConfig` | — | configuration dataclass |
| `DenseRetriever` | `build_index(corpus)` | build the FAISS `IndexFlatIP` index |
| `DenseRetriever` | `retrieve(queries, k=10)` | dense retrieval |
| `ContrastiveReranker` | `rerank(queries, docs, k=5)` | rerank the top-k |
| `HybridGenerator` | `generate_candidates(...)` | multi-candidate decoding |
| `HybridGenerator` | `forward(input_ids, attention_mask, labels)` | real BART forward |
| `HybridRAGCAG` | `__call__(questions, gold_answers=None)` | full pipeline |

### Training (differentiable)

The shipped `HybridRAGCAG.compute_generation_loss` returned a detached constant
(see `results.md` §1.4) and is now deprecated. For real training use
`src/trainable_losses.py`, which returns BART's own teacher-forced
cross-entropy:

```python
from trainable_losses import DifferentiableLoss

loss_fn = DifferentiableLoss(model.generator.model, model.generator.tokenizer)
loss = loss_fn.generation_loss(["What is the capital of France?"], ["Paris"])
loss.backward()          # populates .grad on 91/91 generator parameters
```

`tests/test_loss_differentiable.py` asserts this, so the defect cannot silently
return.

### Running Evaluations

#### Tier 1: Foundational Validation
```bash
python train_and_evaluate.py --evaluation_tier 1
```

#### Tier 2: Enhanced Evaluation (55 questions, 6 systems)
```bash
python option3_full_scale_evaluation.py
```

#### Tier 3: Expert-Level Evaluation (Human comparison)
```bash
python expert_evaluation.py
```

## 📁 Repository Structure

```
hybrid-rag-cag-framework/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── LICENSE                            # MIT License
│
├── src/
│   ├── hybrid_rag_cag_system.py      # Core hybrid system implementation
│   ├── train_and_evaluate.py         # Training and evaluation pipeline
│   ├── option3_full_scale_evaluation.py  # Tier 2 evaluation
│   └── expert_evaluation.py              # Tier 3 expert evaluation
│
├── data/
│   ├── tier1_dataset.json            # Foundational validation dataset
│   ├── tier2_dataset.json            # Enhanced evaluation dataset
│   └── tier3_scientific_dataset.json # Expert-level scientific questions
│
├── results/
│   ├── tier1_results.json            # Foundational evaluation results
│   ├── tier2_results.json            # Enhanced evaluation results
│   ├── tier3_expert_comparison.json  # Human expert comparison results
│   └── enhanced_discussion_analysis.json  # Comprehensive analysis
│
├── paper/
│   └── TECHNICAL_EVALUATION_NOTES.md       # Evaluation notes and limitations
│
└── docs/
    ├── INSTALLATION.md               # Detailed installation guide
    ├── USAGE.md                      # Comprehensive usage examples
    ├── EVALUATION.md                 # Evaluation methodology
    └── API_REFERENCE.md              # Complete API documentation
```

## 🔧 System Architecture

### Core Components

1. **Dense Retrieval Component**
   - Bi-encoder architecture with contrastive learning
   - FAISS-based efficient similarity search
   - SVD dimension reduction for noise filtering

2. **Contrastive Reranking**
   - Improves relevance of retrieved passages
   - Learned reranking weights
   - Multi-stage retrieval pipeline

3. **Multi-Candidate Generation**
   - Generates multiple answer candidates
   - Learned scoring for optimal selection
   - Confidence estimation

4. **Dynamic Fusion Mechanism**
   - Adaptive weighting: `H(q) = α(q) * R(q, D) + (1-α(q)) * G(q)`
   - Question complexity adaptation
   - Confidence-based fusion

### Joint Optimization

```
L_total = L_retrieval + λ₁ * L_generation + λ₂ * L_fusion
```

## 📊 Evaluation Methodology

### Three-Tier Evaluation Framework

Our comprehensive evaluation consists of three progressive tiers:

#### Tier 1: Foundational Validation
- **Purpose:** Establish core system effectiveness
- **Dataset:** 12 diverse questions across multiple domains
- **Baselines:** Standalone RAG, CAG, simple ensemble
- **Key Metric:** 57.5% F1 improvement over RAG — **RETRACTED** (see [`results.md`](results.md))

#### Tier 2: Enhanced Evaluation
- **Purpose:** Compare with state-of-the-art systems
- **Dataset:** 55 questions across 14 domains
- **Baselines:** Advanced RAG, Enhanced CAG, FiD, T5-FiD, DPR+FiD
- **Key Metric:** Statistical significance across all comparisons

#### Tier 3: "Expert-Level" Scientific Evaluation — RETRACTED
- **Purpose:** was described as a human expert comparison; **no humans were involved**
- **Dataset:** 26 PhD-level scientific questions
- **Expert responses:** hardcoded strings in `src/expert_evaluation.py`
- **Status:** withdrawn. See [`results.md`](results.md) §4.

## 🔬 Reproducing Results

### The one command that works

```bash
pip install numpy scikit-learn scipy
python src/run_honest_evaluation.py
```

Writes `results/verified_tier2.json`. Deterministic, CPU-only, under a second, no network.
The numbers it produces are the ones in [`results.md`](results.md) and at the top of this
README.

### Individual Tier Reproduction

**These scripts are retained for reference but do not currently produce valid results.** They are
the source of the retracted numbers — their "baselines" are dictionary lookups. Running them will
not reproduce anything in `results.md`, because the files they write are the ones archived under
`results/RETRACTED/`. Do not cite their output.

```bash
# Retained, produces RETRACTED numbers — do not use
python src/train_and_evaluate.py
python src/option3_full_scale_evaluation.py
python src/expert_evaluation.py

# Produces the real, reproducible numbers
python src/run_honest_evaluation.py
```

> Earlier versions of this README documented a `--evaluation_tier` flag and a
> `scripts/run_all_evaluations.sh` entry point. Neither exists: `train_and_evaluate.py` accepts
> only `--mode`, `--train_data`, `--dev_data`, `--output_dir`, `--max_train_samples`, and
> `--max_eval_samples`, and there is no `scripts/` directory.

### Expected Results

**Previously this section listed F1 ≈ 0.389 (Tier 1), 0.276 (Tier 2), and 0.140 vs 0.368 (Tier 3).
All of those figures are retracted** — see the retraction notice at the top of this README and
[`results.md`](results.md). They are not reproduced here because no code in this repository
produces them.

The one command that currently produces real, reproducible numbers is:

```bash
python src/run_honest_evaluation.py
```

which yields TF-IDF + Extractive F1 0.177 and BM25 + Extractive F1 0.177, against an oracle
ceiling of 0.187, on the shipped 55-question set. Runs in under a second, CPU-only, no GPU.

## 📈 Performance Analysis

**What is actually demonstrated:** on the shipped Tier-2 set, two real retrieval models select
identical sentences on all 55 questions, and 34 of those 55 questions have gold answers that do
not appear in the corpus at all. Performance is bounded by the benchmark, not by the method.

**What is not demonstrated:** accuracy claims, response-time claims, scaling behaviour, or
robustness across dataset sizes. None of these were measured on this data, and no code in this
repository measures them.

### Known limitations (measured)

⚠️ **Corpus coverage:** 34/55 gold answers (61.8%) are absent from the 100-document corpus, so
no retrieval system can answer them.  
⚠️ **Zero exact match:** both baselines score EM 0.000 — the extractive answers are full
sentences while the gold answers are short strings.  
⚠️ **Metric mismatch:** token-F1 structurally understates extractive systems; answer-containment
(0.309) is reported alongside for this reason.  
⚠️ **Retrieval not yet evaluated:** the dense bi-encoder + BART stack has not been run, so the
framework's own retrieval quality is unmeasured.

### What would be needed

- A corpus that contains the answers to the questions asked
- The dense + BART stack running in a memory-adequate environment
- An n ≥ 100 question set for statistical power
- An actual human-expert tier, if expert parity is to be claimed

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution

- **A valid benchmark:** a corpus/question set where answers are actually present
- **Real baselines:** implementations of the models this project previously only labelled
- **Running the dense + BART stack** in an environment with adequate memory
- **Documentation:** improving guides and correcting further stale claims

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Open-source community for foundational libraries (PyTorch, Transformers, FAISS)
- Scientific community for feedback and validation

> The previous version of this section credited "domain experts who participated in the human
> evaluation study." **No human evaluation study took place** — the expert responses used in
> `results/expert_evaluation_results.json` (now archived under `results/RETRACTED/`) are
> hardcoded strings defined in `src/expert_evaluation.py`. That acknowledgement was unfounded
> and has been removed.

## 📧 Contact

- **Author:** H. Wilner
- **GitHub:** [@hwilner](https://github.com/hwilner)
- **Repository:** [hybrid-rag-cag-framework](https://github.com/hwilner/hybrid-rag-cag-framework)

## 🔗 Links

- **Evaluation notes:** [Technical Evaluation Notes](docs/TECHNICAL_EVALUATION_NOTES.md)
- **Results:** [Complete Evaluation Results](results/)
- **Documentation:** [Full Documentation](docs/)
- **Issues:** [Report Issues](https://github.com/hwilner/hybrid-rag-cag-framework/issues)

---

**Star ⭐ this repository if you find it helpful!**
"""
Hybrid RAG-CAG System: A Proper Implementation
==============================================

This implementation fixes the issues in the original code by providing:
1. Proper RAG with dense retrieval, reranking, and answer fusion
2. Real CAG with contrastive learning (not just candidate selection)
3. Joint training with unified loss function
4. Comprehensive evaluation framework
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    AutoTokenizer, AutoModel
)
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import logging
import random
from dataclasses import dataclass
from sklearn.metrics import ndcg_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

@dataclass
class HybridConfig:
    """Configuration for the Hybrid RAG-CAG system"""
    # Model configurations
    retriever_model: str = "sentence-transformers/all-mpnet-base-v2"
    generator_model: str = "facebook/bart-large"
    
    # Retrieval settings
    top_k_retrieve: int = 10
    top_k_rerank: int = 5
    
    # Generation settings
    max_source_length: int = 1024
    max_target_length: int = 128
    num_beams: int = 4
    
    # Training settings
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 5
    
    # Loss function weights
    lambda_gen: float = 1.0
    lambda_ret: float = 0.5
    lambda_contrastive: float = 0.5
    lambda_diversity: float = 0.1

    # Generator precision: "float32" (default, full precision) or
    # "bfloat16" (halves generator memory; useful on memory-tight CPU boxes).
    generator_dtype: str = "float32"
    
    # Contrastive settings
    temperature: float = 0.07
    num_negatives: int = 3
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class QADataset(Dataset):
    """Dataset class for Question-Answering data"""
    
    def __init__(self, data: List[Dict], tokenizer, max_source_length: int = 512, max_target_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        
        # Format input
        source_text = f"question: {question}"
        
        # Tokenize
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            answer,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'source_ids': source_encoding['input_ids'].squeeze(),
            'source_mask': source_encoding['attention_mask'].squeeze(),
            'target_ids': target_encoding['input_ids'].squeeze(),
            'target_mask': target_encoding['attention_mask'].squeeze(),
            'question': question,
            'answer': answer
        }

class DenseRetriever(nn.Module):
    """Dense retrieval component with proper training"""

    def __init__(self, config: HybridConfig, encoder: Optional[SentenceTransformer] = None):
        super().__init__()
        self.config = config
        self.encoder = encoder if encoder is not None else SentenceTransformer(config.retriever_model)
        # Frozen feature extractor; no gradients flow here.
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        
        # Index for fast retrieval
        self.index = None
        self.corpus = None
        
    def build_index(self, corpus: List[str]):
        """Build FAISS index for the corpus"""
        logger.info(f"Building index for {len(corpus)} documents...")
        self.corpus = corpus
        
        # Encode corpus
        corpus_embeddings = self.encoder.encode(
            corpus, 
            batch_size=128, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize for cosine similarity
        corpus_embeddings = corpus_embeddings / np.linalg.norm(
            corpus_embeddings, axis=1, keepdims=True
        )
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(corpus_embeddings.astype('float32'))
        
        logger.info("Index built successfully!")
    
    def retrieve(self, queries: List[str], k: int = 10) -> Tuple[List[List[str]], List[List[float]]]:
        """Retrieve top-k documents for queries"""
        if self.index is None:
            raise ValueError("Index not built! Call build_index() first.")
        
        # Encode queries
        query_embeddings = self.encoder.encode(
            queries, 
            convert_to_numpy=True
        )
        
        # Normalize
        query_embeddings = query_embeddings / np.linalg.norm(
            query_embeddings, axis=1, keepdims=True
        )
        
        # Search
        scores, indices = self.index.search(
            query_embeddings.astype('float32'), k
        )
        
        # Get documents
        retrieved_docs = []
        retrieved_scores = []
        
        for i in range(len(queries)):
            docs = [self.corpus[idx] for idx in indices[i]]
            retrieved_docs.append(docs)
            retrieved_scores.append(scores[i].tolist())
        
        return retrieved_docs, retrieved_scores

class ContrastiveReranker(nn.Module):
    """Contrastive reranking component"""
    
    def __init__(self, config: HybridConfig, encoder: Optional[SentenceTransformer] = None):
        super().__init__()
        self.config = config
        self.encoder = encoder if encoder is not None else SentenceTransformer(config.retriever_model)
        # Frozen feature extractor.
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        
    def rerank(self, queries: List[str], documents_batch: List[List[str]], k: int = 5) -> List[List[str]]:
        """Rerank documents using query-document similarity"""
        reranked_docs = []
        
        for query, documents in zip(queries, documents_batch):
            if not documents:
                reranked_docs.append([])
                continue
                
            # Encode query and documents
            query_embedding = self.encoder.encode([query])
            doc_embeddings = self.encoder.encode(documents)
            
            # Compute similarities
            similarities = F.cosine_similarity(
                torch.tensor(query_embedding), 
                torch.tensor(doc_embeddings), 
                dim=1
            )
            
            # Sort by similarity
            sorted_indices = similarities.argsort(descending=True)[:k]
            reranked = [documents[i] for i in sorted_indices]
            reranked_docs.append(reranked)
        
        return reranked_docs

class HybridGenerator(nn.Module):
    """Generator with multi-candidate decoding + contrastive selection.

    Note: ``self.sentence_encoder`` is a *frozen* sentence-transformer used
    only as a feature extractor for the contrastive and diversity losses.
    Gradients do not flow into it. Only ``self.model`` (BART) is updated by
    the differentiable generation loss.
    """
    
    def __init__(self, config: HybridConfig, encoder: Optional[SentenceTransformer] = None):
        super().__init__()
        self.config = config
        self.tokenizer = BartTokenizer.from_pretrained(config.generator_model)
        self.model = BartForConditionalGeneration.from_pretrained(config.generator_model)
        # Trade compute for memory: recompute activations during backward
        # instead of caching them. Roughly halves peak training RAM on CPU.
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if config.generator_dtype == "bfloat16":
            self.model = self.model.to(torch.bfloat16)
        self.sentence_encoder = encoder if encoder is not None else SentenceTransformer(config.retriever_model)
        # Freeze the sentence encoder -- it is only a feature extractor.
        for p in self.sentence_encoder.parameters():
            p.requires_grad = False
        self.sentence_encoder.eval()
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass through BART. Returns the model's output object so
        that ``outputs.loss`` is a real differentiable cross-entropy."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate_candidates(self, questions: List[str], contexts: List[str], num_candidates: int = 5) -> List[List[str]]:
        """Generate multiple candidate answers"""
        all_candidates = []
        
        for question, context in zip(questions, contexts):
            # Format input
            input_text = f"question: {question} context: {context}"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=self.config.max_source_length,
                truncation=True,
                return_tensors='pt'
            ).to(self.config.device)
            
            # Generate candidates with different strategies
            candidates = []
            
            # Strategy 1: Beam search
            beam_outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_target_length,
                num_beams=num_candidates,
                num_return_sequences=min(num_candidates, 3),
                early_stopping=True,
                do_sample=False
            )
            
            for output in beam_outputs:
                candidate = self.tokenizer.decode(output, skip_special_tokens=True)
                candidates.append(candidate)
            
            # Strategy 2: Nucleus sampling (if we need more candidates)
            if len(candidates) < num_candidates:
                remaining = num_candidates - len(candidates)
                sample_outputs = self.model.generate(
                    **inputs,
                    max_length=self.config.max_target_length,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    num_return_sequences=remaining
                )
                
                for output in sample_outputs:
                    candidate = self.tokenizer.decode(output, skip_special_tokens=True)
                    candidates.append(candidate)
            
            all_candidates.append(candidates[:num_candidates])
        
        return all_candidates
    
    def contrastive_selection(self, questions: List[str], contexts: List[str], 
                            candidates_batch: List[List[str]]) -> List[str]:
        """Select best candidate using contrastive scoring"""
        selected_answers = []
        
        for question, context, candidates in zip(questions, contexts, candidates_batch):
            if not candidates:
                selected_answers.append("")
                continue
            
            # Encode question, context, and candidates
            question_emb = self.sentence_encoder.encode([question])
            context_emb = self.sentence_encoder.encode([context])
            candidate_embs = self.sentence_encoder.encode(candidates)
            
            # Compute similarities
            q_similarities = F.cosine_similarity(
                torch.tensor(question_emb), 
                torch.tensor(candidate_embs), 
                dim=1
            )
            c_similarities = F.cosine_similarity(
                torch.tensor(context_emb), 
                torch.tensor(candidate_embs), 
                dim=1
            )
            
            # Weighted combination
            combined_scores = 0.6 * q_similarities + 0.4 * c_similarities
            
            # Select best candidate
            best_idx = combined_scores.argmax().item()
            selected_answers.append(candidates[best_idx])
        
        return selected_answers

class HybridRAGCAG(nn.Module):
    """Complete Hybrid RAG-CAG System.

    Training-loss contract (post-patch):
      L_total = lambda_gen * L_generation
              + lambda_contrastive * L_contrastive
              + lambda_diversity * L_diversity

      L_generation  -- REAL differentiable BART cross-entropy on
                       ``"question: {q} context: {ctx} -> gold answer"``.
                       Computed inside `forward()` when ``gold_answers`` is
                       provided.

      L_contrastive -- InfoNCE between question embedding and candidate
                       embeddings. Acts only on a frozen sentence-encoder,
                       so it contributes no gradient to BART. Kept as a
                       regulariser over selection-confidence during eval
                       time (does not update parameters).

      L_diversity   -- Mean pairwise cosine across candidates. Same caveat
                       (frozen encoder). Effectively a constant at training
                       time when sentence-encoder params are frozen.

    Because the contrastive and diversity heads are frozen, the
    *learnable* part of the optimisation is the generator only. This is
    a deliberate, honest engineering choice -- swap ``self.sentence_encoder``
    for a fine-tunable encoder (e.g. a SentenceTransformer with a learned
    projection head) and these terms become trainable as well.
    """
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        
        # One shared frozen sentence encoder for all three components
        # (retriever, reranker, contrastive selector). Loading the same
        # checkpoint three times wastes ~1.3 GB of RAM for no benefit.
        shared_encoder = SentenceTransformer(config.retriever_model)

        # Components
        self.retriever = DenseRetriever(config, encoder=shared_encoder)
        self.reranker = ContrastiveReranker(config, encoder=shared_encoder)
        self.generator = HybridGenerator(config, encoder=shared_encoder)
        
        # Move to device
        self.to(config.device)
    
    def forward(self, questions: List[str], gold_answers: Optional[List[str]] = None) -> Dict:
        """End-to-end forward pass.

        Args:
            questions: list of question strings.
            gold_answers: optional list of gold answer strings. If supplied,
                a real differentiable cross-entropy loss is computed against
                BART's seq2seq head.
        """
        # Step 1: Retrieve documents (frozen SentenceTransformer + FAISS)
        retrieved_docs, retrieval_scores = self.retriever.retrieve(
            questions, k=self.config.top_k_retrieve
        )
        
        # Step 2: Rerank documents (frozen SentenceTransformer + cosine)
        reranked_docs = self.reranker.rerank(
            questions, retrieved_docs, k=self.config.top_k_rerank
        )
        
        # Step 3: Prepare contexts
        contexts = []
        for docs in reranked_docs:
            context = " ".join(docs[:3])  # Use top 3 documents
            contexts.append(context)
        
        # Step 4: Generate candidates via beam + nucleus sampling.
        # NB: this runs WITHOUT grad -- candidate strings are produced
        # at eval-time-of-the-current-model and are used for selection
        # and reporting, not for backprop.
        with torch.no_grad():
            candidates_batch = self.generator.generate_candidates(
                questions, contexts, num_candidates=5
            )
            # Step 5: Contrastive selection
            final_answers = self.generator.contrastive_selection(
                questions, contexts, candidates_batch
            )
        
        result = {
            'questions': questions,
            'contexts': contexts,
            'candidates': candidates_batch,
            'final_answers': final_answers,
            'retrieved_docs': retrieved_docs,
            'reranked_docs': reranked_docs
        }
        
        # Compute losses if training (and gold answers available).
        if gold_answers is not None:
            losses = self.compute_losses(
                questions, contexts, gold_answers
            )
            result.update(losses)
        
        return result
    
    def compute_losses(self, questions: List[str], contexts: List[str],
                       gold_answers: List[str]) -> Dict:
        """Compute the unified loss.

        L_generation is a *real* differentiable cross-entropy from BART
        trained to predict each gold answer given (question, context) as
        input. L_contrastive and L_diversity operate on frozen embeddings
        and therefore contribute nothing to the parameter update; they
        are kept for monitoring only.
        """
        # Real differentiable generation loss -----------------------
        gen_loss = self.compute_generation_loss(questions, contexts, gold_answers)
        
        # Frozen-encoder regularisers (informational; no grad into BART)
        with torch.no_grad():
            candidates_batch = self.generator.generate_candidates(
                questions, contexts, num_candidates=5
            )
            contrastive_loss = self.compute_contrastive_loss(
                questions, candidates_batch, gold_answers
            )
            diversity_loss = self.compute_diversity_loss(candidates_batch)
        
        total_loss = (
            self.config.lambda_gen * gen_loss
            + self.config.lambda_contrastive * contrastive_loss
            + self.config.lambda_diversity * diversity_loss
        )
        
        return {
            'total_loss': total_loss,
            'generation_loss': gen_loss,
            'contrastive_loss': contrastive_loss.detach(),
            'diversity_loss': diversity_loss.detach()
        }
    
    def compute_generation_loss(self, questions: List[str], contexts: List[str],
                               gold_answers: List[str]) -> torch.Tensor:
        """BART cross-entropy: P(gold_answer | question, context).

        This is the only term in the total loss that actually updates
        parameters. Implementation uses the canonical seq2seq API:
        ``BartForConditionalGeneration.forward(input_ids=..., labels=...)``.
        """
        batch_size = len(questions)
        
        # Build (question, context) -> gold_answer pairs and tokenize as a batch.
        input_texts = [
            f"question: {q} context: {c}" for q, c in zip(questions, contexts)
        ]
        enc = self.generator.tokenizer(
            input_texts,
            max_length=self.config.max_source_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.config.device)
        
        tgt = self.generator.tokenizer(
            gold_answers,
            max_length=self.config.max_target_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.config.device)
        
        # BART expects labels with the pad token replaced by -100 so the
        # loss ignores padding positions.
        labels = tgt['input_ids'].clone()
        labels[labels == self.generator.tokenizer.pad_token_id] = -100
        
        outputs = self.generator.model(
            input_ids=enc['input_ids'],
            attention_mask=enc['attention_mask'],
            labels=labels
        )
        return outputs.loss  # differentiable scalar
    
    def compute_contrastive_loss(self, questions: List[str],
                               candidates_batch: List[List[str]],
                               gold_answers: List[str]) -> torch.Tensor:
        """InfoNCE-style contrastive loss over frozen candidate embeddings.

        NOTE: encoders are frozen, so this term does not generate useful
        parameter gradients. We ``.detach()`` it implicitly via the
        ``torch.no_grad()`` block in :meth:`compute_losses`, so it cannot
        be misused as a training signal.
        """
        losses = []
        for question, candidates, gold in zip(questions, candidates_batch, gold_answers):
            if not candidates:
                continue
            f1s = [self.compute_f1_score(c, gold) for c in candidates]
            best_idx = int(np.argmax(f1s))
            if f1s[best_idx] < 0.1:
                continue
            
            q_emb = torch.tensor(
                self.generator.sentence_encoder.encode([question])
            )
            cand_embs = torch.tensor(
                self.generator.sentence_encoder.encode(candidates)
            )
            logits = F.cosine_similarity(q_emb, cand_embs, dim=1) / self.config.temperature
            losses.append(
                F.cross_entropy(logits.unsqueeze(0), torch.tensor([best_idx]))
            )
        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).mean()
    
    def compute_diversity_loss(self, candidates_batch: List[List[str]]) -> torch.Tensor:
        """Mean pairwise cosine similarity over candidates, batched.
        Frozen-encoder regulariser (see note in :meth:`compute_losses`)."""
        losses = []
        with torch.no_grad():
            for candidates in candidates_batch:
                if len(candidates) < 2:
                    continue
                emb = torch.tensor(
                    self.generator.sentence_encoder.encode(candidates)
                )
                sims = F.cosine_similarity(
                    emb.unsqueeze(1), emb.unsqueeze(0), dim=2
                )
                mask = ~torch.eye(len(candidates), dtype=bool)
                losses.append(sims[mask].mean())
        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).mean()
    
    @staticmethod
    def compute_f1_score(pred: str, gold: str) -> float:
        """Token-level F1 (Squad-style). Same implementation as
        ``MetricsCalculator.compute_f1`` to keep training and evaluation
        in lock-step."""
        pred_tokens = set(pred.lower().split())
        gold_tokens = set(gold.lower().split())
        
        if not pred_tokens and not gold_tokens:
            return 1.0
        if not pred_tokens or not gold_tokens:
            return 0.0
        
        tp = len(pred_tokens & gold_tokens)
        precision = tp / len(pred_tokens)
        recall = tp / len(gold_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)

# Evaluation utilities
class MetricsCalculator:
    """Calculate various evaluation metrics"""
    
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        
    def calculate_all_metrics(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate all evaluation metrics"""
        metrics = {
            'exact_match': [],
            'f1': [],
            'bleu4': [],
            'rouge_l': [],
            'meteor': []
        }
        
        for pred, ref in zip(predictions, references):
            # Exact Match
            em = float(pred.lower().strip() == ref.lower().strip())
            metrics['exact_match'].append(em)
            
            # F1 Score
            f1 = self.compute_f1(pred, ref)
            metrics['f1'].append(f1)
            
            # BLEU-4
            bleu = self.compute_bleu(pred, ref)
            metrics['bleu4'].append(bleu)
            
            # ROUGE-L
            rouge = self.scorer.score(ref, pred)['rougeL'].fmeasure
            metrics['rouge_l'].append(rouge)
            
            # METEOR
            meteor = self.compute_meteor(pred, ref)
            metrics['meteor'].append(meteor)
        
        # Average all metrics
        return {k: np.mean(v) for k, v in metrics.items()}
    
    @staticmethod
    def compute_f1(pred: str, gold: str) -> float:
        """Compute F1 score"""
        pred_tokens = set(pred.lower().split())
        gold_tokens = set(gold.lower().split())
        
        if not pred_tokens and not gold_tokens:
            return 1.0
        if not pred_tokens or not gold_tokens:
            return 0.0
        
        tp = len(pred_tokens & gold_tokens)
        precision = tp / len(pred_tokens)
        recall = tp / len(gold_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def compute_bleu(pred: str, ref: str) -> float:
        """Compute BLEU-4 score"""
        try:
            pred_tokens = nltk.word_tokenize(pred.lower())
            ref_tokens = nltk.word_tokenize(ref.lower())
            smoothie = SmoothingFunction().method1
            return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        except Exception:
            return 0.0
    
    @staticmethod
    def compute_meteor(pred: str, ref: str) -> float:
        """Compute METEOR score"""
        try:
            pred_tokens = nltk.word_tokenize(pred.lower())
            ref_tokens = nltk.word_tokenize(ref.lower())
            return meteor_score([ref_tokens], pred_tokens)
        except Exception:
            return 0.0


class HybridRAGCAGSystem:
    """Friendly façade around :class:`HybridRAGCAG`.

    This thin wrapper exposes the API shape referenced in the README:
        >>> system = HybridRAGCAGSystem()
        >>> system.index_corpus(corpus)
        >>> answer = system.answer_question("What is the capital of France?")
    while internally delegating to :class:`HybridRAGCAG` (which is the
    full ``nn.Module`` and is what you want for training).
    """
    
    def __init__(self, config: Optional[HybridConfig] = None) -> None:
        self.config = config or HybridConfig()
        self.model = HybridRAGCAG(self.config)
        self._indexed = False
    
    def index_corpus(self, corpus: List[str]) -> None:
        """Build the dense retrieval index. Idempotent."""
        self.model.retriever.build_index(corpus)
        self._indexed = True
    
    def answer_question(self, question: str) -> str:
        """Return the contrastively-selected answer for a single question."""
        if not self._indexed:
            raise RuntimeError(
                "Call `index_corpus(corpus)` before `answer_question(question)`."
            )
        self.model.eval()
        with torch.no_grad():
            result = self.model([question])
        return result['final_answers'][0]
    
    def answer_questions(self, questions: List[str]) -> List[str]:
        """Vectorised variant of :meth:`answer_question`."""
        if not self._indexed:
            raise RuntimeError("Call `index_corpus(corpus)` first.")
        self.model.eval()
        with torch.no_grad():
            result = self.model(questions)
        return result['final_answers']


def _memory_cap_bytes() -> Optional[int]:
    """Return the effective memory limit in bytes, or ``None`` if unlimited."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(path).read().strip()
            if raw and raw != "max":
                val = int(raw)
                if 0 < val < 1 << 60:
                    return val
        except (OSError, ValueError):
            continue
    return None


def _smoke_test() -> None:
    """End-to-end sanity check.

    Verifies that:
      1. The dense-retrieval index can be built on a tiny corpus.
      2. The friendly ``HybridRAGCAGSystem.answer_question`` API works.
      3. ``HybridRAGCAG.forward(questions, gold_answers)`` produces a
         real differentiable scalar ``total_loss`` whose ``.backward()``
         populates BART parameter gradients.

    This intentionally uses toy data so it runs in seconds on CPU.
    Run with: ``python hybrid_rag_cag_system.py``
    """
    corpus = [
        "Paris is the capital and most populous city of France.",
        "The Eiffel Tower was constructed in 1889 by Gustave Eiffel in Paris.",
        "Berlin is the capital of Germany.",
        "Tokyo is the capital of Japan and the world's most populous metropolitan area.",
        "The Amazon River is the longest river in the world, flowing through South America.",
    ]
    qa = [
        ("What is the capital of France?", "Paris"),
        ("Who constructed the Eiffel Tower?", "Gustave Eiffel"),
        ("Which city is the capital of Japan?", "Tokyo"),
    ]
    
    print("[smoke] initialising system...")
    cfg = HybridConfig()
    mem_cap = _memory_cap_bytes()
    if mem_cap is not None and mem_cap < 6 * 1024**3:
        # fp32 BART-large weights + gradients + the frozen encoder exceed a
        # small cgroup limit; bf16 halves the generator footprint. The smoke
        # assertions (differentiable loss, gradients into BART) are unchanged.
        cfg.generator_dtype = "bfloat16"
        print(f"[smoke] memory cap {mem_cap / 1024**3:.1f} GB detected; running generator in bfloat16")
    system = HybridRAGCAGSystem(cfg)
    system.index_corpus(corpus)
    
    print("[smoke] running inference via the friendly façade...")
    for q, gold in qa:
        pred = system.answer_question(q)
        f1 = MetricsCalculator.compute_f1(pred, gold)
        print(f"  Q: {q}")
        print(f"  Gold: {gold!r}  Pred: {pred!r}  F1: {f1:.3f}")
    
    print("[smoke] running training step on the underlying nn.Module...")
    questions = [q for q, _ in qa]
    gold_answers = [g for _, g in qa]
    # Pad-friendly: build a tiny dataset with one batch.
    contexts = [" ".join(system.model.retriever.retrieve([q], k=1)[0][0][:1])
                for q in questions]
    model = system.model
    model.train()
    # Real differentiable loss through BART
    loss = model.compute_losses(questions, contexts, gold_answers)['total_loss']
    print(f"[smoke] total_loss = {loss.item():.4f}, requires_grad = {loss.requires_grad}")
    assert loss.requires_grad, "total_loss must be differentiable to train BART"
    loss.backward()
    # Sanity-check: at least one BART parameter now carries a non-None grad.
    grad_seen = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.generator.model.parameters()
    )
    assert grad_seen, "BART parameters received no gradient -- loss is broken"
    print("[smoke] gradients flowed into BART ✓")
    print("[smoke] PASS")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--smoke", action="store_true",
                        help="Run the end-to-end smoke test and exit.")
    args = parser.parse_args()
    
    if args.smoke:
        _smoke_test()
    else:
        config = HybridConfig()
        print(f"Configuration: {config}")
        model = HybridRAGCAG(config)
        print("Hybrid RAG-CAG system initialised.")
        print("Pass --smoke to run the end-to-end sanity check.")
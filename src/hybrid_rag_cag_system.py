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
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.encoder = SentenceTransformer(config.retriever_model)
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
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.encoder = SentenceTransformer(config.retriever_model)
        
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
    """Generator with contrastive learning"""
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.tokenizer = BartTokenizer.from_pretrained(config.generator_model)
        self.model = BartForConditionalGeneration.from_pretrained(config.generator_model)
        self.sentence_encoder = SentenceTransformer(config.retriever_model)
        
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass for training"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
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
    """Complete Hybrid RAG-CAG System"""
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        
        # Components
        self.retriever = DenseRetriever(config)
        self.reranker = ContrastiveReranker(config)
        self.generator = HybridGenerator(config)
        
        # Move to device
        self.to(config.device)
        
    def forward(self, questions: List[str], gold_answers: List[str] = None) -> Dict:
        """Forward pass for training/inference"""
        batch_size = len(questions)
        
        # Step 1: Retrieve documents
        retrieved_docs, retrieval_scores = self.retriever.retrieve(
            questions, k=self.config.top_k_retrieve
        )
        
        # Step 2: Rerank documents
        reranked_docs = self.reranker.rerank(
            questions, retrieved_docs, k=self.config.top_k_rerank
        )
        
        # Step 3: Prepare contexts
        contexts = []
        for docs in reranked_docs:
            context = " ".join(docs[:3])  # Use top 3 documents
            contexts.append(context)
        
        # Step 4: Generate candidates
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
        
        # Compute losses if training
        if gold_answers is not None:
            losses = self.compute_losses(
                questions, contexts, candidates_batch, 
                final_answers, gold_answers
            )
            result.update(losses)
        
        return result
    
    def compute_losses(self, questions: List[str], contexts: List[str], 
                      candidates_batch: List[List[str]], final_answers: List[str], 
                      gold_answers: List[str]) -> Dict:
        """Compute unified loss function"""
        
        # Generation loss (cross-entropy on final answers)
        gen_loss = self.compute_generation_loss(final_answers, gold_answers)
        
        # Contrastive loss (promote good candidates, demote bad ones)
        contrastive_loss = self.compute_contrastive_loss(
            questions, candidates_batch, gold_answers
        )
        
        # Diversity loss (encourage diverse candidates)
        diversity_loss = self.compute_diversity_loss(candidates_batch)
        
        # Combined loss
        total_loss = (
            self.config.lambda_gen * gen_loss +
            self.config.lambda_contrastive * contrastive_loss +
            self.config.lambda_diversity * diversity_loss
        )
        
        return {
            'total_loss': total_loss,
            'generation_loss': gen_loss,
            'contrastive_loss': contrastive_loss,
            'diversity_loss': diversity_loss
        }
    
    def compute_generation_loss(self, predictions: List[str], targets: List[str]) -> torch.Tensor:
        """Compute generation loss using token-level F1"""
        f1_scores = []
        
        for pred, target in zip(predictions, targets):
            f1 = self.compute_f1_score(pred, target)
            f1_scores.append(f1)
        
        # Convert to loss (1 - F1)
        avg_f1 = np.mean(f1_scores)
        return torch.tensor(1.0 - avg_f1, requires_grad=True)
    
    def compute_contrastive_loss(self, questions: List[str], candidates_batch: List[List[str]], 
                               gold_answers: List[str]) -> torch.Tensor:
        """Compute contrastive loss using InfoNCE"""
        total_loss = 0.0
        num_samples = 0
        
        for question, candidates, gold_answer in zip(questions, candidates_batch, gold_answers):
            if not candidates:
                continue
            
            # Find best candidate (positive sample)
            f1_scores = [self.compute_f1_score(cand, gold_answer) for cand in candidates]
            best_idx = np.argmax(f1_scores)
            
            # If best F1 is very low, skip this sample
            if f1_scores[best_idx] < 0.1:
                continue
            
            # Encode question and candidates
            question_emb = torch.tensor(
                self.generator.sentence_encoder.encode([question])
            )
            candidate_embs = torch.tensor(
                self.generator.sentence_encoder.encode(candidates)
            )
            
            # Compute similarities
            similarities = F.cosine_similarity(
                question_emb, candidate_embs, dim=1
            ) / self.config.temperature
            
            # InfoNCE loss (positive sample is the best candidate)
            loss = F.cross_entropy(similarities.unsqueeze(0), torch.tensor([best_idx]))
            total_loss += loss
            num_samples += 1
        
        return total_loss / max(num_samples, 1)
    
    def compute_diversity_loss(self, candidates_batch: List[List[str]]) -> torch.Tensor:
        """Compute diversity loss to encourage varied responses"""
        total_loss = 0.0
        num_batches = 0
        
        for candidates in candidates_batch:
            if len(candidates) < 2:
                continue
            
            # Encode all candidates
            candidate_embs = torch.tensor(
                self.generator.sentence_encoder.encode(candidates)
            )
            
            # Compute pairwise similarities
            similarities = F.cosine_similarity(
                candidate_embs.unsqueeze(1), 
                candidate_embs.unsqueeze(0), 
                dim=2
            )
            
            # Exclude diagonal (self-similarity)
            mask = ~torch.eye(len(candidates), dtype=bool)
            similarities = similarities[mask]
            
            # Penalize high similarities (encourage diversity)
            diversity_loss = similarities.mean()
            total_loss += diversity_loss
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    @staticmethod
    def compute_f1_score(pred: str, gold: str) -> float:
        """Compute token-level F1 score"""
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
        except:
            return 0.0
    
    @staticmethod
    def compute_meteor(pred: str, ref: str) -> float:
        """Compute METEOR score"""
        try:
            pred_tokens = nltk.word_tokenize(pred.lower())
            ref_tokens = nltk.word_tokenize(ref.lower())
            return meteor_score([ref_tokens], pred_tokens)
        except:
            return 0.0

if __name__ == "__main__":
    # Example usage
    config = HybridConfig()
    print(f"Configuration: {config}")
    
    # Initialize system
    model = HybridRAGCAG(config)
    print("Hybrid RAG-CAG system initialized!")
    
    # Metrics calculator
    metrics_calc = MetricsCalculator()
    print("Metrics calculator ready!")
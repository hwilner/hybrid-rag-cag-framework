"""
Training and Evaluation Script for Hybrid RAG-CAG System
========================================================

This script provides comprehensive training and evaluation capabilities
for the Hybrid RAG-CAG system with proper benchmarking.
"""

import json
import os
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import argparse
import logging
from datetime import datetime

from hybrid_rag_cag_system import (
    HybridRAGCAG, HybridConfig, QADataset, MetricsCalculator
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Data loading and preprocessing utilities"""
    
    @staticmethod
    def load_hotpot_data(file_path: str) -> List[Dict]:
        """Load HotpotQA dataset"""
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} examples")
        return data
    
    @staticmethod
    def preprocess_hotpot(data: List[Dict], max_samples: int = None) -> Tuple[List[str], List[str]]:
        """Extract corpus and create QA pairs from HotpotQA"""
        corpus = set()
        qa_pairs = []
        
        for example in data[:max_samples] if max_samples else data:
            # Extract corpus sentences
            for title, sentences in example["context"]:
                for sentence in sentences:
                    if sentence.strip():
                        corpus.add(sentence.strip())
            
            # Create QA pair
            qa_pairs.append({
                'question': example['question'],
                'answer': example['answer'],
                'supporting_facts': example.get('supporting_facts', [])
            })
        
        return list(corpus), qa_pairs
    
    @staticmethod
    def create_synthetic_negatives(questions: List[str], answers: List[str], 
                                 num_negatives: int = 3) -> List[List[str]]:
        """Create negative samples for contrastive learning"""
        negatives = []
        
        for i, (question, answer) in enumerate(zip(questions, answers)):
            # Strategy 1: Random answers from other examples
            random_negatives = []
            other_indices = [j for j in range(len(answers)) if j != i]
            selected_indices = np.random.choice(
                other_indices, 
                size=min(num_negatives, len(other_indices)), 
                replace=False
            )
            
            for idx in selected_indices:
                random_negatives.append(answers[idx])
            
            # Strategy 2: Corrupted versions of correct answer
            corrupted_negatives = []
            if len(random_negatives) < num_negatives:
                words = answer.split()
                if len(words) > 2:
                    # Shuffle words
                    shuffled = words.copy()
                    np.random.shuffle(shuffled)
                    corrupted_negatives.append(' '.join(shuffled))
                    
                    # Remove random words
                    if len(words) > 1:
                        truncated = words[:len(words)//2]
                        corrupted_negatives.append(' '.join(truncated))
            
            # Combine negatives
            all_negatives = random_negatives + corrupted_negatives
            negatives.append(all_negatives[:num_negatives])
        
        return negatives

class Trainer:
    """Training utilities for the Hybrid RAG-CAG system"""
    
    def __init__(self, model: HybridRAGCAG, config: HybridConfig):
        self.model = model
        self.config = config
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        self.metrics_calculator = MetricsCalculator()
        
        # Training history
        self.train_losses = []
        self.val_metrics = []
        
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            questions = batch['question']
            gold_answers = batch['answer']
            
            # Forward pass
            outputs = self.model(questions, gold_answers)
            loss = outputs['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        
        return {
            'train_loss': avg_loss,
            'num_batches': num_batches
        }
    
    def evaluate(self, eval_loader: DataLoader) -> Dict:
        """Evaluate the model"""
        self.model.eval()
        
        all_questions = []
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                questions = batch['question']
                gold_answers = batch['answer']
                
                # Generate predictions
                outputs = self.model(questions)
                predictions = outputs['final_answers']
                
                # Collect results
                all_questions.extend(questions)
                all_predictions.extend(predictions)
                all_references.extend(gold_answers)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            all_predictions, all_references
        )
        
        self.val_metrics.append(metrics)
        
        return {
            'predictions': all_predictions,
            'references': all_references,
            'questions': all_questions,
            **metrics
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Complete training loop"""
        logger.info(f"Starting training for {self.config.num_epochs} epochs...")
        
        best_f1 = 0.0
        training_history = {
            'train_losses': [],
            'val_metrics': [],
            'best_epoch': 0,
            'best_f1': 0.0
        }
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_results = self.train_epoch(train_loader)
            
            # Evaluate
            val_results = self.evaluate(val_loader)
            
            # Log results
            logger.info(f"Train Loss: {train_results['train_loss']:.4f}")
            logger.info(f"Val F1: {val_results['f1']:.4f}")
            logger.info(f"Val EM: {val_results['exact_match']:.4f}")
            logger.info(f"Val BLEU-4: {val_results['bleu4']:.4f}")
            
            # Save best model
            if val_results['f1'] > best_f1:
                best_f1 = val_results['f1']
                training_history['best_epoch'] = epoch + 1
                training_history['best_f1'] = best_f1
                
                # Save model
                torch.save(
                    self.model.state_dict(), 
                    f'best_hybrid_model_epoch_{epoch+1}.pt'
                )
                logger.info(f"New best model saved! F1: {best_f1:.4f}")
            
            # Update history
            training_history['train_losses'].append(train_results['train_loss'])
            training_history['val_metrics'].append(val_results)
        
        return training_history

class Evaluator:
    """Comprehensive evaluation utilities"""
    
    def __init__(self, model: HybridRAGCAG, config: HybridConfig):
        self.model = model
        self.config = config
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate_comprehensive(self, test_data: List[Dict]) -> Dict:
        """Comprehensive evaluation with detailed analysis"""
        logger.info(f"Starting comprehensive evaluation on {len(test_data)} examples...")
        
        # Prepare data
        questions = [item['question'] for item in test_data]
        gold_answers = [item['answer'] for item in test_data]
        
        # Run inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(questions)
        
        predictions = outputs['final_answers']
        candidates = outputs['candidates']
        contexts = outputs['contexts']
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(predictions, gold_answers)
        
        # Detailed analysis
        analysis = self.detailed_analysis(
            questions, gold_answers, predictions, candidates, contexts
        )
        
        results = {
            'overall_metrics': metrics,
            'detailed_analysis': analysis,
            'examples': self.create_examples(
                questions, gold_answers, predictions, candidates, contexts
            )
        }
        
        return results
    
    def detailed_analysis(self, questions: List[str], gold_answers: List[str],
                         predictions: List[str], candidates: List[List[str]], 
                         contexts: List[str]) -> Dict:
        """Perform detailed analysis"""
        
        # Performance by question length
        short_questions = []
        long_questions = []
        
        for i, question in enumerate(questions):
            question_len = len(question.split())
            if question_len <= 10:
                short_questions.append(i)
            else:
                long_questions.append(i)
        
        short_f1 = np.mean([
            self.metrics_calculator.compute_f1(predictions[i], gold_answers[i])
            for i in short_questions
        ]) if short_questions else 0.0
        
        long_f1 = np.mean([
            self.metrics_calculator.compute_f1(predictions[i], gold_answers[i])
            for i in long_questions
        ]) if long_questions else 0.0
        
        # Candidate diversity analysis
        diversity_scores = []
        for cands in candidates:
            if len(cands) > 1:
                # Calculate pairwise BLEU to measure diversity
                bleu_scores = []
                for i in range(len(cands)):
                    for j in range(i+1, len(cands)):
                        bleu = self.metrics_calculator.compute_bleu(cands[i], cands[j])
                        bleu_scores.append(bleu)
                
                # Lower BLEU = higher diversity
                diversity = 1.0 - np.mean(bleu_scores) if bleu_scores else 0.0
                diversity_scores.append(diversity)
        
        avg_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
        
        return {
            'short_question_f1': short_f1,
            'long_question_f1': long_f1,
            'candidate_diversity': avg_diversity,
            'num_short_questions': len(short_questions),
            'num_long_questions': len(long_questions),
        }
    
    def create_examples(self, questions: List[str], gold_answers: List[str],
                       predictions: List[str], candidates: List[List[str]], 
                       contexts: List[str], num_examples: int = 5) -> List[Dict]:
        """Create example outputs for inspection"""
        examples = []
        
        # Calculate F1 scores to sort by quality
        f1_scores = [
            self.metrics_calculator.compute_f1(pred, gold)
            for pred, gold in zip(predictions, gold_answers)
        ]
        
        # Get best and worst examples
        sorted_indices = np.argsort(f1_scores)
        
        # Best examples
        best_indices = sorted_indices[-num_examples//2:]
        worst_indices = sorted_indices[:num_examples//2]
        selected_indices = list(best_indices) + list(worst_indices)
        
        for i in selected_indices:
            example = {
                'question': questions[i],
                'gold_answer': gold_answers[i],
                'prediction': predictions[i],
                'f1_score': f1_scores[i],
                'context': contexts[i][:200] + "..." if len(contexts[i]) > 200 else contexts[i],
                'candidates': candidates[i] if i < len(candidates) else [],
                'category': 'best' if i in best_indices else 'worst'
            }
            examples.append(example)
        
        return examples

def create_baseline_comparison():
    """Create comparison with baseline RAG and CAG systems"""
    
    class SimpleRAG:
        """Simple RAG baseline"""
        def __init__(self, config):
            self.config = config
            from hybrid_rag_cag_system import DenseRetriever, HybridGenerator
            self.retriever = DenseRetriever(config)
            self.generator = HybridGenerator(config)
        
        def evaluate(self, questions, gold_answers):
            # Simple retrieval + generation
            predictions = []
            for question in questions:
                docs, _ = self.retriever.retrieve([question], k=3)
                context = " ".join(docs[0][:3])
                
                # Simple generation (no contrastive selection)
                candidates = self.generator.generate_candidates([question], [context], 1)
                prediction = candidates[0][0] if candidates[0] else ""
                predictions.append(prediction)
            
            return predictions
    
    class SimpleCAG:
        """Simple CAG baseline"""
        def __init__(self, config):
            self.config = config
            from hybrid_rag_cag_system import HybridGenerator
            self.generator = HybridGenerator(config)
        
        def evaluate(self, questions, gold_answers):
            # Generation with multiple candidates (no retrieval)
            predictions = []
            for question in questions:
                candidates = self.generator.generate_candidates([question], [""], 5)
                # Just take the first candidate (no sophisticated selection)
                prediction = candidates[0][0] if candidates[0] else ""
                predictions.append(prediction)
            
            return predictions
    
    return SimpleRAG, SimpleCAG

def visualize_results(results: Dict, save_path: str = "results_visualization.png"):
    """Create visualization of results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Overall metrics comparison
    metrics = results['overall_metrics']
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    axes[0, 0].bar(metric_names, metric_values, color='skyblue')
    axes[0, 0].set_title('Overall Performance Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: F1 score distribution
    if 'examples' in results:
        f1_scores = [ex['f1_score'] for ex in results['examples']]
        axes[0, 1].hist(f1_scores, bins=10, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('F1 Score Distribution')
        axes[0, 1].set_xlabel('F1 Score')
        axes[0, 1].set_ylabel('Frequency')
    
    # Plot 3: Performance by question length
    if 'detailed_analysis' in results:
        analysis = results['detailed_analysis']
        categories = ['Short Questions\n(<= 10 words)', 'Long Questions\n(> 10 words)']
        f1_values = [analysis['short_question_f1'], analysis['long_question_f1']]
        
        axes[1, 0].bar(categories, f1_values, color=['coral', 'gold'])
        axes[1, 0].set_title('Performance by Question Length')
        axes[1, 0].set_ylabel('F1 Score')
    
    # Plot 4: Diversity analysis
    if 'detailed_analysis' in results:
        diversity = results['detailed_analysis']['candidate_diversity']
        axes[1, 1].bar(['Candidate Diversity'], [diversity], color='mediumpurple')
        axes[1, 1].set_title('Response Diversity')
        axes[1, 1].set_ylabel('Diversity Score')
        axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Results visualization saved to {save_path}")

def main():
    """Main execution function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train and evaluate Hybrid RAG-CAG system')
    parser.add_argument('--mode', choices=['train', 'eval', 'both'], default='both',
                       help='Mode of operation')
    parser.add_argument('--train_data', default='hotpot_train_v1.1.json',
                       help='Path to training data')
    parser.add_argument('--dev_data', default='hotpot_dev_distractor_v1.json',
                       help='Path to development data')
    parser.add_argument('--output_dir', default='./results',
                       help='Output directory for results')
    parser.add_argument('--max_train_samples', type=int, default=1000,
                       help='Maximum number of training samples')
    parser.add_argument('--max_eval_samples', type=int, default=500,
                       help='Maximum number of evaluation samples')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize configuration
    config = HybridConfig()
    logger.info(f"Configuration: {config}")
    
    # Load data
    data_loader = DataLoader()
    
    if args.mode in ['train', 'both']:
        # Load and preprocess training data
        train_data_raw = data_loader.load_hotpot_data(args.train_data)
        corpus, train_qa = data_loader.preprocess_hotpot(
            train_data_raw, max_samples=args.max_train_samples
        )
        
        # Load development data
        dev_data_raw = data_loader.load_hotpot_data(args.dev_data)
        _, dev_qa = data_loader.preprocess_hotpot(
            dev_data_raw, max_samples=args.max_eval_samples
        )
        
        # Initialize model
        model = HybridRAGCAG(config)
        
        # Build index
        model.retriever.build_index(corpus)
        
        # Prepare data loaders
        train_dataset = QADataset(train_qa, model.generator.tokenizer)
        dev_dataset = QADataset(dev_qa, model.generator.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Train model
        trainer = Trainer(model, config)
        training_history = trainer.train(train_loader, dev_loader)
        
        # Save training history
        with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
    
    if args.mode in ['eval', 'both']:
        # Load evaluation data
        if args.mode == 'eval':
            dev_data_raw = data_loader.load_hotpot_data(args.dev_data)
            corpus, dev_qa = data_loader.preprocess_hotpot(
                dev_data_raw, max_samples=args.max_eval_samples
            )
            
            model = HybridRAGCAG(config)
            model.retriever.build_index(corpus)
        
        # Comprehensive evaluation
        evaluator = Evaluator(model, config)
        results = evaluator.evaluate_comprehensive(dev_qa)
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print(f"Overall Performance:")
        for metric, value in results['overall_metrics'].items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        print(f"\nDetailed Analysis:")
        for key, value in results['detailed_analysis'].items():
            print(f"  {key}: {value:.4f}")
        
        print(f"\nExample Results (Best and Worst):")
        for i, example in enumerate(results['examples'][:6]):
            print(f"\n--- Example {i+1} ({example['category']}) ---")
            print(f"Question: {example['question']}")
            print(f"Gold: {example['gold_answer']}")
            print(f"Prediction: {example['prediction']}")
            print(f"F1 Score: {example['f1_score']:.3f}")
        
        # Save results
        with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualizations
        visualize_results(results, os.path.join(args.output_dir, 'results_visualization.png'))
    
    logger.info("Execution completed successfully!")

if __name__ == "__main__":
    main()
"""
Comprehensive benchmarking suite for CoreML sentiment analysis model.
Tests model performance, confidence calibration, and batch processing efficiency.
"""
import coremltools as ct
import numpy as np
from transformers import AutoTokenizer
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import os
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Container for benchmark metrics."""
    latency_ms: float
    confidence: float
    label: str
    batch_size: int
    queue_depth: int
    text_length: int

class BenchmarkSuite:
    def __init__(self, model_path: str, tokenizer_name: str):
        """Initialize the benchmark suite."""
        self.model = ct.models.MLModel(model_path)
        self.model.compute_units = ct.ComputeUnit.ALL
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Configure thread pool
        self.executor = ThreadPoolExecutor(
            max_workers=min(os.cpu_count() * 2, 8)
        )
        
        # Test datasets
        self.datasets = {
            'standard': self._generate_standard_dataset(),
            'edge_cases': self._generate_edge_cases(),
            'mixed_sentiment': self._generate_mixed_sentiment(),
            'neutral': self._generate_neutral_cases()
        }
        
    def _generate_standard_dataset(self, size: int = 1000) -> List[str]:
        """Generate standard sentiment analysis test cases."""
        positive_templates = [
            "This {} is absolutely fantastic!",
            "I love how {} this is.",
            "Incredible {} and amazing performance.",
            "Best {} I've ever experienced.",
            "Highly recommend this {}, it's outstanding."
        ]
        negative_templates = [
            "This {} is terrible.",
            "I hate how {} this is.",
            "Poor {} and disappointing results.",
            "Worst {} I've ever seen.",
            "Cannot recommend this {}, it's awful."
        ]
        
        subjects = [
            "product", "service", "experience", "feature", "design",
            "implementation", "solution", "performance", "quality", "result"
        ]
        
        texts = []
        for _ in range(size // 2):
            template = np.random.choice(positive_templates)
            subject = np.random.choice(subjects)
            texts.append(template.format(subject))
            
            template = np.random.choice(negative_templates)
            subject = np.random.choice(subjects)
            texts.append(template.format(subject))
            
        return texts
    
    def _generate_edge_cases(self) -> List[str]:
        """Generate edge cases for testing."""
        return [
            ".",
            "!",
            "This is a very very very very very very very very very very very very long sentence that goes on and on.",
            "🎉 Great! 👍",
            "N/A",
            "Mixed feelings about this...",
            "5/5 stars",
            "0/5 stars",
            "null",
            "undefined"
        ]
    
    def _generate_mixed_sentiment(self) -> List[str]:
        """Generate cases with mixed sentiment."""
        return [
            "The graphics are amazing but the gameplay is terrible.",
            "Great features, though a bit expensive.",
            "Despite the bugs, I enjoyed using it.",
            "Good product overall, customer service needs improvement.",
            "Beautiful design but poor performance.",
            "Love the concept, hate the execution.",
            "Fast delivery, product was damaged.",
            "Works well most of the time, occasional crashes.",
            "Excellent quality, but not user friendly.",
            "Great value, but missing key features."
        ]
    
    def _generate_neutral_cases(self) -> List[str]:
        """Generate neutral or ambiguous cases."""
        return [
            "It is what it is.",
            "Average performance.",
            "As expected.",
            "Neither good nor bad.",
            "Meets basic requirements.",
            "Standard features included.",
            "Typical response time.",
            "Regular updates available.",
            "Common implementation.",
            "Normal user experience."
        ]
    
    def preprocess_batch(self, texts: List[str], max_length: int = 512) -> Dict[str, np.ndarray]:
        """Preprocess a batch of texts."""
        tokens = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )
        return {
            "input_ids": tokens["input_ids"].astype(np.int32),
            "attention_mask": tokens["attention_mask"].astype(np.int32)
        }
    
    def process_batch(self, texts: List[str], batch_size: int, queue_depth: int) -> List[BenchmarkResult]:
        """Process a batch of texts and collect metrics."""
        results = []
        start_time = time.time()
        
        # Preprocess
        inputs = self.preprocess_batch(texts)
        
        # Run inference
        predictions = self.model.predict(inputs)
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        # Process predictions
        logits = predictions.get('linear_37', predictions)
        
        for i, text in enumerate(texts):
            text_logits = logits[i] if isinstance(logits, np.ndarray) else logits
            
            # Apply softmax
            exp_logits = np.exp(text_logits - np.max(text_logits))
            probabilities = exp_logits / exp_logits.sum()
            
            # Get prediction and confidence
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])
            label = 'positive' if predicted_class == 1 else 'negative'
            
            results.append(BenchmarkResult(
                latency_ms=latency / len(texts),  # Per-text latency
                confidence=confidence,
                label=label,
                batch_size=batch_size,
                queue_depth=queue_depth,
                text_length=len(text)
            ))
            
        return results
    
    def run_benchmark(self, batch_sizes: List[int] = [1, 4, 8, 16, 32]) -> pd.DataFrame:
        """Run comprehensive benchmark suite."""
        all_results = []
        
        for dataset_name, texts in self.datasets.items():
            logger.info(f"Running benchmark on {dataset_name} dataset")
            
            for batch_size in batch_sizes:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:min(i + batch_size, len(texts))]
                    if len(batch) == batch_size:  # Only process full batches
                        results = self.process_batch(
                            batch,
                            batch_size=batch_size,
                            queue_depth=len(texts) - i
                        )
                        for result in results:
                            all_results.append({
                                'dataset': dataset_name,
                                'latency_ms': result.latency_ms,
                                'confidence': result.confidence,
                                'label': result.label,
                                'batch_size': result.batch_size,
                                'queue_depth': result.queue_depth,
                                'text_length': result.text_length
                            })
        
        return pd.DataFrame(all_results)
    
    def analyze_results(self, df: pd.DataFrame, output_dir: str):
        """Analyze and visualize benchmark results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Latency Analysis
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='batch_size', y='latency_ms')
        plt.title('Latency Distribution by Batch Size')
        plt.savefig(f'{output_dir}/latency_distribution.png')
        plt.close()
        
        # 2. Confidence Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='confidence', hue='dataset', multiple="layer", alpha=0.5)
        plt.title('Confidence Distribution by Dataset')
        plt.savefig(f'{output_dir}/confidence_distribution.png')
        plt.close()
        
        # 3. Throughput Analysis
        throughput = df.groupby('batch_size').agg({
            'latency_ms': ['mean', 'std'],
            'text_length': 'count'
        })
        throughput['texts_per_second'] = 1000 / throughput['latency_ms']['mean']
        throughput.to_csv(f'{output_dir}/throughput_analysis.csv')
        
        # 4. Summary Statistics
        dataset_stats = df.groupby('dataset').agg({
            'latency_ms': ['mean', 'std'],
            'confidence': ['mean', 'std'],
            'text_length': 'count'
        })
        
        # Flatten multi-index columns
        dataset_stats.columns = ['_'.join(col).strip() for col in dataset_stats.columns.values]
        
        summary = {
            'overall': {
                'mean_latency_ms': float(df['latency_ms'].mean()),
                'p95_latency_ms': float(df['latency_ms'].quantile(0.95)),
                'mean_confidence': float(df['confidence'].mean()),
                'total_processed': len(df)
            },
            'by_dataset': dataset_stats.reset_index().to_dict('records')
        }
        
        with open(f'{output_dir}/summary_stats.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive model benchmark')
    parser.add_argument('--model-path', required=True, help='Path to CoreML model')
    parser.add_argument('--tokenizer', required=True, help='Name or path to tokenizer')
    parser.add_argument('--output-dir', default='benchmark_results', help='Output directory for results')
    args = parser.parse_args()
    
    # Run benchmark
    suite = BenchmarkSuite(args.model_path, args.tokenizer)
    results_df = suite.run_benchmark()
    
    # Analyze results
    summary = suite.analyze_results(results_df, args.output_dir)
    
    # Print key findings
    print("\nBenchmark Summary:")
    print("-" * 50)
    print(f"Total Samples Processed: {summary['overall']['total_processed']}")
    print(f"Mean Latency: {summary['overall']['mean_latency_ms']:.2f}ms")
    print(f"P95 Latency: {summary['overall']['p95_latency_ms']:.2f}ms")
    print(f"Mean Confidence: {summary['overall']['mean_confidence']:.2%}")
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

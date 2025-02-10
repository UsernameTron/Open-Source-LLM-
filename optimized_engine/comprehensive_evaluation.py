"""
Comprehensive evaluation suite for the enhanced inference engine.
"""
import logging
from enhanced_inference import EnhancedInferenceEngine, EnhancedInferenceConfig
import time
from typing import List, Dict, Any
import json
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import psutil
import GPUtil
import argparse
from dataclasses import dataclass, asdict
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Test case with expected sentiment and metadata."""
    text: str
    expected_sentiment: str
    category: str
    subcategory: str
    priority: int = 0

class ComprehensiveEvaluator:
    def __init__(self, model_path: str, tokenizer_name: str):
        """Initialize evaluator with different temperature configurations."""
        self.temperatures = [0.8, 1.0, 1.2, 1.5]
        self.confidence_thresholds = [0.7, 0.8, 0.9]
        self.batch_sizes = [1, 4, 8, 16, 32, 64]
        
        # Initialize engines with different temperatures
        self.engines = {}
        for temp in self.temperatures:
            config = EnhancedInferenceConfig()
            config.temperature = temp
            config.confidence_threshold = 0.8  # Default threshold
            self.engines[temp] = EnhancedInferenceEngine(
                model_path=model_path,
                tokenizer_name=tokenizer_name,
                config=config
            )
        
        # Test cases
        self.test_cases = self._create_test_cases()
    
    def _create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases."""
        return [
            # 1. Neutral Cases
            TestCase(
                "The product arrived on time and matched the description provided.",
                "neutral",
                "neutral",
                "factual"
            ),
            TestCase(
                "It has four wheels and a steering wheel.",
                "neutral",
                "neutral",
                "descriptive"
            ),
            
            # 2. Borderline Cases
            TestCase(
                "The food was okay, not great but not terrible either.",
                "neutral",
                "borderline",
                "mild_negative"
            ),
            TestCase(
                "It's slightly better than what I expected.",
                "positive",
                "borderline",
                "mild_positive"
            ),
            
            # 3. Long-form Reviews
            TestCase(
                """
                I've been using this product for three months now. The build quality is excellent,
                and the battery life exceeds expectations. However, the software occasionally glitches,
                which can be frustrating. The customer service team has been responsive to my concerns,
                though solutions aren't always immediate. Despite these minor issues, I find myself
                reaching for this device more often than not. The price point is a bit high, but the
                overall value proposition makes it worthwhile for power users.
                """,
                "positive",
                "long_form",
                "mixed_detailed"
            ),
            
            # 4. Sarcasm
            TestCase(
                "Oh great, another perfectly executed update that breaks everything. Just what I needed!",
                "negative",
                "sarcasm",
                "obvious"
            ),
            TestCase(
                "Wow, spending two hours on hold was exactly how I wanted to spend my evening!",
                "negative",
                "sarcasm",
                "subtle"
            ),
            
            # 5. Highly Subjective
            TestCase(
                "The artistic direction shows both innovation and respect for tradition.",
                "neutral",
                "subjective",
                "art_critique"
            ),
            TestCase(
                "The philosophical implications of this design choice are intriguing.",
                "neutral",
                "subjective",
                "abstract"
            ),
            
            # 6. Complex Sentiment
            TestCase(
                "While the core functionality is solid, the UI needs significant improvement.",
                "mixed",
                "complex",
                "balanced"
            ),
            TestCase(
                "Despite its flaws, which are numerous, the system somehow manages to get the job done.",
                "mixed",
                "complex",
                "qualified"
            ),
            
            # 7. Technical Language
            TestCase(
                "The API response latency increased by 15% after the latest deployment.",
                "negative",
                "technical",
                "metrics"
            ),
            TestCase(
                "CPU utilization remained within expected parameters during load testing.",
                "neutral",
                "technical",
                "performance"
            )
        ]
    
    def _measure_system_metrics(self) -> Dict[str, float]:
        """Measure system metrics during inference."""
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_util': None
        }
        
        # Get GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                metrics['gpu_util'] = gpus[0].load * 100
        except Exception:
            pass
        
        return metrics
    
    async def evaluate_temperature_scaling(self) -> Dict[str, Any]:
        """Evaluate different temperature settings."""
        results = defaultdict(list)
        
        for temp in self.temperatures:
            engine = self.engines[temp]
            
            for case in self.test_cases:
                start_time = time.time()
                prediction = engine.infer(case.text)
                latency = (time.time() - start_time) * 1000
                
                results[temp].append({
                    'category': case.category,
                    'subcategory': case.subcategory,
                    'expected': case.expected_sentiment,
                    'predicted': prediction.label,
                    'confidence': prediction.confidence,
                    'latency_ms': latency
                })
        
        return dict(results)
    
    async def evaluate_batch_performance(self) -> Dict[str, Any]:
        """Evaluate performance with different batch sizes."""
        results = {}
        
        # Use default temperature engine
        engine = self.engines[1.2]
        
        for batch_size in self.batch_sizes:
            batch_results = []
            system_metrics = []
            
            # Create larger batch by repeating test cases
            texts = [case.text for case in self.test_cases] * (batch_size // len(self.test_cases) + 1)
            texts = texts[:batch_size]
            
            # Warm-up run
            _ = [engine.infer(text) for text in texts[:min(len(texts), 2)]]
            
            # Actual test
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for text in texts:
                    futures.append(executor.submit(engine.infer, text))
                
                # Collect results and measure system metrics
                for future in futures:
                    try:
                        result = future.result(timeout=5.0)
                        batch_results.append({
                            'latency_ms': result.latency_ms,
                            'confidence': result.confidence
                        })
                        system_metrics.append(self._measure_system_metrics())
                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")
            
            total_time = (time.time() - start_time) * 1000
            
            results[batch_size] = {
                'throughput': len(batch_results) / (total_time / 1000),
                'mean_latency_ms': np.mean([r['latency_ms'] for r in batch_results]),
                'p95_latency_ms': np.percentile([r['latency_ms'] for r in batch_results], 95),
                'mean_confidence': np.mean([r['confidence'] for r in batch_results]),
                'system_metrics': {
                    'cpu_percent': np.mean([m['cpu_percent'] for m in system_metrics]),
                    'memory_percent': np.mean([m['memory_percent'] for m in system_metrics]),
                    'gpu_util': np.mean([m['gpu_util'] for m in system_metrics if m['gpu_util'] is not None])
                }
            }
        
        return results
    
    async def evaluate_neutral_classification(self) -> Dict[str, Any]:
        """Evaluate neutral classification with different confidence thresholds."""
        results = {}
        
        # Use default temperature engine
        engine = self.engines[1.2]
        
        for threshold in self.confidence_thresholds:
            engine.config.confidence_threshold = threshold
            threshold_results = []
            
            for case in [c for c in self.test_cases if c.expected_sentiment == 'neutral']:
                prediction = engine.infer(case.text)
                threshold_results.append({
                    'text': case.text,
                    'category': case.category,
                    'subcategory': case.subcategory,
                    'predicted': prediction.label,
                    'confidence': prediction.confidence,
                    'correct': prediction.label == case.expected_sentiment
                })
            
            results[threshold] = {
                'accuracy': np.mean([r['correct'] for r in threshold_results]),
                'mean_confidence': np.mean([r['confidence'] for r in threshold_results]),
                'predictions': threshold_results
            }
        
        return results
    
    async def run_evaluation(self, output_dir: str):
        """Run comprehensive evaluation."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Run evaluations
        logger.info("Evaluating temperature scaling...")
        temp_results = await self.evaluate_temperature_scaling()
        
        logger.info("Evaluating batch performance...")
        batch_results = await self.evaluate_batch_performance()
        
        logger.info("Evaluating neutral classification...")
        neutral_results = await self.evaluate_neutral_classification()
        
        # Save results
        results = {
            'temperature_scaling': temp_results,
            'batch_performance': batch_results,
            'neutral_classification': neutral_results
        }
        
        with open(output_dir / 'comprehensive_evaluation.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\nEvaluation Summary:")
        print("-" * 50)
        
        # Temperature scaling summary
        print("\n1. Temperature Scaling:")
        for temp, results in temp_results.items():
            correct = sum(1 for r in results if r['predicted'] == r['expected'])
            print(f"\nTemperature {temp}:")
            print(f"  Accuracy: {correct/len(results):.2%}")
            print(f"  Mean Confidence: {np.mean([r['confidence'] for r in results]):.2%}")
            print(f"  Mean Latency: {np.mean([r['latency_ms'] for r in results]):.2f}ms")
        
        # Batch performance summary
        print("\n2. Batch Performance:")
        for size, results in batch_results.items():
            print(f"\nBatch Size {size}:")
            print(f"  Throughput: {results['throughput']:.2f} texts/second")
            print(f"  Mean Latency: {results['mean_latency_ms']:.2f}ms")
            print(f"  P95 Latency: {results['p95_latency_ms']:.2f}ms")
            print(f"  CPU Usage: {results['system_metrics']['cpu_percent']:.1f}%")
        
        # Neutral classification summary
        print("\n3. Neutral Classification:")
        for threshold, results in neutral_results.items():
            print(f"\nConfidence Threshold {threshold}:")
            print(f"  Accuracy: {results['accuracy']:.2%}")
            print(f"  Mean Confidence: {results['mean_confidence']:.2%}")
        
        print(f"\nDetailed results saved to: {output_dir}")

async def main():
    parser = argparse.ArgumentParser(description="Run comprehensive model evaluation")
    parser.add_argument("--model-path", required=True, help="Path to CoreML model")
    parser.add_argument("--tokenizer", required=True, help="Name or path to tokenizer")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    args = parser.parse_args()
    
    evaluator = ComprehensiveEvaluator(args.model_path, args.tokenizer)
    await evaluator.run_evaluation(args.output_dir)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

"""
Test script for the enhanced inference engine.
"""
import asyncio
import logging
from enhanced_inference import EnhancedInferenceEngine, EnhancedInferenceConfig
import time
from typing import List, Dict
import argparse
from pathlib import Path
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestHarness:
    def __init__(self, model_path: str, tokenizer_name: str):
        """Initialize test harness."""
        # Configure enhanced engine
        config = EnhancedInferenceConfig()
        config.confidence_threshold = 0.9
        config.temperature = 1.2  # Slightly higher temperature for better calibration
        
        self.engine = EnhancedInferenceEngine(
            model_path=model_path,
            tokenizer_name=tokenizer_name,
            config=config
        )
        
        # Test datasets
        self.test_cases = {
            'standard': [
                "This product exceeded all my expectations!",
                "Terrible service, would not recommend.",
                "Amazing features and great performance.",
                "Completely disappointed with the quality."
            ],
            'mixed': [
                "Good product but expensive",
                "Great features, occasional bugs",
                "Fast performance, poor battery life",
                "Beautiful design, lacking functionality"
            ],
            'neutral': [
                "Product arrived as described",
                "Standard features included",
                "Regular performance metrics",
                "Typical user experience"
            ],
            'edge': [
                "👍 Great! 🎉",
                "n/a",
                "...",
                "Product is !!!!!!!"
            ]
        }
    
    async def run_load_test(self, duration_seconds: int = 60):
        """Run a load test with mixed traffic patterns."""
        start_time = time.time()
        results = []
        
        # Start dashboard in background
        dashboard_process = subprocess.Popen(
            [sys.executable, 'monitoring_dashboard.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        try:
            logger.info("Starting load test...")
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                while time.time() - start_time < duration_seconds:
                    # Simulate varying load patterns
                    batch_size = np.random.choice([1, 4, 8, 16, 32])
                    
                    # Mix different types of requests
                    for dataset_type, texts in self.test_cases.items():
                        # Random priority (0=highest, 2=lowest)
                        priority = np.random.randint(0, 3)
                        
                        # Submit requests
                        futures = []
                        for text in texts:
                            future = executor.submit(
                                self.engine.infer,
                                text,
                                priority=priority
                            )
                            futures.append(future)
                        
                        # Collect results
                        for future in futures:
                            try:
                                result = future.result(timeout=5.0)
                                results.append({
                                    'text': result.text,
                                    'label': result.label,
                                    'confidence': result.confidence,
                                    'latency_ms': result.latency_ms,
                                    'batch_size': result.batch_size,
                                    'queue_depth': result.queue_depth
                                })
                            except Exception as e:
                                logger.error(f"Request failed: {e}")
                    
                    # Small sleep to control request rate
                    await asyncio.sleep(0.1)
            
            return results
            
        finally:
            # Cleanup
            dashboard_process.send_signal(signal.SIGINT)
            dashboard_process.wait()
    
    def analyze_results(self, results: List[Dict], output_path: str):
        """Analyze and save test results."""
        analysis = {
            'summary': {
                'total_requests': len(results),
                'mean_latency_ms': np.mean([r['latency_ms'] for r in results]),
                'p95_latency_ms': np.percentile([r['latency_ms'] for r in results], 95),
                'mean_confidence': np.mean([r['confidence'] for r in results]),
                'confidence_std': np.std([r['confidence'] for r in results])
            },
            'latency_by_batch_size': {},
            'confidence_distribution': {
                'high_confidence': len([r for r in results if r['confidence'] >= 0.9]),
                'medium_confidence': len([r for r in results if 0.7 <= r['confidence'] < 0.9]),
                'low_confidence': len([r for r in results if r['confidence'] < 0.7])
            }
        }
        
        # Analyze latency by batch size
        batch_sizes = set(r['batch_size'] for r in results)
        for batch_size in batch_sizes:
            batch_results = [r for r in results if r['batch_size'] == batch_size]
            analysis['latency_by_batch_size'][batch_size] = {
                'mean_latency_ms': np.mean([r['latency_ms'] for r in batch_results]),
                'p95_latency_ms': np.percentile([r['latency_ms'] for r in batch_results], 95),
                'request_count': len(batch_results)
            }
        
        # Save analysis
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / 'load_test_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis

def main():
    parser = argparse.ArgumentParser(description='Test enhanced inference engine')
    parser.add_argument('--model-path', required=True, help='Path to CoreML model')
    parser.add_argument('--tokenizer', required=True, help='Name or path to tokenizer')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--output-dir', default='test_results', help='Output directory')
    args = parser.parse_args()
    
    # Run test
    harness = TestHarness(args.model_path, args.tokenizer)
    
    logger.info(f"Running load test for {args.duration} seconds...")
    results = asyncio.run(harness.run_load_test(args.duration))
    
    # Analyze results
    analysis = harness.analyze_results(results, args.output_dir)
    
    # Print summary
    print("\nLoad Test Results:")
    print("-" * 50)
    print(f"Total Requests: {analysis['summary']['total_requests']}")
    print(f"Mean Latency: {analysis['summary']['mean_latency_ms']:.2f}ms")
    print(f"P95 Latency: {analysis['summary']['p95_latency_ms']:.2f}ms")
    print(f"Mean Confidence: {analysis['summary']['mean_confidence']:.2%}")
    print(f"\nConfidence Distribution:")
    print(f"  High (≥0.9): {analysis['confidence_distribution']['high_confidence']}")
    print(f"  Medium (0.7-0.9): {analysis['confidence_distribution']['medium_confidence']}")
    print(f"  Low (<0.7): {analysis['confidence_distribution']['low_confidence']}")
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == '__main__':
    main()

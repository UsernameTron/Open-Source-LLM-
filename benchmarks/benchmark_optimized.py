import numpy as np
import time
import logging
import os
import asyncio
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
from core.inference.engine import InferenceEngine, InferenceConfig
from core.monitoring.performance import PerformanceMonitor
import random

# Set tokenizer parallelism to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelBenchmark:
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str,
        sequence_length: int = 512
    ):
        self.config = InferenceConfig(
            min_batch_size=1,
            max_batch_size=128,
            target_latency_ms=50.0,
            max_queue_size=1000,
            metrics_window=1000,
            adaptive_batching=True
        )
        self.engine = InferenceEngine(model_path, tokenizer_name, self.config)
        self.sequence_length = sequence_length
        self.monitor = PerformanceMonitor()
        self.results_dir = Path('benchmark_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Track warmup metrics
        self.warmup_metrics = None
        
    def generate_random_input(self, length: Optional[int] = None) -> str:
        """Generate random text input."""
        if length is None:
            length = random.randint(10, self.sequence_length)
            
        # Generate random text (simplified for testing)
        words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']
        return ' '.join(random.choices(words, k=length))
        
    async def warmup_model(self):
        """Perform model warmup and collect metrics."""
        logger.info("Starting model warmup phase...")
        
        # Run warmup
        await self.engine.warmup()
        
        # Collect warmup metrics
        self.warmup_metrics = self.engine._warmup.get_performance_summary()
        logger.info(f"Warmup complete. Summary: {self.warmup_metrics}")
        
    async def run_latency_test(
        self,
        num_requests: int = 1000,
        concurrent_requests: int = 10
    ) -> Dict:
        """Run latency test with concurrent requests."""
        logger.info(f"Running latency test with {num_requests} requests, {concurrent_requests} concurrent...")
        
        try:
            # Ensure model is warmed up
            if not self.engine._is_warmed_up:
                await self.warmup_model()
            
            start_time = time.time()
            results = []
            tasks = []
            completed_requests = 0
            
            # Create tasks with varying priorities
            for i in range(num_requests):
                text = self.generate_random_input()
                priority = i % 3  # Simulate different priority levels
                tasks.append(self.engine.infer_async(text, priority))
                
                if len(tasks) >= concurrent_requests:
                    try:
                        logger.debug(f"Processing batch of {len(tasks)} requests...")
                        batch_results = await asyncio.gather(*tasks)
                        results.extend(batch_results)
                        completed_requests += len(batch_results)
                        tasks = []
                        
                        # Log progress
                        progress = (completed_requests / num_requests) * 100
                        logger.info(f"Progress: {progress:.1f}% ({completed_requests}/{num_requests} requests)")
                        
                        # Update monitor with batch metrics
                        metal_metrics = self.engine._warmup._get_metal_metrics()
                        # Update metrics for each result in the batch
                        for result in batch_results:
                            self.monitor.update_metrics({
                                'batch_size': self.engine._current_batch_size,
                                'latency_ms': result.get('latency', 0),
                                'queue_depth': self.engine._request_queue.qsize(),
                                'compute_util': metal_metrics['compute_util'],
                                'temperature_c': metal_metrics['temperature_c'],
                                'power_w': metal_metrics['power_w'],
                                'timestamp': time.time()
                            })
                        
                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")
                        raise
                
            # Process remaining tasks
            if tasks:
                try:
                    logger.debug(f"Processing final batch of {len(tasks)} requests...")
                    batch_results = await asyncio.gather(*tasks)
                    results.extend(batch_results)
                    completed_requests += len(batch_results)
                    
                    # Update monitor with final batch metrics
                    metal_metrics = self.engine._warmup._get_metal_metrics()
                    # Update metrics for each result in the final batch
                    for result in batch_results:
                        self.monitor.update_metrics({
                            'batch_size': self.engine._current_batch_size,
                            'latency_ms': result.get('latency', 0),
                            'queue_depth': self.engine._request_queue.qsize(),
                            'compute_util': metal_metrics['compute_util'],
                            'temperature_c': metal_metrics['temperature_c'],
                            'power_w': metal_metrics['power_w'],
                            'timestamp': time.time()
                        })
                    
                except Exception as e:
                    logger.error(f"Error processing final batch: {e}")
                    raise
            
            # Calculate final metrics
            total_time = time.time() - start_time
            logger.info(f"Test completed in {total_time:.2f} seconds")
            
            try:
                report = self.monitor.generate_report(self.monitor.metrics_window)
                
                # Save report if available
                if report:
                    report_path = self.results_dir / f"latency_report_{int(time.time())}.json"
                    with open(report_path, 'w') as f:
                        json.dump(report.to_dict(), f, indent=2)
                    logger.info(f"Saved latency report to {report_path}")
                    
                    # Generate plots
                    plot_path = self.results_dir / f"latency_plots_{int(time.time())}.png"
                    self.monitor.plot_metrics(self.monitor.metrics_window, str(plot_path))
                    logger.info(f"Generated performance plots at {plot_path}")
                
                return {
                    'results': results,
                    'metrics': report.to_dict() if report else None,
                    'warmup_metrics': self.warmup_metrics,
                    'throughput': num_requests / total_time,
                    'total_time': total_time,
                    'completed_requests': completed_requests
                }
                
            except Exception as e:
                logger.error(f"Error generating final report: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Latency test failed: {e}")
            raise
        
    async def run_throughput_test(
        self,
        batch_sizes: List[int] = [1, 8, 16, 32, 64, 128],
        requests_per_batch: int = 100,
        test_duration: int = 60  # seconds per batch size
    ) -> Dict:
        """Run throughput test with different batch sizes."""
        logger.info("Running throughput test...")
        
        results = {}
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Configure engine for this batch size
            self.engine._current_batch_size = batch_size
            start_time = time.time()
            end_time = start_time + test_duration
            
            tasks = []
            completed_requests = 0
            batch_latencies = []
            metal_metrics_samples = []
            
            while time.time() < end_time:
                # Generate concurrent requests
                for _ in range(min(batch_size * 2, requests_per_batch)):
                    text = self.generate_random_input()
                    tasks.append(self.engine.infer_async(text))
                    
                if len(tasks) >= requests_per_batch:
                    # Get Metal metrics before inference
                    metal_metrics = self.engine._warmup._get_metal_metrics()
                    metal_metrics_samples.append(metal_metrics)
                    
                    # Run inference
                    batch_start = time.time()
                    batch_results = await asyncio.gather(*tasks)
                    batch_end = time.time()
                    
                    # Process results
                    batch_latency = (batch_end - batch_start) * 1000  # Convert to ms
                    batch_latencies.append(batch_latency)
                    completed_requests += len(batch_results)
                    
                    # Update metrics for each result
                    for result in batch_results:
                        self.monitor.update_metrics({
                            'batch_size': batch_size,
                            'latency_ms': result.get('latency', 0),
                            'queue_depth': self.engine._request_queue.qsize(),
                            'compute_util': metal_metrics['compute_util'],
                            'temperature_c': metal_metrics['temperature_c'],
                            'power_w': metal_metrics['power_w'],
                            'timestamp': time.time()
                        })
                    
                    tasks = []
                    
            # Process any remaining tasks
            if tasks:
                metal_metrics = self.engine._warmup._get_metal_metrics()
                metal_metrics_samples.append(metal_metrics)
                
                batch_start = time.time()
                batch_results = await asyncio.gather(*tasks)
                batch_end = time.time()
                
                batch_latency = (batch_end - batch_start) * 1000
                batch_latencies.append(batch_latency)
                completed_requests += len(batch_results)
                
                for result in batch_results:
                    self.monitor.update_metrics({
                        'batch_size': batch_size,
                        'latency_ms': result.get('latency', 0),
                        'queue_depth': self.engine._request_queue.qsize(),
                        'compute_util': metal_metrics['compute_util'],
                        'temperature_c': metal_metrics['temperature_c'],
                        'power_w': metal_metrics['power_w'],
                        'timestamp': time.time()
                    })
                
            # Generate performance report
            report = self.monitor.generate_report(self.monitor.metrics_window)
            if report:
                # Save report
                report_path = self.results_dir / f"throughput_report_{batch_size}_{int(time.time())}.json"
                with open(report_path, 'w') as f:
                    json.dump(report.to_dict(), f, indent=2)
                
                # Generate plots
                plot_path = self.results_dir / f"throughput_plots_{batch_size}_{int(time.time())}.png"
                self.monitor.plot_metrics(report, str(plot_path))
                
                # Calculate metrics for this batch size
                results[batch_size] = {
                    'avg_latency_ms': np.mean(report.latencies),
                    'p95_latency_ms': np.percentile(report.latencies, 95),
                    'p99_latency_ms': np.percentile(report.latencies, 99),
                    'throughput': np.mean(report.throughputs),
                    'gpu_utilization': np.mean(report.gpu_utilization),
                    'memory_utilization': np.mean(report.memory_utilization),
                    'queue_size': np.mean(report.queue_sizes)
                }
            
        return results
        
    def save_results(self, latency_results: Dict, throughput_results: Dict):
        """Save benchmark results and generate summary."""
        # Save raw results
        with open(self.results_dir / 'latency_results.json', 'w') as f:
            json.dump(latency_results, f, indent=2)
            
        with open(self.results_dir / 'throughput_results.json', 'w') as f:
            json.dump(throughput_results, f, indent=2)
            
        # Generate summary report
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'latency_test': {
                'num_requests': len(latency_results['results']),
                'avg_confidence': np.mean([r.get('confidence', 0) for r in latency_results['results']]),
                'metrics': latency_results.get('metrics', {})
            },
            'throughput_test': {
                'batch_sizes_tested': list(throughput_results.keys()),
                'optimal_batch_size': max(
                    throughput_results.items(),
                    key=lambda x: x[1]['throughput']
                )[0],
                'recommendations': [
                    rec for batch in throughput_results.values()
                    for rec in batch.get('recommendations', [])
                ]
            }
        }
        
        with open(self.results_dir / 'benchmark_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Benchmark complete! Results saved to {self.results_dir}/")
        
async def main():
    # Setup
    model_path = 'models/sentiment_model_metal.mlpackage'
    tokenizer_name = 'bert-base-uncased'  # Update with your tokenizer
    
    # Initialize benchmark
    benchmark = ModelBenchmark(model_path, tokenizer_name)
    
    # Warmup model first
    await benchmark.warmup_model()
    
    # Run latency test
    logger.info("\nRunning latency test...")
    latency_results = await benchmark.run_latency_test(
        num_requests=1000,
        concurrent_requests=10
    )
    
    # Print warmup results
    print("\nWarmup Phase Results:")
    for key, value in benchmark.warmup_metrics.items():
        print(f"{key}: {value:.2f}")
    
    # Print latency results
    print("\nLatency Test Results:")
    metrics = latency_results['metrics']
    if metrics:
        print(f"Throughput: {latency_results['throughput']:.2f} requests/sec")
        print(f"Mean Latency: {metrics.get('avg_latency_ms', 0.0):.2f} ms")
        print(f"P50 Latency: {metrics.get('p50_latency_ms', 0.0):.2f} ms")
        print(f"P95 Latency: {metrics.get('p95_latency_ms', 0.0):.2f} ms")
        print(f"P99 Latency: {metrics.get('p99_latency_ms', 0.0):.2f} ms")
        
        print("\nResource Utilization:")
        print(f"Final Batch Size: {benchmark.engine._current_batch_size}")
        print(f"Mean Compute Utilization: {metrics.get('mean_compute_util', 0):.2f}%")
        print(f"Mean Temperature: {metrics.get('mean_temperature_c', 0):.2f}°C")
        print(f"Mean Power Usage: {metrics.get('mean_power_w', 0):.2f}W")
    
    # Run throughput test
    logger.info("\nRunning throughput test...")
    throughput_results = await benchmark.run_throughput_test(
        batch_sizes=[1, 4, 8, 16, 32, 64, 128],
        requests_per_batch=100,
        test_duration=30  # 30 seconds per batch size
    )
    
    # Print throughput results
    print("\nThroughput Test Results:")
    for batch_size, results in throughput_results.items():
        print(f"\nBatch Size {batch_size}:")
        print(f"Throughput: {results['throughput']:.2f} requests/sec")
        print(f"P95 Latency: {results['p95_latency_ms']:.2f} ms")
        print(f"GPU Utilization: {results['gpu_utilization']:.2f}%")
        print(f"Memory Utilization: {results['memory_utilization']:.2f}%")
        
        if 'recommendations' in results:
            print("\nRecommendations:")
            for rec in results['recommendations']:
                print(f"- {rec}")
    
    # Save results and generate summary
    benchmark.save_results(latency_results, throughput_results)

if __name__ == '__main__':
    asyncio.run(main())

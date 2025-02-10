"""Benchmark script to measure the impact of batch optimization changes."""
import asyncio
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import logging
import uuid
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import seaborn as sns
from core.inference.engine import InferenceEngine, InferenceConfig
from core.monitoring.metrics_collector import MetricsCollector, RequestTrace, BatchMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchOptimizationBenchmark:
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str,
        output_dir: str = "benchmark_results"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create two configurations for comparison
        self.baseline_config = InferenceConfig(
            min_batch_size=1,
            max_batch_size=32,
            target_latency_ms=50.0,
            adaptive_batching=True
        )
        
        self.optimized_config = InferenceConfig(
            min_batch_size=4,
            max_batch_size=64,
            target_latency_ms=100.0,
            adaptive_batching=True
        )
        
        # Initialize engines
        self.baseline_engine = InferenceEngine(model_path, tokenizer_name, self.baseline_config)
        self.optimized_engine = InferenceEngine(model_path, tokenizer_name, self.optimized_config)
        
        # Initialize metrics collectors
        self.baseline_metrics = MetricsCollector(export_path=output_dir + "/baseline")
        self.optimized_metrics = MetricsCollector(export_path=output_dir + "/optimized")

    async def _generate_load(
        self,
        engine: InferenceEngine,
        metrics: MetricsCollector,
        num_requests: int,
        concurrency: int,
        priority_distribution: List[float]
    ):
        """Generate a specific load pattern."""
        async def single_request():
            priority = np.random.choice(len(priority_distribution), p=priority_distribution)
            request_id = str(uuid.uuid4())
            start_time = time.time()
            
            try:
                result = await engine.infer_async("Sample text for inference", priority=priority)
                end_time = time.time()
                
                trace = RequestTrace(
                    request_id=request_id,
                    priority=priority,
                    batch_size=result.get("batch_size", 1),
                    queue_time=result.get("queue_time", 0),
                    processing_time=result.get("processing_time", 0),
                    total_latency=(end_time - start_time) * 1000
                )
                metrics.add_request_trace(trace)
                
            except Exception as e:
                logger.error(f"Request {request_id} failed: {e}")

        # Create request batches
        tasks = []
        for _ in range(num_requests):
            if len(tasks) >= concurrency:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            tasks.append(asyncio.create_task(single_request()))
        
        # Wait for remaining tasks
        if tasks:
            await asyncio.wait(tasks)

    async def run_benchmark(self):
        """Run the complete benchmark suite."""
        scenarios = [
            {
                "name": "Low Load",
                "requests": 100,
                "concurrency": 5,
                "priority_dist": [0.2, 0.3, 0.5]  # High, Medium, Low priority
            },
            {
                "name": "Medium Load",
                "requests": 500,
                "concurrency": 20,
                "priority_dist": [0.1, 0.3, 0.6]
            },
            {
                "name": "High Load",
                "requests": 1000,
                "concurrency": 50,
                "priority_dist": [0.05, 0.15, 0.8]
            }
        ]
        
        results = []
        for scenario in scenarios:
            logger.info(f"\nRunning scenario: {scenario['name']}")
            
            # Run baseline
            logger.info("Running baseline configuration...")
            await self._generate_load(
                self.baseline_engine,
                self.baseline_metrics,
                scenario["requests"],
                scenario["concurrency"],
                scenario["priority_dist"]
            )
            baseline_stats = self.baseline_metrics.get_current_stats()
            
            # Run optimized
            logger.info("Running optimized configuration...")
            await self._generate_load(
                self.optimized_engine,
                self.optimized_metrics,
                scenario["requests"],
                scenario["concurrency"],
                scenario["priority_dist"]
            )
            optimized_stats = self.optimized_metrics.get_current_stats()
            
            # Collect results
            results.append({
                "scenario": scenario["name"],
                "metric": "avg_latency",
                "baseline": baseline_stats["avg_latency"],
                "optimized": optimized_stats["avg_latency"],
                "improvement": (baseline_stats["avg_latency"] - optimized_stats["avg_latency"]) / baseline_stats["avg_latency"] * 100
            })
            
            results.append({
                "scenario": scenario["name"],
                "metric": "throughput",
                "baseline": baseline_stats["throughput"],
                "optimized": optimized_stats["throughput"],
                "improvement": (optimized_stats["throughput"] - baseline_stats["throughput"]) / baseline_stats["throughput"] * 100
            })
        
        # Generate report
        self._generate_report(results)

    def _generate_report(self, results: List[Dict[str, Any]]):
        """Generate a comprehensive benchmark report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"report_{timestamp}"
        report_dir.mkdir(exist_ok=True)
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Create summary plots
        plt.figure(figsize=(12, 6))
        
        # Latency comparison
        latency_df = df[df["metric"] == "avg_latency"]
        plt.subplot(1, 2, 1)
        sns.barplot(data=latency_df, x="scenario", y="improvement")
        plt.title("Latency Improvement (%)")
        plt.xticks(rotation=45)
        
        # Throughput comparison
        throughput_df = df[df["metric"] == "throughput"]
        plt.subplot(1, 2, 2)
        sns.barplot(data=throughput_df, x="scenario", y="improvement")
        plt.title("Throughput Improvement (%)")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(report_dir / "improvements.png")
        
        # Generate HTML report
        html_report = f"""
        <html>
        <head>
            <title>Batch Optimization Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .improvement-positive {{ color: green; }}
                .improvement-negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Batch Optimization Benchmark Report</h1>
            <p>Generated on: {timestamp}</p>
            
            <h2>Performance Improvements</h2>
            <img src="improvements.png" style="width: 100%; max-width: 800px;">
            
            <h2>Detailed Results</h2>
            {df.to_html(classes='dataframe', float_format=lambda x: '{:.2f}'.format(x))}
        </body>
        </html>
        """
        
        with open(report_dir / "report.html", "w") as f:
            f.write(html_report)
        
        logger.info(f"Benchmark report generated at {report_dir}/report.html")

async def main():
    benchmark = BatchOptimizationBenchmark(
        model_path="path/to/your/model",
        tokenizer_name="your-tokenizer-name"
    )
    await benchmark.run_benchmark()

if __name__ == "__main__":
    asyncio.run(main())

"""
Run optimized inference with real-time monitoring and benchmarking.
"""
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import argparse
from core.inference.engine import InferenceEngine, InferenceConfig
from core.monitoring.dashboard import start_dashboard
from benchmarks.batch_optimization_benchmark import BatchOptimizationBenchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(description='Run optimized inference with monitoring')
    parser.add_argument('--model-path', required=True, help='Path to CoreML model')
    parser.add_argument('--tokenizer', required=True, help='Name or path to tokenizer')
    parser.add_argument('--output-dir', default='benchmark_results', help='Output directory for results')
    parser.add_argument('--dashboard-port', type=int, default=8050, help='Dashboard port')
    parser.add_argument('--run-benchmark', action='store_true', help='Run benchmark suite')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    # Start dashboard
    dashboard_thread = start_dashboard(
        metrics_dir=str(metrics_dir),
        port=args.dashboard_port
    )
    logger.info(f"Dashboard started at http://localhost:{args.dashboard_port}")

    # Initialize inference engine with optimized config
    config = InferenceConfig(
        min_batch_size=4,
        max_batch_size=64,
        target_latency_ms=100.0,
        max_queue_size=2000,
        cache_size=2048,
        num_threads=8,
        metrics_window=200,
        adaptive_batching=True
    )

    engine = InferenceEngine(
        model_path=args.model_path,
        tokenizer_name=args.tokenizer,
        config=config,
        metrics_export_path=str(metrics_dir)
    )

    # Warm up the model
    logger.info("Warming up model...")
    await engine.warmup()

    if args.run_benchmark:
        # Run comprehensive benchmark
        logger.info("Running benchmark suite...")
        benchmark = BatchOptimizationBenchmark(
            model_path=args.model_path,
            tokenizer_name=args.tokenizer,
            output_dir=str(output_dir)
        )
        await benchmark.run_benchmark()
        logger.info(f"Benchmark results available in {output_dir}/report_*.html")

    # Keep the script running for dashboard
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())

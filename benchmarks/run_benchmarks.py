#!/usr/bin/env python3

import argparse
from benchmark import ModelBenchmark, BenchmarkConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run LLM Engine benchmarks")
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="Name of the PyTorch model to benchmark"
    )
    parser.add_argument(
        "--coreml-model",
        default="models/sentiment_model_metal.mlpackage",
        help="Path to the CoreML model"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32],
        help="Batch sizes to benchmark"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per batch size"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=512,
        help="Sequence length for input texts"
    )
    parser.add_argument(
        "--warm-up",
        type=int,
        default=10,
        help="Number of warm-up iterations"
    )
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        batch_sizes=args.batch_sizes,
        num_iterations=args.iterations,
        sequence_length=args.sequence_length,
        warm_up_iterations=args.warm_up
    )
    
    benchmark = ModelBenchmark(
        pytorch_model_name=args.model_name,
        coreml_model_path=args.coreml_model,
        config=config
    )
    
    logger.info("Starting benchmarks...")
    results = benchmark.run_benchmarks()
    
    # Print summary
    print("\nBenchmark Results Summary:")
    print("=" * 80)
    for batch_size in args.batch_sizes:
        print(f"\nBatch Size: {batch_size}")
        print("-" * 40)
        
        for framework in ["pytorch", "coreml"]:
            df_subset = results[
                (results["batch_size"] == batch_size) &
                (results["framework"] == framework)
            ]
            
            print(f"\n{framework.upper()}:")
            print(f"Mean Latency: {df_subset['mean_latency'].iloc[0]:.2f} ms")
            print(f"P95 Latency: {df_subset['p95_latency'].iloc[0]:.2f} ms")
            print(f"P99 Latency: {df_subset['p99_latency'].iloc[0]:.2f} ms")
    
    print("\nDetailed results and plots have been saved to benchmarks/results/")

if __name__ == "__main__":
    main()

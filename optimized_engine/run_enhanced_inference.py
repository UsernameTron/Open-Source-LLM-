"""
Demo script for running inference with the enhanced engine.
"""
from enhanced_inference import EnhancedInferenceEngine, EnhancedInferenceConfig
import logging
import argparse
from typing import List, Dict
import json
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_inference_demo(model_path: str, tokenizer_name: str):
    """Run inference demo with various types of input."""
    # Configure enhanced engine
    config = EnhancedInferenceConfig()
    config.confidence_threshold = 0.9
    config.temperature = 1.2
    
    # Initialize engine
    engine = EnhancedInferenceEngine(
        model_path=model_path,
        tokenizer_name=tokenizer_name,
        config=config
    )
    
    # Test cases
    test_cases = [
        # Standard cases
        {
            "text": "This product is absolutely amazing! The quality is outstanding.",
            "type": "standard",
            "priority": 0
        },
        {
            "text": "Terrible experience, would not recommend to anyone.",
            "type": "standard",
            "priority": 0
        },
        # Mixed sentiment
        {
            "text": "Good product but a bit expensive for what you get.",
            "type": "mixed",
            "priority": 1
        },
        {
            "text": "Fast shipping but the packaging was damaged.",
            "type": "mixed",
            "priority": 1
        },
        # Neutral cases
        {
            "text": "Product arrived on the specified date.",
            "type": "neutral",
            "priority": 2
        },
        {
            "text": "The item matches the description provided.",
            "type": "neutral",
            "priority": 2
        },
        # Edge cases
        {
            "text": "👍 Love it! 🎉 Best purchase ever! ⭐️⭐️⭐️⭐️⭐️",
            "type": "edge",
            "priority": 0
        },
        {
            "text": "n/a",
            "type": "edge",
            "priority": 2
        }
    ]
    
    results = []
    
    # Run both async and sync inference
    logger.info("Running async inference...")
    async_results = []
    request_ids = []
    
    # Submit async requests
    for case in test_cases:
        request_id = engine.infer_async(
            case["text"],
            priority=case["priority"],
            dataset_type=case["type"]
        )
        request_ids.append((request_id, case))
    
    # Collect async results
    for request_id, case in request_ids:
        result = engine.get_result(request_id)
        if result:
            async_results.append({
                "type": "async",
                "case_type": case["type"],
                "priority": case["priority"],
                "text": case["text"],
                "label": result.label,
                "confidence": result.confidence,
                "latency_ms": result.latency_ms,
                "batch_size": result.batch_size,
                "queue_depth": result.queue_depth
            })
    
    # Run sync inference
    logger.info("Running sync inference...")
    sync_results = []
    for case in test_cases:
        try:
            result = engine.infer(case["text"], priority=case["priority"])
            sync_results.append({
                "type": "sync",
                "case_type": case["type"],
                "priority": case["priority"],
                "text": case["text"],
                "label": result.label,
                "confidence": result.confidence,
                "latency_ms": result.latency_ms,
                "batch_size": result.batch_size,
                "queue_depth": result.queue_depth
            })
        except Exception as e:
            logger.error(f"Sync inference failed: {e}")
    
    results = async_results + sync_results
    
    # Analyze results
    analysis = {
        "summary": {
            "total_requests": len(results),
            "async_requests": len(async_results),
            "sync_requests": len(sync_results),
            "mean_latency_ms": sum(r["latency_ms"] for r in results) / len(results),
            "mean_confidence": sum(r["confidence"] for r in results) / len(results)
        },
        "by_type": {}
    }
    
    # Analyze by type
    for case_type in ["standard", "mixed", "neutral", "edge"]:
        type_results = [r for r in results if r["case_type"] == case_type]
        if type_results:
            analysis["by_type"][case_type] = {
                "count": len(type_results),
                "mean_confidence": sum(r["confidence"] for r in type_results) / len(type_results),
                "mean_latency_ms": sum(r["latency_ms"] for r in type_results) / len(type_results),
                "labels": [r["label"] for r in type_results]
            }
    
    # Print results
    print("\nInference Results:")
    print("-" * 50)
    print(f"Total Requests: {analysis['summary']['total_requests']}")
    print(f"Mean Latency: {analysis['summary']['mean_latency_ms']:.2f}ms")
    print(f"Mean Confidence: {analysis['summary']['mean_confidence']:.2%}")
    print("\nResults by Type:")
    for case_type, stats in analysis["by_type"].items():
        print(f"\n{case_type.upper()}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean Confidence: {stats['mean_confidence']:.2%}")
        print(f"  Mean Latency: {stats['mean_latency_ms']:.2f}ms")
        print(f"  Labels: {stats['labels']}")
    
    # Save detailed results
    output_dir = Path("inference_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run enhanced inference demo")
    parser.add_argument("--model-path", required=True, help="Path to CoreML model")
    parser.add_argument("--tokenizer", required=True, help="Name or path to tokenizer")
    args = parser.parse_args()
    
    run_inference_demo(args.model_path, args.tokenizer)

if __name__ == "__main__":
    main()

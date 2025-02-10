"""
Optimized batch inference for CoreML model.
"""
import coremltools as ct
import numpy as np
from transformers import AutoTokenizer
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
import concurrent.futures
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchOptimizedInference:
    def __init__(self, model_path: str, tokenizer_name: str, batch_size: int = 8):
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = ct.models.MLModel(model_path)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set compute units and batch size
        self.model.compute_units = ct.ComputeUnit.ALL
        self.batch_size = batch_size
        
        # Set environment variable for tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Initialize thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(os.cpu_count() * 2, 8)
        )
        
    def preprocess_batch(self, texts: List[str], max_length: int = 512):
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
        
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Run batch prediction with performance monitoring."""
        start_time = time.time()
        
        try:
            # Preprocess batch
            inputs = self.preprocess_batch(texts)
            preprocess_time = time.time() - start_time
            
            # Batch inference
            inference_start = time.time()
            predictions = self.model.predict(inputs)
            inference_time = time.time() - inference_start
            
            # Process predictions
            results = []
            logits = predictions.get('linear_37', predictions)
            
            for i, text in enumerate(texts):
                # Get logits for current text
                text_logits = logits[i] if isinstance(logits, np.ndarray) else logits
                
                # Apply softmax to get probabilities
                exp_logits = np.exp(text_logits - np.max(text_logits))  # Subtract max for numerical stability
                probabilities = exp_logits / exp_logits.sum()
                
                # Get predicted class and confidence
                predicted_class = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_class])
                
                # Map to sentiment labels (assuming binary classification: 0=negative, 1=positive)
                label = 'positive' if predicted_class == 1 else 'negative'
                    
                results.append({
                    'text': text,
                    'label': label,
                    'confidence': confidence,
                    'metrics': {
                        'preprocess_time_ms': preprocess_time * 1000 / len(texts),
                        'inference_time_ms': inference_time * 1000 / len(texts),
                        'total_time_ms': (time.time() - start_time) * 1000 / len(texts)
                    }
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
            
    def predict_large_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Handle large batches by splitting into optimal batch sizes."""
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = self.predict_batch(batch)
            results.extend(batch_results)
        return results

def main():
    parser = argparse.ArgumentParser(description='Run optimized CoreML batch inference')
    parser.add_argument('--model-path', required=True, help='Path to CoreML model')
    parser.add_argument('--tokenizer', required=True, help='Name or path to tokenizer')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for inference')
    args = parser.parse_args()
    
    # Test texts
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "The service was terrible and the food was cold.",
        "A masterpiece of modern cinema, truly outstanding!",
        "I can't believe how bad this product is.",
        "The customer support team was very helpful and professional.",
        "Not worth the money, complete waste of time.",
        "The graphics are stunning and the gameplay is smooth.",
        "Disappointing performance and lack of features."
    ]
    
    # Initialize inference
    engine = BatchOptimizedInference(args.model_path, args.tokenizer, args.batch_size)
    
    # Run batch prediction
    logger.info(f"Running batch prediction for {len(test_texts)} texts")
    start_time = time.time()
    results = engine.predict_large_batch(test_texts)
    total_time = time.time() - start_time
    
    # Print results
    print("\nBatch Prediction Results:")
    print("-" * 50)
    
    total_inference_time = 0
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        total_inference_time += result['metrics']['inference_time_ms']
    
    print("\nPerformance Summary:")
    print(f"Total Texts Processed: {len(results)}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Average Inference Time per Text: {total_inference_time/len(results):.2f}ms")
    print(f"Total Processing Time: {total_time*1000:.2f}ms")
    print(f"Throughput: {len(results)/total_time:.2f} texts/second")

if __name__ == "__main__":
    main()

"""
Simplified optimized inference script for CoreML model.
"""
import coremltools as ct
import numpy as np
from transformers import AutoTokenizer
import time
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedInference:
    def __init__(self, model_path: str, tokenizer_name: str):
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = ct.models.MLModel(model_path)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set compute units
        self.model.compute_units = ct.ComputeUnit.ALL
        
    def preprocess(self, text: str, max_length: int = 512):
        """Preprocess input text."""
        tokens = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )
        return {
            "input_ids": tokens["input_ids"].astype(np.int32),
            "attention_mask": tokens["attention_mask"].astype(np.int32)
        }
        
    def predict(self, text: str):
        """Run prediction with performance monitoring."""
        start_time = time.time()
        
        try:
            # Preprocess
            inputs = self.preprocess(text)
            preprocess_time = time.time() - start_time
            
            # Inference
            inference_start = time.time()
            prediction = self.model.predict(inputs)
            inference_time = time.time() - inference_start
            
            # Process prediction
            if isinstance(prediction, dict):
                confidence = float(prediction.get('confidence', 0.0))
                label = prediction.get('label', 'unknown')
            else:
                confidence = float(prediction[0]) if prediction else 0.0
                label = 'positive' if confidence > 0.5 else 'negative'
                
            total_time = time.time() - start_time
            
            return {
                'text': text,
                'label': label,
                'confidence': confidence,
                'metrics': {
                    'preprocess_time_ms': preprocess_time * 1000,
                    'inference_time_ms': inference_time * 1000,
                    'total_time_ms': total_time * 1000
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Run optimized CoreML inference')
    parser.add_argument('--model-path', required=True, help='Path to CoreML model')
    parser.add_argument('--tokenizer', required=True, help='Name or path to tokenizer')
    parser.add_argument('--text', default='This is a test sentence.', help='Text to analyze')
    args = parser.parse_args()
    
    # Initialize inference
    engine = OptimizedInference(args.model_path, args.tokenizer)
    
    # Run prediction
    logger.info(f"Running prediction for text: {args.text}")
    result = engine.predict(args.text)
    
    # Print results
    print("\nPrediction Results:")
    print(f"Text: {result['text']}")
    print(f"Label: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nPerformance Metrics:")
    print(f"Preprocessing Time: {result['metrics']['preprocess_time_ms']:.2f}ms")
    print(f"Inference Time: {result['metrics']['inference_time_ms']:.2f}ms")
    print(f"Total Time: {result['metrics']['total_time_ms']:.2f}ms")

if __name__ == "__main__":
    main()

"""
Inspect CoreML model structure and outputs.
"""
import coremltools as ct
import numpy as np
from transformers import AutoTokenizer
import logging
import argparse
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_model(model_path: str, tokenizer_name: str):
    """Inspect model structure and sample outputs."""
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = ct.models.MLModel(model_path)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Print model info
    print("\nModel Information:")
    print("-" * 50)
    print(f"Model Type: {type(model)}")
    print(f"Input Description:")
    for input_name, input_desc in model.input_description.items():
        print(f"  {input_name}: {input_desc}")
    print("\nOutput Description:")
    for output_name, output_desc in model.output_description.items():
        print(f"  {output_name}: {output_desc}")
        
    # Test with sample input
    sample_text = "This is a test sentence."
    tokens = tokenizer(
        sample_text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )
    
    inputs = {
        "input_ids": tokens["input_ids"].astype(np.int32),
        "attention_mask": tokens["attention_mask"].astype(np.int32)
    }
    
    # Run prediction
    print("\nSample Prediction:")
    print("-" * 50)
    print(f"Input Text: {sample_text}")
    try:
        prediction = model.predict(inputs)
        print("\nRaw Prediction Output:")
        print(json.dumps(prediction, indent=2, default=str))
        
        # Analyze prediction structure
        print("\nPrediction Structure:")
        if isinstance(prediction, dict):
            for key, value in prediction.items():
                print(f"\nKey: {key}")
                print(f"Type: {type(value)}")
                if isinstance(value, np.ndarray):
                    print(f"Shape: {value.shape}")
                    print(f"Sample Values: {value.flatten()[:5]}")
        else:
            print(f"Type: {type(prediction)}")
            if isinstance(prediction, np.ndarray):
                print(f"Shape: {prediction.shape}")
                print(f"Sample Values: {prediction.flatten()[:5]}")
                
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Inspect CoreML model')
    parser.add_argument('--model-path', required=True, help='Path to CoreML model')
    parser.add_argument('--tokenizer', required=True, help='Name or path to tokenizer')
    args = parser.parse_args()
    
    inspect_model(args.model_path, args.tokenizer)

if __name__ == "__main__":
    main()

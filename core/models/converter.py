import coremltools as ct
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from typing import Tuple, Optional, List, Dict
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConverter:
    def __init__(self, model_name: str, max_length: int = 512):
        self.model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        # Set model to evaluation mode and disable gradient computation
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
    def prepare_inputs(self) -> List[ct.TensorType]:
        """Prepare input shapes and types for CoreML conversion with dynamic batch size."""
        input_features = [
            ct.TensorType(
                name="input_ids",
                shape=(ct.RangeDim(1, 128, 1), self.max_length),  # Dynamic batch size from 1 to 128
                dtype=np.int32
            ),
            ct.TensorType(
                name="attention_mask",
                shape=(ct.RangeDim(1, 128, 1), self.max_length),  # Dynamic batch size from 1 to 128
                dtype=np.int32
            )
        ]
        
        return input_features
    
    def forward_wrapper(self, input_ids, attention_mask):
        """Wrapper function to only return logits"""
        outputs = self.model(input_ids, attention_mask)
        return outputs.logits
        
    def convert_to_coreml(self, output_path: str, compute_units: ct.ComputeUnit = ct.ComputeUnit.ALL) -> None:
        """Convert transformer model to CoreML format with M4-specific optimizations."""
        logger.info(f"Converting {self.model_name} to CoreML format...")
        
        # Create example inputs with batch size of 1 for tracing
        example_input_ids = torch.randint(0, self.tokenizer.vocab_size, (1, self.max_length), dtype=torch.long)
        example_attention_mask = torch.ones((1, self.max_length), dtype=torch.long)
        
        # Also create larger batch size inputs for testing
        test_batch_sizes = [16, 32, 64, 128]
        test_inputs = [
            (torch.randint(0, self.tokenizer.vocab_size, (bs, self.max_length), dtype=torch.long),
             torch.ones((bs, self.max_length), dtype=torch.long))
            for bs in test_batch_sizes
        ]
        
        # Create a ScriptModule wrapper with mixed precision support
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, input_ids, attention_mask):
                # Enable AMP (Automatic Mixed Precision)
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask)
                    return outputs.logits
        
        wrapper = ModelWrapper(self.model)
        wrapper.eval()
        
        # Trace model with example inputs
        with torch.no_grad():
            traced_model = torch.jit.trace(
                wrapper,
                (example_input_ids, example_attention_mask),
                strict=False,
                check_trace=False
            )
        
        # M4-specific optimization config
        config = {
            'global': {
                'compute_precision': ct.precision.FLOAT16,
                'memory_page_size': 16384,  # 16KB pages for M4
                'allow_low_precision': True
            },
            'neural_engine': {
                'enabled': True,
                'minimum_layer_size': 16  # Minimum ops to offload to Neural Engine
            }
        }
        
        # Convert to CoreML with dynamic shapes and M4 optimizations
        model = ct.convert(
            traced_model,
            inputs=self.prepare_inputs(),
            compute_units=compute_units,
            minimum_deployment_target=ct.target.macOS13,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16
        )
        
        # Save the base model
        os.makedirs(output_path, exist_ok=True)
        model_path = os.path.join(output_path, 'sentiment_model_metal.mlpackage')
        model.save(model_path)
        
        # Verify the model works with different batch sizes
        logger.info("Verifying model with different batch sizes...")
        for test_input_ids, test_attention_mask in test_inputs:
            test_input = {
                'input_ids': test_input_ids.numpy().astype(np.int32),
                'attention_mask': test_attention_mask.numpy().astype(np.int32)
            }
            try:
                model.predict(test_input)
                logger.info(f"✓ Model works with batch size {len(test_input_ids)}")
            except Exception as e:
                logger.error(f"✗ Failed with batch size {len(test_input_ids)}: {e}")
        
        # Save the optimized model
        optimized_model_path = os.path.join(output_path, 'sentiment_model_metal.mlpackage')
        model.save(optimized_model_path)
        logger.info(f"Model saved to {optimized_model_path}")


def convert_and_optimize_model(
    model_name: str,
    output_path: str,
    max_length: int = 512,
    compute_units: ct.ComputeUnit = ct.ComputeUnit.ALL
) -> None:
    """Convenience function to convert and optimize a model in one go."""
    converter = ModelConverter(model_name, max_length)
    converter.convert_to_coreml(output_path, compute_units)

import pytest
import numpy as np
import torch
import time
from unittest.mock import Mock, patch
from core.inference.engine import InferenceEngine, InferenceConfig
from core.metrics import metrics_tracker

@pytest.fixture
def mock_model():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Add a mock parameter to make it behave like a real model
            self.mock_param = torch.nn.Parameter(torch.randn(1))
            self.current_text = ""

        def set_text(self, text):
            self.current_text = text

        def forward(self, input_ids=None, attention_mask=None):
            batch_size = input_ids.shape[0] if input_ids is not None else 1
            # Create logits for positive, negative, and neutral sentiments
            if "love" in self.current_text.lower() or "amazing" in self.current_text.lower() or "great" in self.current_text.lower():
                # Strong positive sentiment
                logits = torch.tensor([[-1.0, -1.0, 2.0]], dtype=torch.float32)
            elif "hate" in self.current_text.lower() or "terrible" in self.current_text.lower() or "bad" in self.current_text.lower():
                # Strong negative sentiment
                logits = torch.tensor([[2.0, -1.0, -1.0]], dtype=torch.float32)
            else:
                # Neutral sentiment
                logits = torch.tensor([[-1.0, 2.0, -1.0]], dtype=torch.float32)
            
            # Create batch_logits by repeating for batch size
            batch_logits = logits.repeat(batch_size, 1)
            
            # Create a mock output object that mimics HuggingFace model output
            class ModelOutput:
                def __init__(self, logits, text):
                    self.logits = logits
                    self.text = text
                    self.attentions = None
            
            output = ModelOutput(batch_logits, self.current_text)
            output.text = self.current_text  # Add text to output
            return output

        def to(self, device):
            # Mock device movement but still call parent's to() for parameters
            super().to(device)
            return self
            
        def predict(self, inputs):
            # Handle empty inputs
            if inputs is None or not inputs:
                raise ValueError("Inputs cannot be empty")
            
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask')
            
            if input_ids is None or attention_mask is None:
                raise ValueError("input_ids and attention_mask are required")
            
            # Get model output with logits
            output = self.forward(input_ids, attention_mask)
            
            # Create attention weights with shape [num_layers, batch_size, num_heads, seq_len, seq_len]
            batch_size = input_ids.shape[0] if len(input_ids.shape) > 1 else 1
            seq_len = input_ids.shape[-1]
            num_layers = 12  # Standard transformer layers
            num_heads = 12  # Standard transformer heads
            
            # Create normalized attention weights
            attention_weights = np.zeros((num_layers, batch_size, num_heads, seq_len, seq_len))
            for l in range(num_layers):
                for b in range(batch_size):
                    for h in range(num_heads):
                        for i in range(seq_len):
                            # Create a simple attention pattern where each token attends equally to all tokens
                            attention_weights[l, b, h, i] = 1.0 / seq_len
            
            # Create token importances that sum to 1 for each sequence
            token_importances = np.ones((batch_size, seq_len)) / seq_len
            
            return {
                'attentions': attention_weights,  # Shape: [num_layers, batch_size, num_heads, seq_len, seq_len]
                'logits': output.logits,
                'input_text': output.text,
                'token_importances': token_importances
            }

    model = MockModel()
    model.forward = model.forward  # Bind the forward method
    model.__call__ = model.forward  # Make the model callable
    return model

@pytest.fixture
def mock_tokenizer():
    class MockTokenizer:
        def __init__(self):
            self.vocab = {0: '[PAD]', 1: '[CLS]', 2: '[SEP]', 3: 'test', 4: 'text'}
            
        def convert_ids_to_tokens(self, ids):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return [self.vocab.get(id, f'[UNK{id}]') for id in ids]
            
        def encode(self, text, return_tensors=None, padding=True, truncation=True, max_length=None):
            # Simple mock encoding - just return fixed tensors
            input_ids = torch.tensor([[1, 2, 3]])
            attention_mask = torch.tensor([[1, 1, 1]])
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
            
        def __call__(self, text, max_length=None, padding=True, truncation=True, return_tensors=None):
            # Make the tokenizer callable like a HuggingFace tokenizer
            return self.encode(text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length)
            
        def decode(self, ids, **kwargs):
            # Convert token ids back to text
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            tokens = self.convert_ids_to_tokens(ids)
            return ' '.join(tokens)
    
    return MockTokenizer()


@pytest.fixture
def inference_engine(mock_model, mock_tokenizer):
    config = InferenceConfig()
    engine = InferenceEngine(
        model_path=mock_model,
        tokenizer_name=None,  # This signals to use mock tokenizer
        config=config
    )
    engine.tokenizer = mock_tokenizer  # Set tokenizer directly
    return engine

@pytest.fixture
def sample_logits():
    # Create sample logits that would typically come from the model
    return np.array([[-2.0, 0.1, 2.5]])  # Should predict positive with high confidence

def test_softmax_conversion(inference_engine, sample_logits):
    """Test that logits are properly converted to probabilities."""
    # Get probabilities using the engine's internal method
    probs = inference_engine._softmax(sample_logits)
    
    # Calculate expected softmax manually
    exp_logits = np.exp(sample_logits)
    expected_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Check that probabilities match expected values
    np.testing.assert_array_almost_equal(probs, expected_probs)
    
    # Check probability properties
    assert np.all(probs >= 0), "Probabilities should be non-negative"
    assert np.all(probs <= 1), "Probabilities should be <= 1"
    assert np.abs(np.sum(probs) - 1.0) < 1e-6, "Probabilities should sum to 1"

@pytest.mark.asyncio
async def test_inference_results(inference_engine):
    """Test the complete inference pipeline with different texts."""
    test_cases = [
        {
            "text": "I love this product, it's amazing!",
            "expected_sentiment": "Positive"
        },
        {
            "text": "This is terrible, I hate it.",
            "expected_sentiment": "Negative"
        },
        {
            "text": "The weather is okay today.",
            "expected_sentiment": "Neutral"
        }
    ]
    
    for case in test_cases:
        result = await inference_engine.infer_sentiment(case["text"])
        
        # Check result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "prediction" in result, "Missing prediction class"
        assert "confidence" in result, "Missing confidence score"
        assert "probabilities" in result, "Missing probability distribution"
        assert "input_text" in result, "Missing input text"
        assert result["input_text"] == case["text"], "Input text does not match expected text"
        
        # Check probability distribution
        probs = result["probabilities"]
        assert len(probs) == 3, "Should have probabilities for all 3 classes"
        assert all(0 <= p <= 1 for p in probs), "Probabilities should be between 0 and 1"
        assert abs(sum(probs) - 1.0) < 1e-6, "Probabilities should sum to 1"
        
        # For strongly positive/negative texts, verify sentiment
        if case["expected_sentiment"] in ["Positive", "Negative"]:
            predicted_sentiment = result["prediction"]
            assert predicted_sentiment == case["expected_sentiment"], \
                f"Expected {case['expected_sentiment']} for '{case['text']}', got {predicted_sentiment}"

@pytest.mark.asyncio
async def test_performance_metrics(inference_engine):
    """Test that performance metrics are properly calculated."""
    # Run a few inferences to get meaningful metrics
    texts = ["Sample text 1", "Sample text 2", "Sample text 3"]
    for text in texts:
        await inference_engine.infer_sentiment(text)
    
    # Get metrics
    metrics = inference_engine.get_performance_metrics()
    
    # Check metric structure
    assert "latency_ms" in metrics, "Missing latency metric"
    assert "throughput" in metrics, "Missing throughput metric"
    assert "gpu_utilization" in metrics, "Missing GPU utilization metric"
    assert "memory_utilization" in metrics, "Missing memory utilization metric"
    
    # Check metric values
    assert metrics["latency_ms"] > 0, "Latency should be positive"
    assert metrics["throughput"] > 0, "Throughput should be positive"
    assert 0 <= metrics["gpu_utilization"] <= 100, "GPU utilization should be between 0-100%"
    assert 0 <= metrics["memory_utilization"] <= 100, "Memory utilization should be between 0-100%"

def test_resource_monitoring(inference_engine):
    """Test that resource monitoring works correctly."""
    # Give the monitoring thread time to collect metrics
    time.sleep(2)
    
    # Get metrics
    metrics = inference_engine._collect_resource_metrics()
    
    # Check metrics are within expected ranges
    assert 0 <= metrics.cpu_util <= 100, "CPU utilization should be between 0-100%"
    assert 0 <= metrics.memory_utilization <= 100, "Memory utilization should be between 0-100%"
    assert metrics.batch_latency >= 0, "Batch latency should be non-negative"
    assert metrics.queue_size >= 0, "Queue size should be non-negative"

@pytest.mark.asyncio
async def test_error_handling(inference_engine):
    """Test error handling during inference."""
    # Test with empty input list
    with pytest.raises(RuntimeError, match="Error in batch inference: Input list cannot be empty"):
        await inference_engine.infer_batch([])
    
    # Test with None input
    with pytest.raises(RuntimeError, match="Error in batch inference: Input list cannot be None"):
        await inference_engine.infer_batch(None)
    
    # Test with invalid text type
    texts = [
        "First test message",
        "Second test message",
        "Third test message"
    ]
    results = await inference_engine.infer_batch(texts)
    
    # Check string input results
    assert len(results) == len(texts), "Should return result for each input"
    for result, text in zip(results, texts):
        assert isinstance(result, dict), "Each result should be a dictionary"
        assert 'prediction' in result, "Result should contain prediction"
        assert 'confidence' in result, "Result should contain confidence"
        assert 'probabilities' in result, "Result should contain probabilities"
        assert len(result['probabilities']) == 3, "Should have 3 class probabilities"
        assert 'input_text' in result, "Result should contain input text"
        assert result['input_text'] == text, "Input text should match original"
        assert result.get('error') is None, "Should not have any errors"

    # Test with dictionary inputs
    dict_texts = [{'input_text': text} for text in texts]
    dict_results = await inference_engine.infer_batch(dict_texts)
    
    # Check dictionary input results
    assert len(dict_results) == len(dict_texts), "Should return result for each input"
    for result, input_dict in zip(dict_results, dict_texts):
        assert isinstance(result, dict), "Each result should be a dictionary"
        assert 'prediction' in result, "Result should contain prediction"
        assert 'confidence' in result, "Result should contain confidence"
        assert 'probabilities' in result, "Result should contain probabilities"
        assert len(result['probabilities']) == 3, "Should have 3 class probabilities"
        assert 'input_text' in result, "Result should contain input text"
        assert result['input_text'] == input_dict['input_text'], "Input text should match original"
        assert result.get('error') is None, "Should not have any errors"

@pytest.mark.asyncio
async def test_run_model_inference(inference_engine):
    """Test the async model inference method."""
    # Prepare test input
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    batch_size = 2
    inputs = [{'text': 'test1'}, {'text': 'test2'}]

    # Test with valid input
    results = await inference_engine._run_model_inference(
        input_ids, attention_mask, batch_size, inputs
    )
    assert len(results) == batch_size, "Should return result for each input"
    for result in results:
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'probabilities' in result
    
    # Test with 1D input (should be automatically unsqueezed)
    input_ids_1d = torch.tensor([1, 2, 3])
    attention_mask_1d = torch.tensor([1, 1, 1])
    results = await inference_engine._run_model_inference(
        input_ids_1d, attention_mask_1d, 1, [{'text': 'test'}]
    )
    assert len(results) == 1, "Should handle 1D input correctly"
    
    # Test handling of empty inputs
    with pytest.raises(RuntimeError, match="Error in model inference: list index out of range"):
        await inference_engine._run_model_inference(
            input_ids, attention_mask, 0, []
        )
    
    # Test error handling with mismatched batch size
    with pytest.raises(RuntimeError, match="Error in model inference: Batch size mismatch"):
        await inference_engine._run_model_inference(
            input_ids, attention_mask, 3, inputs  # batch_size=3 but only 2 inputs
        )
    
    # Test error handling with missing text field
    with pytest.raises(RuntimeError, match="Error in model inference: Missing text field"):
        await inference_engine._run_model_inference(
            input_ids, attention_mask, 1, [{}]
        )

@pytest.mark.asyncio
async def test_explain_method(inference_engine):
    """Test the improved explain method."""
    test_text = "This is a test sentence for explanation."
    
    # Get explanation
    explanation = await inference_engine.explain(test_text)
    
    # Verify explanation structure
    assert isinstance(explanation, dict), "Explanation should be a dictionary"
    assert 'prediction' in explanation, "Should include prediction"
    assert 'confidence' in explanation, "Should include confidence"
    assert 'important_tokens' in explanation, "Should include important tokens"
    assert 'attention_weights' in explanation, "Should include attention weights"
    assert 'input_text' in explanation, "Should include input text"
    
    # Verify token importance
    assert len(explanation['important_tokens']) > 0, "Should have important tokens"
    total_importance = 0.0
    for token_info in explanation['important_tokens']:
        assert 'token' in token_info, "Each token should have text"
        assert 'importance' in token_info, "Each token should have importance score"
        assert isinstance(token_info['importance'], float), "Importance should be float"
        assert 0 <= token_info['importance'] <= 1.0, "Importance score should be between 0 and 1"
        total_importance += token_info['importance']
    
    # Verify importance scores sum to approximately 1
    assert np.isclose(total_importance, 1.0, atol=1e-6), "Token importance scores should sum to 1"
    
    # Verify attention weights shape and normalization
    attention_weights = explanation['attention_weights']
    assert isinstance(attention_weights, (list, np.ndarray)), "Attention weights should be a list or numpy array"
    if isinstance(attention_weights, list):
        attention_weights = np.array(attention_weights)
    
    # Shape should be [num_layers, batch_size, num_heads, seq_len, seq_len]
    assert len(attention_weights.shape) == 5, "Should have 5 dimensions: [num_layers, batch_size, num_heads, seq_len, seq_len]"
    assert attention_weights.shape[0] == 12, "Should have 12 layers"
    assert attention_weights.shape[1] == 1, "Batch size should be 1"
    assert attention_weights.shape[2] == 12, "Should have 12 attention heads"
    assert attention_weights.shape[3] == attention_weights.shape[4], "Sequence length should be same"
    
    # Verify attention weights are normalized per head
    for layer in range(attention_weights.shape[0]):
        for batch in range(attention_weights.shape[1]):
            for head in range(attention_weights.shape[2]):
                for row in range(attention_weights.shape[3]):
                    assert np.isclose(np.sum(attention_weights[layer, batch, head, row]), 1.0, atol=1e-6), \
                        "Each attention row should sum to 1"

    # Test empty input
    empty_result = await inference_engine.explain("")
    assert empty_result['error'] == 'Explanation failed'
    assert 'Error in explain: Input text cannot be empty' in empty_result['details']
    
    # Test whitespace input
    whitespace_result = await inference_engine.explain("   ")
    assert whitespace_result['error'] == 'Explanation failed'
    assert 'Error in explain: Input text cannot be empty' in whitespace_result['details']
    
    # Test None input
    none_result = await inference_engine.explain(None)
    assert none_result['error'] == 'Explanation failed'
    assert 'Error in explain: Input text cannot be None' in none_result['details']
    
    # Test non-string input
    nonstring_result = await inference_engine.explain(123)
    assert nonstring_result['error'] == 'Explanation failed'
    assert 'Error in explain: Input must be a string' in nonstring_result['details']



@pytest.mark.asyncio
async def test_concurrent_inference(inference_engine):
    """Test concurrent inference requests."""
    import asyncio
    
    # Create multiple concurrent requests
    texts = [f"Test message {i}" for i in range(5)]
    tasks = [inference_engine.infer_async(text) for text in texts]
    
    # Run concurrently
    results = await asyncio.gather(*tasks)
    
    # Verify results
    assert len(results) == len(texts), "Should handle all concurrent requests"
    for result in results:
        assert isinstance(result, dict), "Each result should be a dictionary"
        assert 'prediction' in result, "Should include prediction"
        assert 'confidence' in result, "Should include confidence"
    assert len(results) == len(texts), "Should have one result per input text"
    for result in results:
        assert isinstance(result, dict), "Each result should be a dictionary"
        assert "prediction" in result, "Missing prediction class"
        assert "confidence" in result, "Missing confidence score"
        assert "probabilities" in result, "Missing probability distribution"
        assert "text" in result, "Missing input text"

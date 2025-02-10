import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from core.inference.engine import InferenceEngine, InferenceConfig, ResourceMetrics
from hypothesis import given, strategies as st, settings, HealthCheck

@pytest.fixture
def mock_model_spec():
    spec = MagicMock()
    input_type = MagicMock()
    input_type.multiArrayType.shape = [1, 512]  # Example input shape
    spec.description.input = [input_type]
    return spec

@pytest.fixture
def mock_model(mock_model_spec):
    model = Mock()
    def predict(**kwargs):
        # Get input size from kwargs
        input_ids = kwargs.get('input_ids', [])
        if isinstance(input_ids, torch.Tensor):
            batch_size = input_ids.shape[0]
        else:
            batch_size = len(input_ids) if isinstance(input_ids, list) else 1
        # Return batch_size predictions with probabilities
        return {
            'logits': torch.tensor([[0.7, 0.3]] * batch_size),
            'prediction': {
                'prediction': 'Positive',
                'confidence': 0.8,
                'probabilities': {'Positive': 0.7, 'Negative': 0.3}
            },
            'performance': {
                'latency_ms': 10.0,
                'throughput': 100.0,
                'gpu_utilization': 0.5,
                'memory_utilization': 0.3
            }
        }
    
    def getitem(key):
        if key == 'logits':
            return torch.tensor([[0.7, 0.3]])
        return predict()[key]
    
    model.__call__ = predict
    model.__getitem__ = getitem
    model.get_spec.return_value = mock_model_spec
    model.type = Mock()
    model.type.multiArrayType = Mock()
    model.type.multiArrayType.shape = [1, 2]
    model.predict = predict
    model.get_performance_metrics = lambda: {
        'latency_ms': 10.0,
        'throughput': 100.0,
        'gpu_utilization': 0.5,
        'memory_utilization': 0.3
    }
    return model

@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.encode.return_value = {'input_ids': np.array([[1, 2, 3]]), 'attention_mask': np.array([[1, 1, 1]])}
    tokenizer.decode.return_value = "test output"
    return tokenizer

@pytest.fixture
def inference_config():
    return InferenceConfig(
        min_batch_size=8,
        max_batch_size=32,
        target_latency_ms=15.0,
        enable_caching=True,
        cache_size=100
    )

@pytest.fixture
def mock_metrics_collector():
    collector = Mock()
    collector.record_latency = Mock()
    collector.record_batch_size = Mock()
    collector.record_cache_hit = Mock()
    collector.record_cache_miss = Mock()
    collector.record_resource_usage = Mock()
    return collector

@pytest.fixture
def inference_engine(mock_model, mock_tokenizer, mock_metrics_collector, inference_config):
    with patch('coremltools.models.MLModel') as mock_mlmodel, \
         patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_init, \
         patch('core.inference.engine.MetricsCollector') as mock_metrics:
        mock_mlmodel.return_value = mock_model
        mock_tokenizer_init.return_value = mock_tokenizer
        mock_metrics.return_value = mock_metrics_collector
        engine = InferenceEngine(
            model_path="dummy_path",
            tokenizer_name="dummy_tokenizer",
            config=inference_config
        )
        return engine

def test_normalize_confidence(inference_engine):
    """Test confidence score normalization"""
    test_cases = [0.1, 0.5, 0.9]
    for confidence in test_cases:
        normalized = inference_engine._normalize_confidence(confidence)
        assert 0 <= normalized <= 1, f"Normalized confidence {normalized} out of range"

@pytest.mark.asyncio
async def test_batch_inference(inference_engine, mock_model):
    """Test batch inference functionality"""
    inputs = [
        {
            'input_ids': np.array([[1, 2, 3]]),
            'attention_mask': np.array([[1, 1, 1]])
        },
        {
            'input_ids': np.array([[4, 5, 6]]),
            'attention_mask': np.array([[1, 1, 1]])
        }
    ]
    
    results = await inference_engine._batch_inference(inputs)
    
    assert len(results) == 2, "Should return results for all inputs"
    for result in results:
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'raw_confidence' in result
        assert 'probabilities' in result
        assert 0 <= result['confidence'] <= 1

@pytest.mark.asyncio
async def test_caching(inference_engine):
    """Test caching functionality"""
    # Create test inputs
    inputs = [
        {
            'input_ids': np.array([[1, 2, 3]]),
            'attention_mask': np.array([[1, 1, 1]])
        }
    ]
    
    # First call should compute result
    first_result = await inference_engine._batch_inference(inputs)
    
    # Second call should return cached result
    with patch.object(inference_engine.model, 'predict') as mock_predict:
        second_result = await inference_engine._batch_inference(inputs)
        mock_predict.assert_not_called()
    
    assert first_result == second_result, "Cached result should match original"

@pytest.mark.hypothesis
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(confidence=st.floats(min_value=0.0, max_value=1.0))
def test_confidence_normalization_properties(inference_engine, confidence):
    """Property-based test for confidence normalization"""
    # Test with temperature=1.0 for identity mapping
    normalized = inference_engine._normalize_confidence(confidence, temperature=1.0)
    assert 0 <= normalized <= 1, "Normalized confidence should be between 0 and 1"
    assert abs(normalized - confidence) < 1e-6, "At temperature=1.0, normalized should equal input"
    
    # Test with temperature=0.5 for contrast enhancement
    normalized_t05 = inference_engine._normalize_confidence(confidence, temperature=0.5)
    assert 0 <= normalized_t05 <= 1, "Normalized confidence should be between 0 and 1"
    
    # Test that temperature scaling makes values more extreme
    if confidence > 0:
        # For temperature < 1, values get pushed lower due to raising to power > 1
        assert normalized_t05 <= confidence, "Temperature<1 should push values lower"
    elif confidence == 0:
        assert normalized_t05 == 0, "Zero confidence should remain zero"
        assert normalized_t05 <= confidence, "Temperature<1 should push low values lower"

@pytest.mark.asyncio
async def test_resource_monitoring(inference_engine):
    """Test resource monitoring functionality"""
    metrics = inference_engine._collect_resource_metrics()
    
    assert hasattr(metrics, 'gpu_utilization')
    assert hasattr(metrics, 'memory_utilization')
    assert hasattr(metrics, 'batch_latency')
    assert hasattr(metrics, 'queue_size')
    assert hasattr(metrics, 'throughput')
    
    assert 0 <= metrics.memory_utilization <= 100
    assert metrics.queue_size >= 0
    assert metrics.throughput >= 0

def test_batch_size_adjustment(inference_engine):
    """Test dynamic batch size adjustment"""
    # Mock metrics for testing
    metrics = ResourceMetrics(
        gpu_utilization=50.0,
        memory_utilization=60.0,
        batch_latency=10.0,
        queue_size=5,
        throughput=100.0
    )
    
    # Initial batch size should be min_batch_size
    assert inference_engine._current_batch_size == inference_engine.config.min_batch_size
    
    # Test adjustment
    inference_engine._adjust_batch_size(metrics)
    
    # Batch size should stay within bounds
    assert inference_engine.config.min_batch_size <= inference_engine._current_batch_size <= inference_engine.config.max_batch_size

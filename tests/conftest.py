import pytest
import os
import sys
import duckdb
import torch
import asyncio
from fastapi.testclient import TestClient
from api.main import app, get_db

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Configure pytest
def pytest_configure(config):
    """Configure pytest for our test suite."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )

@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture providing path to test data directory."""
    return os.path.join(os.path.dirname(__file__), 'data')

@pytest.fixture(scope="session")
def test_db():
    # Use an in-memory database for testing
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS inference_results_id_seq;
            CREATE TABLE IF NOT EXISTS inference_results (
                id INTEGER PRIMARY KEY DEFAULT(nextval('inference_results_id_seq')),
                text TEXT NOT NULL,
                prediction JSON,
                explanation JSON,
                metrics JSON,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        yield conn
    finally:
        conn.close()

@pytest.fixture(scope="session")
def override_get_db(test_db):
    def _get_db():
        try:
            yield test_db
        finally:
            pass  # Don't close here, we'll close in the test_db fixture
    return _get_db

from core.metrics import metrics_tracker

@pytest.fixture(scope="session")
def mock_engine():
    from core.inference.engine import InferenceEngine, InferenceConfig
    from unittest.mock import Mock, AsyncMock
    import numpy as np
    import torch

    # Mock model setup
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device('cpu')
            self.embedding = torch.nn.Parameter(torch.randn(1))

        def parameters(self):
            return iter([self.embedding])

        def forward(self, input_ids=None, attention_mask=None):
            """Generate sentiment predictions based on input text content.
            
            Args:
                input_ids: Tensor of token IDs [batch_size, seq_len]
                attention_mask: Tensor of attention mask [batch_size, seq_len]
                
            Returns:
                Object with logits attribute
            """
            batch_size = input_ids.shape[0]
            logits = []
            
            # Define token ranges for sentiment
            positive_tokens = range(100, 105)  # love, great, amazing, excellent, best
            negative_tokens = range(200, 205)  # terrible, awful, bad, poor, worst
            
            for i in range(batch_size):
                tokens = input_ids[i].tolist()
                
                # Count sentiment indicators
                pos_count = sum(1 for t in tokens if t in positive_tokens)
                neg_count = sum(1 for t in tokens if t in negative_tokens)
                
                # Convert counts to logits
                # Logits order: [negative, neutral, positive]
                if pos_count > 0 and neg_count == 0:
                    # Strong positive sentiment
                    logits.append([-5.0, -3.0, 5.0])  # High positive confidence
                elif neg_count > 0 and pos_count == 0:
                    # Strong negative sentiment
                    logits.append([5.0, -3.0, -5.0])  # High negative confidence
                elif pos_count == 0 and neg_count == 0:
                    # Neutral sentiment (no sentiment words)
                    logits.append([-3.0, 5.0, -3.0])  # Strong neutral preference
                else:
                    # Mixed sentiment (equal positive and negative)
                    logits.append([-1.0, 3.0, -1.0])  # Weak neutral
            
            logits = torch.tensor(logits, dtype=torch.float32)
            # Create output with logits attribute and make it subscriptable
            class MockOutput:
                def __init__(self, logits):
                    self.logits = logits
                def __getitem__(self, key):
                    return getattr(self, key)
            return MockOutput(logits)

    class TokenizerOutput(dict):
        def __init__(self, input_ids, attention_mask):
            super().__init__()
            self['input_ids'] = input_ids
            self['attention_mask'] = attention_mask

    class MockTokenizer:
        def __init__(self):
            self.vocab = {
                '[PAD]': 0,
                '[UNK]': 1,
                'love': 100,
                'great': 101,
                'amazing': 102,
                'excellent': 103,
                'best': 104,
                'terrible': 200,
                'awful': 201,
                'bad': 202,
                'poor': 203,
                'worst': 204
            }
            self.max_length = 128

        def __call__(self, text, max_length=None, padding=None, truncation=None, return_tensors=None):
            if isinstance(text, str):
                texts = [text]
            else:
                texts = text
                
            batch_size = len(texts)
            seq_length = max_length or self.max_length
            
            # Initialize tensors
            input_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
            attention_mask = torch.zeros((batch_size, seq_length), dtype=torch.long)
            
            for i, text in enumerate(texts):
                # Tokenize by looking for known words
                words = text.lower().split()
                tokens = []
                for word in words:
                    if word in self.vocab:
                        tokens.append(self.vocab[word])
                    else:
                        tokens.append(self.vocab['[UNK]'])
                        
                # Pad or truncate
                if len(tokens) > seq_length:
                    tokens = tokens[:seq_length]
                else:
                    tokens.extend([self.vocab['[PAD]']] * (seq_length - len(tokens)))
                    
                # Set attention mask
                mask = [1] * min(len(words), seq_length) + [0] * max(0, seq_length - len(words))
                
                # Convert to tensors
                input_ids[i] = torch.tensor(tokens)
                attention_mask[i] = torch.tensor(mask)
            
            return TokenizerOutput(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
    
    # Create mock components
    mock_model = MockModel()
    mock_tokenizer = MockTokenizer()
    
    # Create config
    config = InferenceConfig(
        min_batch_size=1,
        max_batch_size=4,
        max_length=128,
        enable_caching=True
    )
    
    # Create engine with mock model
    engine = InferenceEngine(
        model_path=mock_model,  # Pass mock model directly
        tokenizer_name=None,    # We'll set the tokenizer directly
        config=config
    )
    
    # Set mock tokenizer directly
    engine.tokenizer = mock_tokenizer
    
    # Return the engine for testing
    return engine

@pytest.fixture(scope="session")
def client(override_get_db, mock_engine):
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)

@pytest.fixture(scope="function")
def mock_gpu_utils(monkeypatch):
    """Mock GPU utilities to avoid actual GPU checks in tests."""
    from unittest.mock import Mock
    mock_gpu = Mock()
    mock_gpu.get_gpu_memory_info.return_value = (1000, 2000)  # Free, Total in MB
    mock_gpu.get_gpu_utilization.return_value = 50.0  # Percentage
    monkeypatch.setattr("core.utils.gpu_utils", mock_gpu)


    
    async def mock_infer_async(text):
        # Tokenize the input
        tokens = engine.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        
        # Run model inference
        outputs = engine.model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
            
        return {
            'prediction': int(pred),
            'confidence': float(probs.max()),
            'probabilities': probs[0].tolist(),
            'performance': metrics_tracker.get_metrics()
        }
    
    engine.infer_async = AsyncMock(side_effect=mock_infer_async)
    
    async def explain(text):
        result = await mock_infer_async(text)
        pred = result['prediction']
        probs = result['probabilities']
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiment = sentiment_map[pred]
        
        return {
            'summary': f'This text appears to be {sentiment}',
            'confidence_analysis': f'High confidence prediction ({max(probs):.1f})',
            'sentiment_breakdown': {
                'Negative': f'{probs[0]*100:.0f}%',
                'Neutral': f'{probs[1]*100:.0f}%',
                'Positive': f'{probs[2]*100:.0f}%'
            }
        }
    
    async def mock_batch_inference(inputs):
        """Process a batch of inputs for inference.
        
        Args:
            inputs: List of preprocessed inputs (dictionaries with 'input_ids' and 'attention_mask')
            or list of strings to be tokenized
            
        Returns:
            List of dictionaries containing inference results with standardized format
        """
        if not inputs:
            return []
            
        # Get current metrics
        metrics = metrics_tracker.get_metrics()
        
        # Standardize input format
        if isinstance(inputs[0], str):
            # Batch tokenize all texts at once
            tokenized = engine.tokenizer(
                inputs,
                max_length=engine.config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            batch_input_ids = tokenized['input_ids']
            batch_attention_mask = tokenized['attention_mask']
            texts = inputs
        else:
            # Combine preprocessed tensors into batches
            if torch.is_tensor(inputs[0]['input_ids']):
                batch_input_ids = torch.stack([x['input_ids'] for x in inputs], dim=0)
                batch_attention_mask = torch.stack([x['attention_mask'] for x in inputs], dim=0)
            else:
                batch_input_ids = torch.tensor(np.stack([x['input_ids'] for x in inputs], axis=0))
                batch_attention_mask = torch.tensor(np.stack([x['attention_mask'] for x in inputs], axis=0))
            texts = [x.get('text', '') for x in inputs]
        
        # Run model inference
        with torch.no_grad():
            outputs = engine.model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask
            )
        
        # Process results
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        
        # Map predictions to sentiment labels
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        results = []
        for idx, (text, pred, prob) in enumerate(zip(texts, preds, probs)):
            results.append({
                'text': text,
                'prediction': sentiment_map[int(pred)],
                'confidence': float(prob.max()),
                'probabilities': prob.tolist(),
                'performance': metrics,
                'logits': logits[idx].tolist()
            })
            
        # Return results after a small delay to simulate async behavior
        await asyncio.sleep(0.01)
        return results

    async def mock_infer_async(text):
        """Mock async inference for a single text input."""
        # Get current metrics
        metrics = metrics_tracker.get_metrics()
        
        # Tokenize input
        tokens = engine.tokenizer(
            text,
            max_length=engine.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Run model inference
        with torch.no_grad():
            outputs = engine.model(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask']
            )
        
        # Process results
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
        
        # Map prediction to sentiment
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        result = {
            'text': text,
            'prediction': sentiment_map[int(pred)],
            'confidence': float(probs.max()),
            'probabilities': probs[0].tolist(),
            'performance': metrics,
            'logits': logits[0].tolist()
        }
        
        # Add small delay to simulate async behavior
        await asyncio.sleep(0.01)
        return result

    async def explain(text):
        """Generate explanation for model prediction."""
        result = await mock_infer_async(text)
        sentiment = result['prediction']
        confidence = result['confidence']
        
        return {
            'summary': f'This text appears to be {sentiment}',
            'confidence_analysis': f'Prediction confidence: {confidence:.2f}',
            'sentiment_breakdown': {
                'Negative': f"{result['probabilities'][0]*100:.1f}%",
                'Neutral': f"{result['probabilities'][1]*100:.1f}%",
                'Positive': f"{result['probabilities'][2]*100:.1f}%"
            }
        }

    # Set up the async mock methods
    engine._batch_inference = mock_batch_inference
    engine.infer_batch = AsyncMock(side_effect=mock_batch_inference)
    engine.infer_sentiment = AsyncMock(side_effect=mock_infer_async)
    engine.explain = AsyncMock(side_effect=explain)

    # Mock config
    engine.config = Mock()
    engine.config.max_length = 128

    # Set up the tokenizer and model
    engine.tokenizer = MockTokenizer()
    engine.model = MockModel()

    return engine

@pytest.fixture(scope="session")
def client(override_get_db, mock_engine):
    app.dependency_overrides[get_db] = override_get_db
    # Override the engine
    import api.main
    api.main.engine = mock_engine
    try:
        client = TestClient(app)
        yield client
    finally:
        app.dependency_overrides.clear()

@pytest.fixture(autouse=True)
def mock_gpu_utils(monkeypatch):
    """Mock GPU utilities to avoid actual GPU checks in tests."""
    class MockGPU:
        def __init__(self):
            self.load = 0.0
            
    def mock_get_gpus():
        return [MockGPU()]
        
    monkeypatch.setattr("GPUtil.getGPUs", mock_get_gpus)

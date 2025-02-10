import pytest

@pytest.fixture
def sample_texts():
    return [
        ("I love this product, it's amazing!", "Positive"),
        ("This is terrible, I hate it.", "Negative"),
        ("The weather is okay today.", "Neutral")
    ]

def test_sentiment_predictions(client, sample_texts):
    """Test that each sample text gets appropriate sentiment predictions."""
    for text, expected_sentiment in sample_texts:
        response = client.post("/analyze", json={"text": text})
        assert response.status_code == 200
        
        data = response.json()
        prediction = data["output"]
        
        # Check prediction structure
        assert "prediction" in prediction, "Missing prediction class"
        assert "confidence" in prediction, "Missing confidence score"
        assert "probabilities" in prediction, "Missing probability distribution"
        assert "text" in prediction, "Missing input text"
        
        # Check probability distribution
        probs = prediction["probabilities"]
        assert len(probs) == 3, "Should have probabilities for all 3 classes"
        assert all(0 <= p <= 1 for p in probs), "Probabilities should be between 0 and 1"
        assert abs(sum(probs) - 1.0) < 1e-6, "Probabilities should sum to 1"
        
        # Check confidence score
        assert 0 <= prediction["confidence"] <= 1, "Confidence should be between 0 and 1"
        
        # Map prediction class to sentiment label
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        predicted_sentiment = sentiment_map[prediction["prediction"]]
        
        # For strongly positive/negative texts, check sentiment matches
        if expected_sentiment in ["Positive", "Negative"]:
            assert predicted_sentiment == expected_sentiment, \
                f"Expected {expected_sentiment} for '{text}', got {predicted_sentiment}"

def test_metrics_structure(client):
    """Test that performance metrics have the correct structure."""
    response = client.post("/analyze", json={"text": "Test text"})
    assert response.status_code == 200
    
    data = response.json()
    assert "metrics" in data, "Response should include metrics"
    metrics = data["metrics"]
    
    # Check required metrics fields
    required_metrics = ["latency_ms", "throughput", "gpu_utilization", "memory_utilization"]
    for metric in required_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        assert isinstance(metrics[metric], (int, float)), f"Metric {metric} should be numeric"
        
    # Check metric ranges
    assert metrics["latency_ms"] >= 0, "Latency should be non-negative"
    assert metrics["throughput"] >= 0, "Throughput should be non-negative"
    assert 0 <= metrics["gpu_utilization"] <= 100, "GPU utilization should be between 0-100%"
    assert 0 <= metrics["memory_utilization"] <= 100, "Memory utilization should be between 0-100%"

def test_visualization_data(client):
    """Test that the response includes all necessary data for visualization."""
    text = "This is a test message."
    response = client.post("/analyze", json={"text": text})
    assert response.status_code == 200
    
    data = response.json()
    output = data["output"]
    
    # Check data needed for bar chart
    assert "probabilities" in output, "Missing probabilities for bar chart"
    probs = output["probabilities"]
    assert len(probs) == 3, "Need exactly 3 probabilities for sentiment classes"
    
    # Check data needed for confidence meter
    assert "confidence" in output, "Missing confidence for confidence meter"
    assert isinstance(output["confidence"], (int, float)), "Confidence should be numeric"
    
    # Check sentiment label data
    assert "prediction" in output, "Missing prediction for sentiment label"
    assert output["prediction"] in [0, 1, 2], "Invalid prediction class"
    
    # Check input text display
    assert "text" in output, "Missing input text for display"
    assert output["text"] == text, "Input text should match original"

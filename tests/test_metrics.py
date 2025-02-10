import pytest
import time

def test_metrics_update(client):
    """Test that metrics are updated after each inference."""
    # First request to establish baseline
    response1 = client.post("/analyze", json={"text": "Initial test"})
    assert response1.status_code == 200
    metrics1 = response1.json()["metrics"]
    
    # Wait a moment to ensure metrics will be different
    time.sleep(0.1)
    
    # Second request to check metric updates
    response2 = client.post("/analyze", json={"text": "Second test"})
    assert response2.status_code == 200
    metrics2 = response2.json()["metrics"]
    
    # Metrics should be different between requests
    assert metrics1 != metrics2, "Metrics should update between requests"
    
    # Throughput should be calculated
    assert metrics2["throughput"] > 0, "Throughput should be positive"

def test_metrics_format(client):
    """Test that metrics are properly formatted."""
    response = client.post("/analyze", json={"text": "Test text"})
    assert response.status_code == 200
    metrics = response.json()["metrics"]
    
    # Check metric formatting
    assert isinstance(metrics["latency_ms"], (int, float)), "Latency should be numeric"
    assert isinstance(metrics["throughput"], (int, float)), "Throughput should be numeric"
    assert isinstance(metrics["gpu_utilization"], (int, float)), "GPU utilization should be numeric"
    assert isinstance(metrics["memory_utilization"], (int, float)), "Memory utilization should be numeric"
    
    # Check decimal places
    latency_str = str(metrics["latency_ms"])
    assert len(latency_str.split(".")[-1]) <= 2, "Latency should have at most 2 decimal places"
    
    throughput_str = str(metrics["throughput"])
    assert len(throughput_str.split(".")[-1]) <= 2, "Throughput should have at most 2 decimal places"

def test_metrics_ranges(client):
    """Test that metrics stay within expected ranges."""
    response = client.post("/analyze", json={"text": "Test text"})
    assert response.status_code == 200
    metrics = response.json()["metrics"]
    
    # Check value ranges
    assert metrics["latency_ms"] >= 0, "Latency should be non-negative"
    assert metrics["throughput"] >= 0, "Throughput should be non-negative"
    assert 0 <= metrics["gpu_utilization"] <= 100, "GPU utilization should be between 0-100%"
    assert 0 <= metrics["memory_utilization"] <= 100, "Memory utilization should be between 0-100%"

def test_metrics_under_load(client):
    """Test metrics behavior under multiple rapid requests."""
    texts = [f"Test text {i}" for i in range(5)]
    responses = []
    
    # Send multiple requests in quick succession
    for text in texts:
        response = client.post("/analyze", json={"text": text})
        assert response.status_code == 200
        responses.append(response.json()["metrics"])
    
    # Check that throughput increases under load
    assert responses[-1]["throughput"] > responses[0]["throughput"], \
        "Throughput should increase under load"
    
    # Check that latency remains reasonable
    for metrics in responses:
        assert metrics["latency_ms"] < 5000, "Latency should not exceed 5 seconds"

def test_metrics_persistence(client):
    """Test that metrics persist between requests."""
    # Initial request
    response1 = client.post("/analyze", json={"text": "First test"})
    assert response1.status_code == 200
    
    # Get metrics directly
    response_stats = client.get("/stats")
    assert response_stats.status_code == 200
    stats = response_stats.json()
    
    # Compare metrics from both endpoints
    metrics1 = response1.json()["metrics"]
    assert abs(metrics1["throughput"] - stats["throughput"]) < 0.1, \
        "Metrics should be consistent between endpoints"

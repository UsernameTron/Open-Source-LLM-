import json
import pytest

@pytest.fixture
def valid_text():
    return "This is a test message that should work perfectly fine."

@pytest.fixture
def empty_text():
    return ""

@pytest.fixture
def long_text():
    return "A" * 10001

def test_analyze_valid_input(client, valid_text):
    """Test that a valid text input returns 200 and correct prediction structure."""
    response = client.post("/analyze", json={"text": valid_text})
    data = response.json()
    
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    assert "prediction" in data, "Response should contain 'prediction'"
    assert isinstance(data["prediction"], dict), "Prediction should be a dictionary"

def test_analyze_invalid_input_empty_text(client, empty_text):
    """Test that empty text triggers a 400 error."""
    response = client.post("/analyze", json={"text": empty_text})
    assert response.status_code == 400, f"Empty text should trigger a 400 error, got: {response.status_code}"
    
def test_analyze_invalid_input_too_long_text(client, long_text):
    """Test that overly long text triggers a 400 error."""
    response = client.post("/analyze", json={"text": long_text})
    assert response.status_code == 400, f"Too long text should trigger a 400 error, got: {response.status_code}"

def test_analyze_invalid_input_wrong_type(client):
    """Test that non-string input triggers a 400 error."""
    response = client.post("/analyze", json={"text": 123})
    assert response.status_code == 400, f"Non-string input should trigger a 400 error, got: {response.status_code}"

def test_analyze_missing_text_field(client):
    """Test that missing text field triggers a 422 error."""
    response = client.post("/analyze", json={})
    assert response.status_code == 422, f"Missing text field should trigger a 422 error, got: {response.status_code}"

def test_analyze_malformed_json(client):
    """Test that malformed JSON triggers a 422 error."""
    response = client.post("/analyze", data="not json")
    assert response.status_code == 422, f"Malformed JSON should trigger a 422 error, got: {response.status_code}"

def test_analyze_response_structure(client, valid_text):
    """Test that the response has the correct structure and data types."""
    response = client.post("/analyze", json={"text": valid_text})
    data = response.json()
    
    assert response.status_code == 200
    assert isinstance(data, dict), "Response should be a dictionary"
    assert "prediction" in data, "Response should contain 'prediction'"
    assert isinstance(data["prediction"], dict), "Prediction should be a dictionary"
    
    prediction = data["prediction"]
    assert "confidence" in prediction, "Prediction should contain confidence"
    assert isinstance(prediction["confidence"], (int, float)), "Confidence should be numeric"
    assert 0 <= prediction["confidence"] <= 1, "Confidence should be between 0 and 1"

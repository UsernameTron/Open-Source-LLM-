import requests
import json
import time

BASE_URL = "http://localhost:8000"
API_KEY = "8FFDbzL-cfJc8wkNo9gcGSMvKOvJhG7ZLzqWeuU2fBY"

def test_text_analysis():
    print("\nTesting Text Analysis:")
    url = f"{BASE_URL}/api/analyze-text"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    with open("test_complex.txt", "r") as f:
        text = f.read()
    
    data = {"text": text}
    response = requests.post(url, headers=headers, json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

def test_file_upload(file_path, content_type):
    print(f"\nTesting {content_type} Upload:")
    url = f"{BASE_URL}/api/analyze-file"
    headers = {
        "X-API-Key": API_KEY
    }
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, content_type)}
        response = requests.post(url, headers=headers, files=files)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()

def main():
    # Wait for server to be ready
    time.sleep(2)
    
    # Test text analysis
    text_result = test_text_analysis()
    
    # Test file uploads
    txt_result = test_file_upload("test_complex.txt", "text/plain")
    csv_result = test_file_upload("test_complex.csv", "text/csv")
    pdf_result = test_file_upload("test_complex.pdf", "application/pdf")
    
    # Print summary
    print("\nTest Summary:")
    print("=" * 50)
    results = {
        "Text Analysis": text_result,
        "TXT Upload": txt_result,
        "CSV Upload": csv_result,
        "PDF Upload": pdf_result
    }
    
    for test_name, result in results.items():
        confidence = result.get("confidence", 0) * 100
        prediction = result.get("prediction", "unknown")
        print(f"{test_name}:")
        print(f"  Prediction: {prediction}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Passed: {'Yes' if confidence >= 80 else 'No'}")
        print("-" * 50)

if __name__ == "__main__":
    main()

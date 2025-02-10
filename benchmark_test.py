import time
import statistics
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

BASE_URL = "http://localhost:8000"
API_KEY = "8FFDbzL-cfJc8wkNo9gcGSMvKOvJhG7ZLzqWeuU2fBY"

def analyze_text(text: str) -> Dict[str, Any]:
    url = f"{BASE_URL}/api/analyze-text"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    start_time = time.time()
    response = requests.post(url, headers=headers, json={"text": text})
    end_time = time.time()
    result = response.json()
    result['actual_processing_time'] = (end_time - start_time) * 1000
    return result

def run_performance_test(num_requests: int = 100):
    # Test data with varying complexity
    test_texts = [
        # Short text
        "The system works well.",
        
        # Medium text with mixed sentiment
        "Initially had problems but now working great after fixes.",
        
        # Long text with technical terms
        """The system initially exhibited concerning performance issues and critical
        bottlenecks. However, after implementing innovative optimizations and
        architectural improvements, we achieved exceptional throughput and
        remarkable stability. The final solution exceeded expectations.""",
        
        # Complex text with sarcasm
        """Oh great, another 'innovative' solution that's supposed to revolutionize
        everything. Surprisingly though, it actually works really well."""
    ]
    
    all_results = []
    processing_times = []
    
    print(f"\nRunning performance test with {num_requests} requests...")
    
    def process_request(text: str) -> Dict[str, Any]:
        try:
            return analyze_text(text)
        except Exception as e:
            return {"error": str(e)}
    
    start_time = time.time()
    
    # Run requests in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for _ in range(num_requests // len(test_texts)):
            for text in test_texts:
                futures.append(executor.submit(process_request, text))
        
        for future in futures:
            result = future.result()
            if "error" not in result:
                all_results.append(result)
                processing_times.append(result["actual_processing_time"])
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_processing_time = statistics.mean(processing_times)
    p95_time = sorted(processing_times)[int(len(processing_times) * 0.95)]
    p99_time = sorted(processing_times)[int(len(processing_times) * 0.99)]
    requests_per_second = len(processing_times) / total_time
    
    print("\nPerformance Metrics:")
    print(f"Total Requests: {len(processing_times)}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Processing Time: {avg_processing_time:.2f} ms")
    print(f"95th Percentile: {p95_time:.2f} ms")
    print(f"99th Percentile: {p99_time:.2f} ms")
    print(f"Requests per Second: {requests_per_second:.2f}")
    
    # Analyze accuracy consistency
    sentiment_consistency = {}
    for text in test_texts:
        results = [r for r in all_results if text in r.get("text", "")]
        predictions = [r["prediction"] for r in results]
        confidences = [r["confidence"] for r in results]
        
        dominant_prediction = max(set(predictions), key=predictions.count)
        consistency = predictions.count(dominant_prediction) / len(predictions)
        avg_confidence = sum(confidences) / len(confidences)
        
        print(f"\nAccuracy Analysis for text: {text[:50]}...")
        print(f"Prediction Consistency: {consistency * 100:.1f}%")
        print(f"Average Confidence: {avg_confidence * 100:.1f}%")

if __name__ == "__main__":
    run_performance_test(100)

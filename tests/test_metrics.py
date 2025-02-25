import pytest
import time
import threading
from pathlib import Path
from core.monitoring.metrics import MetricsCollector, RequestTrace, BatchMetrics

@pytest.fixture
def metrics_collector():
    return MetricsCollector(metrics_window=1000, export_path="test_metrics")

def test_request_trace_collection(metrics_collector):
    """Test that request traces are collected properly."""
    trace = RequestTrace(
        request_id="test1",
        priority=1,
        batch_size=32,
        queue_time=0.1,
        processing_time=0.2,
        total_latency=0.3,
        batch_id="batch1"
    )
    
    metrics_collector.add_request_trace(trace)
    stats = metrics_collector.get_current_stats()
    
    assert stats["avg_latency"] > 0, "Average latency should be positive"
    assert stats["avg_batch_size"] == 32, "Batch size should match"

def test_batch_metrics_collection(metrics_collector):
    """Test that batch metrics are collected properly."""
    batch_metrics = BatchMetrics(
        batch_id="batch1",
        size=32,
        processing_time=0.2,
        gpu_utilization=45.5,
        memory_utilization=60.0,
        queue_size=5,
        priorities=[1, 2, 1],
        throughput=100.0,
        avg_latency_ms=50.0,
        p95_latency_ms=75.0
    )
    
    metrics_collector.add_batch_metrics(batch_metrics)
    stats = metrics_collector.get_current_stats()
    
    assert stats["gpu_utilization"] == 45.5, "GPU utilization should match"
    assert stats["memory_utilization"] == 60.0, "Memory utilization should match"

def test_metrics_thread_safety(metrics_collector):
    """Test thread safety of metrics collection."""
    def add_traces():
        for i in range(100):
            trace = RequestTrace(
                request_id=f"test{i}",
                priority=1,
                batch_size=32,
                queue_time=0.1,
                processing_time=0.2,
                total_latency=0.3,
                batch_id=f"batch{i}"
            )
            metrics_collector.add_request_trace(trace)
    
    threads = [threading.Thread(target=add_traces) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    stats = metrics_collector.get_current_stats()
    assert stats["avg_latency"] > 0, "Metrics should be collected from all threads"

def test_metrics_export(metrics_collector):
    """Test metrics export functionality."""
    # Add some test data
    trace = RequestTrace(
        request_id="test1",
        priority=1,
        batch_size=32,
        queue_time=0.1,
        processing_time=0.2,
        total_latency=0.3,
        batch_id="batch1"
    )
    metrics_collector.add_request_trace(trace)
    
    # Force an export
    metrics_collector._export_metrics()
    
    # Check that export directory exists and contains files
    export_path = Path("test_metrics")
    assert export_path.exists(), "Export directory should be created"
    assert any(export_path.glob("metrics_*.json")), "Export files should be created"

def test_metrics_window_size(metrics_collector):
    """Test that metrics window size is respected."""
    for i in range(2000):  # More than window size
        trace = RequestTrace(
            request_id=f"test{i}",
            priority=1,
            batch_size=32,
            queue_time=0.1,
            processing_time=0.2,
            total_latency=0.3,
            batch_id=f"batch{i}"
        )
        metrics_collector.add_request_trace(trace)
    
    # Access private member for testing
    assert len(metrics_collector._request_traces) == 1000, "Window size should be respected"

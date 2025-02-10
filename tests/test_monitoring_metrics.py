import pytest
import time
import numpy as np
from core.monitoring.metrics import InferenceMetrics, BatchMetrics, MetricsCollector

def test_inference_metrics():
    # Test InferenceMetrics creation and default values
    metrics = InferenceMetrics(
        latency_ms=10.5,
        batch_size=8,
        confidence_scores=[0.8, 0.9, 0.7],
        gpu_utilization=0.6,
        memory_utilization=0.4
    )
    
    assert metrics.latency_ms == 10.5
    assert metrics.batch_size == 8
    assert metrics.confidence_scores == [0.8, 0.9, 0.7]
    assert metrics.gpu_utilization == 0.6
    assert metrics.memory_utilization == 0.4
    assert isinstance(metrics.timestamp, float)
    assert metrics.timestamp <= time.time()

def test_batch_metrics():
    # Test BatchMetrics creation
    metrics = BatchMetrics(
        batch_size=16,
        throughput=100.0,
        avg_latency_ms=15.0,
        p95_latency_ms=20.0,
        gpu_utilization=0.7,
        memory_utilization=0.5,
        accuracy=0.95,
        confidence_mean=0.85,
        confidence_std=0.1
    )
    
    assert metrics.batch_size == 16
    assert metrics.throughput == 100.0
    assert metrics.avg_latency_ms == 15.0
    assert metrics.p95_latency_ms == 20.0
    assert metrics.gpu_utilization == 0.7
    assert metrics.memory_utilization == 0.5
    assert metrics.accuracy == 0.95
    assert metrics.confidence_mean == 0.85
    assert metrics.confidence_std == 0.1

def test_metrics_collector():
    collector = MetricsCollector(window_size=5)
    
    # Test window size initialization
    assert len(collector.metrics_window) == 0
    assert collector.window_size == 5
    
    # Add metrics and test window behavior
    for i in range(7):
        metrics = InferenceMetrics(
            latency_ms=10.0 + i,
            batch_size=8,
            confidence_scores=[0.8],
            gpu_utilization=0.6,
            memory_utilization=0.4
        )
        collector.add_inference_metrics(metrics)
    
    # Check that window maintains max size
    assert len(collector.metrics_window) == 5
    
    # Check that oldest metrics were dropped
    latencies = [m.latency_ms for m in collector.metrics_window]
    assert min(latencies) == 12.0  # First two entries (10.0, 11.0) should be dropped
    assert max(latencies) == 16.0

def test_metrics_collector_report_generation():
    collector = MetricsCollector(window_size=100)
    
    # Add some test metrics
    for _ in range(10):
        metrics = InferenceMetrics(
            latency_ms=np.random.normal(10, 2),
            batch_size=8,
            confidence_scores=[np.random.uniform(0.7, 0.9)],
            gpu_utilization=np.random.uniform(0.5, 0.7),
            memory_utilization=np.random.uniform(0.3, 0.5)
        )
        collector.add_inference_metrics(metrics)
    
    # Force report generation
    collector.last_report_time = 0  # Reset last report time
    collector.report_interval = 0    # Set interval to 0 to force immediate report
    
    # Add one more metric to trigger report
    metrics = InferenceMetrics(
        latency_ms=10.0,
        batch_size=8,
        confidence_scores=[0.8],
        gpu_utilization=0.6,
        memory_utilization=0.4
    )
    collector.add_inference_metrics(metrics)
    
    # Verify that metrics were collected
    assert len(collector.metrics_window) == 11
    
    # Test report content
    report = collector._generate_performance_report()
    assert isinstance(report, dict)
    assert 'overall_metrics' in report
    assert 'batch_metrics' in report
    assert 'recommendations' in report
    
    # Check overall metrics
    assert 'avg_latency_ms' in report['overall_metrics']
    assert 'p95_latency_ms' in report['overall_metrics']
    assert 'avg_gpu_utilization' in report['overall_metrics']
    assert 'avg_memory_utilization' in report['overall_metrics']

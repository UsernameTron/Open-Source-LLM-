import pytest
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from core.monitoring.metrics_collector import RequestTrace, BatchMetrics, MetricsCollector

@pytest.fixture
def temp_metrics_dir(tmp_path):
    metrics_dir = tmp_path / "test_metrics"
    metrics_dir.mkdir()
    return metrics_dir

def test_request_trace():
    trace = RequestTrace(
        request_id="req1",
        priority=1,
        batch_size=8,
        queue_time=0.5,
        processing_time=1.0,
        total_latency=1.5,
        batch_id="batch1"
    )
    
    assert trace.request_id == "req1"
    assert trace.priority == 1
    assert trace.batch_size == 8
    assert trace.queue_time == 0.5
    assert trace.processing_time == 1.0
    assert trace.total_latency == 1.5
    assert trace.batch_id == "batch1"
    assert isinstance(trace.timestamp, float)

def test_batch_metrics():
    metrics = BatchMetrics(
        batch_id="batch1",
        size=8,
        processing_time=1.0,
        gpu_utilization=0.6,
        memory_utilization=0.4,
        queue_size=5,
        priorities=[1, 2, 1]
    )
    
    assert metrics.batch_id == "batch1"
    assert metrics.size == 8
    assert metrics.processing_time == 1.0
    assert metrics.gpu_utilization == 0.6
    assert metrics.memory_utilization == 0.4
    assert metrics.queue_size == 5
    assert metrics.priorities == [1, 2, 1]
    assert isinstance(metrics.timestamp, float)

def test_metrics_collector_initialization(temp_metrics_dir):
    collector = MetricsCollector(metrics_window=100, export_path=str(temp_metrics_dir))
    
    assert collector.metrics_window == 100
    assert collector.export_path == Path(temp_metrics_dir)
    assert len(collector._request_traces) == 0
    assert len(collector._batch_metrics) == 0
    assert collector._current_stats["avg_latency"] == 0.0
    assert collector._current_stats["p95_latency"] == 0.0

def test_metrics_collector_add_request_trace(temp_metrics_dir):
    collector = MetricsCollector(metrics_window=3, export_path=str(temp_metrics_dir))
    
    # Add multiple traces
    for i in range(5):
        trace = RequestTrace(
            request_id=f"req{i}",
            priority=1,
            batch_size=8,
            queue_time=0.5,
            processing_time=1.0,
            total_latency=1.5
        )
        collector.add_request_trace(trace)
    
    # Check window size is maintained
    assert len(collector._request_traces) == 3
    
    # Check most recent traces are kept
    traces = list(collector._request_traces)
    assert traces[-1].request_id == "req4"
    assert traces[0].request_id == "req2"

def test_metrics_collector_add_batch_metrics(temp_metrics_dir):
    collector = MetricsCollector(metrics_window=3, export_path=str(temp_metrics_dir))
    
    # Add multiple batch metrics
    for i in range(5):
        metrics = BatchMetrics(
            batch_id=f"batch{i}",
            size=8,
            processing_time=1.0,
            gpu_utilization=0.6,
            memory_utilization=0.4,
            queue_size=5,
            priorities=[1]
        )
        collector.add_batch_metrics(metrics)
    
    # Check window size is maintained
    assert len(collector._batch_metrics) == 3
    
    # Check most recent metrics are kept
    batches = list(collector._batch_metrics)
    assert batches[-1].batch_id == "batch4"
    assert batches[0].batch_id == "batch2"

def test_metrics_collector_export(temp_metrics_dir):
    collector = MetricsCollector(metrics_window=100, export_path=str(temp_metrics_dir))
    
    # Add some test data
    trace = RequestTrace(
        request_id="req1",
        priority=1,
        batch_size=8,
        queue_time=0.5,
        processing_time=1.0,
        total_latency=1.5
    )
    collector.add_request_trace(trace)
    
    metrics = BatchMetrics(
        batch_id="batch1",
        size=8,
        processing_time=1.0,
        gpu_utilization=0.6,
        memory_utilization=0.4,
        queue_size=5,
        priorities=[1]
    )
    collector.add_batch_metrics(metrics)
    
    # Export metrics
    collector.export_metrics()
    
    # Check that files were created
    metrics_file = temp_metrics_dir / f"metrics_{datetime.now().strftime('%Y%m%d')}.json"
    assert metrics_file.exists()
    
    # Verify file contents
    with open(metrics_file) as f:
        data = json.load(f)
        assert "request_traces" in data
        assert "batch_metrics" in data
        assert len(data["request_traces"]) == 1
        assert len(data["batch_metrics"]) == 1

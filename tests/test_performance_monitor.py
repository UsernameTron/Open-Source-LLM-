import pytest
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from core.monitoring.performance import PerformanceReport, PerformanceMonitor

@pytest.fixture
def sample_performance_report():
    return PerformanceReport(
        timestamp=datetime.now().isoformat(),
        batch_sizes=[8, 16, 32],
        latencies=[10.0, 15.0, 25.0],
        throughputs=[800, 1067, 1280],
        gpu_utilization=[0.5, 0.7, 0.9],
        memory_utilization=[0.4, 0.6, 0.8],
        queue_sizes=[0, 5, 10]
    )

@pytest.fixture
def temp_output_dir(tmp_path):
    output_dir = tmp_path / "test_performance"
    output_dir.mkdir()
    return output_dir

def test_performance_report_creation(sample_performance_report):
    report = sample_performance_report
    
    assert len(report.batch_sizes) == 3
    assert len(report.latencies) == 3
    assert len(report.throughputs) == 3
    assert len(report.gpu_utilization) == 3
    assert len(report.memory_utilization) == 3
    assert len(report.queue_sizes) == 3
    
    # Test to_dict method
    report_dict = report.to_dict()
    assert "timestamp" in report_dict
    assert "metrics" in report_dict
    assert all(key in report_dict["metrics"] for key in [
        "batch_sizes", "latencies", "throughputs",
        "gpu_utilization", "memory_utilization", "queue_sizes"
    ])

def test_performance_monitor_initialization(temp_output_dir):
    monitor = PerformanceMonitor(output_dir=str(temp_output_dir))
    
    assert monitor.output_dir == Path(temp_output_dir)
    assert isinstance(monitor.metrics_window, list)
    assert len(monitor.metrics_window) == 0

def test_performance_monitor_update_metrics(temp_output_dir):
    monitor = PerformanceMonitor(output_dir=str(temp_output_dir))
    
    # Add test metrics
    test_metrics = {
        "batch_size": 8,
        "latency": 10.0,
        "throughput": 800,
        "gpu_util": 0.6,
        "memory_util": 0.4,
        "queue_size": 5
    }
    
    monitor.update_metrics(test_metrics)
    assert len(monitor.metrics_window) == 1
    
    # Verify metrics were stored correctly
    stored_metrics = monitor.metrics_window[0]
    assert "timestamp" in stored_metrics
    assert stored_metrics["batch_size"] == test_metrics["batch_size"]
    assert stored_metrics["latency"] == test_metrics["latency"]

def test_performance_monitor_generate_report(temp_output_dir):
    monitor = PerformanceMonitor(output_dir=str(temp_output_dir))
    
    # Add multiple test metrics
    for i in range(3):
        test_metrics = {
            "batch_size": 8 * (i + 1),
            "latency": 10.0 * (i + 1),
            "throughput": 800 * (i + 1),
            "gpu_util": 0.2 * (i + 1),
            "memory_util": 0.15 * (i + 1),
            "queue_size": 5 * i
        }
        monitor.update_metrics(test_metrics)
    
    # Generate report
    report = monitor.generate_report()
    
    # Verify report structure
    assert isinstance(report, PerformanceReport)
    assert len(report.batch_sizes) == 3
    assert len(report.latencies) == 3
    assert len(report.throughputs) == 3
    assert len(report.gpu_utilization) == 3
    assert len(report.memory_utilization) == 3
    assert len(report.queue_sizes) == 3

def test_performance_monitor_save_report(temp_output_dir, sample_performance_report):
    monitor = PerformanceMonitor(output_dir=str(temp_output_dir))
    
    # Save report
    monitor.save_report(sample_performance_report)
    
    # Check that report file was created
    report_files = list(temp_output_dir.glob("performance_report_*.json"))
    assert len(report_files) == 1
    
    # Verify file contents
    with open(report_files[0]) as f:
        data = json.load(f)
        assert "timestamp" in data
        assert "metrics" in data
        assert all(key in data["metrics"] for key in [
            "batch_sizes", "latencies", "throughputs",
            "gpu_utilization", "memory_utilization", "queue_sizes"
        ])

def test_performance_monitor_plot_metrics(temp_output_dir):
    monitor = PerformanceMonitor(output_dir=str(temp_output_dir))
    
    # Add test metrics
    for i in range(10):
        test_metrics = {
            "batch_size": 8,
            "latency_ms": 10.0 + np.random.normal(0, 1),
            "throughput": 800 + np.random.normal(0, 50),
            "gpu_utilization": 0.6 + np.random.normal(0, 0.1),
            "memory_utilization": 0.4 + np.random.normal(0, 0.1),
            "queue_size": 5 + np.random.poisson(2)
        }
        monitor.update_metrics(test_metrics)
    
    # Generate and save plots
    monitor.plot_metrics()
    
    # Check that plot files were created
    plot_files = list(temp_output_dir.glob("*.png"))
    assert len(plot_files) > 0  # At least one plot should be created

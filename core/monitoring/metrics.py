import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import psutil
import logging
from collections import deque
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class InferenceMetrics:
    latency_ms: float
    batch_size: int
    confidence_scores: List[float]
    gpu_utilization: float
    memory_utilization: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class BatchMetrics:
    batch_size: int
    throughput: float  # requests/second
    avg_latency_ms: float
    p95_latency_ms: float
    gpu_utilization: float
    memory_utilization: float
    accuracy: float
    confidence_mean: float
    confidence_std: float

class MetricsCollector:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_window: deque = deque(maxlen=window_size)
        self.batch_metrics: Dict[int, List[BatchMetrics]] = {}
        self.last_report_time = time.time()
        self.report_interval = 60  # Generate report every 60 seconds
        
    def add_inference_metrics(self, metrics: InferenceMetrics):
        """Add new inference metrics to the collection."""
        self.metrics_window.append(metrics)
        
        # Check if it's time to generate a report
        if time.time() - self.last_report_time > self.report_interval:
            self._generate_performance_report()
            self.last_report_time = time.time()
    
    def calculate_batch_metrics(self, batch_size: int) -> BatchMetrics:
        """Calculate aggregate metrics for a specific batch size."""
        relevant_metrics = [m for m in self.metrics_window if m.batch_size == batch_size]
        if not relevant_metrics:
            return None
            
        latencies = [m.latency_ms for m in relevant_metrics]
        confidences = [c for m in relevant_metrics for c in m.confidence_scores]
        
        return BatchMetrics(
            batch_size=batch_size,
            throughput=len(relevant_metrics) / (relevant_metrics[-1].timestamp - relevant_metrics[0].timestamp),
            avg_latency_ms=np.mean(latencies),
            p95_latency_ms=np.percentile(latencies, 95),
            gpu_utilization=np.mean([m.gpu_utilization for m in relevant_metrics]),
            memory_utilization=np.mean([m.memory_utilization for m in relevant_metrics]),
            accuracy=np.mean([1 if c > 0.9 else 0 for c in confidences]),  # Using 0.9 as confidence threshold
            confidence_mean=np.mean(confidences),
            confidence_std=np.std(confidences)
        )
    
    def _generate_performance_report(self):
        """Generate a comprehensive performance report."""
        if not self.metrics_window:
            return {}
            
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_metrics": {},
            "batch_metrics": {},
            "recommendations": []
        }
        
        # Calculate overall metrics
        all_latencies = [m.latency_ms for m in self.metrics_window]
        report["overall_metrics"] = {
            "avg_latency_ms": float(np.mean(all_latencies)),
            "p95_latency_ms": float(np.percentile(all_latencies, 95)),
            "avg_gpu_utilization": float(np.mean([m.gpu_utilization for m in self.metrics_window])),
            "avg_memory_utilization": float(np.mean([m.memory_utilization for m in self.metrics_window]))
        }
        
        # Calculate metrics per batch size
        unique_batch_sizes = set(m.batch_size for m in self.metrics_window)
        for batch_size in unique_batch_sizes:
            metrics = self.calculate_batch_metrics(batch_size)
            if metrics:
                report["batch_metrics"][str(batch_size)] = {
                    "throughput": float(metrics.throughput),
                    "avg_latency_ms": float(metrics.avg_latency_ms),
                    "p95_latency_ms": float(metrics.p95_latency_ms),
                    "gpu_utilization": float(metrics.gpu_utilization),
                    "memory_utilization": float(metrics.memory_utilization),
                    "accuracy": metrics.accuracy,
                    "confidence_mean": metrics.confidence_mean,
                    "confidence_std": metrics.confidence_std
                }
        
        # Generate recommendations
        self._add_recommendations(report)
        
        # Save report if needed
        if os.getenv('SAVE_REPORTS', 'false').lower() == 'true':
            report_path = f"performance_reports/report_{int(time.time())}.json"
            os.makedirs("performance_reports", exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Performance report generated: {report_path}")
        
        return report
        
    def _add_recommendations(self, report: Dict):
        """Add performance optimization recommendations based on metrics."""
        metrics = report["overall_metrics"]
        
        # Check latency targets
        if metrics["p95_latency_ms"] > 100:
            report["recommendations"].append({
                "type": "latency",
                "severity": "high",
                "message": "P95 latency exceeds target of 100ms. Consider reducing batch size or implementing request queuing."
            })
            
        # Check GPU utilization
        if metrics["avg_gpu_utilization"] < 80:
            report["recommendations"].append({
                "type": "gpu_utilization",
                "severity": "medium",
                "message": "GPU utilization below 80%. Consider increasing batch size or concurrent requests."
            })
            
        # Check memory utilization
        if metrics["avg_memory_utilization"] > 90:
            report["recommendations"].append({
                "type": "memory",
                "severity": "high",
                "message": "Memory utilization above 90%. Implement more aggressive garbage collection or reduce batch size."
            })
            
        # Analyze batch size efficiency
        batch_metrics = report["batch_metrics"]
        if batch_metrics:
            max_throughput = max(m["throughput"] for m in batch_metrics.values())
            optimal_batch = max(batch_metrics.items(), key=lambda x: x[1]["throughput"])[0]
            report["recommendations"].append({
                "type": "batch_size",
                "severity": "info",
                "message": f"Optimal batch size for throughput is {optimal_batch}. Consider adjusting dynamic batching thresholds."
            })

"""Metrics collector for tracking model performance and resource usage."""

import os
import time
import json
from typing import Dict, Optional, List
import numpy as np

class MetricsCollector:
    """Collects and stores metrics about model performance and resource usage."""

    def __init__(self, metrics_window: int = 1000, export_path: str = "metrics"):
        """Initialize the metrics collector.
        
        Args:
            metrics_window: Number of samples to keep in the sliding window
            export_path: Directory to export metrics to
        """
        self.metrics_window = metrics_window
        self.export_path = export_path
        os.makedirs(export_path, exist_ok=True)

        # Initialize metric storage
        self.latencies: List[float] = []
        self.batch_sizes: List[int] = []
        self.cache_hits: List[bool] = []
        self.resource_usage: List[Dict] = []
        self.last_export_time = time.time()
        self.export_interval = 60  # Export metrics every minute

    def record_latency(self, latency: float) -> None:
        """Record a new latency measurement.
        
        Args:
            latency: Inference latency in milliseconds
        """
        self.latencies.append(latency)
        if len(self.latencies) > self.metrics_window:
            self.latencies.pop(0)
        self._maybe_export_metrics()

    def record_batch_size(self, batch_size: int) -> None:
        """Record a batch size measurement.
        
        Args:
            batch_size: Size of the processed batch
        """
        self.batch_sizes.append(batch_size)
        if len(self.batch_sizes) > self.metrics_window:
            self.batch_sizes.pop(0)
        self._maybe_export_metrics()

    def record_cache_hit(self, hit: bool) -> None:
        """Record a cache hit/miss.
        
        Args:
            hit: True if cache hit, False if miss
        """
        self.cache_hits.append(hit)
        if len(self.cache_hits) > self.metrics_window:
            self.cache_hits.pop(0)
        self._maybe_export_metrics()

    def record_resource_usage(self, cpu_percent: float, memory_percent: float, 
                            gpu_utilization: Optional[float] = None) -> None:
        """Record resource usage metrics.
        
        Args:
            cpu_percent: CPU utilization percentage
            memory_percent: Memory utilization percentage
            gpu_utilization: Optional GPU utilization percentage
        """
        usage = {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
        }
        if gpu_utilization is not None:
            usage["gpu_utilization"] = gpu_utilization
            
        self.resource_usage.append(usage)
        if len(self.resource_usage) > self.metrics_window:
            self.resource_usage.pop(0)
        self._maybe_export_metrics()

    def get_summary_metrics(self) -> Dict:
        """Get summary statistics of collected metrics.
        
        Returns:
            Dictionary containing summary metrics
        """
        metrics = {
            "latency": {
                "mean": np.mean(self.latencies) if self.latencies else 0,
                "p50": np.percentile(self.latencies, 50) if self.latencies else 0,
                "p95": np.percentile(self.latencies, 95) if self.latencies else 0,
                "p99": np.percentile(self.latencies, 99) if self.latencies else 0,
            },
            "batch_size": {
                "mean": np.mean(self.batch_sizes) if self.batch_sizes else 0,
                "min": min(self.batch_sizes) if self.batch_sizes else 0,
                "max": max(self.batch_sizes) if self.batch_sizes else 0,
            },
            "cache": {
                "hit_rate": (sum(self.cache_hits) / len(self.cache_hits) 
                            if self.cache_hits else 0),
            },
        }

        if self.resource_usage:
            latest_usage = self.resource_usage[-1]
            metrics["resources"] = {
                "cpu_percent": latest_usage["cpu_percent"],
                "memory_percent": latest_usage["memory_percent"],
            }
            if "gpu_utilization" in latest_usage:
                metrics["resources"]["gpu_utilization"] = latest_usage["gpu_utilization"]

        return metrics

    def _maybe_export_metrics(self) -> None:
        """Export metrics to disk if enough time has passed."""
        current_time = time.time()
        if current_time - self.last_export_time >= self.export_interval:
            metrics = self.get_summary_metrics()
            export_file = os.path.join(self.export_path, 
                                     f"metrics_{int(current_time)}.json")
            with open(export_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            self.last_export_time = current_time

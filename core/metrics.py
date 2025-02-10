"""Module for tracking performance metrics."""

from time import time

class PerformanceMetricsTracker:
    """Class to track and update performance metrics"""
    def __init__(self):
        self.request_count = 0
        self.start_time = time()
        self._last_metrics = None
        self._last_request_time = None
        self._last_update_time = None
        
    def update_metrics(self):
        """Update metrics after processing a request"""
        current_time = time()
        self.request_count += 1
        
        # Calculate elapsed time since start
        elapsed_time = max(0.001, current_time - self.start_time)
        
        # Calculate throughput (requests per second)
        throughput = self.request_count / elapsed_time
        
        # Calculate latency (time since last request)
        latency = 0 if self._last_request_time is None else \
                 (current_time - self._last_request_time) * 1000  # Convert to ms
        
        # Update timestamps
        self._last_request_time = current_time
        self._last_update_time = current_time
        
        metrics = {
            'latency_ms': round(latency, 2),
            'throughput': round(throughput, 2),
            'gpu_utilization': 0.5,  # Mock values for testing
            'memory_utilization': 0.3
        }
        self._last_metrics = metrics
        return metrics
        
    def get_metrics(self):
        """Get current performance metrics without updating counters"""
        current_time = time()
        
        # If no metrics exist or too much time has passed, update them
        if self._last_metrics is None or \
           (self._last_update_time and current_time - self._last_update_time > 5.0):
            return self.update_metrics()
            
        # Return cached metrics
        return self._last_metrics

# Create a global metrics tracker
metrics_tracker = PerformanceMetricsTracker()

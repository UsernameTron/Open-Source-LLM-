import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import traceback
from dataclasses import dataclass
from core.config import settings
from threading import Lock
import json
from pathlib import Path
import psutil
import GPUtil

@dataclass
class ErrorMetrics:
    timestamp: float
    error_type: str
    error_message: str
    endpoint: str
    input_data: Dict[str, Any]
    latency: float

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoringSystem:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_dir: str = "logs"):
        if MonitoringSystem._initialized:
            return
        MonitoringSystem._initialized = True
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("inference_monitor")
        self.logger.setLevel(logging.DEBUG)  # Capture all levels of logs
        
        # File handler for all logs
        fh = logging.FileHandler(self.log_dir / "inference.log")
        fh.setLevel(logging.INFO)
        
        # Error file handler
        error_fh = logging.FileHandler(self.log_dir / "errors.log")
        error_fh.setLevel(logging.DEBUG)  # Capture all levels in error log
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        error_fh.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(error_fh)
        
        # Prometheus metrics
        self.request_counter = Counter(
            'inference_requests_total',
            'Total number of inference requests',
            ['endpoint']
        )
        self.latency_histogram = Histogram(
            'inference_latency_seconds',
            'Request latency in seconds',
            ['endpoint']
        )
        self.error_counter = Counter(
            'inference_errors_total',
            'Total number of errors',
            ['error_type', 'endpoint']
        )
        self.active_requests = Gauge(
            'inference_active_requests',
            'Number of active requests',
            ['endpoint']
        )
        self.memory_usage = Gauge(
            'inference_memory_usage_bytes',
            'Memory usage in bytes'
        )
        self.cpu_usage = Gauge(
            'inference_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        # Initialize metrics
        self._active_request_count = 0
        self._total_request_count = 0
        self._error_count = 0
        
        # Error tracking
        self.recent_errors: List[ErrorMetrics] = []
        
        # Start Prometheus HTTP server
        try:
            start_http_server(8001)
        except Exception as e:
            logger.warning(f"Could not start Prometheus server: {e}"
                         f" - metrics collection will be disabled")
        self.error_lock = Lock()
        self.max_stored_errors = settings.MAX_STORED_ERRORS
        
        # Start resource monitoring
        self._start_resource_monitoring()
        
    def _start_resource_monitoring(self):
        """Start monitoring system resources."""
        def update_resource_metrics():
            while True:
                # Update memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.set(memory.used)
                
                # Update CPU usage
                self.cpu_usage.set(psutil.cpu_percent())
                
                time.sleep(1)  # Update every second
                
        import threading
        thread = threading.Thread(target=update_resource_metrics, daemon=True)
        thread.start()
        
    def log_request_start(self, endpoint: str, input_data: Dict[str, Any]) -> float:
        """Log the start of a request and return the start time."""
        start_time = time.time()
        self.active_requests.labels(endpoint=endpoint).inc()
        self.logger.info(f"Request started - Endpoint: {endpoint}, Input: {input_data}")
        return start_time
        
    def log_request_end(self, endpoint: str, start_time: float,
                       status: str = "success", error: Optional[Exception] = None,
                       input_data: Optional[Dict[str, Any]] = None):
        """Log the end of a request with its outcome."""
        end_time = time.time()
        duration = end_time - start_time
        
        self.active_requests.labels(endpoint=endpoint).dec()
        self.request_counter.labels(endpoint=endpoint).inc()
        self.latency_histogram.labels(endpoint=endpoint).observe(duration)
        
        if status == "success":
            self.logger.info(
                f"Request completed - Endpoint: {endpoint}, "
                f"Duration: {duration:.3f}s"
            )
        else:
            error_type = type(error).__name__ if error else "Unknown"
            error_msg = str(error) if error else "Unknown error"
            self.error_counter.labels(
                error_type=error_type,
                endpoint=endpoint
            ).inc()
            
            error_metrics = ErrorMetrics(
                timestamp=end_time,
                error_type=error_type,
                error_message=error_msg,
                endpoint=endpoint,
                input_data=input_data or {},
                latency=duration
            )
            
            with self.error_lock:
                self.recent_errors.append(error_metrics)
                if len(self.recent_errors) > self.max_stored_errors:
                    self.recent_errors.pop(0)
            
            self.logger.error(
                f"Request failed - Endpoint: {endpoint}, "
                f"Error: {error_type}: {error_msg}, "
                f"Duration: {duration:.3f}s, "
                f"Input: {input_data}"
            )
            
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent errors."""
        with self.error_lock:
            errors = self.recent_errors[-limit:]
            return [
                {
                    "timestamp": datetime.fromtimestamp(e.timestamp).isoformat(),
                    "error_type": e.error_type,
                    "error_message": e.error_message,
                    "endpoint": e.endpoint,
                    "latency": e.latency,
                    "stack_trace": e.stack_trace,
                    "input_data": e.input_data
                }
                for e in errors
            ]
            
    def export_error_report(self) -> str:
        """Export error report as JSON."""
        report_path = self.log_dir / f"error_report_{int(time.time())}.json"
        with self.error_lock:
            report = {
                "generated_at": datetime.now().isoformat(),
                "total_errors": len(self.recent_errors),
                "errors": self.get_recent_errors(limit=self.max_stored_errors)
            }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return str(report_path)

# Global monitoring instance
monitor = MonitoringSystem()

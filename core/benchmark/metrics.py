"""
Performance and resource metrics for benchmarking.
"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from core.metrics import performance_monitor
from core.thermal.monitor import ThermalMetrics

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance-related metrics."""
    # Timing metrics
    total_time_ms: float
    compute_time_ms: float
    io_time_ms: float
    overhead_time_ms: float
    
    # Throughput metrics
    operations_per_second: float
    items_processed: int
    batch_size: int
    
    # Efficiency metrics
    compute_efficiency: float  # 0-1
    memory_efficiency: float  # 0-1
    energy_efficiency: float  # operations/joule
    
    # Error metrics
    error_rate: float
    retry_count: int
    
    @property
    def total_efficiency(self) -> float:
        """Calculate overall efficiency score."""
        weights = {
            'compute': 0.4,
            'memory': 0.3,
            'energy': 0.3
        }
        return (
            weights['compute'] * self.compute_efficiency +
            weights['memory'] * self.memory_efficiency +
            weights['energy'] * self.energy_efficiency
        )

@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    # CPU metrics
    cpu_util_percent: float
    cpu_freq_mhz: float
    cpu_temp_c: float
    cpu_power_w: float
    
    # Memory metrics
    memory_used_gb: float
    memory_peak_gb: float
    memory_bandwidth_gbs: float
    
    # GPU metrics
    gpu_util_percent: float
    gpu_freq_mhz: float
    gpu_temp_c: float
    gpu_power_w: float
    gpu_memory_used_gb: float
    
    # System metrics
    total_power_w: float
    fan_speed_rpm: int
    thermal_pressure: str
    
    @classmethod
    def from_thermal_metrics(
        cls,
        thermal: ThermalMetrics,
        memory_stats: Dict
    ) -> 'ResourceMetrics':
        """Create from thermal metrics."""
        return cls(
            cpu_util_percent=memory_stats['cpu_percent'],
            cpu_freq_mhz=memory_stats['cpu_freq'],
            cpu_temp_c=thermal.cpu_temperature,
            cpu_power_w=thermal.cpu_power,
            memory_used_gb=memory_stats['memory_used'] / 1024,
            memory_peak_gb=memory_stats['memory_peak'] / 1024,
            memory_bandwidth_gbs=memory_stats['memory_bandwidth'],
            gpu_util_percent=memory_stats['gpu_percent'],
            gpu_freq_mhz=memory_stats['gpu_freq'],
            gpu_temp_c=thermal.gpu_temperature,
            gpu_power_w=thermal.gpu_power,
            gpu_memory_used_gb=memory_stats['gpu_memory'] / 1024,
            total_power_w=thermal.total_power,
            fan_speed_rpm=thermal.fan_speed,
            thermal_pressure=thermal.thermal_pressure
        )

class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.reset()
        logger.info("Initialized metrics collector")
        
    def reset(self):
        """Reset all metrics."""
        self._start_time = None
        self._end_time = None
        self._operation_times = []
        self._batch_sizes = []
        self._error_counts = 0
        self._retry_counts = 0
        self._resource_samples = []
        
    @performance_monitor
    def start_operation(self):
        """Start timing an operation."""
        if not self._start_time:
            self._start_time = time.perf_counter()
        self._operation_start = time.perf_counter()
        
    @performance_monitor
    def end_operation(
        self,
        batch_size: int,
        success: bool = True,
        retried: bool = False
    ):
        """End timing an operation."""
        duration = time.perf_counter() - self._operation_start
        self._operation_times.append(duration)
        self._batch_sizes.append(batch_size)
        
        if not success:
            self._error_counts += 1
        if retried:
            self._retry_counts += 1
            
    @performance_monitor
    def add_resource_sample(
        self,
        thermal_metrics: ThermalMetrics,
        memory_stats: Dict
    ):
        """Add resource utilization sample."""
        metrics = ResourceMetrics.from_thermal_metrics(
            thermal_metrics,
            memory_stats
        )
        self._resource_samples.append(metrics)
        
    @performance_monitor
    def compute_metrics(self) -> tuple[PerformanceMetrics, ResourceMetrics]:
        """
        Compute final metrics.
        
        Returns:
            Tuple of (performance metrics, resource metrics)
        """
        try:
            self._end_time = time.perf_counter()
            
            # Calculate timing metrics
            total_time = (self._end_time - self._start_time) * 1000
            compute_time = sum(self._operation_times) * 1000
            overhead_time = total_time - compute_time
            
            # Calculate throughput metrics
            total_items = sum(self._batch_sizes)
            ops_per_second = len(self._operation_times) / (total_time / 1000)
            avg_batch_size = np.mean(self._batch_sizes)
            
            # Calculate error metrics
            error_rate = (
                self._error_counts / len(self._operation_times)
                if self._operation_times else 0
            )
            
            # Calculate efficiency metrics
            if self._resource_samples:
                avg_power = np.mean([
                    m.total_power_w for m in self._resource_samples
                ])
                energy_efficiency = ops_per_second / avg_power
                
                avg_cpu_util = np.mean([
                    m.cpu_util_percent for m in self._resource_samples
                ])
                compute_efficiency = min(avg_cpu_util / 100, 1.0)
                
                avg_memory_util = np.mean([
                    m.memory_used_gb / m.memory_peak_gb
                    for m in self._resource_samples
                ])
                memory_efficiency = 1.0 - min(avg_memory_util, 1.0)
            else:
                energy_efficiency = 0
                compute_efficiency = 0
                memory_efficiency = 0
                
            # Create performance metrics
            perf_metrics = PerformanceMetrics(
                total_time_ms=total_time,
                compute_time_ms=compute_time,
                io_time_ms=0,  # TODO: Add I/O tracking
                overhead_time_ms=overhead_time,
                operations_per_second=ops_per_second,
                items_processed=total_items,
                batch_size=int(avg_batch_size),
                compute_efficiency=compute_efficiency,
                memory_efficiency=memory_efficiency,
                energy_efficiency=energy_efficiency,
                error_rate=error_rate,
                retry_count=self._retry_counts
            )
            
            # Calculate average resource metrics
            if self._resource_samples:
                resource_metrics = ResourceMetrics(
                    cpu_util_percent=np.mean([
                        m.cpu_util_percent for m in self._resource_samples
                    ]),
                    cpu_freq_mhz=np.mean([
                        m.cpu_freq_mhz for m in self._resource_samples
                    ]),
                    cpu_temp_c=np.mean([
                        m.cpu_temp_c for m in self._resource_samples
                    ]),
                    cpu_power_w=np.mean([
                        m.cpu_power_w for m in self._resource_samples
                    ]),
                    memory_used_gb=np.mean([
                        m.memory_used_gb for m in self._resource_samples
                    ]),
                    memory_peak_gb=np.max([
                        m.memory_peak_gb for m in self._resource_samples
                    ]),
                    memory_bandwidth_gbs=np.mean([
                        m.memory_bandwidth_gbs for m in self._resource_samples
                    ]),
                    gpu_util_percent=np.mean([
                        m.gpu_util_percent for m in self._resource_samples
                    ]),
                    gpu_freq_mhz=np.mean([
                        m.gpu_freq_mhz for m in self._resource_samples
                    ]),
                    gpu_temp_c=np.mean([
                        m.gpu_temp_c for m in self._resource_samples
                    ]),
                    gpu_power_w=np.mean([
                        m.gpu_power_w for m in self._resource_samples
                    ]),
                    gpu_memory_used_gb=np.mean([
                        m.gpu_memory_used_gb for m in self._resource_samples
                    ]),
                    total_power_w=np.mean([
                        m.total_power_w for m in self._resource_samples
                    ]),
                    fan_speed_rpm=int(np.mean([
                        m.fan_speed_rpm for m in self._resource_samples
                    ])),
                    thermal_pressure=max(
                        set(m.thermal_pressure for m in self._resource_samples),
                        key=lambda x: [
                            "Nominal",
                            "Moderate",
                            "Heavy",
                            "Critical"
                        ].index(x)
                    )
                )
            else:
                resource_metrics = None
                
            return perf_metrics, resource_metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            raise
            
    def get_raw_data(self) -> Dict:
        """Get raw metrics data for analysis."""
        return {
            'operation_times': self._operation_times,
            'batch_sizes': self._batch_sizes,
            'error_counts': self._error_counts,
            'retry_counts': self._retry_counts,
            'resource_samples': [
                vars(sample) for sample in self._resource_samples
            ]
        }

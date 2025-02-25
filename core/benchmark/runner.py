"""
Benchmark runner for cross-model performance testing.
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
from core.metrics import performance_monitor
from core.thermal.monitor import ThermalMonitor
from core.thermal.profile import ProfileManager, ThermalProfile
from .metrics import MetricsCollector, PerformanceMetrics, ResourceMetrics

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    name: str
    description: str
    iterations: int = 5
    warmup_iterations: int = 1
    cooldown_seconds: float = 5.0
    thermal_profile: Optional[str] = "balanced"
    collect_resource_metrics: bool = True
    resource_sample_interval_ms: float = 100.0

@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config: BenchmarkConfig
    performance: PerformanceMetrics
    resources: Optional[ResourceMetrics]
    raw_data: Dict
    thermal_profile: Optional[ThermalProfile]
    start_time: float
    end_time: float
    
    @property
    def duration_seconds(self) -> float:
        """Get total benchmark duration."""
        return self.end_time - self.start_time

class BenchmarkRunner:
    """Runs performance benchmarks across different models."""
    
    def __init__(
        self,
        thermal_monitor: ThermalMonitor,
        profile_manager: ProfileManager
    ):
        """
        Initialize benchmark runner.
        
        Args:
            thermal_monitor: System thermal monitor
            profile_manager: Thermal profile manager
        """
        self.thermal_monitor = thermal_monitor
        self.profile_manager = profile_manager
        self.metrics_collector = MetricsCollector()
        self._current_config = None
        logger.info("Initialized benchmark runner")
        
    @performance_monitor
    async def run_benchmark(
        self,
        config: BenchmarkConfig,
        benchmark_func: Callable,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run a benchmark.
        
        Args:
            config: Benchmark configuration
            benchmark_func: Function to benchmark
            *args: Arguments for benchmark function
            **kwargs: Keyword arguments for benchmark function
            
        Returns:
            Benchmark results
        """
        try:
            self._current_config = config
            start_time = time.time()
            
            # Apply thermal profile if specified
            thermal_profile = None
            if config.thermal_profile:
                thermal_profile = await self.profile_manager.select_profile(
                    config.thermal_profile,
                    apply=True
                )
                
            # Reset metrics collector
            self.metrics_collector.reset()
            
            # Start resource monitoring if enabled
            if config.collect_resource_metrics:
                monitoring_task = asyncio.create_task(
                    self._monitor_resources(
                        config.resource_sample_interval_ms / 1000
                    )
                )
            
            # Run warmup iterations
            logger.info(f"Running {config.warmup_iterations} warmup iterations")
            for i in range(config.warmup_iterations):
                await self._run_iteration(benchmark_func, *args, **kwargs)
                await asyncio.sleep(1)  # Brief pause between warmups
                
            # Reset metrics for actual runs
            self.metrics_collector.reset()
            
            # Run benchmark iterations
            logger.info(f"Running {config.iterations} benchmark iterations")
            for i in range(config.iterations):
                await self._run_iteration(benchmark_func, *args, **kwargs)
                
                # Cooldown between iterations
                if i < config.iterations - 1:
                    await asyncio.sleep(config.cooldown_seconds)
                    
            # Stop resource monitoring
            if config.collect_resource_metrics:
                monitoring_task.cancel()
                try:
                    await monitoring_task
                except asyncio.CancelledError:
                    pass
                    
            # Compute final metrics
            perf_metrics, resource_metrics = (
                self.metrics_collector.compute_metrics()
            )
            
            # Create result
            result = BenchmarkResult(
                config=config,
                performance=perf_metrics,
                resources=resource_metrics,
                raw_data=self.metrics_collector.get_raw_data(),
                thermal_profile=thermal_profile,
                start_time=start_time,
                end_time=time.time()
            )
            
            logger.info(
                f"Benchmark completed in {result.duration_seconds:.2f}s with "
                f"{perf_metrics.operations_per_second:.2f} ops/sec"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error running benchmark: {str(e)}")
            raise
        finally:
            self._current_config = None
            
    async def _run_iteration(
        self,
        benchmark_func: Callable,
        *args,
        **kwargs
    ):
        """Run a single benchmark iteration."""
        try:
            # Start timing
            self.metrics_collector.start_operation()
            
            # Run benchmark function
            result = await benchmark_func(*args, **kwargs)
            
            # End timing
            self.metrics_collector.end_operation(
                batch_size=getattr(result, 'batch_size', 1),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in benchmark iteration: {str(e)}")
            self.metrics_collector.end_operation(
                batch_size=1,
                success=False
            )
            raise
            
    async def _monitor_resources(self, interval: float):
        """Monitor system resources."""
        try:
            while True:
                # Get thermal metrics
                thermal_metrics = await self.thermal_monitor.get_metrics()
                
                # Get memory stats
                memory_stats = self._get_memory_stats()
                
                # Add sample
                self.metrics_collector.add_resource_sample(
                    thermal_metrics,
                    memory_stats
                )
                
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error monitoring resources: {str(e)}")
            
    def _get_memory_stats(self) -> Dict:
        """Get current memory statistics."""
        try:
            import psutil
            
            # Get process memory info
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Get system memory info
            system = psutil.virtual_memory()
            
            # Get CPU info
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_freq = psutil.cpu_freq()
            
            # Get GPU info (mock values for now)
            # TODO: Add actual GPU monitoring
            gpu_stats = {
                'percent': 0,
                'freq': 0,
                'memory': 0
            }
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_freq': cpu_freq.current if cpu_freq else 0,
                'memory_used': memory_info.rss / (1024 * 1024),  # MB
                'memory_peak': memory_info.vms / (1024 * 1024),  # MB
                'memory_bandwidth': 0,  # TODO: Add bandwidth monitoring
                'gpu_percent': gpu_stats['percent'],
                'gpu_freq': gpu_stats['freq'],
                'gpu_memory': gpu_stats['memory']
            }
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}")
            return {
                'cpu_percent': 0,
                'cpu_freq': 0,
                'memory_used': 0,
                'memory_peak': 0,
                'memory_bandwidth': 0,
                'gpu_percent': 0,
                'gpu_freq': 0,
                'gpu_memory': 0
            }

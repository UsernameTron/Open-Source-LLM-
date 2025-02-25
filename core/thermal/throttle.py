"""
Dynamic thermal throttling for M4 Pro optimization.
"""

import logging
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from core.metrics import performance_monitor
from .monitor import ThermalMonitor, ThermalMetrics

logger = logging.getLogger(__name__)

@dataclass
class ThrottleConfig:
    """Configuration for thermal throttling."""
    # Temperature thresholds (Celsius)
    cpu_temp_warning: float = 80.0
    cpu_temp_critical: float = 90.0
    gpu_temp_warning: float = 75.0
    gpu_temp_critical: float = 85.0
    
    # Power thresholds (Watts)
    total_power_target: float = 35.0  # M4 Pro typical TDP
    power_warning: float = 40.0
    power_critical: float = 45.0
    
    # Batch size limits
    min_batch_size: int = 1
    max_batch_size: int = 64
    default_batch_size: int = 32
    
    # Throttling parameters
    throttle_factor_min: float = 0.1
    throttle_cooldown_sec: float = 30.0
    
    # Recovery parameters
    recovery_threshold_temp: float = 70.0
    recovery_threshold_power: float = 30.0
    recovery_step: float = 0.1

class ThermalThrottler:
    def __init__(
        self,
        monitor: ThermalMonitor,
        config: Optional[ThrottleConfig] = None
    ):
        """
        Initialize thermal throttler.
        
        Args:
            monitor: Thermal monitor instance
            config: Optional throttling configuration
        """
        self.monitor = monitor
        self.config = config or ThrottleConfig()
        self._last_throttle_time = 0
        self._current_throttle_factor = 1.0
        logger.info("Initialized thermal throttler")

    @performance_monitor
    async def get_batch_size(self, requested_size: int) -> int:
        """
        Get thermally optimized batch size.
        
        Args:
            requested_size: Requested batch size
            
        Returns:
            Adjusted batch size based on thermal conditions
        """
        try:
            # Get current metrics
            metrics = await self.monitor.get_metrics()
            
            # Calculate throttle factor
            throttle_factor = self._calculate_throttle_factor(metrics)
            
            # Apply throttling with cooldown
            if self._should_update_throttle():
                self._current_throttle_factor = throttle_factor
                self._last_throttle_time = time.time()
            
            # Calculate adjusted batch size
            adjusted_size = int(
                requested_size * self._current_throttle_factor
            )
            
            # Clamp to limits
            final_size = max(
                self.config.min_batch_size,
                min(adjusted_size, self.config.max_batch_size)
            )
            
            if final_size != requested_size:
                logger.info(
                    f"Adjusted batch size from {requested_size} to {final_size} "
                    f"(throttle factor: {self._current_throttle_factor:.2f})"
                )
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error calculating batch size: {str(e)}")
            return self.config.min_batch_size

    def _calculate_throttle_factor(self, metrics: ThermalMetrics) -> float:
        """Calculate throttling factor based on thermal metrics."""
        factors = []
        
        # CPU temperature factor
        if metrics.cpu_temperature >= self.config.cpu_temp_critical:
            factors.append(self.config.throttle_factor_min)
        elif metrics.cpu_temperature >= self.config.cpu_temp_warning:
            severity = (metrics.cpu_temperature - self.config.cpu_temp_warning) / (
                self.config.cpu_temp_critical - self.config.cpu_temp_warning
            )
            factors.append(1.0 - severity * 0.8)
            
        # GPU temperature factor
        if metrics.gpu_temperature >= self.config.gpu_temp_critical:
            factors.append(self.config.throttle_factor_min)
        elif metrics.gpu_temperature >= self.config.gpu_temp_warning:
            severity = (metrics.gpu_temperature - self.config.gpu_temp_warning) / (
                self.config.gpu_temp_critical - self.config.gpu_temp_warning
            )
            factors.append(1.0 - severity * 0.8)
            
        # Power factor
        if metrics.total_power >= self.config.power_critical:
            factors.append(self.config.throttle_factor_min)
        elif metrics.total_power >= self.config.power_warning:
            severity = (metrics.total_power - self.config.power_warning) / (
                self.config.power_critical - self.config.power_warning
            )
            factors.append(1.0 - severity * 0.8)
            
        # Use most restrictive factor
        if factors:
            return max(min(factors), self.config.throttle_factor_min)
        return 1.0

    def _should_update_throttle(self) -> bool:
        """Check if throttle factor should be updated."""
        return (
            time.time() - self._last_throttle_time >
            self.config.throttle_cooldown_sec
        )

    @performance_monitor
    async def get_performance_headroom(self) -> Tuple[float, str]:
        """
        Calculate available performance headroom.
        
        Returns:
            Tuple of (headroom factor, limitation reason)
        """
        try:
            metrics = await self.monitor.get_metrics()
            
            # Check different limitations
            limitations = []
            
            # CPU temperature
            cpu_headroom = (
                self.config.cpu_temp_critical - metrics.cpu_temperature
            ) / self.config.cpu_temp_critical
            if cpu_headroom < 0.2:
                limitations.append(
                    ("CPU Temperature", cpu_headroom)
                )
                
            # GPU temperature
            gpu_headroom = (
                self.config.gpu_temp_critical - metrics.gpu_temperature
            ) / self.config.gpu_temp_critical
            if gpu_headroom < 0.2:
                limitations.append(
                    ("GPU Temperature", gpu_headroom)
                )
                
            # Power consumption
            power_headroom = (
                self.config.power_critical - metrics.total_power
            ) / self.config.power_critical
            if power_headroom < 0.2:
                limitations.append(
                    ("Power Consumption", power_headroom)
                )
                
            # Find most restrictive limitation
            if limitations:
                limitations.sort(key=lambda x: x[1])
                return (
                    max(limitations[0][1], 0.0),
                    limitations[0][0]
                )
                
            return 1.0, "No Limitations"
            
        except Exception as e:
            logger.error(f"Error calculating performance headroom: {str(e)}")
            return 0.0, "Error"

    async def should_throttle(self) -> Tuple[bool, str]:
        """
        Check if throttling is needed.
        
        Returns:
            Tuple of (should_throttle, reason)
        """
        try:
            metrics = await self.monitor.get_metrics()
            
            # Check temperature thresholds
            if metrics.cpu_temperature >= self.config.cpu_temp_warning:
                return True, f"CPU temperature ({metrics.cpu_temperature:.1f}°C)"
                
            if metrics.gpu_temperature >= self.config.gpu_temp_warning:
                return True, f"GPU temperature ({metrics.gpu_temperature:.1f}°C)"
                
            # Check power threshold
            if metrics.total_power >= self.config.power_warning:
                return True, f"Power consumption ({metrics.total_power:.1f}W)"
                
            return False, "Normal operation"
            
        except Exception as e:
            logger.error(f"Error checking throttle condition: {str(e)}")
            return True, "Error checking conditions"

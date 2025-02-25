"""
Thermal monitoring for Apple Silicon M4 Pro.
"""

import logging
import time
from typing import Dict, Optional, List
import psutil
import subprocess
from dataclasses import dataclass
from core.metrics import performance_monitor

logger = logging.getLogger(__name__)

@dataclass
class ThermalMetrics:
    """Thermal and power metrics."""
    cpu_temperature: float  # Celsius
    gpu_temperature: float  # Celsius
    soc_temperature: float  # Celsius
    cpu_power: float  # Watts
    gpu_power: float  # Watts
    total_power: float  # Watts
    fan_speed: int  # RPM
    thermal_pressure: str  # Nominal, Moderate, Heavy, Critical
    timestamp: float

    @property
    def is_throttling(self) -> bool:
        """Check if system is thermally throttling."""
        return self.thermal_pressure in ["Heavy", "Critical"]

    @property
    def throttle_factor(self) -> float:
        """Get throttling factor (0.0-1.0)."""
        factors = {
            "Nominal": 1.0,
            "Moderate": 0.8,
            "Heavy": 0.5,
            "Critical": 0.2
        }
        return factors.get(self.thermal_pressure, 0.2)

class ThermalMonitor:
    def __init__(self):
        """Initialize thermal monitor."""
        self._history: List[ThermalMetrics] = []
        self._max_history_size = 1000
        self._powermetrics_proc = None
        self._start_monitoring()
        logger.info("Initialized thermal monitor")

    def _start_monitoring(self):
        """Start powermetrics monitoring in background."""
        try:
            # Start powermetrics with sampling
            cmd = [
                "sudo",
                "powermetrics",
                "--samplers",
                "cpu_power,gpu_power,thermal",
                "--format",
                "json"
            ]
            
            self._powermetrics_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info("Started powermetrics monitoring")
            
        except Exception as e:
            logger.error(f"Error starting powermetrics: {str(e)}")
            raise

    @performance_monitor
    async def get_metrics(self) -> ThermalMetrics:
        """
        Get current thermal metrics.
        
        Returns:
            ThermalMetrics object
        """
        try:
            # Get thermal metrics from powermetrics
            metrics = self._read_powermetrics()
            
            # Create ThermalMetrics object
            thermal_metrics = ThermalMetrics(
                cpu_temperature=metrics["cpu_temp"],
                gpu_temperature=metrics["gpu_temp"],
                soc_temperature=metrics["soc_temp"],
                cpu_power=metrics["cpu_power"],
                gpu_power=metrics["gpu_power"],
                total_power=metrics["total_power"],
                fan_speed=metrics["fan_speed"],
                thermal_pressure=self._get_thermal_pressure(),
                timestamp=time.time()
            )
            
            # Update history
            self._update_history(thermal_metrics)
            
            return thermal_metrics
            
        except Exception as e:
            logger.error(f"Error getting thermal metrics: {str(e)}")
            raise

    def _read_powermetrics(self) -> Dict:
        """Read metrics from powermetrics output."""
        try:
            if not self._powermetrics_proc:
                raise RuntimeError("powermetrics not running")
                
            # Read latest output
            output = self._powermetrics_proc.stdout.readline()
            data = json.loads(output)
            
            return {
                "cpu_temp": data["cpu_temp"],
                "gpu_temp": data["gpu_temp"],
                "soc_temp": data["soc_temp"],
                "cpu_power": data["cpu_power"],
                "gpu_power": data["gpu_power"],
                "total_power": data["total_power"],
                "fan_speed": data["fan_speed"]
            }
            
        except Exception as e:
            logger.error(f"Error reading powermetrics: {str(e)}")
            # Return fallback values
            return self._get_fallback_metrics()

    def _get_fallback_metrics(self) -> Dict:
        """Get fallback metrics using psutil."""
        temps = psutil.sensors_temperatures()
        power = psutil.sensors_power()
        
        return {
            "cpu_temp": temps.get("coretemp", [{"current": 0}])[0]["current"],
            "gpu_temp": temps.get("amdgpu", [{"current": 0}])[0]["current"],
            "soc_temp": temps.get("acpitz", [{"current": 0}])[0]["current"],
            "cpu_power": power.get("power1", {"current": 0})["current"],
            "gpu_power": power.get("power2", {"current": 0})["current"],
            "total_power": power.get("total", {"current": 0})["current"],
            "fan_speed": psutil.sensors_fans().get("system", [{"current": 0}])[0]["current"]
        }

    def _get_thermal_pressure(self) -> str:
        """Get thermal pressure level."""
        try:
            # Run thermal pressure command
            result = subprocess.run(
                ["sysctl", "kern.thermal_pressure"],
                capture_output=True,
                text=True
            )
            
            # Parse output
            pressure = result.stdout.split(": ")[1].strip()
            return pressure
            
        except Exception as e:
            logger.error(f"Error getting thermal pressure: {str(e)}")
            return "Unknown"

    def _update_history(self, metrics: ThermalMetrics):
        """Update metrics history."""
        self._history.append(metrics)
        
        # Maintain max history size
        if len(self._history) > self._max_history_size:
            self._history = self._history[-self._max_history_size:]

    def get_average_metrics(self, window_size: int = 60) -> ThermalMetrics:
        """
        Get average metrics over specified window.
        
        Args:
            window_size: Number of samples to average
            
        Returns:
            ThermalMetrics with averaged values
        """
        if not self._history:
            return None
            
        # Get recent metrics
        recent = self._history[-window_size:]
        
        # Calculate averages
        avg_metrics = ThermalMetrics(
            cpu_temperature=sum(m.cpu_temperature for m in recent) / len(recent),
            gpu_temperature=sum(m.gpu_temperature for m in recent) / len(recent),
            soc_temperature=sum(m.soc_temperature for m in recent) / len(recent),
            cpu_power=sum(m.cpu_power for m in recent) / len(recent),
            gpu_power=sum(m.gpu_power for m in recent) / len(recent),
            total_power=sum(m.total_power for m in recent) / len(recent),
            fan_speed=int(sum(m.fan_speed for m in recent) / len(recent)),
            thermal_pressure=recent[-1].thermal_pressure,  # Use latest pressure
            timestamp=time.time()
        )
        
        return avg_metrics

    def __del__(self):
        """Cleanup monitoring process."""
        if self._powermetrics_proc:
            self._powermetrics_proc.terminate()
            self._powermetrics_proc.wait()

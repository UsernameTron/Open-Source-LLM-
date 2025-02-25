"""Thermal management system for M4 Pro optimization."""

import asyncio
import logging
import subprocess
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ThermalState(Enum):
    """Thermal states for system operation."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class ThermalConfig:
    """Configuration for thermal management."""
    warning_temp: float = 75.0  # Celsius
    critical_temp: float = 85.0
    sampling_interval: float = 1.0  # seconds
    base_batch_size: int = 64

class ThermalManager:
    """Manages thermal conditions and optimizes performance."""
    
    def __init__(
        self,
        config: Optional[ThermalConfig] = None,
        on_state_change: Optional[Callable[[ThermalState], None]] = None
    ):
        self.config = config or ThermalConfig()
        self.current_temperature = 0.0
        self.current_state = ThermalState.NORMAL
        self.on_state_change = on_state_change
        self._monitoring_task = None
        self._subscribers: Dict[str, Callable] = {}
        
    async def start_monitoring(self):
        """Start thermal monitoring."""
        if self._monitoring_task:
            return
            
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info("Started thermal monitoring")
        
    async def stop_monitoring(self):
        """Stop thermal monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("Stopped thermal monitoring")
            
    def subscribe(
        self,
        subscriber_id: str,
        callback: Callable[[Dict[str, Any]], None]
    ):
        """Subscribe to thermal updates."""
        self._subscribers[subscriber_id] = callback
        
    def unsubscribe(self, subscriber_id: str):
        """Unsubscribe from thermal updates."""
        self._subscribers.pop(subscriber_id, None)
        
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on thermal state."""
        if self.current_state == ThermalState.CRITICAL:
            return max(1, self.config.base_batch_size // 4)
        elif self.current_state == ThermalState.WARNING:
            return max(1, self.config.base_batch_size // 2)
        return self.config.base_batch_size
        
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                # Get current temperature
                self.current_temperature = await self._get_cpu_temperature()
                
                # Update state
                new_state = self._get_thermal_state()
                if new_state != self.current_state:
                    await self._handle_state_change(new_state)
                    
                # Notify subscribers
                await self._notify_subscribers()
                
                # Wait for next check
                await asyncio.sleep(self.config.sampling_interval)
                
            except Exception as e:
                logger.error(f"Error in thermal monitoring: {str(e)}")
                await asyncio.sleep(self.config.sampling_interval * 2)
                
    async def _get_cpu_temperature(self) -> float:
        """Get current CPU temperature."""
        try:
            # For macOS/M4
            result = subprocess.run(
                ['osx-cpu-temp'],
                capture_output=True,
                text=True
            )
            temp_string = result.stdout.strip()
            return float(temp_string.split(':')[1].replace('°C', '').strip())
        except Exception as e:
            logger.error(f"Error getting CPU temperature: {str(e)}")
            return 0.0
            
    def _get_thermal_state(self) -> ThermalState:
        """Determine current thermal state."""
        if self.current_temperature >= self.config.critical_temp:
            return ThermalState.CRITICAL
        elif self.current_temperature >= self.config.warning_temp:
            return ThermalState.WARNING
        return ThermalState.NORMAL
        
    async def _handle_state_change(self, new_state: ThermalState):
        """Handle thermal state changes."""
        logger.warning(
            f"Thermal state changed: {self.current_state.value} -> "
            f"{new_state.value} ({self.current_temperature}°C)"
        )
        
        self.current_state = new_state
        
        if self.on_state_change:
            try:
                self.on_state_change(new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {str(e)}")
                
    async def _notify_subscribers(self):
        """Notify subscribers of current thermal state."""
        update = {
            'temperature': self.current_temperature,
            'state': self.current_state.value,
            'optimal_batch_size': self.get_optimal_batch_size()
        }
        
        for callback in self._subscribers.values():
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {str(e)}")
                
    async def __aenter__(self):
        """Context manager entry."""
        await self.start_monitoring()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.stop_monitoring()

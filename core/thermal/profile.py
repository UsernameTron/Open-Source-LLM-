"""
Thermal performance profiles for M4 Pro optimization.
"""

import logging
import json
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from core.metrics import performance_monitor
from .monitor import ThermalMonitor, ThermalMetrics
from .throttle import ThrottleConfig

logger = logging.getLogger(__name__)

@dataclass
class ThermalProfile:
    """Thermal performance profile."""
    name: str
    description: str
    
    # Batch processing
    batch_size: int
    max_concurrent_batches: int
    
    # Memory limits
    max_memory_usage: float  # GB
    memory_buffer: float  # GB
    
    # Power targets
    power_target: float  # Watts
    gpu_power_share: float  # 0-1
    
    # Thermal limits
    max_cpu_temp: float  # Celsius
    max_gpu_temp: float  # Celsius
    
    # Performance settings
    clock_speed_factor: float  # 0-1
    thermal_throttle_point: float  # Celsius
    
    @classmethod
    def create_balanced(cls) -> 'ThermalProfile':
        """Create balanced profile."""
        return cls(
            name="balanced",
            description="Balanced performance and thermal profile",
            batch_size=32,
            max_concurrent_batches=2,
            max_memory_usage=32.0,
            memory_buffer=4.0,
            power_target=35.0,
            gpu_power_share=0.4,
            max_cpu_temp=85.0,
            max_gpu_temp=80.0,
            clock_speed_factor=0.8,
            thermal_throttle_point=80.0
        )
        
    @classmethod
    def create_performance(cls) -> 'ThermalProfile':
        """Create performance-oriented profile."""
        return cls(
            name="performance",
            description="Maximum performance profile",
            batch_size=64,
            max_concurrent_batches=4,
            max_memory_usage=40.0,
            memory_buffer=2.0,
            power_target=45.0,
            gpu_power_share=0.5,
            max_cpu_temp=90.0,
            max_gpu_temp=85.0,
            clock_speed_factor=1.0,
            thermal_throttle_point=85.0
        )
        
    @classmethod
    def create_efficiency(cls) -> 'ThermalProfile':
        """Create efficiency-oriented profile."""
        return cls(
            name="efficiency",
            description="Maximum efficiency profile",
            batch_size=16,
            max_concurrent_batches=1,
            max_memory_usage=24.0,
            memory_buffer=8.0,
            power_target=25.0,
            gpu_power_share=0.3,
            max_cpu_temp=75.0,
            max_gpu_temp=70.0,
            clock_speed_factor=0.6,
            thermal_throttle_point=70.0
        )

class ProfileManager:
    def __init__(
        self,
        monitor: ThermalMonitor,
        config_path: Optional[str] = None
    ):
        """
        Initialize profile manager.
        
        Args:
            monitor: Thermal monitor instance
            config_path: Optional path to profile configurations
        """
        self.monitor = monitor
        self.config_path = config_path
        self._current_profile = None
        self._profiles: Dict[str, ThermalProfile] = {}
        self._load_profiles()
        logger.info("Initialized profile manager")

    def _load_profiles(self):
        """Load profiles from configuration or defaults."""
        try:
            if self.config_path:
                with open(self.config_path, 'r') as f:
                    profiles_data = json.load(f)
                    
                for name, data in profiles_data.items():
                    self._profiles[name] = ThermalProfile(**data)
            else:
                # Load default profiles
                self._profiles["balanced"] = ThermalProfile.create_balanced()
                self._profiles["performance"] = ThermalProfile.create_performance()
                self._profiles["efficiency"] = ThermalProfile.create_efficiency()
                
            logger.info(f"Loaded {len(self._profiles)} thermal profiles")
            
        except Exception as e:
            logger.error(f"Error loading profiles: {str(e)}")
            # Load defaults on error
            self._profiles["balanced"] = ThermalProfile.create_balanced()

    @performance_monitor
    async def select_profile(
        self,
        name: str,
        apply: bool = True
    ) -> ThermalProfile:
        """
        Select and optionally apply a thermal profile.
        
        Args:
            name: Profile name
            apply: Whether to apply the profile immediately
            
        Returns:
            Selected profile
        """
        try:
            if name not in self._profiles:
                raise ValueError(f"Profile '{name}' not found")
                
            profile = self._profiles[name]
            
            if apply:
                await self.apply_profile(profile)
                
            self._current_profile = profile
            logger.info(f"Selected profile: {name}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error selecting profile: {str(e)}")
            raise

    @performance_monitor
    async def apply_profile(self, profile: ThermalProfile):
        """
        Apply thermal profile settings.
        
        Args:
            profile: Profile to apply
        """
        try:
            # Create throttle config from profile
            throttle_config = ThrottleConfig(
                cpu_temp_warning=profile.thermal_throttle_point,
                cpu_temp_critical=profile.max_cpu_temp,
                gpu_temp_warning=profile.thermal_throttle_point,
                gpu_temp_critical=profile.max_gpu_temp,
                total_power_target=profile.power_target,
                power_warning=profile.power_target * 1.1,
                power_critical=profile.power_target * 1.2,
                min_batch_size=max(1, profile.batch_size // 4),
                max_batch_size=profile.batch_size,
                default_batch_size=profile.batch_size
            )
            
            # Apply system settings
            await self._apply_system_settings(profile)
            
            logger.info(f"Applied profile: {profile.name}")
            
        except Exception as e:
            logger.error(f"Error applying profile: {str(e)}")
            raise

    async def _apply_system_settings(self, profile: ThermalProfile):
        """Apply profile settings to system."""
        try:
            # Set power limits
            await self._set_power_limits(
                total_power=profile.power_target,
                gpu_power=profile.power_target * profile.gpu_power_share
            )
            
            # Set memory limits
            await self._set_memory_limits(
                max_memory=profile.max_memory_usage,
                buffer=profile.memory_buffer
            )
            
            # Set performance settings
            await self._set_performance_settings(
                clock_factor=profile.clock_speed_factor
            )
            
        except Exception as e:
            logger.error(f"Error applying system settings: {str(e)}")
            raise

    async def _set_power_limits(self, total_power: float, gpu_power: float):
        """Set system power limits."""
        try:
            # Use powermetrics to set power limits
            # Note: This requires sudo access
            cmd = [
                "sudo",
                "powermetrics",
                "--set-power-limits",
                f"--total-power={total_power}",
                f"--gpu-power={gpu_power}"
            ]
            
            subprocess.run(cmd, check=True)
            logger.info(f"Set power limits: total={total_power}W, gpu={gpu_power}W")
            
        except Exception as e:
            logger.error(f"Error setting power limits: {str(e)}")
            raise

    async def _set_memory_limits(self, max_memory: float, buffer: float):
        """Set memory limits."""
        try:
            # Set memory limits using memory pressure
            total_memory = max_memory + buffer
            cmd = [
                "sudo",
                "sysctl",
                "-w",
                f"kern.memorystatus_pressure_threshold={int(total_memory * 1024)}"
            ]
            
            subprocess.run(cmd, check=True)
            logger.info(f"Set memory limits: max={max_memory}GB, buffer={buffer}GB")
            
        except Exception as e:
            logger.error(f"Error setting memory limits: {str(e)}")
            raise

    async def _set_performance_settings(self, clock_factor: float):
        """Set performance-related settings."""
        try:
            # Set CPU performance using cpufreq
            cmd = [
                "sudo",
                "cpufreq-set",
                "-g",
                "performance" if clock_factor > 0.8 else "powersave"
            ]
            
            subprocess.run(cmd, check=True)
            logger.info(f"Set performance settings: clock_factor={clock_factor}")
            
        except Exception as e:
            logger.error(f"Error setting performance settings: {str(e)}")
            raise

    @performance_monitor
    async def get_recommended_profile(self) -> str:
        """
        Get recommended profile based on current conditions.
        
        Returns:
            Name of recommended profile
        """
        try:
            metrics = await self.monitor.get_metrics()
            
            # Check thermal conditions
            if metrics.is_throttling:
                return "efficiency"
                
            # Check power consumption
            if metrics.total_power > 40:
                return "balanced"
                
            # Check temperature
            if max(metrics.cpu_temperature, metrics.gpu_temperature) > 80:
                return "balanced"
                
            return "performance"
            
        except Exception as e:
            logger.error(f"Error getting recommended profile: {str(e)}")
            return "balanced"  # Safe default

    def save_profiles(self, path: Optional[str] = None):
        """Save profiles to configuration file."""
        try:
            save_path = path or self.config_path
            if not save_path:
                raise ValueError("No configuration path specified")
                
            profiles_data = {
                name: asdict(profile)
                for name, profile in self._profiles.items()
            }
            
            with open(save_path, 'w') as f:
                json.dump(profiles_data, f, indent=2)
                
            logger.info(f"Saved {len(self._profiles)} profiles to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving profiles: {str(e)}")
            raise

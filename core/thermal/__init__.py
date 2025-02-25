"""
Thermal management system for M4 Pro optimization.
"""

from .monitor import ThermalMonitor, ThermalMetrics
from .throttle import ThermalThrottler
from .profile import ThermalProfile, ProfileManager

__all__ = [
    'ThermalMonitor',
    'ThermalMetrics',
    'ThermalThrottler',
    'ThermalProfile',
    'ProfileManager'
]

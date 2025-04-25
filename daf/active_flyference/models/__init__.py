"""
Generative models for Active Inference in flybody.

This module contains generative models for different flybody tasks.
"""

from .generative_model import GenerativeModel
from .walk_model import WalkOnBallModel
from .flight_model import FlightModel

__all__ = [
    'GenerativeModel',
    'WalkOnBallModel',
    'FlightModel'
] 
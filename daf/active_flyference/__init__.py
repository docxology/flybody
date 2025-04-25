"""
Active Flyference: An Active Inference framework for flybody models.

This package provides an implementation of the Active Inference framework
for flybody tasks, modeling perception and action as a POMDP.
"""

from .active_inference_agent import ActiveInferenceAgent, create_agent_for_task

__all__ = [
    'ActiveInferenceAgent',
    'create_agent_for_task'
] 
#!/usr/bin/env python3
"""
Run Active Inference agents on flybody tasks.
This is a wrapper script for deploy_active_inference.py that handles imports correctly.
"""

import os
import sys
import argparse

# Add the current directory to the path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Add parent directory to sys.path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import from daf_agent_utils
from daf_agent_utils import create_run_dirs, run_episode, MetricLogger

try:
    # Try to import the ActiveInferenceAgent class with absolute imports
    from active_flyference.active_inference_agent import ActiveInferenceAgent, create_agent_for_task
    from active_flyference.models.generative_model import GenerativeModel
    from active_flyference.models.walk_model import WalkOnBallModel
    from active_flyference.models.flight_model import FlightModel
    from active_flyference.utils.visualization import plot_active_inference_summary
    
    # Now import the main function from deploy_active_inference
    from active_flyference.deploy_active_inference import main
    
    if __name__ == "__main__":
        # Execute the main function
        main()
except Exception as e:
    print(f"Error importing modules: {e}")
    print("\nPossible fixes:")
    print("1. Make sure you're running this script from the flybody root directory")
    print("2. Make sure you have installed all required dependencies")
    print("3. Try using: cd /path/to/flybody && python -m daf.active_flyference.deploy_active_inference")
    sys.exit(1) 
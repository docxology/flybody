"""
Filter implementations for flybody.
"""

import numpy as np
from scipy import signal

class ButterworthFilter:
    """
    Butterworth filter implementation for smoothing time series data.
    
    This is typically used for joint angle filtering to reduce noise and produce
    smoother motions in the simulation.
    """
    
    def __init__(self, cutoff=10.0, fs=100.0, order=4):
        """
        Initialize a Butterworth filter.
        
        Args:
            cutoff: Cutoff frequency in Hz
            fs: Sampling frequency in Hz
            order: Filter order
        """
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self._setup_filter()
        self.reset()
        
    def _setup_filter(self):
        """Create the filter coefficients."""
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        self.b, self.a = signal.butter(self.order, normal_cutoff, btype='low', analog=False)
        
    def reset(self):
        """Reset filter state."""
        self.zi = signal.lfilter_zi(self.b, self.a)
        self.last_value = None
        
    def filter(self, x):
        """
        Apply filter to input data.
        
        Args:
            x: Input data (can be scalar or array)
            
        Returns:
            Filtered output
        """
        # Handle scalar inputs
        if np.isscalar(x):
            x = np.array([x])
            
        # Initialize zi state if this is the first value
        if self.last_value is None:
            self.last_value = x[0]
            self.zi = self.zi * self.last_value
            
        # Apply filter
        y, self.zi = signal.lfilter(self.b, self.a, x, zi=self.zi)
        self.last_value = y[-1]
        
        # Return scalar if input was scalar
        if len(y) == 1:
            return y[0]
        
        return y
        
    def __call__(self, x):
        """Call interface for the filter."""
        return self.filter(x) 
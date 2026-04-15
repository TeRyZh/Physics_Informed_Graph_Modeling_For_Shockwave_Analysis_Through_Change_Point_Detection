import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import math
from numba import jit
import warnings
from scipy.signal import find_peaks
from .segmented_regression import segmented_regression_dp


import numpy as np
from typing import List, Tuple, Dict, Any


class PureDPDetection:
    """
    Pure Dynamic Programming method for segmented regression.
    """
    
    def __init__(self, penalty=100, min_segment_length=20):
        """
        Initialize Pure DP detector.
        
        Args:
            penalty: Penalty for adding a new segment
            min_segment_length: Minimum segment length
        """
        self.penalty = penalty
        self.min_segment_length = min_segment_length
    
    def detect(self, time, data):
        """
        Detect change points using pure DP segmented regression.
        
        Args:
            time: Array of time points
            data: Array of data (speed, position, etc.)
            
        Returns:
            breakpoints: Indices where change points are detected
            diagnostics: Additional information about the detection
        """
        breakpoints, diagnostics = segmented_regression_dp(
            time, data, self.penalty, self.min_segment_length
        )
        
        return breakpoints, diagnostics
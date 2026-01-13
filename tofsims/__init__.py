"""
Three-dimensional ToF-SIMS Imaging Toolkit

A Python toolkit for analyzing and visualizing three-dimensional 
Time-of-Flight Secondary Ion Mass Spectrometry (ToF-SIMS) imaging data.

This package provides modules for:
- Data loading from various ToF-SIMS formats
- Preprocessing and normalization
- Region of Interest (ROI) and line-profile analysis
- 3D visualization of molecular distributions
"""

__version__ = "0.1.0"
__author__ = "ToF-SIMS Research Team"

from . import data_loader
from . import preprocessing
from . import roi_analysis
from . import visualization

__all__ = [
    'data_loader',
    'preprocessing',
    'roi_analysis',
    'visualization',
]

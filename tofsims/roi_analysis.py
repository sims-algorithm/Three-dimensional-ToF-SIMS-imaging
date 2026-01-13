"""
ROI Analysis Module for ToF-SIMS Imaging

This module provides functions for Region of Interest (ROI) selection and analysis,
including line profile extraction and statistical analysis.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class ROI:
    """
    Class representing a Region of Interest.
    
    Attributes
    ----------
    z_range : tuple of int
        (start, end) indices in depth dimension
    y_range : tuple of int
        (start, end) indices in height dimension
    x_range : tuple of int
        (start, end) indices in width dimension
    name : str
        Name/label for this ROI
    """
    z_range: Tuple[int, int]
    y_range: Tuple[int, int]
    x_range: Tuple[int, int]
    name: str = "ROI"
    
    def extract(self, data: np.ndarray) -> np.ndarray:
        """
        Extract data within this ROI.
        
        Parameters
        ----------
        data : np.ndarray
            3D data array
            
        Returns
        -------
        np.ndarray
            Data within ROI boundaries
        """
        z_slice = slice(*self.z_range)
        y_slice = slice(*self.y_range)
        x_slice = slice(*self.x_range)
        return data[z_slice, y_slice, x_slice]
    
    def volume(self) -> int:
        """Calculate the volume (number of voxels) in this ROI."""
        z_size = self.z_range[1] - self.z_range[0]
        y_size = self.y_range[1] - self.y_range[0]
        x_size = self.x_range[1] - self.x_range[0]
        return z_size * y_size * x_size


def define_roi(z_range: Tuple[int, int],
               y_range: Tuple[int, int],
               x_range: Tuple[int, int],
               name: str = "ROI") -> ROI:
    """
    Define a Region of Interest.
    
    Parameters
    ----------
    z_range : tuple of int
        (start, end) indices in depth dimension
    y_range : tuple of int
        (start, end) indices in height dimension
    x_range : tuple of int
        (start, end) indices in width dimension
    name : str, default="ROI"
        Name/label for this ROI
        
    Returns
    -------
    ROI
        ROI object
        
    Examples
    --------
    >>> roi = define_roi(z_range=(20, 80), y_range=(50, 150), 
    ...                  x_range=(50, 150), name="Cell_1")
    """
    return ROI(z_range, y_range, x_range, name)


def extract_roi_statistics(data: np.ndarray, roi: ROI) -> Dict[str, float]:
    """
    Calculate statistics for data within an ROI.
    
    Parameters
    ----------
    data : np.ndarray
        3D data array
    roi : ROI
        Region of Interest
        
    Returns
    -------
    dict
        Dictionary containing statistical measures
        
    Examples
    --------
    >>> stats = extract_roi_statistics(data, roi)
    >>> print(f"Mean intensity: {stats['mean']:.2f}")
    """
    roi_data = roi.extract(data)
    
    stats = {
        'mean': float(np.mean(roi_data)),
        'median': float(np.median(roi_data)),
        'std': float(np.std(roi_data)),
        'min': float(np.min(roi_data)),
        'max': float(np.max(roi_data)),
        'sum': float(np.sum(roi_data)),
        'volume': roi.volume(),
        'total_intensity': float(np.sum(roi_data))
    }
    
    return stats


def extract_line_profile(data: np.ndarray,
                        start: Tuple[int, int, int],
                        end: Tuple[int, int, int],
                        width: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract intensity profile along a line in 3D space.
    
    Parameters
    ----------
    data : np.ndarray
        3D data array
    start : tuple of int
        Starting point coordinates (z, y, x)
    end : tuple of int
        Ending point coordinates (z, y, x)
    width : int, default=1
        Width of the line profile (averaging perpendicular to line)
        
    Returns
    -------
    distances : np.ndarray
        Distance values along the line
    intensities : np.ndarray
        Intensity values along the line
        
    Examples
    --------
    >>> distances, intensities = extract_line_profile(
    ...     data, start=(0, 50, 50), end=(99, 200, 200))
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(distances, intensities)
    """
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    
    # Calculate number of samples along the line
    length = np.linalg.norm(end - start)
    num_samples = int(np.ceil(length)) + 1
    
    # Generate points along the line
    t = np.linspace(0, 1, num_samples)
    points = start[:, np.newaxis] + t * (end - start)[:, np.newaxis]
    
    # Sample intensities at these points
    from scipy import ndimage
    intensities = ndimage.map_coordinates(data, points, order=1)
    
    # Calculate distances
    distances = np.linspace(0, length, num_samples)
    
    return distances, intensities


def extract_depth_profile(data: np.ndarray,
                          y_range: Optional[Tuple[int, int]] = None,
                          x_range: Optional[Tuple[int, int]] = None,
                          aggregation: str = 'mean') -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract depth profile (intensity vs. depth) from 3D data.
    
    Parameters
    ----------
    data : np.ndarray
        3D data array with shape (depth, height, width)
    y_range : tuple of int, optional
        (start, end) indices for y range. If None, uses entire range.
    x_range : tuple of int, optional
        (start, end) indices for x range. If None, uses entire range.
    aggregation : str, default='mean'
        Method to aggregate xy plane: 'mean', 'median', 'sum', or 'max'
        
    Returns
    -------
    depths : np.ndarray
        Depth indices
    intensities : np.ndarray
        Aggregated intensity at each depth
        
    Examples
    --------
    >>> depths, intensities = extract_depth_profile(data, 
    ...                                             y_range=(50, 200),
    ...                                             x_range=(50, 200))
    """
    # Select region
    y_slice = slice(*y_range) if y_range else slice(None)
    x_slice = slice(*x_range) if x_range else slice(None)
    region = data[:, y_slice, x_slice]
    
    # Aggregate along xy plane
    if aggregation == 'mean':
        intensities = np.mean(region, axis=(1, 2))
    elif aggregation == 'median':
        intensities = np.median(region, axis=(1, 2))
    elif aggregation == 'sum':
        intensities = np.sum(region, axis=(1, 2))
    elif aggregation == 'max':
        intensities = np.max(region, axis=(1, 2))
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    depths = np.arange(len(intensities))
    
    return depths, intensities


def compare_rois(data: np.ndarray, rois: List[ROI]) -> Dict[str, Dict[str, float]]:
    """
    Compare statistics across multiple ROIs.
    
    Parameters
    ----------
    data : np.ndarray
        3D data array
    rois : list of ROI
        List of ROI objects to compare
        
    Returns
    -------
    dict
        Dictionary mapping ROI names to their statistics
        
    Examples
    --------
    >>> roi1 = define_roi((0, 50), (0, 100), (0, 100), name="Region_A")
    >>> roi2 = define_roi((50, 100), (0, 100), (0, 100), name="Region_B")
    >>> comparison = compare_rois(data, [roi1, roi2])
    >>> for name, stats in comparison.items():
    ...     print(f"{name}: mean={stats['mean']:.2f}")
    """
    comparison = {}
    for roi in rois:
        comparison[roi.name] = extract_roi_statistics(data, roi)
    return comparison


def create_roi_mask(shape: Tuple[int, int, int], roi: ROI) -> np.ndarray:
    """
    Create a binary mask for an ROI.
    
    Parameters
    ----------
    shape : tuple of int
        Shape of the full volume (depth, height, width)
    roi : ROI
        Region of Interest
        
    Returns
    -------
    np.ndarray
        Binary mask with 1 inside ROI and 0 outside
        
    Examples
    --------
    >>> mask = create_roi_mask(data.shape, roi)
    >>> masked_data = data * mask
    """
    mask = np.zeros(shape, dtype=bool)
    z_slice = slice(*roi.z_range)
    y_slice = slice(*roi.y_range)
    x_slice = slice(*roi.x_range)
    mask[z_slice, y_slice, x_slice] = True
    return mask.astype(int)

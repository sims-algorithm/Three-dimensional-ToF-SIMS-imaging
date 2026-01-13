"""
Preprocessing Module for ToF-SIMS Imaging

This module provides functions for preprocessing ToF-SIMS data including
normalization, filtering, background correction, and noise reduction.
"""

import numpy as np
from typing import Optional, Tuple, Union
from scipy import ndimage


def normalize_data(data: np.ndarray, 
                   method: str = 'minmax',
                   axis: Optional[int] = None) -> np.ndarray:
    """
    Normalize ToF-SIMS data using various methods.
    
    Parameters
    ----------
    data : np.ndarray
        Input data to normalize
    method : str, default='minmax'
        Normalization method: 'minmax', 'zscore', 'l2', or 'total_ion_count'
    axis : int, optional
        Axis along which to normalize. If None, normalize entire array.
        
    Returns
    -------
    np.ndarray
        Normalized data
        
    Examples
    --------
    >>> normalized = normalize_data(data, method='minmax')
    >>> print(f"Range: {normalized.min()} to {normalized.max()}")
    """
    data = data.astype(np.float64)
    
    if method == 'minmax':
        # Scale to [0, 1] range
        data_min = np.min(data, axis=axis, keepdims=True)
        data_max = np.max(data, axis=axis, keepdims=True)
        data_range = data_max - data_min
        # Avoid division by zero
        data_range[data_range == 0] = 1.0
        normalized = (data - data_min) / data_range
        
    elif method == 'zscore':
        # Standardize to zero mean and unit variance
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        std[std == 0] = 1.0
        normalized = (data - mean) / std
        
    elif method == 'l2':
        # L2 normalization
        norm = np.linalg.norm(data, axis=axis, keepdims=True)
        norm[norm == 0] = 1.0
        normalized = data / norm
        
    elif method == 'total_ion_count':
        # Normalize by total ion count (sum)
        total = np.sum(data, axis=axis, keepdims=True)
        total[total == 0] = 1.0
        normalized = data / total
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def gaussian_filter_3d(data: np.ndarray, 
                       sigma: Union[float, Tuple[float, float, float]] = 1.0) -> np.ndarray:
    """
    Apply 3D Gaussian filter for noise reduction.
    
    Parameters
    ----------
    data : np.ndarray
        3D input data
    sigma : float or tuple of float
        Standard deviation for Gaussian kernel. Can be a single value or
        tuple (sigma_z, sigma_y, sigma_x) for anisotropic filtering.
        
    Returns
    -------
    np.ndarray
        Filtered data
        
    Examples
    --------
    >>> filtered = gaussian_filter_3d(data, sigma=1.5)
    >>> # Anisotropic filtering (less smoothing in z direction)
    >>> filtered = gaussian_filter_3d(data, sigma=(0.5, 1.0, 1.0))
    """
    return ndimage.gaussian_filter(data, sigma=sigma)


def median_filter_3d(data: np.ndarray, 
                     size: Union[int, Tuple[int, int, int]] = 3) -> np.ndarray:
    """
    Apply 3D median filter for outlier removal.
    
    Parameters
    ----------
    data : np.ndarray
        3D input data
    size : int or tuple of int
        Size of the median filter kernel. Can be a single value or
        tuple (size_z, size_y, size_x) for anisotropic filtering.
        
    Returns
    -------
    np.ndarray
        Filtered data
        
    Examples
    --------
    >>> filtered = median_filter_3d(data, size=3)
    """
    return ndimage.median_filter(data, size=size)


def background_correction(data: np.ndarray,
                         method: str = 'rolling_ball',
                         radius: int = 50) -> np.ndarray:
    """
    Remove background from ToF-SIMS data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    method : str, default='rolling_ball'
        Background correction method: 'rolling_ball' or 'percentile'
    radius : int, default=50
        Radius for rolling ball algorithm or percentile value
        
    Returns
    -------
    np.ndarray
        Background-corrected data
        
    Examples
    --------
    >>> corrected = background_correction(data, method='rolling_ball', radius=50)
    """
    if method == 'rolling_ball':
        # Approximate rolling ball with morphological opening
        structure = ndimage.generate_binary_structure(data.ndim, 1)
        structure = ndimage.iterate_structure(structure, radius)
        background = ndimage.grey_opening(data, structure=structure)
        corrected = data - background
        corrected[corrected < 0] = 0
        
    elif method == 'percentile':
        # Use percentile as background estimate
        if data.ndim == 3:
            background = np.percentile(data, radius, axis=(1, 2), keepdims=True)
        else:
            background = np.percentile(data, radius)
        corrected = data - background
        corrected[corrected < 0] = 0
        
    else:
        raise ValueError(f"Unknown background correction method: {method}")
    
    return corrected


def remove_hot_pixels(data: np.ndarray, 
                     threshold: float = 3.0) -> np.ndarray:
    """
    Remove hot pixels (outliers) from ToF-SIMS data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    threshold : float, default=3.0
        Number of standard deviations above mean to consider as hot pixel
        
    Returns
    -------
    np.ndarray
        Data with hot pixels replaced by median of neighbors
        
    Examples
    --------
    >>> cleaned = remove_hot_pixels(data, threshold=3.0)
    """
    data_cleaned = data.copy()
    
    # Calculate median and MAD (median absolute deviation)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    # Identify hot pixels
    hot_pixels = data > (median + threshold * mad * 1.4826)
    
    # Replace with median filtered values
    median_filtered = ndimage.median_filter(data, size=3)
    data_cleaned[hot_pixels] = median_filtered[hot_pixels]
    
    return data_cleaned


def crop_volume(data: np.ndarray,
                z_range: Optional[Tuple[int, int]] = None,
                y_range: Optional[Tuple[int, int]] = None,
                x_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Crop a 3D volume to specified ranges.
    
    Parameters
    ----------
    data : np.ndarray
        3D input data with shape (depth, height, width)
    z_range : tuple of int, optional
        (start, end) indices for depth dimension
    y_range : tuple of int, optional
        (start, end) indices for height dimension
    x_range : tuple of int, optional
        (start, end) indices for width dimension
        
    Returns
    -------
    np.ndarray
        Cropped volume
        
    Examples
    --------
    >>> # Crop to central region
    >>> cropped = crop_volume(data, z_range=(10, 90), 
    ...                       y_range=(50, 200), x_range=(50, 200))
    """
    z_slice = slice(*z_range) if z_range else slice(None)
    y_slice = slice(*y_range) if y_range else slice(None)
    x_slice = slice(*x_range) if x_range else slice(None)
    
    return data[z_slice, y_slice, x_slice]


def resample_volume(data: np.ndarray,
                   scale_factors: Tuple[float, float, float]) -> np.ndarray:
    """
    Resample 3D volume by given scale factors.
    
    Parameters
    ----------
    data : np.ndarray
        3D input data
    scale_factors : tuple of float
        Scaling factors for each dimension (z, y, x)
        
    Returns
    -------
    np.ndarray
        Resampled volume
        
    Examples
    --------
    >>> # Downsample by factor of 2 in all dimensions
    >>> resampled = resample_volume(data, scale_factors=(0.5, 0.5, 0.5))
    """
    return ndimage.zoom(data, scale_factors, order=1)

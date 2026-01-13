"""
Data Loading Module for ToF-SIMS Imaging

This module provides functions to load ToF-SIMS data from various file formats
including raw binary files, CSV, and common image formats.
"""

import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path


def load_raw_data(filepath: Union[str, Path], 
                  shape: Optional[Tuple[int, int, int]] = None,
                  dtype: str = 'float32') -> np.ndarray:
    """
    Load raw ToF-SIMS data from a binary file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the raw data file
    shape : tuple of int, optional
        Expected shape of the 3D data (depth, height, width).
        If None, data is loaded as 1D array.
    dtype : str, default='float32'
        Data type for the array
        
    Returns
    -------
    np.ndarray
        3D array containing the ToF-SIMS data
        
    Examples
    --------
    >>> data = load_raw_data('data.raw', shape=(100, 256, 256))
    >>> print(data.shape)
    (100, 256, 256)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    data = np.fromfile(filepath, dtype=dtype)
    
    if shape is not None:
        if data.size != np.prod(shape):
            raise ValueError(
                f"Data size {data.size} does not match expected shape {shape} "
                f"(product: {np.prod(shape)})"
            )
        data = data.reshape(shape)
    
    return data


def load_image_stack(directory: Union[str, Path], 
                     pattern: str = "*.tif",
                     depth_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load a stack of 2D images to form a 3D volume.
    
    Parameters
    ----------
    directory : str or Path
        Directory containing the image files
    pattern : str, default='*.tif'
        Glob pattern for matching image files
    depth_range : tuple of int, optional
        (start, end) indices for loading subset of images
        
    Returns
    -------
    np.ndarray
        3D array with shape (depth, height, width)
        
    Examples
    --------
    >>> data = load_image_stack('images/', pattern='slice_*.tif')
    >>> print(f"Loaded {data.shape[0]} slices")
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL/Pillow is required for loading images. "
                         "Install with: pip install Pillow")
    
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Get sorted list of matching files
    files = sorted(directory.glob(pattern))
    if not files:
        raise ValueError(f"No files matching pattern '{pattern}' found in {directory}")
    
    # Apply depth range filter if specified
    if depth_range is not None:
        start, end = depth_range
        files = files[start:end]
    
    # Load first image to get dimensions
    first_img = np.array(Image.open(files[0]))
    height, width = first_img.shape[:2]
    
    # Initialize 3D array
    data = np.zeros((len(files), height, width), dtype=first_img.dtype)
    data[0] = first_img if first_img.ndim == 2 else first_img[:, :, 0]
    
    # Load remaining images
    for i, filepath in enumerate(files[1:], start=1):
        img = np.array(Image.open(filepath))
        data[i] = img if img.ndim == 2 else img[:, :, 0]
    
    return data


def load_csv_data(filepath: Union[str, Path],
                  shape: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Load ToF-SIMS data from a CSV file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file
    shape : tuple of int, optional
        Expected shape to reshape the data into (depth, height, width)
        
    Returns
    -------
    np.ndarray
        Data array (1D or 3D depending on shape parameter)
        
    Examples
    --------
    >>> data = load_csv_data('data.csv', shape=(50, 128, 128))
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    data = np.loadtxt(filepath, delimiter=',')
    
    if shape is not None:
        data = data.reshape(shape)
    
    return data


def get_data_info(data: np.ndarray) -> dict:
    """
    Get summary information about loaded ToF-SIMS data.
    
    Parameters
    ----------
    data : np.ndarray
        The ToF-SIMS data array
        
    Returns
    -------
    dict
        Dictionary containing data statistics
        
    Examples
    --------
    >>> info = get_data_info(data)
    >>> print(f"Data range: {info['min']} to {info['max']}")
    """
    info = {
        'shape': data.shape,
        'dtype': data.dtype,
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'size_mb': data.nbytes / (1024 * 1024)
    }
    return info

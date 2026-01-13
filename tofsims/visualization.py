"""
Visualization Module for ToF-SIMS Imaging

This module provides functions for 3D visualization of ToF-SIMS data,
including volume rendering, slice viewing, and projection plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Optional, Tuple, Union
from mpl_toolkits.mplot3d import Axes3D


def plot_slice(data: np.ndarray,
               slice_idx: int,
               axis: int = 0,
               cmap: str = 'viridis',
               title: Optional[str] = None,
               colorbar: bool = True,
               figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot a 2D slice from 3D data.
    
    Parameters
    ----------
    data : np.ndarray
        3D data array
    slice_idx : int
        Index of the slice to plot
    axis : int, default=0
        Axis along which to slice: 0 (z), 1 (y), or 2 (x)
    cmap : str, default='viridis'
        Colormap name
    title : str, optional
        Plot title
    colorbar : bool, default=True
        Whether to show colorbar
    figsize : tuple of int, default=(8, 6)
        Figure size
        
    Examples
    --------
    >>> plot_slice(data, slice_idx=50, axis=0, title='Depth slice at z=50')
    """
    plt.figure(figsize=figsize)
    
    if axis == 0:
        slice_data = data[slice_idx, :, :]
        axis_labels = ('X', 'Y')
    elif axis == 1:
        slice_data = data[:, slice_idx, :]
        axis_labels = ('X', 'Z')
    elif axis == 2:
        slice_data = data[:, :, slice_idx]
        axis_labels = ('Y', 'Z')
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or 2")
    
    im = plt.imshow(slice_data, cmap=cmap, aspect='auto', origin='lower')
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    
    if title:
        plt.title(title)
    else:
        axis_names = ['Z', 'Y', 'X']
        plt.title(f'Slice along {axis_names[axis]} axis at index {slice_idx}')
    
    if colorbar:
        plt.colorbar(im, label='Intensity')
    
    plt.tight_layout()


def plot_orthogonal_slices(data: np.ndarray,
                          z_idx: Optional[int] = None,
                          y_idx: Optional[int] = None,
                          x_idx: Optional[int] = None,
                          cmap: str = 'viridis',
                          figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Plot three orthogonal slices (XY, XZ, YZ planes).
    
    Parameters
    ----------
    data : np.ndarray
        3D data array with shape (depth, height, width)
    z_idx : int, optional
        Z index for XY slice. If None, uses middle slice.
    y_idx : int, optional
        Y index for XZ slice. If None, uses middle slice.
    x_idx : int, optional
        X index for YZ slice. If None, uses middle slice.
    cmap : str, default='viridis'
        Colormap name
    figsize : tuple of int, default=(15, 5)
        Figure size
        
    Examples
    --------
    >>> plot_orthogonal_slices(data, z_idx=50, y_idx=128, x_idx=128)
    """
    depth, height, width = data.shape
    
    # Use middle slices if not specified
    if z_idx is None:
        z_idx = depth // 2
    if y_idx is None:
        y_idx = height // 2
    if x_idx is None:
        x_idx = width // 2
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # XY slice (at z_idx)
    im1 = axes[0].imshow(data[z_idx, :, :], cmap=cmap, aspect='auto', origin='lower')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title(f'XY slice (Z={z_idx})')
    plt.colorbar(im1, ax=axes[0], label='Intensity')
    
    # XZ slice (at y_idx)
    im2 = axes[1].imshow(data[:, y_idx, :], cmap=cmap, aspect='auto', origin='lower')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].set_title(f'XZ slice (Y={y_idx})')
    plt.colorbar(im2, ax=axes[1], label='Intensity')
    
    # YZ slice (at x_idx)
    im3 = axes[2].imshow(data[:, :, x_idx], cmap=cmap, aspect='auto', origin='lower')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    axes[2].set_title(f'YZ slice (X={x_idx})')
    plt.colorbar(im3, ax=axes[2], label='Intensity')
    
    plt.tight_layout()


def plot_maximum_intensity_projection(data: np.ndarray,
                                     axis: int = 0,
                                     cmap: str = 'viridis',
                                     title: Optional[str] = None,
                                     figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Create Maximum Intensity Projection (MIP) along specified axis.
    
    Parameters
    ----------
    data : np.ndarray
        3D data array
    axis : int, default=0
        Axis along which to project: 0 (z), 1 (y), or 2 (x)
    cmap : str, default='viridis'
        Colormap name
    title : str, optional
        Plot title
    figsize : tuple of int, default=(8, 6)
        Figure size
        
    Examples
    --------
    >>> plot_maximum_intensity_projection(data, axis=0, 
    ...                                   title='MIP along Z axis')
    """
    plt.figure(figsize=figsize)
    
    mip = np.max(data, axis=axis)
    
    im = plt.imshow(mip, cmap=cmap, aspect='auto', origin='lower')
    
    if axis == 0:
        plt.xlabel('X')
        plt.ylabel('Y')
        default_title = 'Maximum Intensity Projection (Z axis)'
    elif axis == 1:
        plt.xlabel('X')
        plt.ylabel('Z')
        default_title = 'Maximum Intensity Projection (Y axis)'
    else:
        plt.xlabel('Y')
        plt.ylabel('Z')
        default_title = 'Maximum Intensity Projection (X axis)'
    
    plt.title(title if title else default_title)
    plt.colorbar(im, label='Max Intensity')
    plt.tight_layout()


def plot_volume_rendering(data: np.ndarray,
                         threshold: Optional[float] = None,
                         alpha: float = 0.3,
                         cmap: str = 'viridis',
                         figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Create a simple 3D volume rendering using scatter plot.
    
    Parameters
    ----------
    data : np.ndarray
        3D data array
    threshold : float, optional
        Only show voxels above this threshold. If None, uses 50th percentile.
    alpha : float, default=0.3
        Transparency of points
    cmap : str, default='viridis'
        Colormap name
    figsize : tuple of int, default=(10, 8)
        Figure size
        
    Examples
    --------
    >>> # Show only high-intensity regions
    >>> plot_volume_rendering(data, threshold=0.5, alpha=0.5)
    """
    if threshold is None:
        threshold = np.percentile(data, 50)
    
    # Find voxels above threshold
    z, y, x = np.where(data > threshold)
    values = data[z, y, x]
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot
    scatter = ax.scatter(x, y, z, c=values, cmap=cmap, 
                        alpha=alpha, s=1, marker='.')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Depth)')
    ax.set_title('3D Volume Rendering')
    
    plt.colorbar(scatter, ax=ax, label='Intensity', shrink=0.5)
    plt.tight_layout()


def plot_line_profile(distances: np.ndarray,
                     intensities: np.ndarray,
                     title: str = 'Line Profile',
                     xlabel: str = 'Distance',
                     ylabel: str = 'Intensity',
                     figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot a line profile.
    
    Parameters
    ----------
    distances : np.ndarray
        Distance values
    intensities : np.ndarray
        Intensity values
    title : str, default='Line Profile'
        Plot title
    xlabel : str, default='Distance'
        X-axis label
    ylabel : str, default='Intensity'
        Y-axis label
    figsize : tuple of int, default=(10, 6)
        Figure size
        
    Examples
    --------
    >>> from tofsims.roi_analysis import extract_line_profile
    >>> distances, intensities = extract_line_profile(data, (0,50,50), (99,200,200))
    >>> plot_line_profile(distances, intensities)
    """
    plt.figure(figsize=figsize)
    plt.plot(distances, intensities, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_depth_profile(depths: np.ndarray,
                      intensities: np.ndarray,
                      title: str = 'Depth Profile',
                      figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot a depth profile.
    
    Parameters
    ----------
    depths : np.ndarray
        Depth indices
    intensities : np.ndarray
        Intensity values at each depth
    title : str, default='Depth Profile'
        Plot title
    figsize : tuple of int, default=(10, 6)
        Figure size
        
    Examples
    --------
    >>> from tofsims.roi_analysis import extract_depth_profile
    >>> depths, intensities = extract_depth_profile(data)
    >>> plot_depth_profile(depths, intensities)
    """
    plt.figure(figsize=figsize)
    plt.plot(depths, intensities, linewidth=2, marker='o', markersize=3)
    plt.xlabel('Depth (slice index)')
    plt.ylabel('Intensity')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def create_montage(data: np.ndarray,
                  axis: int = 0,
                  n_rows: Optional[int] = None,
                  n_cols: Optional[int] = None,
                  cmap: str = 'viridis',
                  figsize: Tuple[int, int] = (15, 12)) -> None:
    """
    Create a montage of slices from 3D data.
    
    Parameters
    ----------
    data : np.ndarray
        3D data array
    axis : int, default=0
        Axis along which to slice
    n_rows : int, optional
        Number of rows in montage. If None, auto-calculated.
    n_cols : int, optional
        Number of columns in montage. If None, auto-calculated.
    cmap : str, default='viridis'
        Colormap name
    figsize : tuple of int, default=(15, 12)
        Figure size
        
    Examples
    --------
    >>> create_montage(data, axis=0, n_rows=4, n_cols=5)
    """
    n_slices = data.shape[axis]
    
    # Auto-calculate grid dimensions if not provided
    if n_rows is None and n_cols is None:
        n_cols = int(np.ceil(np.sqrt(n_slices)))
        n_rows = int(np.ceil(n_slices / n_cols))
    elif n_rows is None:
        n_rows = int(np.ceil(n_slices / n_cols))
    elif n_cols is None:
        n_cols = int(np.ceil(n_slices / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    # Limit number of slices to available subplots
    max_slices = min(n_slices, len(axes))
    
    for i in range(max_slices):
        if axis == 0:
            slice_data = data[i, :, :]
        elif axis == 1:
            slice_data = data[:, i, :]
        else:
            slice_data = data[:, :, i]
        
        axes[i].imshow(slice_data, cmap=cmap, aspect='auto')
        axes[i].set_title(f'Slice {i}', fontsize=8)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(max_slices, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()


def save_figure(filename: str, dpi: int = 300, bbox_inches: str = 'tight') -> None:
    """
    Save the current figure to file.
    
    Parameters
    ----------
    filename : str
        Output filename (e.g., 'figure.png', 'plot.pdf')
    dpi : int, default=300
        Resolution in dots per inch
    bbox_inches : str, default='tight'
        Bounding box adjustment
        
    Examples
    --------
    >>> plot_slice(data, 50)
    >>> save_figure('slice_50.png', dpi=300)
    """
    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Figure saved to: {filename}")

# Three-dimensional ToF-SIMS Imaging Toolkit

A clean, well-documented Python research toolkit for analyzing and visualizing three-dimensional Time-of-Flight Secondary Ion Mass Spectrometry (ToF-SIMS) imaging data. This toolkit reconstructs 3D molecular distributions from serial 2D images and provides comprehensive analysis capabilities for tissue and cell data.

## Features

- **Data Loading**: Support for multiple formats including raw binary files, image stacks, and CSV data
- **Preprocessing**: Noise reduction, normalization, background correction, and filtering
- **ROI Analysis**: Region of Interest selection, statistical analysis, and comparison
- **Line Profiles**: Extract and analyze intensity profiles along arbitrary lines in 3D space
- **3D Visualization**: Volume rendering, orthogonal slices, maximum intensity projections, and montages

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/ZhangYuxuanMUC/Three-dimensional-ToF-SIMS-imaging.git
cd Three-dimensional-ToF-SIMS-imaging

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Requirements

- Python >= 3.7
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0
- Pillow >= 8.0.0

## Quick Start

### Basic Workflow

```python
import numpy as np
from tofsims import data_loader, preprocessing, roi_analysis, visualization

# Load data from image stack
data = data_loader.load_image_stack('path/to/images/', pattern='*.tif')

# Preprocess data
data_filtered = preprocessing.gaussian_filter_3d(data, sigma=1.5)
data_normalized = preprocessing.normalize_data(data_filtered, method='minmax')

# Define and analyze ROI
roi = roi_analysis.define_roi(
    z_range=(20, 80),
    y_range=(50, 200),
    x_range=(50, 200),
    name="Sample_Region"
)
stats = roi_analysis.extract_roi_statistics(data_normalized, roi)

# Visualize
visualization.plot_orthogonal_slices(data_normalized, z_idx=50)
visualization.plot_maximum_intensity_projection(data_normalized, axis=0)
```

### Running Examples

The `examples/` directory contains comprehensive demonstrations:

```bash
# Run basic workflow example
python examples/basic_workflow.py

# Run ROI analysis example
python examples/roi_analysis_example.py

# Run visualization example
python examples/visualization_example.py
```

## Module Overview

### `tofsims.data_loader`

Functions for loading ToF-SIMS data:

- `load_raw_data()` - Load binary raw data files
- `load_image_stack()` - Load stacks of 2D images (TIFF, PNG, etc.)
- `load_csv_data()` - Load data from CSV files
- `get_data_info()` - Get summary statistics about loaded data

### `tofsims.preprocessing`

Data preprocessing and filtering:

- `normalize_data()` - Normalize using minmax, z-score, L2, or total ion count
- `gaussian_filter_3d()` - Apply 3D Gaussian smoothing
- `median_filter_3d()` - Apply 3D median filter for outlier removal
- `background_correction()` - Remove background using rolling ball or percentile methods
- `remove_hot_pixels()` - Detect and remove outlier pixels
- `crop_volume()` - Crop 3D volume to specified ranges
- `resample_volume()` - Resample volume to different resolution

### `tofsims.roi_analysis`

Region of Interest analysis:

- `define_roi()` - Create ROI objects with spatial boundaries
- `extract_roi_statistics()` - Calculate mean, median, std, etc. for ROI
- `extract_line_profile()` - Extract intensity along a line in 3D space
- `extract_depth_profile()` - Get intensity vs depth profile
- `compare_rois()` - Compare statistics across multiple ROIs
- `create_roi_mask()` - Generate binary mask for ROI

### `tofsims.visualization`

Visualization functions:

- `plot_slice()` - Display single 2D slice from 3D data
- `plot_orthogonal_slices()` - Show XY, XZ, and YZ planes
- `plot_maximum_intensity_projection()` - Create MIP along any axis
- `plot_volume_rendering()` - 3D scatter plot visualization
- `plot_line_profile()` - Plot intensity along extracted line
- `plot_depth_profile()` - Plot intensity vs depth
- `create_montage()` - Create grid of slices
- `save_figure()` - Save current figure with specified DPI

## Usage Examples

### Loading Different Data Formats

```python
from tofsims.data_loader import load_raw_data, load_image_stack, load_csv_data

# Load raw binary file
data = load_raw_data('data.raw', shape=(100, 256, 256), dtype='float32')

# Load image stack (automatically sorted)
data = load_image_stack('images/', pattern='slice_*.tif')

# Load CSV data
data = load_csv_data('data.csv', shape=(50, 128, 128))
```

### Preprocessing Pipeline

```python
from tofsims import preprocessing

# Apply Gaussian filter for noise reduction
data_smooth = preprocessing.gaussian_filter_3d(data, sigma=(1.0, 1.5, 1.5))

# Remove background
data_corrected = preprocessing.background_correction(data_smooth, method='rolling_ball', radius=50)

# Normalize to [0, 1] range
data_normalized = preprocessing.normalize_data(data_corrected, method='minmax')

# Remove hot pixels
data_clean = preprocessing.remove_hot_pixels(data_normalized, threshold=3.0)
```

### ROI Analysis

```python
from tofsims import roi_analysis

# Define multiple ROIs
roi1 = roi_analysis.define_roi((10, 50), (50, 150), (50, 150), name="Region_A")
roi2 = roi_analysis.define_roi((50, 90), (50, 150), (50, 150), name="Region_B")

# Compare ROIs
comparison = roi_analysis.compare_rois(data, [roi1, roi2])
for name, stats in comparison.items():
    print(f"{name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

# Extract line profile
distances, intensities = roi_analysis.extract_line_profile(
    data, start=(0, 100, 100), end=(99, 200, 200)
)

# Extract depth profile
depths, intensities = roi_analysis.extract_depth_profile(
    data, y_range=(50, 200), x_range=(50, 200), aggregation='mean'
)
```

### Visualization

```python
from tofsims import visualization
import matplotlib.pyplot as plt

# Plot orthogonal slices
visualization.plot_orthogonal_slices(data, z_idx=50, y_idx=128, x_idx=128)
plt.show()

# Create maximum intensity projection
visualization.plot_maximum_intensity_projection(data, axis=0)
plt.show()

# 3D volume rendering
visualization.plot_volume_rendering(data, threshold=0.5, alpha=0.3)
plt.show()

# Create montage of slices
visualization.create_montage(data, axis=0, n_rows=5, n_cols=4)
plt.show()

# Save figure
visualization.save_figure('output_figure.png', dpi=300)
```

## Project Structure

```
Three-dimensional-ToF-SIMS-imaging/
├── tofsims/                      # Main package
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Data loading functions
│   ├── preprocessing.py         # Preprocessing and filtering
│   ├── roi_analysis.py          # ROI and line profile analysis
│   └── visualization.py         # Visualization functions
├── examples/                     # Example scripts
│   ├── basic_workflow.py        # Complete analysis workflow
│   ├── roi_analysis_example.py  # ROI analysis demonstration
│   └── visualization_example.py # Visualization techniques
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation script
├── .gitignore                   # Git ignore patterns
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## Best Practices for Research Code

This toolkit follows best practices for academic research code:

1. **Clear Documentation**: Every function includes detailed docstrings with parameters, returns, and examples
2. **Modular Design**: Separate modules for distinct functionalities
3. **Type Hints**: Function signatures include type hints for clarity
4. **Reproducibility**: Deterministic algorithms and documented random seeds where applicable
5. **No External Services**: Pure Python implementation without web frameworks or GUI dependencies
6. **Citation-Friendly**: Clear module boundaries suitable for methods sections

## Contributing

Contributions are welcome! Please ensure:

- Code follows existing style and documentation patterns
- Functions include comprehensive docstrings
- Examples demonstrate new features
- Changes are suitable for academic research use

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this toolkit in your research, please cite:

```
Zhang, Y. et al. (2026). Three-dimensional ToF-SIMS Imaging Toolkit.
GitHub repository: https://github.com/ZhangYuxuanMUC/Three-dimensional-ToF-SIMS-imaging
```

## Support

For issues, questions, or suggestions:
- Open an issue on [GitHub](https://github.com/ZhangYuxuanMUC/Three-dimensional-ToF-SIMS-imaging/issues)
- Consult the example scripts in the `examples/` directory
- Review the docstrings in each module for detailed API documentation

## Acknowledgments

This toolkit is designed for academic research in mass spectrometry imaging, supporting the analysis of complex biological samples including tissues and cells at the molecular level.

"""
Basic Workflow Example for ToF-SIMS Analysis

This script demonstrates a complete workflow for loading, preprocessing,
analyzing, and visualizing 3D ToF-SIMS data.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import tofsims modules
from tofsims import data_loader, preprocessing, roi_analysis, visualization


def main():
    """
    Demonstrate a basic ToF-SIMS data analysis workflow.
    """
    
    print("=" * 60)
    print("ToF-SIMS 3D Imaging Analysis - Basic Workflow")
    print("=" * 60)
    
    # ========================================================================
    # Step 1: Create synthetic data for demonstration
    # ========================================================================
    print("\n1. Generating synthetic ToF-SIMS data...")
    
    # Create synthetic 3D data (depth, height, width)
    depth, height, width = 100, 256, 256
    
    # Create a gradient pattern with some structure
    z = np.linspace(0, 1, depth)[:, np.newaxis, np.newaxis]
    y = np.linspace(0, 1, height)[np.newaxis, :, np.newaxis]
    x = np.linspace(0, 1, width)[np.newaxis, np.newaxis, :]
    
    # Synthetic data with spatial structure
    data = (np.sin(4 * np.pi * x) * np.cos(4 * np.pi * y) * z + 1) / 2
    
    # Add some noise
    data += np.random.normal(0, 0.05, data.shape)
    
    # Add some "hot spots" to simulate molecular distributions
    data[40:60, 100:150, 100:150] += 0.3
    data[60:80, 50:100, 150:200] += 0.4
    
    # Clip to valid range
    data = np.clip(data, 0, 1)
    
    print(f"   Data shape: {data.shape}")
    print(f"   Data range: [{data.min():.3f}, {data.max():.3f}]")
    
    # Get data info
    info = data_loader.get_data_info(data)
    print(f"   Mean intensity: {info['mean']:.3f}")
    print(f"   Std deviation: {info['std']:.3f}")
    
    # ========================================================================
    # Step 2: Preprocessing
    # ========================================================================
    print("\n2. Preprocessing data...")
    
    # Apply Gaussian filter for noise reduction
    print("   - Applying Gaussian filter...")
    data_filtered = preprocessing.gaussian_filter_3d(data, sigma=(1.0, 1.5, 1.5))
    
    # Normalize data
    print("   - Normalizing data...")
    data_normalized = preprocessing.normalize_data(data_filtered, method='minmax')
    
    # Remove hot pixels
    print("   - Removing hot pixels...")
    data_clean = preprocessing.remove_hot_pixels(data_normalized, threshold=3.0)
    
    print("   Preprocessing complete!")
    
    # ========================================================================
    # Step 3: ROI Analysis
    # ========================================================================
    print("\n3. Performing ROI analysis...")
    
    # Define ROIs for different regions
    roi1 = roi_analysis.define_roi(
        z_range=(40, 60),
        y_range=(100, 150),
        x_range=(100, 150),
        name="Region_A"
    )
    
    roi2 = roi_analysis.define_roi(
        z_range=(60, 80),
        y_range=(50, 100),
        x_range=(150, 200),
        name="Region_B"
    )
    
    # Calculate statistics for each ROI
    stats1 = roi_analysis.extract_roi_statistics(data_clean, roi1)
    stats2 = roi_analysis.extract_roi_statistics(data_clean, roi2)
    
    print(f"\n   {roi1.name} statistics:")
    print(f"     Mean: {stats1['mean']:.4f}")
    print(f"     Median: {stats1['median']:.4f}")
    print(f"     Std: {stats1['std']:.4f}")
    
    print(f"\n   {roi2.name} statistics:")
    print(f"     Mean: {stats2['mean']:.4f}")
    print(f"     Median: {stats2['median']:.4f}")
    print(f"     Std: {stats2['std']:.4f}")
    
    # Compare ROIs
    comparison = roi_analysis.compare_rois(data_clean, [roi1, roi2])
    print(f"\n   Intensity ratio ({roi1.name}/{roi2.name}): "
          f"{comparison[roi1.name]['mean'] / comparison[roi2.name]['mean']:.2f}")
    
    # Extract depth profile
    print("\n   Extracting depth profile...")
    depths, intensities = roi_analysis.extract_depth_profile(
        data_clean, 
        y_range=(50, 200),
        x_range=(50, 200),
        aggregation='mean'
    )
    
    # ========================================================================
    # Step 4: Visualization
    # ========================================================================
    print("\n4. Creating visualizations...")
    
    # Plot orthogonal slices
    print("   - Plotting orthogonal slices...")
    visualization.plot_orthogonal_slices(
        data_clean,
        z_idx=50,
        y_idx=128,
        x_idx=128
    )
    plt.savefig('output_orthogonal_slices.png', dpi=150, bbox_inches='tight')
    print("     Saved: output_orthogonal_slices.png")
    
    # Plot maximum intensity projection
    print("   - Creating maximum intensity projection...")
    visualization.plot_maximum_intensity_projection(
        data_clean,
        axis=0,
        title='MIP - Z axis projection'
    )
    plt.savefig('output_mip.png', dpi=150, bbox_inches='tight')
    print("     Saved: output_mip.png")
    
    # Plot depth profile
    print("   - Plotting depth profile...")
    visualization.plot_depth_profile(
        depths,
        intensities,
        title='Average Intensity vs Depth'
    )
    plt.savefig('output_depth_profile.png', dpi=150, bbox_inches='tight')
    print("     Saved: output_depth_profile.png")
    
    # Create a montage
    print("   - Creating slice montage...")
    visualization.create_montage(
        data_clean,
        axis=0,
        n_rows=5,
        n_cols=5,
        cmap='viridis'
    )
    plt.savefig('output_montage.png', dpi=150, bbox_inches='tight')
    print("     Saved: output_montage.png")
    
    # 3D volume rendering
    print("   - Creating 3D volume rendering...")
    visualization.plot_volume_rendering(
        data_clean,
        threshold=0.6,
        alpha=0.3
    )
    plt.savefig('output_3d_volume.png', dpi=150, bbox_inches='tight')
    print("     Saved: output_3d_volume.png")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - output_orthogonal_slices.png")
    print("  - output_mip.png")
    print("  - output_depth_profile.png")
    print("  - output_montage.png")
    print("  - output_3d_volume.png")
    
    # Note: In a real scenario, you would replace the synthetic data generation
    # with actual data loading:
    # data = data_loader.load_raw_data('path/to/data.raw', shape=(100, 256, 256))
    # or
    # data = data_loader.load_image_stack('path/to/images/', pattern='*.tif')


if __name__ == '__main__':
    main()

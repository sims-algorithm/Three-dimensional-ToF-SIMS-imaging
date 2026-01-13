"""
Visualization Example for ToF-SIMS Data

This script demonstrates various visualization techniques for 
3D ToF-SIMS imaging data including slices, projections, and volume rendering.
"""

import numpy as np
import matplotlib.pyplot as plt

from tofsims import data_loader, preprocessing, visualization


def main():
    """
    Demonstrate various visualization techniques for ToF-SIMS data.
    """
    
    print("=" * 60)
    print("ToF-SIMS 3D Visualization Examples")
    print("=" * 60)
    
    # ========================================================================
    # Create synthetic data with interesting features
    # ========================================================================
    print("\n1. Creating synthetic ToF-SIMS data with features...")
    
    depth, height, width = 100, 256, 256
    
    # Create base data
    data = np.zeros((depth, height, width))
    
    # Add spherical structures at different depths
    print("   - Adding spherical features...")
    for z0, y0, x0, radius, intensity in [
        (25, 128, 128, 30, 0.8),
        (50, 80, 180, 25, 0.6),
        (50, 180, 80, 25, 0.7),
        (75, 128, 128, 20, 0.9),
    ]:
        z, y, x = np.ogrid[:depth, :height, :width]
        distance = np.sqrt((z - z0)**2 + (y - y0)**2 + (x - x0)**2)
        data += intensity * np.exp(-(distance / radius)**2)
    
    # Add some background with spatial variation
    print("   - Adding background pattern...")
    y = np.linspace(0, 2*np.pi, height)[np.newaxis, :, np.newaxis]
    x = np.linspace(0, 2*np.pi, width)[np.newaxis, np.newaxis, :]
    background = 0.2 * (1 + 0.3 * np.sin(2*x) * np.cos(2*y))
    data += background
    
    # Add noise
    data += np.random.normal(0, 0.02, data.shape)
    data = np.clip(data, 0, 1)
    
    # Preprocess
    print("   - Preprocessing...")
    data = preprocessing.gaussian_filter_3d(data, sigma=(1.0, 1.5, 1.5))
    data = preprocessing.normalize_data(data, method='minmax')
    
    print(f"   Data shape: {data.shape}")
    print(f"   Data range: [{data.min():.3f}, {data.max():.3f}]")
    
    # ========================================================================
    # Single slice visualization
    # ========================================================================
    print("\n2. Creating single slice visualizations...")
    
    # XY slice at different depths
    print("   - Plotting XY slices at different depths...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, z_idx in enumerate([25, 50, 75]):
        im = axes[i].imshow(data[z_idx, :, :], cmap='viridis', origin='lower')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].set_title(f'XY slice at Z={z_idx}')
        plt.colorbar(im, ax=axes[i], label='Intensity')
    
    plt.tight_layout()
    plt.savefig('viz_xy_slices.png', dpi=150, bbox_inches='tight')
    print("     Saved: viz_xy_slices.png")
    plt.close()
    
    # ========================================================================
    # Orthogonal slices
    # ========================================================================
    print("\n3. Creating orthogonal slice views...")
    
    visualization.plot_orthogonal_slices(
        data,
        z_idx=50,
        y_idx=128,
        x_idx=128,
        cmap='hot'
    )
    plt.savefig('viz_orthogonal_slices.png', dpi=150, bbox_inches='tight')
    print("     Saved: viz_orthogonal_slices.png")
    plt.close()
    
    # ========================================================================
    # Maximum Intensity Projections
    # ========================================================================
    print("\n4. Creating Maximum Intensity Projections...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # MIP along Z axis
    plt.subplot(1, 3, 1)
    visualization.plot_maximum_intensity_projection(
        data,
        axis=0,
        cmap='viridis',
        title='MIP along Z axis'
    )
    plt.gcf().set_size_inches(5, 4)
    
    # MIP along Y axis
    plt.subplot(1, 3, 2)
    visualization.plot_maximum_intensity_projection(
        data,
        axis=1,
        cmap='viridis',
        title='MIP along Y axis'
    )
    plt.gcf().set_size_inches(5, 4)
    
    # MIP along X axis
    plt.subplot(1, 3, 3)
    visualization.plot_maximum_intensity_projection(
        data,
        axis=2,
        cmap='viridis',
        title='MIP along X axis'
    )
    
    plt.gcf().set_size_inches(15, 4)
    plt.tight_layout()
    plt.savefig('viz_mip_all_axes.png', dpi=150, bbox_inches='tight')
    print("     Saved: viz_mip_all_axes.png")
    plt.close()
    
    # ========================================================================
    # Slice montage
    # ========================================================================
    print("\n5. Creating slice montage...")
    
    # Show every 5th slice
    subset = data[::5, :, :]
    visualization.create_montage(
        subset,
        axis=0,
        n_rows=4,
        n_cols=5,
        cmap='plasma'
    )
    plt.savefig('viz_montage.png', dpi=150, bbox_inches='tight')
    print("     Saved: viz_montage.png")
    plt.close()
    
    # ========================================================================
    # 3D Volume Rendering
    # ========================================================================
    print("\n6. Creating 3D volume renderings...")
    
    # High threshold - show only bright features
    print("   - High threshold rendering...")
    visualization.plot_volume_rendering(
        data,
        threshold=0.5,
        alpha=0.4,
        cmap='hot'
    )
    plt.savefig('viz_3d_high_threshold.png', dpi=150, bbox_inches='tight')
    print("     Saved: viz_3d_high_threshold.png")
    plt.close()
    
    # Low threshold - show more structure
    print("   - Low threshold rendering...")
    visualization.plot_volume_rendering(
        data,
        threshold=0.3,
        alpha=0.2,
        cmap='viridis'
    )
    plt.savefig('viz_3d_low_threshold.png', dpi=150, bbox_inches='tight')
    print("     Saved: viz_3d_low_threshold.png")
    plt.close()
    
    # ========================================================================
    # Depth profile visualization
    # ========================================================================
    print("\n7. Creating depth profile...")
    
    from tofsims.roi_analysis import extract_depth_profile
    
    # Full volume depth profile
    depths, intensities = extract_depth_profile(data, aggregation='mean')
    visualization.plot_depth_profile(
        depths,
        intensities,
        title='Mean Intensity vs Depth (Full Volume)'
    )
    plt.savefig('viz_depth_profile_full.png', dpi=150, bbox_inches='tight')
    print("     Saved: viz_depth_profile_full.png")
    plt.close()
    
    # Regional depth profiles
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    regions = [
        ((50, 150), (50, 150), 'Center'),
        ((0, 100), (0, 100), 'Top-Left'),
        ((0, 100), (156, 256), 'Top-Right'),
        ((156, 256), (156, 256), 'Bottom-Right'),
    ]
    
    for i, (y_range, x_range, name) in enumerate(regions):
        ax = axes[i // 2, i % 2]
        depths, intensities = extract_depth_profile(
            data, 
            y_range=y_range,
            x_range=x_range,
            aggregation='mean'
        )
        ax.plot(depths, intensities, linewidth=2)
        ax.set_xlabel('Depth (slice index)')
        ax.set_ylabel('Mean Intensity')
        ax.set_title(f'Depth Profile - {name}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('viz_depth_profiles_regional.png', dpi=150, bbox_inches='tight')
    print("     Saved: viz_depth_profiles_regional.png")
    plt.close()
    
    # ========================================================================
    # Custom colormap comparison
    # ========================================================================
    print("\n8. Comparing different colormaps...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colormaps = ['viridis', 'plasma', 'hot', 'coolwarm', 'gray', 'jet']
    
    slice_data = data[50, :, :]
    
    for i, cmap in enumerate(colormaps):
        ax = axes[i // 3, i % 3]
        im = ax.imshow(slice_data, cmap=cmap, origin='lower')
        ax.set_title(f'Colormap: {cmap}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('viz_colormap_comparison.png', dpi=150, bbox_inches='tight')
    print("     Saved: viz_colormap_comparison.png")
    plt.close()
    
    print("\n" + "=" * 60)
    print("Visualization examples complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - viz_xy_slices.png")
    print("  - viz_orthogonal_slices.png")
    print("  - viz_mip_all_axes.png")
    print("  - viz_montage.png")
    print("  - viz_3d_high_threshold.png")
    print("  - viz_3d_low_threshold.png")
    print("  - viz_depth_profile_full.png")
    print("  - viz_depth_profiles_regional.png")
    print("  - viz_colormap_comparison.png")
    
    print("\nNote: To work with real data, use:")
    print("  data = data_loader.load_raw_data('file.raw', shape=(100, 256, 256))")
    print("  or")
    print("  data = data_loader.load_image_stack('directory/', pattern='*.tif')")


if __name__ == '__main__':
    main()

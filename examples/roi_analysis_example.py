"""
ROI Analysis Example for ToF-SIMS Data

This script demonstrates Region of Interest (ROI) analysis and 
line profile extraction for ToF-SIMS imaging data.
"""

import numpy as np
import matplotlib.pyplot as plt

from tofsims import data_loader, preprocessing, roi_analysis, visualization


def main():
    """
    Demonstrate ROI and line profile analysis on ToF-SIMS data.
    """
    
    print("=" * 60)
    print("ToF-SIMS ROI and Line Profile Analysis")
    print("=" * 60)
    
    # ========================================================================
    # Generate synthetic data with distinct regions
    # ========================================================================
    print("\n1. Creating synthetic sample with multiple regions...")
    
    depth, height, width = 80, 200, 200
    data = np.zeros((depth, height, width))
    
    # Background
    data += 0.2
    
    # Region 1: High intensity region (e.g., cell membrane)
    data[20:40, 60:140, 60:140] = 0.7
    
    # Region 2: Medium intensity region (e.g., cytoplasm)
    data[40:60, 80:120, 80:120] = 0.5
    
    # Region 3: Low intensity region (e.g., nucleus)
    data[50:70, 90:110, 90:110] = 0.3
    
    # Add some depth-dependent variation
    z_gradient = np.linspace(0.9, 1.1, depth)[:, np.newaxis, np.newaxis]
    data = data * z_gradient
    
    # Add noise
    data += np.random.normal(0, 0.03, data.shape)
    data = np.clip(data, 0, 1)
    
    # Preprocess
    data = preprocessing.gaussian_filter_3d(data, sigma=(0.5, 1.0, 1.0))
    
    print(f"   Data shape: {data.shape}")
    
    # ========================================================================
    # Define and analyze multiple ROIs
    # ========================================================================
    print("\n2. Defining and analyzing ROIs...")
    
    # Define ROIs for different biological structures
    roi_membrane = roi_analysis.define_roi(
        z_range=(20, 40),
        y_range=(60, 140),
        x_range=(60, 140),
        name="Membrane"
    )
    
    roi_cytoplasm = roi_analysis.define_roi(
        z_range=(40, 60),
        y_range=(80, 120),
        x_range=(80, 120),
        name="Cytoplasm"
    )
    
    roi_nucleus = roi_analysis.define_roi(
        z_range=(50, 70),
        y_range=(90, 110),
        x_range=(90, 110),
        name="Nucleus"
    )
    
    rois = [roi_membrane, roi_cytoplasm, roi_nucleus]
    
    # Calculate and compare statistics
    comparison = roi_analysis.compare_rois(data, rois)
    
    print("\n   ROI Statistics:")
    print("   " + "-" * 55)
    print(f"   {'Region':<15} {'Mean':<10} {'Median':<10} {'Std':<10}")
    print("   " + "-" * 55)
    
    for name, stats in comparison.items():
        print(f"   {name:<15} {stats['mean']:<10.4f} "
              f"{stats['median']:<10.4f} {stats['std']:<10.4f}")
    
    # ========================================================================
    # Extract and analyze line profiles
    # ========================================================================
    print("\n3. Extracting line profiles...")
    
    # Horizontal line profile through center
    print("   - Extracting horizontal line profile...")
    distances_h, intensities_h = roi_analysis.extract_line_profile(
        data,
        start=(30, 100, 20),
        end=(30, 100, 180)
    )
    
    # Vertical line profile through center
    print("   - Extracting vertical line profile...")
    distances_v, intensities_v = roi_analysis.extract_line_profile(
        data,
        start=(30, 20, 100),
        end=(30, 180, 100)
    )
    
    # Depth line profile
    print("   - Extracting depth line profile...")
    distances_d, intensities_d = roi_analysis.extract_line_profile(
        data,
        start=(10, 100, 100),
        end=(70, 100, 100)
    )
    
    # ========================================================================
    # Extract depth profiles for each ROI
    # ========================================================================
    print("\n4. Extracting depth profiles for ROIs...")
    
    depth_profiles = {}
    for roi in rois:
        roi_data = roi.extract(data)
        depths = np.arange(roi_data.shape[0])
        intensities = np.mean(roi_data, axis=(1, 2))
        depth_profiles[roi.name] = (depths + roi.z_range[0], intensities)
    
    # ========================================================================
    # Visualization
    # ========================================================================
    print("\n5. Creating visualizations...")
    
    # Plot slice with ROI overlays
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    slice_idx = 30
    im = ax.imshow(data[slice_idx, :, :], cmap='viridis', origin='lower')
    
    # Draw ROI boundaries
    for roi in rois:
        if roi.z_range[0] <= slice_idx < roi.z_range[1]:
            y_min, y_max = roi.y_range
            x_min, x_max = roi.x_range
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_min + 5, y_min + 5, roi.name, 
                   color='red', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Slice at Z={slice_idx} with ROI boundaries')
    plt.colorbar(im, ax=ax, label='Intensity')
    plt.tight_layout()
    plt.savefig('roi_slice_with_boundaries.png', dpi=150, bbox_inches='tight')
    print("     Saved: roi_slice_with_boundaries.png")
    
    # Plot line profiles
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(distances_h, intensities_h, linewidth=2)
    axes[0].set_xlabel('Distance (pixels)')
    axes[0].set_ylabel('Intensity')
    axes[0].set_title('Horizontal Line Profile')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(distances_v, intensities_v, linewidth=2)
    axes[1].set_xlabel('Distance (pixels)')
    axes[1].set_ylabel('Intensity')
    axes[1].set_title('Vertical Line Profile')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(distances_d, intensities_d, linewidth=2)
    axes[2].set_xlabel('Distance (pixels)')
    axes[2].set_ylabel('Intensity')
    axes[2].set_title('Depth Line Profile')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roi_line_profiles.png', dpi=150, bbox_inches='tight')
    print("     Saved: roi_line_profiles.png")
    
    # Plot depth profiles for each ROI
    plt.figure(figsize=(10, 6))
    for roi_name, (depths, intensities) in depth_profiles.items():
        plt.plot(depths, intensities, marker='o', markersize=4, 
                linewidth=2, label=roi_name)
    
    plt.xlabel('Depth (slice index)')
    plt.ylabel('Mean Intensity')
    plt.title('Depth Profiles by ROI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roi_depth_profiles.png', dpi=150, bbox_inches='tight')
    print("     Saved: roi_depth_profiles.png")
    
    # Create ROI masks visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, roi in enumerate(rois):
        mask = roi_analysis.create_roi_mask(data.shape, roi)
        mip = np.max(mask, axis=0)
        axes[i].imshow(mip, cmap='Reds', origin='lower')
        axes[i].set_title(f'{roi.name} Mask')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('roi_masks.png', dpi=150, bbox_inches='tight')
    print("     Saved: roi_masks.png")
    
    print("\n" + "=" * 60)
    print("ROI Analysis complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - roi_slice_with_boundaries.png")
    print("  - roi_line_profiles.png")
    print("  - roi_depth_profiles.png")
    print("  - roi_masks.png")


if __name__ == '__main__':
    main()

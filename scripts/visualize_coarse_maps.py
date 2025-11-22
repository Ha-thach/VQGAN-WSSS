import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def visualize_image_with_coarse_map(img_path, coarse_map_path, indices_path, save_path, cmap='tab20'):
    """
    Visualize original image alongside its coarse segmentation map

    Args:
        img_path: Path to original image
        coarse_map_path: Path to coarse map PNG (optional)
        indices_path: Path to indices .npy file
        save_path: Path to save combined visualization
        cmap: Colormap for indices
    """
    # Load original image
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)

    # Load indices
    indices = np.load(indices_path)

    # Infer spatial dimensions
    h = w = int(np.sqrt(len(indices)))
    indices_2d = indices.reshape(h, w)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Plot coarse map
    im = axes[1].imshow(indices_2d, cmap=cmap, interpolation='nearest')
    axes[1].set_title('Coarse Segmentation Map', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Code Index')

    # Plot overlay (semi-transparent coarse map on original image)
    axes[2].imshow(img_np)
    im2 = axes[2].imshow(indices_2d, cmap=cmap, alpha=0.5, interpolation='nearest')
    axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Add filename as suptitle
    fig.suptitle(f'{os.path.basename(img_path)}', fontsize=16, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization: {save_path}")


def visualize_batch(img_dir, coarse_dir, indices_dir, output_dir, n_samples=10, cmap='tab20'):
    """
    Visualize multiple images with their coarse maps

    Args:
        img_dir: Directory containing original images
        coarse_dir: Directory containing coarse map PNGs
        indices_dir: Directory containing indices .npy files
        output_dir: Directory to save visualizations
        n_samples: Number of samples to visualize
        cmap: Colormap
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all indices files
    indices_files = sorted(Path(indices_dir).glob('*.npy'))[:n_samples]

    for i, indices_path in enumerate(indices_files):
        # Get corresponding image path
        stem = indices_path.stem.replace('_indices', '')

        # Find original image in img_dir
        img_candidates = list(Path(img_dir).glob(f'{stem}*'))
        if not img_candidates:
            # Try without suffix
            base_stem = stem.split('_coarse')[0]
            img_candidates = list(Path(img_dir).glob(f'*{base_stem}*.png'))

        if not img_candidates:
            print(f"Warning: Could not find image for {stem}")
            continue

        img_path = img_candidates[0]
        coarse_map_path = Path(coarse_dir) / f'{stem}_coarse.png'
        save_path = os.path.join(output_dir, f'{stem}_combined.png')

        visualize_image_with_coarse_map(
            str(img_path),
            str(coarse_map_path) if coarse_map_path.exists() else None,
            str(indices_path),
            save_path,
            cmap=cmap
        )

        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{len(indices_files)} images")

    print(f"\n{'='*50}")
    print(f"Visualization complete!")
    print(f"Saved {len(indices_files)} combined images to: {output_dir}")
    print(f"{'='*50}")


def create_grid_visualization(img_dir, indices_dir, output_path, n_samples=9, cmap='tab20'):
    """
    Create a grid visualization showing multiple images and their coarse maps

    Args:
        img_dir: Directory containing original images
        indices_dir: Directory containing indices .npy files
        output_path: Path to save grid visualization
        n_samples: Number of samples (must be perfect square, e.g., 4, 9, 16)
        cmap: Colormap
    """
    indices_files = sorted(Path(indices_dir).glob('*.npy'))[:n_samples]

    grid_size = int(np.sqrt(n_samples))
    fig, axes = plt.subplots(grid_size, grid_size * 2, figsize=(grid_size * 4, grid_size * 2))

    for i, indices_path in enumerate(indices_files):
        row = i // grid_size
        col = (i % grid_size) * 2

        # Get corresponding image
        stem = indices_path.stem.replace('_indices', '')
        img_candidates = list(Path(img_dir).glob(f'*{stem}*.png'))

        if not img_candidates:
            continue

        img_path = img_candidates[0]

        # Load image and indices
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        indices = np.load(indices_path)
        h = w = int(np.sqrt(len(indices)))
        indices_2d = indices.reshape(h, w)

        # Plot original image
        axes[row, col].imshow(img_np)
        axes[row, col].set_title(f'{os.path.basename(img_path)[:30]}...', fontsize=8)
        axes[row, col].axis('off')

        # Plot coarse map
        axes[row, col + 1].imshow(indices_2d, cmap=cmap, interpolation='nearest')
        axes[row, col + 1].set_title('Coarse Map', fontsize=8)
        axes[row, col + 1].axis('off')

    plt.suptitle('Original Images vs Coarse Segmentation Maps', fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved grid visualization: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize coarse maps with original images")
    parser.add_argument("--img_dir", type=str, default="data/sub_BCSS_WSSS/training",
                        help="Directory containing original images")
    parser.add_argument("--analysis_dir", type=str, default="codebook_analysis_bcss_gumbel8",
                        help="Directory containing analysis results")
    parser.add_argument("--output_dir", type=str, default="codebook_analysis_bcss_gumbel8/visualizations",
                        help="Output directory for combined visualizations")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of samples to visualize")
    parser.add_argument("--cmap", type=str, default="tab20",
                        help="Colormap for coarse maps")
    parser.add_argument("--grid", action="store_true",
                        help="Create grid visualization")
    parser.add_argument("--grid_samples", type=int, default=9,
                        help="Number of samples for grid (must be perfect square)")

    args = parser.parse_args()

    coarse_dir = os.path.join(args.analysis_dir, "coarse_maps")
    indices_dir = os.path.join(args.analysis_dir, "index_npy")

    # Check directories exist
    if not os.path.exists(args.img_dir):
        print(f"Error: Image directory not found: {args.img_dir}")
        sys.exit(1)
    if not os.path.exists(indices_dir):
        print(f"Error: Indices directory not found: {indices_dir}")
        sys.exit(1)

    print(f"Image directory: {args.img_dir}")
    print(f"Coarse maps directory: {coarse_dir}")
    print(f"Indices directory: {indices_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Create individual visualizations
    visualize_batch(
        args.img_dir,
        coarse_dir,
        indices_dir,
        args.output_dir,
        n_samples=args.n_samples,
        cmap=args.cmap
    )

    # Create grid visualization if requested
    if args.grid:
        grid_path = os.path.join(args.analysis_dir, "coarse_maps_grid.png")
        create_grid_visualization(
            args.img_dir,
            indices_dir,
            grid_path,
            n_samples=args.grid_samples,
            cmap=args.cmap
        )

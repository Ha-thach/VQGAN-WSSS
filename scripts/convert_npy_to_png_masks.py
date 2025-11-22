"""
Convert .npy pseudo-labels to .png masks (P mode)
For use with train_segmentation_model.py
"""

import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse


def convert_npy_to_png_mask(npy_path, output_path):
    """
    Convert .npy segmentation to .png P-mode mask

    Args:
        npy_path: Path to .npy file with class indices (H, W)
        output_path: Path to save .png mask
    """
    # Load .npy (class indices: 0-4)
    mask = np.load(npy_path)

    # Ensure uint8
    mask = mask.astype(np.uint8)

    # Save as P mode (palette) PNG
    mask_img = Image.fromarray(mask, mode='P')

    # Set palette (optional, for visualization)
    palette = [
        255, 0, 0,    # 0: TUM - Red
        0, 255, 0,    # 1: STR - Green
        0, 0, 255,    # 2: LYM - Blue
        255, 255, 0,  # 3: NEC - Yellow
        0, 0, 0       # 4: Background - Black
    ]
    # Pad palette to 256 colors
    palette += [0] * (256 * 3 - len(palette))
    mask_img.putpalette(palette)

    mask_img.save(output_path)


def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .npy files
    npy_files = sorted(list(input_dir.glob('*_seg.npy')))

    if len(npy_files) == 0:
        print(f"No *_seg.npy files found in {input_dir}")
        return

    print(f"Found {len(npy_files)} .npy files")
    print(f"Converting to .png masks in {output_dir}")

    for npy_path in tqdm(npy_files, desc='Converting'):
        # Extract base name (remove _seg.npy suffix)
        base_name = npy_path.stem.replace('_seg', '')

        # Output path
        output_path = output_dir / f'{base_name}.png'

        # Convert
        convert_npy_to_png_mask(npy_path, output_path)

    print(f"\n✓ Converted {len(npy_files)} masks to {output_dir}")
    print(f"\nYou can now use these masks for training:")
    print(f"python token_classification/train_segmentation_model.py \\")
    print(f"  --train-mask-dir {output_dir} \\")
    print(f"  ...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert .npy pseudo-labels to .png masks'
    )

    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory with *_seg.npy files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for .png masks')

    args = parser.parse_args()
    main(args)

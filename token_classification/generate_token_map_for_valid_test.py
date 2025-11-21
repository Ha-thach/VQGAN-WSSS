"""
Generate Token Maps for Validation/Test Data
- Reads images and corresponding masks
- Extracts class labels from masks
- Saves token maps with labels in filename: image_name[0101].npy
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to import taming
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
import albumentations as A
from tqdm import tqdm

from taming.models.vqgan import GumbelVQ
from omegaconf import OmegaConf


# Class definitions
CLASSES = ["TUM", "STR", "LYM", "NEC"]

MASK_CLASSES = {
    0: "BACKGROUND",
    1: "TUM",
    2: "STR",
    3: "LYM",
    4: "NEC"
}


def get_class_labels_from_mask(mask):
    """
    Extract class labels from segmentation mask (P mode palette indices)

    Args:
        mask: numpy array [H, W] with palette indices
              0 = TUM, 1 = STR, 2 = LYM, 3 = NEC, 4 = Background

    Returns:
        cls_label: numpy array [4] with binary values [TUM, STR, LYM, NEC]
        cls_names: list of active class names
    """
    unique_classes = np.unique(mask)

    # Remove background (index 4)
    unique_classes = unique_classes[unique_classes < 4]

    cls_label = np.zeros(4, dtype=np.int32)
    for class_id in unique_classes:
        if 0 <= class_id <= 3:
            cls_label[class_id] = 1  # Direct mapping: 0->TUM, 1->STR, 2->LYM, 3->NEC

    cls_names = [CLASSES[i] for i, v in enumerate(cls_label) if v == 1]

    return cls_label, cls_names


def format_label_string(cls_label):
    """
    Convert class label array to filename format [0101]

    Args:
        cls_label: numpy array [4] with binary values

    Returns:
        str: formatted label like "[0101]"
    """
    label_str = ''.join([str(int(x)) for x in cls_label])
    return f"[{label_str}]"


def extract_indices(info):
    """Extract token indices from quantizer output"""
    if isinstance(info, (tuple, list)):
        for x in info[::-1]:
            if torch.is_tensor(x) and x.dtype == torch.long:
                return x
        for x in info:
            got = extract_indices(x)
            if got is not None:
                return got
    elif isinstance(info, dict):
        if 'min_encoding_indices' in info:
            return info['min_encoding_indices']
        if 'indices' in info:
            return info['indices']
    return None


@torch.no_grad()
def load_vqgan(config_path, ckpt_path, device='cpu'):
    """Load pretrained VQ-VAE model"""
    config = OmegaConf.load(config_path)
    model = GumbelVQ(**config.model.params)

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded - missing: {len(missing)}, unexpected: {len(unexpected)}")

    model = model.to(device).eval()

    n_embed = config.model.params.n_embed
    embed_dim = config.model.params.embed_dim

    return model, n_embed, embed_dim


def filter_images_with_masks(img_dir, mask_dir):
    """
    Filter images that have corresponding mask files

    Args:
        img_dir: Path to image directory
        mask_dir: Path to mask directory

    Returns:
        list of (img_path, mask_path) tuples
    """
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)

    # Get all images
    img_paths = sorted(list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')))

    # Match with masks
    paired_paths = []
    missing_masks = []

    for img_path in img_paths:
        img_stem = img_path.stem

        # Try to find corresponding mask
        mask_path = mask_dir / f"{img_stem}.png"

        if not mask_path.exists():
            # Try other extensions
            possible_masks = list(mask_dir.glob(f"{img_stem}.*"))
            if possible_masks:
                mask_path = possible_masks[0]
            else:
                missing_masks.append(img_stem)
                continue

        paired_paths.append((img_path, mask_path))

    if missing_masks:
        print(f"\nWarning: {len(missing_masks)} images without masks (will be skipped)")
        if len(missing_masks) <= 10:
            for name in missing_masks:
                print(f"  - {name}")
        else:
            for name in missing_masks[:10]:
                print(f"  - {name}")
            print(f"  ... and {len(missing_masks) - 10} more")

    return paired_paths


@torch.no_grad()
def generate_token_maps_with_labels(args):
    """
    Generate token maps with labels from masks for validation/test data
    """

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"\n{'='*60}")
    print("Loading VQ-VAE model...")
    print(f"{'='*60}")
    model, n_embed, embed_dim = load_vqgan(args.config, args.checkpoint, device)
    print(f"Codebook: {n_embed} tokens, {embed_dim} dims")

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Image transform
    transform = A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    # Filter images with masks
    print(f"\n{'='*60}")
    print("Filtering images with masks...")
    print(f"{'='*60}")
    print(f"Image dir: {args.img_dir}")
    print(f"Mask dir: {args.mask_dir}")

    paired_paths = filter_images_with_masks(args.img_dir, args.mask_dir)

    if not paired_paths:
        print(f"\nError: No valid image-mask pairs found!")
        return

    print(f"\nFound {len(paired_paths)} valid image-mask pairs")

    # Limit if specified
    if args.max_images and args.max_images > 0:
        paired_paths = paired_paths[:args.max_images]
        print(f"Processing first {len(paired_paths)} images")

    # Statistics
    token_histogram = np.zeros(n_embed, dtype=np.int64)
    class_distribution = np.zeros(4, dtype=np.int64)
    processed = 0
    skipped = 0

    # Process images
    print(f"\n{'='*60}")
    print("Processing images...")
    print(f"{'='*60}")

    for img_path, mask_path in tqdm(paired_paths, desc='Encoding'):
        try:
            # Load and transform image
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img, dtype=np.uint8)
            transformed = transform(image=img_np)['image']

            # To tensor
            img_tensor = torch.from_numpy(transformed).permute(2, 0, 1).unsqueeze(0).float()
            x = img_tensor.to(device)

            # Encode to tokens
            quant, _, info = model.encode(x)
            idx = extract_indices(info)

            if idx is None:
                print(f"\nWarning: No indices for {img_path.name}")
                skipped += 1
                continue

            # Convert to numpy
            idx_cpu = idx.cpu().numpy().reshape(-1)

            # Update histogram
            hist = np.bincount(idx_cpu, minlength=n_embed)
            token_histogram += hist

            # Reshape to spatial
            h, w = quant.shape[2], quant.shape[3]
            token_map = idx_cpu.reshape(h, w)

            # Load mask and extract label (keep in P mode to get palette indices)
            mask = Image.open(mask_path)  # Don't convert to 'L' - keep P mode!
            mask_np = np.array(mask, dtype=np.int64)

            cls_label, cls_names = get_class_labels_from_mask(mask_np)
            label_str = format_label_string(cls_label)

            # Update class distribution
            class_distribution += cls_label

            # Create filename with label: image_name[0101].npy
            img_name = img_path.stem
            save_name = f"{img_name}{label_str}.npy"

            # Save token map
            npy_path = out_dir / save_name
            np.save(npy_path, token_map)

            processed += 1

        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            skipped += 1
            continue

    # Save histogram if requested
    if args.save_histogram:
        np.save(out_dir / 'token_histogram.npy', token_histogram)
        print(f"\nSaved token histogram")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {processed}/{len(paired_paths)} images")
    print(f"Skipped: {skipped}")
    print(f"Tokens used: {np.count_nonzero(token_histogram)}/{n_embed}")
    print(f"Most frequent token: {token_histogram.argmax()} ({token_histogram.max()} times)")

    print(f"\nClass distribution:")
    for cls_name, count in zip(CLASSES, class_distribution):
        print(f"  {cls_name}: {count} samples")

    print(f"\nOutput directory: {out_dir}")
    print(f"Token files saved with format: image_name[0101].npy")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate token maps with labels for validation/test data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate validation token maps with labels
  python token_classification/generate_token_map_for_valid_test.py \\
      --config logs/vqgan_gumbel_f8/configs/model.yaml \\
      --checkpoint logs/vqgan_gumbel_f8/checkpoints/last.ckpt \\
      --img-dir data/sub_BCSS_WSSS/validation \\
      --mask-dir data/sub_BCSS_WSSS/validation/mask \\
      --output-dir token_maps/val

  # Generate test token maps with labels
  python token_classification/generate_token_map_for_valid_test.py \\
      --config logs/vqgan_gumbel_f8/configs/model.yaml \\
      --checkpoint logs/vqgan_gumbel_f8/checkpoints/last.ckpt \\
      --img-dir data/sub_BCSS_WSSS/test \\
      --mask-dir data/sub_BCSS_WSSS/test/mask \\
      --output-dir token_maps/test
        """
    )

    parser.add_argument('--config', type=str, required=True,
                        help='Path to model config YAML')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--img-dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--mask-dir', type=str, required=True,
                        help='Directory containing masks (for label extraction)')
    parser.add_argument('--output-dir', type=str, default='token_maps',
                        help='Output directory')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Max images to process (None = all)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU')
    parser.add_argument('--save-histogram', action='store_true',
                        help='Save token histogram')

    args = parser.parse_args()
    generate_token_maps_with_labels(args)

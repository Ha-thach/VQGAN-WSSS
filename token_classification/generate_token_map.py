"""
Generate Token Maps from Images
Uses pretrained Gumbel-8 VQ-VAE model to encode images into token indices
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


@torch.no_grad()
def extract_codebook(model):
    """Extract codebook embeddings"""
    if hasattr(model.quantize, 'embed'):
        return model.quantize.embed.weight.detach().cpu().numpy()
    elif hasattr(model.quantize, 'embedding'):
        return model.quantize.embedding.weight.detach().cpu().numpy()
    else:
        raise AttributeError("Quantizer has no 'embed' or 'embedding'")


@torch.no_grad()
def generate_token_maps(args):
    """Generate token maps from images"""

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

    # Save codebook
    codebook = extract_codebook(model)
    #np.save(out_dir /'codebook_prototypes.npy', codebook)
    #print(f"Codebook saved: {codebook.shape}")

    # Image transform
    transform = A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    # Get image paths
    img_dir = Path(args.img_dir)
    img_paths = sorted(list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')))

    if not img_paths:
        print(f"Error: No images found in {img_dir}")
        return

    print(f"\nFound {len(img_paths)} images")

    # Limit if specified
    if args.max_images and args.max_images > 0:
        img_paths = img_paths[:args.max_images]
        print(f"Processing first {len(img_paths)} images")

    # Statistics
    token_histogram = np.zeros(n_embed, dtype=np.int64)
    processed = 0

    # Process images
    print(f"\n{'='*60}")
    print("Processing images...")
    print(f"{'='*60}")

    for img_path in tqdm(img_paths, desc='Encoding'):
        try:
            # Load and transform
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img, dtype=np.uint8)
            transformed = transform(image=img_np)['image']

            # To tensor
            img_tensor = torch.from_numpy(transformed).permute(2, 0, 1).unsqueeze(0).float()
            x = img_tensor.to(device)

            # Encode
            quant, _, info = model.encode(x)
            idx = extract_indices(info)

            if idx is None:
                print(f"Warning: No indices for {img_path.name}")
                continue

            # Convert to numpy
            idx_cpu = idx.cpu().numpy().reshape(-1)

            # Update histogram
            hist = np.bincount(idx_cpu, minlength=n_embed)
            token_histogram += hist

            # Reshape to spatial
            h, w = quant.shape[2], quant.shape[3]
            token_map = idx_cpu.reshape(h, w)

            # Save .npy
            img_name = img_path.stem
            npy_path = out_dir / f"{img_name}.npy"
            np.save(npy_path, token_map)

            processed += 1

        except Exception as e:
            print(f"Error: {img_path.name} - {e}")
            continue

    # Save histogram
    #np.save(out_dir / 'token_histogram.npy', token_histogram)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {processed}/{len(img_paths)} images")
    print(f"Tokens used: {np.count_nonzero(token_histogram)}/{n_embed}")
    print(f"Most frequent: token {token_histogram.argmax()} ({token_histogram.max()} times)")
    print(f"\nOutput: {out_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate token maps from images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python token_classification/generate_token_mask.py \\
      --config logs/vqgan_gumbel_f8/configs/model.yaml \\
      --checkpoint logs/vqgan_gumbel_f8/checkpoints/last.ckpt \\
      --img-dir data/sub_BCSS_WSSS/training \\
      --output-dir token_maps
        """
    )

    parser.add_argument('--config', type=str, required=True,
                        help='Path to model config YAML')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--img-dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output-dir', type=str, default='token_maps',
                        help='Output directory')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Max images to process (None = all)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU')

    args = parser.parse_args()
    generate_token_maps(args)

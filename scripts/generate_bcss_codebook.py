import os
import sys
import argparse
import numpy as np

# Add parent directory to path to import taming
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from taming.models.vqgan import GumbelVQ
from omegaconf import OmegaConf
import yaml
import matplotlib.pyplot as plt



def extract_indices(info):
    """Extract indices from quantizer output info"""
    # case tuple/list
    if isinstance(info, (tuple, list)):
        # theo class này: info = (perplexity, min_encodings, min_encoding_indices)
        for x in info[::-1]:  # duyệt từ cuối về đầu
            if torch.is_tensor(x) and x.dtype == torch.long:
                return x
        # hoặc đệ quy nếu lồng sâu
        for x in info:
            got = extract_indices(x)
            if got is not None:
                return got
    # case dict
    elif isinstance(info, dict):
        if 'min_encoding_indices' in info:
            return info['min_encoding_indices']
        if 'indices' in info:
            return info['indices']
    return None

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

@torch.no_grad()
def load_vqgan(config_path, ckpt_path, device='cpu'):
    """Load VQGAN model from config and checkpoint"""
    config = load_config(config_path)
    model = GumbelVQ(**config.model.params)
    # Load checkpoint
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    if ckpt_path is not None:
        model_state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
        print(f"Loaded checkpoint - missing: {len(missing)}, unexpected: {len(unexpected)}")

        model = model.to(device).eval()
        n_embed = config.model.params.n_embed
        embed_dim = config.model.params.embed_dim
    return model, (n_embed, embed_dim)


@torch.no_grad()
def extract_codebook(model):
    """Extract codebook embeddings (prototypes)"""
    # codebook/prototype: [n_embed, embed_dim]
    # GumbelQuantize uses 'embed' instead of 'embedding'
    if hasattr(model.quantize, 'embed'):
        return model.quantize.embed.weight.detach().cpu().numpy()
    elif hasattr(model.quantize, 'embedding'):
        return model.quantize.embedding.weight.detach().cpu().numpy()
    else:
        raise AttributeError("Quantizer has no 'embed' or 'embedding' attribute")


def generate_token_index(idx, quant_tensor, outout_path, cmap="tab20"):
    """
    Save coarse segmentation map
    idx: indices array
    quant_tensor: Tensor [B,C,h,w] để suy ra h,w nếu cần
    """
    h, w = quant_tensor.shape[2], quant_tensor.shape[3]
    idx_hw = idx.reshape(h, w)

    plt.figure(figsize=(4, 4))
    plt.imshow(idx_hw, cmap=cmap)
    plt.axis("off")
    os.makedirs(os.path.dirname(outout_path), exist_ok=True)
    plt.savefig(outout_path, bbox_inches="tight", pad_inches=0)
    plt.close()

@torch.no_grad()
def eval_dataset(args):
    """Process images from a single folder and generate token maps"""
    from PIL import Image
    import albumentations as A
    import glob

    # Device
    device = torch.device("cpu")
    print("Using device:", device)

    # Model
    model, (n_embed, embed_dim) = load_vqgan(args.config, args.checkpoint, device)

    # Extract and save codebook
    codebook = extract_codebook(model)
    print(f"\nCodebook shape: {codebook.shape}  (n_embed={n_embed}, embed_dim={embed_dim})")

    # Get folder name from path (e.g., "TUM" from "data/BCSS-WSSS/training_single_class/TUM")
    sub_folder_name = os.path.basename(os.path.normpath(args.img_root))
    print(f"Processing folder: {sub_folder_name}")

    # Create output subfolder with same name
    out_dir = os.path.join(args.out_dir, sub_folder_name)
    os.makedirs(out_dir, exist_ok=True)

    codebook_path = os.path.join(out_dir, "codebook_prototypes.npy")
    np.save(codebook_path, codebook)
    print(f"Saved codebook to: {codebook_path}")

    # Transform for images
    transform = A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    # Get all images in input folder
    img_paths = sorted(glob.glob(os.path.join(args.img_root, "*.png")))
    if not img_paths:
        img_paths = sorted(glob.glob(os.path.join(args.img_root, "*.jpg")))

    if not img_paths:
        print(f"Error: No images found in {args.img_root}")
        return

    print(f"\nFound {len(img_paths)} images in {args.img_root}")

    # Limit images if specified
    if args.max_images and args.max_images > 0:
        img_paths = img_paths[:args.max_images]
        print(f"Processing first {len(img_paths)} images")

    # Process each image
    rows = []
    token_hist = np.zeros(n_embed, dtype=np.int64)

    for i, img_path in enumerate(img_paths):
        # Load and transform image
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.uint8)
        transformed = transform(image=img_np)["image"]

        # Convert to tensor [1, C, H, W]
        img_tensor = torch.from_numpy(transformed).permute(2, 0, 1).unsqueeze(0).float()
        x = img_tensor.to(device)

        # Encode
        quant, _, info = model.encode(x)
        idx = extract_indices(info)

        if idx is None:
            print(f"Warning: Could not extract indices for {img_path}")
            continue

        idx_cpu = idx.cpu().numpy().reshape(-1)
        hist = np.bincount(idx_cpu, minlength=n_embed)
        token_hist += hist

        # Get spatial dimensions
        h, w = quant.shape[2], quant.shape[3]
        idx_hw = idx_cpu.reshape(h, w)

        # Save token map as .npy
        stem = os.path.splitext(os.path.basename(img_path))[0]
        npy_path = os.path.join(out_dir, f"{stem}_tokens.npy")
        np.save(npy_path, idx_hw)

        # Save visualization
        png_path = os.path.join(out_dir, f"{stem}_tokens.png")
        generate_token_index(idx_cpu, quant, png_path, cmap="tab20")

        # Save to rows
        topk_ids = hist.argsort()[-args.topk:][::-1]
        rows.append({
            "file": os.path.basename(img_path),
            "top_codes": ",".join(map(str, topk_ids.tolist())),
            "top_counts": ",".join(map(str, hist[topk_ids].tolist()))
        })

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(img_paths)}] processed...")

    # Save statistics
    print(f"\n{'='*50}")
    print(f"Total processed: {len(rows)} images")
    print(f"{'='*50}")

    # Save CSV
    csv_path = os.path.join(out_dir, "codes_per_image.csv")
    with open(csv_path, "w") as f:
        f.write("file,top_codes,top_counts\n")
        for r in rows:
            f.write(f"{r['file']},{r['top_codes']},{r['top_counts']}\n")
    print(f"Saved: {csv_path}")

    # Save histogram
    np.save(os.path.join(out_dir, "token_histogram.npy"), token_hist)

    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate codebook and coarse maps for BCSS_WSSS using Gumbel-8 checkpoint")
    parser.add_argument("--config", type=str,
                        default="logs/vqgan_gumbel_f8/configs/model.yaml",
                        help="Path to model config yaml")
    parser.add_argument("--checkpoint", type=str,
                        default="logs/vqgan_gumbel_f8/checkpoints/last.ckpt",
                        help="Path to checkpoint")
    parser.add_argument("--img_root", type=str,
                        default="data/BCSS-WSSS/training_single_class",
                        help="Path to image directory (with class subfolders)")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size")
    parser.add_argument("--out_dir", type=str,
                        default="codebook_analysis_bcss_gumbel8",
                        help="Output directory")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of images to process (None for all)")
    parser.add_argument("--topk", type=int, default=10,
                        help="Top-k codes to save per image")
    parser.add_argument("--dump_zq", action="store_true",
                        help="Save z_q mean per image")
    parser.add_argument("--dump_indices", action="store_true",
                        help="Save raw indices .npy per image")

    args = parser.parse_args()
    eval_dataset(args)



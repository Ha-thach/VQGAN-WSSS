"""
Analyze codebook tokens to determine which class each token belongs to.

This script:
1. Processes all images and extracts token indices
2. Accumulates token usage per class
3. Creates token-to-class mapping based on statistics
4. Visualizes token class distribution
"""

import os
import sys
import argparse
import json
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import yaml
import matplotlib.pyplot as plt
from PIL import Image

from taming.models.vqgan import GumbelVQ
from taming.data.bcss_wsss import BCSSWSSSTrainDataset


CLASSES = ["TUM", "STR", "LYM", "NEC"]
CLASS_COLORS = ['red', 'green', 'blue', 'orange']


def load_config(config_path):
    return OmegaConf.load(config_path)


def collate_fn(batch):
    """Custom collate function to handle None values"""
    collated = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if values[0] is None:
            collated[key] = None
        elif torch.is_tensor(values[0]):
            collated[key] = torch.stack(values)
        elif isinstance(values[0], np.ndarray):
            collated[key] = np.stack(values)
        elif isinstance(values[0], (list, str)):
            collated[key] = values
        else:
            collated[key] = values
    return collated


@torch.no_grad()
def load_vqgan(config_path, ckpt_path, device='cpu'):
    config = load_config(config_path)
    model = GumbelVQ(**config.model.params)

    print(f"Loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded - missing: {len(missing)}, unexpected: {len(unexpected)}")

    model = model.to(device).eval()
    n_embed = config.model.params.n_embed
    embed_dim = config.model.params.embed_dim

    return model, (n_embed, embed_dim)


def extract_indices(info):
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
def analyze_tokens(args):
    device = torch.device("cpu")
    print("Using device:", device)

    # Load model
    model, (n_embed, embed_dim) = load_vqgan(args.config, args.checkpoint, device)
    print(f"Codebook: {n_embed} tokens × {embed_dim} dims")

    # Dataset
    ds = BCSSWSSSTrainDataset(img_root=args.img_root, size=args.image_size, has_mask=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f"Dataset: {len(ds)} images\n")

    # Statistics: count token usage per class
    # token_class_count[token_id][class_id] = count
    token_class_count = np.zeros((n_embed, len(CLASSES)), dtype=np.int64)

    # Also track per-image dominant class for each token
    token_images = defaultdict(list)  # token_id -> [(image_name, class_id), ...]

    max_images = args.max_images if args.max_images else len(ds)

    for i, batch in enumerate(dl):
        if i >= max_images:
            break

        # Get image
        image = batch["image"]
        if not torch.is_tensor(image):
            image = torch.from_numpy(image).permute(0, 3, 1, 2).float()

        x = image.to(device)
        labels = batch["cls_label"].numpy()[0]
        fname = batch["file_path_"][0]

        # Encode
        quant, emb_loss, info = model.encode(x)
        idx = extract_indices(info)

        if idx is None:
            continue

        idx_cpu = idx.cpu().numpy().reshape(-1)

        # Get class (may be multi-label, use all active classes)
        active_classes = np.where(labels > 0.5)[0]

        # Count token usage for each active class
        for token_id in idx_cpu:
            for class_id in active_classes:
                token_class_count[token_id, class_id] += 1

            # Track which images use this token
            if len(token_images[token_id]) < 10:  # Keep max 10 examples
                token_images[token_id].append((fname, active_classes.tolist()))

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{max_images} images...")

    print(f"\nAnalysis complete for {min(max_images, len(ds))} images\n")

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Determine dominant class for each token
    token_to_class = np.argmax(token_class_count, axis=1)  # [n_embed]
    token_confidence = np.max(token_class_count, axis=1) / (np.sum(token_class_count, axis=1) + 1e-8)

    # 2. Save token-to-class mapping
    mapping = {}
    for token_id in range(n_embed):
        total_usage = token_class_count[token_id].sum()
        if total_usage > 0:
            mapping[int(token_id)] = {
                "class": CLASSES[token_to_class[token_id]],
                "confidence": float(token_confidence[token_id]),
                "usage": int(total_usage),
                "per_class": {CLASSES[j]: int(token_class_count[token_id, j]) for j in range(len(CLASSES))}
            }

    mapping_path = os.path.join(args.out_dir, "token_to_class_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved token mapping to: {mapping_path}")

    # 3. Statistics
    print("\n" + "="*60)
    print("TOKEN STATISTICS")
    print("="*60)

    # Count tokens per class
    for class_id, class_name in enumerate(CLASSES):
        class_tokens = np.sum(token_to_class == class_id)
        high_conf_tokens = np.sum((token_to_class == class_id) & (token_confidence > 0.5))
        print(f"{class_name}: {class_tokens} tokens total, {high_conf_tokens} high-confidence (>50%)")

    # Unused tokens
    unused = np.sum(token_class_count.sum(axis=1) == 0)
    print(f"\nUnused tokens: {unused}/{n_embed} ({unused/n_embed*100:.1f}%)")

    # 4. Visualize token distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Token count per class
    ax = axes[0, 0]
    class_counts = [np.sum(token_to_class == i) for i in range(len(CLASSES))]
    bars = ax.bar(CLASSES, class_counts, color=CLASS_COLORS)
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Tokens')
    ax.set_title('Dominant Class Assignment per Token')
    for bar, count in zip(bars, class_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(count), ha='center', va='bottom')

    # Plot 2: Confidence distribution
    ax = axes[0, 1]
    used_tokens = token_class_count.sum(axis=1) > 0
    ax.hist(token_confidence[used_tokens], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0.5, color='r', linestyle='--', label='50% threshold')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Number of Tokens')
    ax.set_title('Token Classification Confidence Distribution')
    ax.legend()

    # Plot 3: Top tokens per class
    ax = axes[1, 0]
    top_k = 20
    x_positions = np.arange(top_k)
    width = 0.2

    for class_id, class_name in enumerate(CLASSES):
        class_usage = token_class_count[:, class_id]
        top_tokens = np.argsort(class_usage)[-top_k:][::-1]
        top_counts = class_usage[top_tokens]
        ax.bar(x_positions + class_id * width, top_counts, width,
               label=class_name, color=CLASS_COLORS[class_id], alpha=0.8)

    ax.set_xlabel('Token Rank')
    ax.set_ylabel('Usage Count')
    ax.set_title(f'Top {top_k} Most Used Tokens per Class')
    ax.legend()
    ax.set_xticks(x_positions + 1.5 * width)
    ax.set_xticklabels([str(i+1) for i in range(top_k)])

    # Plot 4: Token usage heatmap (sample)
    ax = axes[1, 1]
    sample_tokens = 100
    sample_data = token_class_count[:sample_tokens, :]
    im = ax.imshow(sample_data.T, aspect='auto', cmap='YlOrRd')
    ax.set_xlabel('Token ID (first 100)')
    ax.set_ylabel('Class')
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    ax.set_title('Token Usage Heatmap (First 100 Tokens)')
    plt.colorbar(im, ax=ax, label='Usage Count')

    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "token_class_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {fig_path}")

    # 5. Save raw statistics
    np.save(os.path.join(args.out_dir, "token_class_count.npy"), token_class_count)
    np.save(os.path.join(args.out_dir, "token_to_class.npy"), token_to_class)
    np.save(os.path.join(args.out_dir, "token_confidence.npy"), token_confidence)

    # 6. Print top tokens for each class
    print("\n" + "="*60)
    print("TOP 10 TOKENS PER CLASS")
    print("="*60)

    for class_id, class_name in enumerate(CLASSES):
        print(f"\n{class_name}:")
        class_usage = token_class_count[:, class_id]
        top_tokens = np.argsort(class_usage)[-10:][::-1]

        for rank, token_id in enumerate(top_tokens, 1):
            usage = class_usage[token_id]
            total = token_class_count[token_id].sum()
            conf = usage / total if total > 0 else 0
            print(f"  {rank}. Token {token_id}: {usage} uses ({conf*100:.1f}% of token's total)")

    # 7. Print class-specific tokens (high confidence)
    print("\n" + "="*60)
    print("CLASS-SPECIFIC TOKENS (>80% confidence)")
    print("="*60)

    for class_id, class_name in enumerate(CLASSES):
        class_specific = (token_to_class == class_id) & (token_confidence > 0.8)
        specific_tokens = np.where(class_specific)[0]

        if len(specific_tokens) > 0:
            # Sort by usage
            usages = token_class_count[specific_tokens, class_id]
            sorted_idx = np.argsort(usages)[::-1]
            top_specific = specific_tokens[sorted_idx[:10]]

            print(f"\n{class_name}: {len(specific_tokens)} class-specific tokens")
            for token_id in top_specific:
                usage = token_class_count[token_id, class_id]
                conf = token_confidence[token_id]
                print(f"  Token {token_id}: {usage} uses, {conf*100:.1f}% confidence")
        else:
            print(f"\n{class_name}: No class-specific tokens found")

    print("\n" + "="*60)
    print(f"Analysis saved to: {args.out_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze codebook tokens by class")
    parser.add_argument("--config", type=str,
                        default="logs/vqgan_gumbel_f8/configs/model.yaml")
    parser.add_argument("--checkpoint", type=str,
                        default="logs/vqgan_gumbel_f8/checkpoints/last.ckpt")
    parser.add_argument("--img_root", type=str,
                        default="/Users/thachha/Desktop/AIO2025-official/AIMA/CP-WSSS/PBIP/data/BCSS-WSSS/training")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--out_dir", type=str,
                        default="codebook_analysis_bcss_gumbel8/token_class_analysis")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Max images to process (None for all)")

    args = parser.parse_args()
    analyze_tokens(args)

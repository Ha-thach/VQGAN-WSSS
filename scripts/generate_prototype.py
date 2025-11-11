import os
import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import yaml

import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
from PIL import Image

# Import from taming repo
from taming.models.vqgan import VQModel
from taming.util import instantiate_from_config


def load_config(config_path, display=False):
    """Load OmegaConf config from yaml file"""
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_vqgan_from_lightning_checkpoint(config_path, ckpt_path, device='cpu'):
    """
    Load VQGAN model from Lightning checkpoint format
    
    Args:
        config_path: Path to config yaml file
        ckpt_path: Path to Lightning checkpoint (.ckpt file)
        device: Device to load model on
    
    Returns:
        model: Loaded VQGAN model in eval mode
        config: Model configuration
    """
    # Load config
    config = load_config(config_path)
    
    # Instantiate model from config
    model = instantiate_from_config(config.model)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Lightning saves model state_dict under 'state_dict' key
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Fallback for non-Lightning checkpoints
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Codebook size: {model.quantize.n_e}")
    print(f"Embedding dim: {model.quantize.e_dim}")
    
    return model, config


@torch.no_grad()
def extract_codebook_prototypes(model):
    """
    Extract codebook embeddings (prototypes)
    
    Args:
        model: VQGAN model
    
    Returns:
        codebook: numpy array of shape [n_embed, embed_dim]
    """
    # The codebook is stored in the quantizer's embedding
    codebook = model.quantize.embedding.weight.detach().cpu().numpy()
    return codebook


@torch.no_grad()
def encode_image(model, image_tensor):
    """
    Encode image and get quantized indices
    
    Args:
        model: VQGAN model
        image_tensor: Tensor of shape [B, C, H, W] normalized to [-1, 1]
    
    Returns:
        indices: Codebook indices [B, h, w] or [B, h*w]
        quant: Quantized features [B, embed_dim, h, w]
    """
    # Encode
    h = model.encoder(image_tensor)
    h = model.quant_conv(h)
    
    # Quantize
    quant, emb_loss, info = model.quantize(h)
    
    # Extract indices - depends on quantizer implementation
    # For VectorQuantizer2, indices are in info tuple
    if isinstance(info, tuple):
        # Usually (perplexity, min_encodings, min_encoding_indices)
        indices = info[2]  # min_encoding_indices
    elif isinstance(info, dict):
        indices = info.get('min_encoding_indices', None)
    else:
        indices = info
    
    return indices, quant


def visualize_codebook_usage(indices_list, n_embed, save_path):
    """
    Visualize codebook usage histogram across dataset
    
    Args:
        indices_list: List of index arrays from all images
        n_embed: Total number of codebook entries
        save_path: Path to save visualization
    """
    # Concatenate all indices
    all_indices = np.concatenate([idx.reshape(-1) for idx in indices_list])
    
    # Count usage
    hist = np.bincount(all_indices, minlength=n_embed)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # Full histogram
    axes[0].bar(range(n_embed), hist)
    axes[0].set_xlabel('Codebook Index')
    axes[0].set_ylabel('Usage Count')
    axes[0].set_title(f'Codebook Usage Distribution (Total entries: {n_embed})')
    axes[0].grid(True, alpha=0.3)
    
    # Top used codes
    top_k = min(50, n_embed)
    top_indices = np.argsort(hist)[-top_k:][::-1]
    axes[1].bar(range(top_k), hist[top_indices])
    axes[1].set_xlabel(f'Top {top_k} Codebook Indices')
    axes[1].set_ylabel('Usage Count')
    axes[1].set_title(f'Top {top_k} Most Used Codebook Entries')
    axes[1].set_xticks(range(top_k))
    axes[1].set_xticklabels(top_indices, rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved codebook usage visualization to: {save_path}")
    
    # Statistics
    unused = np.sum(hist == 0)
    print(f"\nCodebook Statistics:")
    print(f"  Total entries: {n_embed}")
    print(f"  Used entries: {n_embed - unused}")
    print(f"  Unused entries: {unused} ({unused/n_embed*100:.2f}%)")
    print(f"  Mean usage: {hist.mean():.2f}")
    print(f"  Std usage: {hist.std():.2f}")
    print(f"  Max usage: {hist.max()}")
    
    return hist


def visualize_index_map(indices, save_path, cmap='tab20'):
    """
    Visualize spatial index map
    
    Args:
        indices: Index tensor [h, w] or [h*w]
        save_path: Path to save visualization
        cmap: Matplotlib colormap
    """
    if indices.ndim == 1:
        # Need to infer spatial dimensions
        h = w = int(np.sqrt(len(indices)))
        indices = indices.reshape(h, w)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(indices, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Codebook Index')
    plt.title('Codebook Index Map')
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_prototype_grid(codebook, save_path, n_samples=64, grid_size=8):
    """
    Visualize codebook prototypes as a grid (PCA or t-SNE visualization)
    
    Args:
        codebook: Codebook array [n_embed, embed_dim]
        save_path: Path to save visualization
        n_samples: Number of prototypes to visualize
        grid_size: Grid size for layout
    """
    from sklearn.decomposition import PCA
    
    # Select subset if codebook is large
    n_embed = codebook.shape[0]
    if n_embed > n_samples:
        indices = np.linspace(0, n_embed-1, n_samples, dtype=int)
        subset = codebook[indices]
    else:
        subset = codebook
        n_samples = n_embed
    
    # Reduce to 3D using PCA for RGB visualization
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(subset)
    
    # Normalize to [0, 1]
    reduced = (reduced - reduced.min()) / (reduced.max() - reduced.min())
    
    # Create grid
    grid_h = grid_w = grid_size
    fig, axes = plt.subplots(grid_h, grid_w, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(min(n_samples, grid_h * grid_w)):
        color = reduced[i]
        axes[i].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        axes[i].axis('off')
        axes[i].set_title(f'#{i}', fontsize=8)
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Codebook Prototypes (PCA-reduced to RGB)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved prototype visualization to: {save_path}")


@torch.no_grad()
def analyze_dataset(model, data_loader, output_dir, device='cpu', max_images=None):
    """
    Analyze a dataset using the VQGAN model
    
    Args:
        model: VQGAN model
        data_loader: PyTorch DataLoader
        output_dir: Directory to save outputs
        device: Device to run on
        max_images: Maximum number of images to process (None = all)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'index_maps'), exist_ok=True)
    
    n_embed = model.quantize.n_e
    all_indices = []
    results = []
    
    for i, batch in enumerate(data_loader):
        if max_images and i >= max_images:
            break
        
        # Get image
        if isinstance(batch, dict):
            images = batch.get('image', batch.get('img', None))
            img_paths = batch.get('img_path', batch.get('path', [f'image_{i}']))
        else:
            images = batch
            img_paths = [f'image_{i}']
        
        images = images.to(device)
        
        # Encode
        indices, quant = encode_image(model, images)
        indices_np = indices.cpu().numpy()
        
        # Process each image in batch
        for b in range(images.shape[0]):
            idx = indices_np[b]
            if idx.ndim > 1:
                idx = idx.reshape(-1)
            
            all_indices.append(idx)
            
            # Count usage
            hist = np.bincount(idx, minlength=n_embed)
            top_k = 10
            top_codes = np.argsort(hist)[-top_k:][::-1]
            
            # Save result
            img_name = os.path.basename(img_paths[b]) if b < len(img_paths) else f'image_{i}_{b}'
            results.append({
                'image': img_name,
                'unique_codes': int(np.sum(hist > 0)),
                'top_codes': top_codes.tolist(),
                'top_counts': hist[top_codes].tolist(),
            })
            
            # Save index map for first few images
            if i < 10:
                map_path = os.path.join(output_dir, 'index_maps', f'{Path(img_name).stem}_indices.png')
                visualize_index_map(idx, map_path)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} batches...")
    
    # Save results
    results_path = os.path.join(output_dir, 'analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved analysis results to: {results_path}")
    
    # Visualize codebook usage
    usage_path = os.path.join(output_dir, 'codebook_usage.png')
    hist = visualize_codebook_usage(all_indices, n_embed, usage_path)
    
    # Save histogram
    np.save(os.path.join(output_dir, 'codebook_histogram.npy'), hist)
    
    return hist, results


def create_simple_dataloader(img_dir, image_size=256, batch_size=4):
    """
    Create a simple dataloader for a directory of images
    
    Args:
        img_dir: Directory containing images
        image_size: Size to resize images to
        batch_size: Batch size
    
    Returns:
        DataLoader
    """
    from torch.utils.data import Dataset
    
    class SimpleImageDataset(Dataset):
        def __init__(self, img_dir, transform):
            self.img_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.PNG']:
                self.img_paths.extend(Path(img_dir).glob(ext))
            self.img_paths = sorted(self.img_paths)
            self.transform = transform
        
        def __len__(self):
            return len(self.img_paths)
        
        def __getitem__(self, idx):
            img_path = str(self.img_paths[idx])
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            
            if self.transform:
                image = self.transform(image=image)['image']
            
            return {
                'image': image,
                'img_path': img_path
            }
    
    # Transform: resize and normalize to [-1, 1]
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    dataset = SimpleImageDataset(img_dir, transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=False
    )
    
    return dataloader


def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_vqgan_from_lightning_checkpoint(
        args.config, 
        args.checkpoint, 
        device
    )
    
    # Extract codebook
    print("\n" + "="*50)
    print("Extracting Codebook Prototypes")
    print("="*50)
    codebook = extract_codebook_prototypes(model)
    print(f"Codebook shape: {codebook.shape}")
    
    # Save codebook
    os.makedirs(args.output_dir, exist_ok=True)
    codebook_path = os.path.join(args.output_dir, 'codebook_prototypes.npy')
    np.save(codebook_path, codebook)
    print(f"Saved codebook to: {codebook_path}")
    
    # Visualize prototypes
    if args.visualize_prototypes:
        proto_path = os.path.join(args.output_dir, 'prototype_grid.png')
        visualize_prototype_grid(codebook, proto_path, n_samples=64, grid_size=8)
    
    # Analyze dataset if provided
    if args.image_dir:
        print("\n" + "="*50)
        print("Analyzing Dataset")
        print("="*50)
        
        dataloader = create_simple_dataloader(
            args.image_dir,
            image_size=args.image_size,
            batch_size=args.batch_size
        )
        print(f"Dataset size: {len(dataloader.dataset)} images")
        
        analyze_dataset(
            model, 
            dataloader, 
            args.output_dir,
            device,
            max_images=args.max_images
        )
    
    print("\n" + "="*50)
    print("Analysis Complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract and analyze codebook prototypes from VQGAN Lightning checkpoint'
    )
    
    # Model loading
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to model config yaml file'
    )
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        required=True,
        help='Path to Lightning checkpoint (.ckpt file)'
    )
    
    # Data
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='Directory containing images to analyze (optional)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='Image size for analysis'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for analysis'
    )
    parser.add_argument(
        '--max_images',
        type=int,
        default=None,
        help='Maximum number of images to analyze (None = all)'
    )
    
    # Output
    parser.add_argument(
        '--output_dir',
        type=str,
        default='codebook_analysis',
        help='Output directory for results'
    )
    parser.add_argument(
        '--visualize_prototypes',
        action='store_true',
        help='Generate prototype visualization grid'
    )
    
    # Device
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage'
    )
    
    args = parser.parse_args()
    main(args)
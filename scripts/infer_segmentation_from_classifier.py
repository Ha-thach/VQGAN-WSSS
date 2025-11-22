"""
Inference: Convert Token Classification to Pixel-Level Segmentation

Three approaches:
1. CAM (Class Activation Mapping) - Use attention from trained image classifier
2. Token-level prediction - Modify classifier to predict per-token
3. Dual-head model - Train both image and segmentation heads
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from tqdm import tqdm

from token_classification.train_token_classifier import TokenViT
from taming.models.token_classifier_segmentation import (
    TokenViTSegmentation,
    TokenViTWithImageClassification,
    generate_pseudo_labels_from_image_classifier
)


# Class info
CLASSES = ["TUM", "STR", "LYM", "NEC"]
CLASS_COLORS = {
    0: [0, 0, 0],        # Background - black
    1: [255, 0, 0],      # TUM - red
    2: [0, 255, 0],      # STR - green
    3: [0, 0, 255],      # LYM - blue
    4: [255, 255, 0]     # NEC - yellow
}


def colorize_mask(mask, colors=CLASS_COLORS):
    """Convert class indices to RGB"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in colors.items():
        rgb[mask == class_id] = color

    return rgb


@torch.no_grad()
def method1_cam_from_image_classifier(args):
    """
    Method 1: Use CAM on trained image classifier

    Pros: Use existing trained model, no retraining
    Cons: Less accurate, only highlights discriminative regions
    """
    print("="*60)
    print("Method 1: CAM from Image Classifier")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Load trained image classifier
    model = TokenViT(
        num_tokens=args.num_tokens,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        pretrained_model=args.pretrained_model,
        freeze_backbone=False
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Loaded checkpoint: {args.checkpoint}")

    # Load token map
    token_map = np.load(args.token_map_path)
    tokens = torch.from_numpy(token_map).unsqueeze(0).long().to(device)  # (1, H, W)

    print(f"Token map shape: {tokens.shape}")

    # Generate pseudo labels using CAM
    pseudo_mask, probs, cam = generate_pseudo_labels_from_image_classifier(
        model, tokens, threshold=args.threshold, cam_method='simple-cam'
    )

    # Convert to numpy
    pseudo_mask = pseudo_mask[0].cpu().numpy()  # (H, W)
    probs = probs[0].cpu().numpy()

    print(f"\nImage-level predictions:")
    for i, cls_name in enumerate(CLASSES):
        print(f"  {cls_name}: {probs[i]:.4f}")

    # Upsample to original size (32x32 -> 256x256)
    pseudo_mask_upsampled = cv2.resize(
        pseudo_mask.astype(np.uint8),
        (args.output_size, args.output_size),
        interpolation=cv2.INTER_NEAREST
    )

    # Colorize
    colored_mask = colorize_mask(pseudo_mask_upsampled)

    # Save
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    Image.fromarray(colored_mask).save(output_path / 'cam_segmentation.png')
    np.save(output_path / 'cam_segmentation.npy', pseudo_mask_upsampled)

    print(f"\n✓ Saved to {output_path}")
    print(f"  - cam_segmentation.png (RGB visualization)")
    print(f"  - cam_segmentation.npy (class indices)")


@torch.no_grad()
def method2_token_level_prediction(args):
    """
    Method 2: Use token-level segmentation model

    Pros: More accurate, end-to-end
    Cons: Requires training new model
    """
    print("="*60)
    print("Method 2: Token-Level Segmentation")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Load segmentation model
    model = TokenViTSegmentation(
        num_tokens=args.num_tokens,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        pretrained_model=args.pretrained_model,
        freeze_backbone=False,
        output_stride=8
    ).to(device)

    # Load checkpoint (if available)
    if args.checkpoint and Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint: {args.checkpoint}")
    else:
        print("⚠ No checkpoint provided, using random weights (need to train first!)")

    model.eval()

    # Load token map
    token_map = np.load(args.token_map_path)
    tokens = torch.from_numpy(token_map).unsqueeze(0).long().to(device)  # (1, H, W)

    print(f"Token map shape: {tokens.shape}")

    # Forward
    seg_logits, token_logits = model(tokens)

    # Get predictions
    seg_probs = torch.sigmoid(seg_logits)  # (1, num_classes, 256, 256)
    seg_pred = seg_probs.argmax(dim=1)  # (1, 256, 256)

    # Convert to numpy
    seg_pred = seg_pred[0].cpu().numpy()  # (256, 256)
    seg_probs = seg_probs[0].cpu().numpy()  # (4, 256, 256)

    # Image-level prediction (max pool over spatial dimensions)
    image_probs = seg_probs.max(axis=(1, 2))

    print(f"\nImage-level predictions:")
    for i, cls_name in enumerate(CLASSES):
        print(f"  {cls_name}: {image_probs[i]:.4f}")

    # Colorize
    colored_mask = colorize_mask(seg_pred)

    # Save
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    Image.fromarray(colored_mask).save(output_path / 'token_segmentation.png')
    np.save(output_path / 'token_segmentation.npy', seg_pred)

    # Save per-class probability maps
    for i, cls_name in enumerate(CLASSES):
        prob_map = (seg_probs[i] * 255).astype(np.uint8)
        Image.fromarray(prob_map).save(output_path / f'{cls_name}_prob.png')

    print(f"\n✓ Saved to {output_path}")
    print(f"  - token_segmentation.png (RGB visualization)")
    print(f"  - token_segmentation.npy (class indices)")
    print(f"  - *_prob.png (per-class probability maps)")


@torch.no_grad()
def method3_dual_head_model(args):
    """
    Method 3: Dual-head model (image classification + segmentation)

    Pros: Best of both worlds, can use weak supervision
    Cons: More complex training
    """
    print("="*60)
    print("Method 3: Dual-Head Model")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Load dual-head model
    model = TokenViTWithImageClassification(
        num_tokens=args.num_tokens,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        pretrained_model=args.pretrained_model,
        freeze_backbone=False,
        output_stride=8
    ).to(device)

    # Load checkpoint (if available)
    if args.checkpoint and Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint: {args.checkpoint}")
    else:
        print("⚠ No checkpoint provided, using random weights (need to train first!)")

    model.eval()

    # Load token map
    token_map = np.load(args.token_map_path)
    tokens = torch.from_numpy(token_map).unsqueeze(0).long().to(device)

    print(f"Token map shape: {tokens.shape}")

    # Forward
    image_logits, seg_logits = model(tokens)

    # Predictions
    image_probs = torch.sigmoid(image_logits)[0].cpu().numpy()
    seg_probs = torch.sigmoid(seg_logits)
    seg_pred = seg_probs.argmax(dim=1)[0].cpu().numpy()

    print(f"\nImage-level predictions:")
    for i, cls_name in enumerate(CLASSES):
        print(f"  {cls_name}: {image_probs[i]:.4f}")

    # Colorize
    colored_mask = colorize_mask(seg_pred)

    # Save
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    Image.fromarray(colored_mask).save(output_path / 'dual_segmentation.png')
    np.save(output_path / 'dual_segmentation.npy', seg_pred)

    print(f"\n✓ Saved to {output_path}")


def batch_inference(args):
    """Process entire directory of token maps"""
    print("="*60)
    print("Batch Inference")
    print("="*60)

    token_dir = Path(args.token_dir)
    token_paths = sorted(list(token_dir.glob('*.npy')))
    token_paths = [p for p in token_paths if not p.name.startswith('codebook')]

    print(f"Found {len(token_paths)} token maps")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use method 1 (CAM) for batch processing
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    model = TokenViT(
        num_tokens=args.num_tokens,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        pretrained_model=args.pretrained_model,
        freeze_backbone=False
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Loaded checkpoint")

    for token_path in tqdm(token_paths, desc='Processing'):
        # Load
        token_map = np.load(token_path)
        tokens = torch.from_numpy(token_map).unsqueeze(0).long().to(device)

        # Generate pseudo mask
        pseudo_mask, probs, _ = generate_pseudo_labels_from_image_classifier(
            model, tokens, threshold=args.threshold, cam_method='simple-cam'
        )

        pseudo_mask = pseudo_mask[0].cpu().numpy()

        # Upsample
        pseudo_mask_upsampled = cv2.resize(
            pseudo_mask.astype(np.uint8),
            (args.output_size, args.output_size),
            interpolation=cv2.INTER_NEAREST
        )

        # Save
        stem = token_path.stem.split('[')[0]  # Remove label suffix
        np.save(output_path / f'{stem}_seg.npy', pseudo_mask_upsampled)

        # Save colored version
        colored = colorize_mask(pseudo_mask_upsampled)
        Image.fromarray(colored).save(output_path / f'{stem}_seg.png')

    print(f"\n✓ Processed {len(token_paths)} files")
    print(f"✓ Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert token classification to segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Method 1: CAM from existing image classifier (no retraining needed)
  python scripts/infer_segmentation_from_classifier.py \\
      --method cam \\
      --checkpoint outputs/token_vit/best_model.pth \\
      --token-map-path token_maps/test/sample[0101].npy \\
      --output-dir outputs/segmentation

  # Method 2: Token-level segmentation (requires trained segmentation model)
  python scripts/infer_segmentation_from_classifier.py \\
      --method token-level \\
      --checkpoint outputs/token_seg/best_model.pth \\
      --token-map-path token_maps/test/sample[0101].npy \\
      --output-dir outputs/segmentation

  # Batch processing
  python scripts/infer_segmentation_from_classifier.py \\
      --method batch \\
      --checkpoint outputs/token_vit/best_model.pth \\
      --token-dir token_maps/test \\
      --output-dir outputs/segmentation_batch
        """
    )

    parser.add_argument('--method', type=str, default='cam',
                        choices=['cam', 'token-level', 'dual-head', 'batch'],
                        help='Inference method')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--num-tokens', type=int, default=8192)
    parser.add_argument('--embed-dim', type=int, default=768)
    parser.add_argument('--num-classes', type=int, default=4)
    parser.add_argument('--pretrained-model', type=str,
                        default='google/vit-base-patch16-224')

    # Input
    parser.add_argument('--token-map-path', type=str,
                        help='Path to single token map (.npy)')
    parser.add_argument('--token-dir', type=str,
                        help='Directory of token maps (for batch)')

    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--output-size', type=int, default=256,
                        help='Output segmentation size')

    # Parameters
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for CAM')
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    if args.method == 'cam':
        method1_cam_from_image_classifier(args)
    elif args.method == 'token-level':
        method2_token_level_prediction(args)
    elif args.method == 'dual-head':
        method3_dual_head_model(args)
    elif args.method == 'batch':
        batch_inference(args)

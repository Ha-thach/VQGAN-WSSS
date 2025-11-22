"""
Train Token-Level Segmentation Model

Uses token maps + pixel-level masks to train end-to-end segmentation
"""

import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from PIL import Image
import albumentations as A

sys.path.insert(0, str(Path(__file__).parent.parent))

from taming.models.token_classifier_segmentation import TokenViTSegmentation


CLASSES = ["TUM", "STR", "LYM", "NEC"]


class TokenMapSegmentationDataset(Dataset):
    """
    Dataset for token-level segmentation training
    Requires: token maps + pixel-level segmentation masks
    """

    def __init__(
        self,
        token_dir,
        mask_dir,
        img_dir=None,
        target_size=256
    ):
        """
        Args:
            token_dir: Directory with token .npy files
            mask_dir: Directory with segmentation masks (.png)
            img_dir: Optional image directory (for visualization)
            target_size: Target mask size (default 256)
        """
        self.token_dir = Path(token_dir)
        self.mask_dir = Path(mask_dir)
        self.img_dir = Path(img_dir) if img_dir else None
        self.target_size = target_size

        # Load data
        self._load_data()

    def _load_data(self):
        """Match token maps with masks"""
        # Get all token maps
        token_paths = sorted(list(self.token_dir.glob('*.npy')))
        token_paths = [
            p for p in token_paths
            if not p.name.startswith('codebook')
            and not p.name.startswith('token_histogram')
        ]

        print(f"Found {len(token_paths)} token files")

        # Match with masks
        self.token_paths = []
        self.mask_paths = []
        missing = 0

        for token_path in token_paths:
            # Extract image name (remove label suffix if exists)
            stem = token_path.stem
            if '[' in stem:
                stem = stem.split('[')[0]

            # Find mask
            mask_path = self.mask_dir / f'{stem}.png'

            if not mask_path.exists():
                # Try other extensions
                possible = list(self.mask_dir.glob(f'{stem}.*'))
                if possible:
                    mask_path = possible[0]
                else:
                    missing += 1
                    continue

            self.token_paths.append(token_path)
            self.mask_paths.append(mask_path)

        print(f"Matched {len(self.token_paths)} token-mask pairs")
        if missing > 0:
            print(f"Warning: {missing} tokens without masks")

    def __len__(self):
        return len(self.token_paths)

    def __getitem__(self, idx):
        # Load token map
        token_map = np.load(self.token_paths[idx])
        tokens = torch.from_numpy(token_map).long()  # (H, W)

        # Load mask (P mode - palette indices)
        mask = Image.open(self.mask_paths[idx])
        mask_np = np.array(mask, dtype=np.int64)  # (H, W)

        # Resize mask to target size
        if mask_np.shape[0] != self.target_size or mask_np.shape[1] != self.target_size:
            mask_pil = Image.fromarray(mask_np.astype(np.uint8))
            mask_pil = mask_pil.resize((self.target_size, self.target_size), Image.NEAREST)
            mask_np = np.array(mask_pil, dtype=np.int64)

        mask = torch.from_numpy(mask_np).long()  # (H, W)

        return {
            'tokens': tokens,
            'mask': mask,
            'file_path': self.token_paths[idx].name
        }


def dice_loss(pred, target, num_classes=5):
    """
    Multi-class Dice loss

    Args:
        pred: (B, num_classes, H, W) logits
        target: (B, H, W) class indices
    """
    pred_probs = torch.softmax(pred, dim=1)  # (B, C, H, W)

    # One-hot encode target
    target_one_hot = torch.zeros_like(pred_probs)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)

    # Dice coefficient
    smooth = 1.0
    intersection = (pred_probs * target_one_hot).sum(dim=(2, 3))
    union = pred_probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice.mean()

    return dice_loss


def train_epoch(model, dataloader, optimizer, device, alpha=0.7):
    """
    Train one epoch

    Args:
        alpha: Weight for CE loss (1-alpha for Dice)
    """
    model.train()

    total_loss = 0
    total_ce_loss = 0
    total_dice_loss = 0

    ce_criterion = nn.CrossEntropyLoss(ignore_index=255)

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        tokens = batch['tokens'].to(device)  # (B, H, W)
        masks = batch['mask'].to(device)  # (B, H, W)

        optimizer.zero_grad()

        # Forward
        seg_logits, token_logits = model(tokens)  # (B, C, 256, 256)

        # CE loss
        ce_loss = ce_criterion(seg_logits, masks)

        # Dice loss
        dice = dice_loss(seg_logits, masks, num_classes=5)

        # Combined loss
        loss = alpha * ce_loss + (1 - alpha) * dice

        # Backward
        loss.backward()
        optimizer.step()

        # Log
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_dice_loss += dice.item()

        pbar.set_postfix({
            'loss': loss.item(),
            'ce': ce_loss.item(),
            'dice': dice.item()
        })

    avg_loss = total_loss / len(dataloader)
    avg_ce = total_ce_loss / len(dataloader)
    avg_dice = total_dice_loss / len(dataloader)

    return avg_loss, avg_ce, avg_dice


@torch.no_grad()
def validate(model, dataloader, device, alpha=0.7):
    """Validate model"""
    model.eval()

    total_loss = 0
    total_ce_loss = 0
    total_dice_loss = 0
    total_iou = 0

    ce_criterion = nn.CrossEntropyLoss(ignore_index=255)

    pbar = tqdm(dataloader, desc='Validating')
    for batch in pbar:
        tokens = batch['tokens'].to(device)
        masks = batch['mask'].to(device)

        # Forward
        seg_logits, _ = model(tokens)

        # CE loss
        ce_loss = ce_criterion(seg_logits, masks)

        # Dice loss
        dice = dice_loss(seg_logits, masks, num_classes=5)

        # Combined loss
        loss = alpha * ce_loss + (1 - alpha) * dice

        # IoU
        preds = seg_logits.argmax(dim=1)
        iou = compute_iou(preds, masks, num_classes=5)

        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_dice_loss += dice.item()
        total_iou += iou

    avg_loss = total_loss / len(dataloader)
    avg_ce = total_ce_loss / len(dataloader)
    avg_dice = total_dice_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return {
        'loss': avg_loss,
        'ce_loss': avg_ce,
        'dice_loss': avg_dice,
        'iou': avg_iou
    }


def compute_iou(pred, target, num_classes=5):
    """Compute mean IoU"""
    ious = []

    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)

        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()

        if union > 0:
            iou = intersection / union
            ious.append(iou.item())

    return np.mean(ious) if ious else 0.0


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("LOADING DATASETS")
    print(f"{'='*60}")

    # Build datasets
    train_dataset = TokenMapSegmentationDataset(
        token_dir=args.train_token_dir,
        mask_dir=args.train_mask_dir,
        target_size=args.target_size
    )

    val_dataset = TokenMapSegmentationDataset(
        token_dir=args.val_token_dir,
        mask_dir=args.val_mask_dir,
        target_size=args.target_size
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

    # Build model
    print(f"\n{'='*60}")
    print("BUILDING MODEL")
    print(f"{'='*60}")

    model = TokenViTSegmentation(
        num_tokens=args.num_tokens,
        embed_dim=args.embed_dim,
        num_classes=5,  # 4 tissue classes + background
        pretrained_model=args.pretrained_model,
        freeze_backbone=args.freeze_backbone,
        output_stride=8
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': []
    }

    best_val_iou = 0

    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss, train_ce, train_dice = train_epoch(
            model, train_loader, optimizer, device, alpha=args.alpha
        )

        # Validate
        val_metrics = validate(model, val_loader, device, alpha=args.alpha)

        # Update scheduler
        scheduler.step()

        # Log
        print(f"Train Loss: {train_loss:.4f} (CE: {train_ce:.4f}, Dice: {train_dice:.4f})")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val IoU: {val_metrics['iou']:.4f}")

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_iou'].append(val_metrics['iou'])

        # Save best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                output_dir / 'best_model.pth'
            )
            print(f"✓ Saved best model (IoU: {best_val_iou:.4f})")

        # Save last model
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            output_dir / 'last_model.pth'
        )

    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train token-level segmentation model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data
    parser.add_argument('--train-token-dir', type=str, required=True)
    parser.add_argument('--train-mask-dir', type=str, required=True)
    parser.add_argument('--val-token-dir', type=str, required=True)
    parser.add_argument('--val-mask-dir', type=str, required=True)
    parser.add_argument('--target-size', type=int, default=256)

    # Model
    parser.add_argument('--num-tokens', type=int, default=8192)
    parser.add_argument('--embed-dim', type=int, default=768)
    parser.add_argument('--pretrained-model', type=str,
                        default='google/vit-base-patch16-224')
    parser.add_argument('--freeze-backbone', action='store_true')

    # Training
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Weight for CE loss (1-alpha for Dice)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--cpu', action='store_true')

    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/token_seg')

    args = parser.parse_args()
    main(args)

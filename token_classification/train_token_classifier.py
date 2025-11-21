"""
Train Token Classifier using Pretrained Vision Transformer
Simple training loop without PyTorch Lightning
"""

import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from token_classification.dataset import build_token_dataset


def collate_fn(batch):
    """
    Custom collate function to handle variable-length cls_names lists
    """
    tokens = torch.stack([item['tokens'] for item in batch])
    cls_labels = torch.stack([item['cls_label'] for item in batch])
    cls_names = [item['cls_names'] for item in batch]  # Keep as list of lists
    file_paths = [item['file_path_'] for item in batch]

    return {
        'tokens': tokens,
        'cls_label': cls_labels,
        'cls_names': cls_names,
        'file_path_': file_paths
    }


class TokenViT(nn.Module):
    """
    Vision Transformer adapted for token map classification
    Uses pretrained ViT but replaces input layer for discrete tokens
    """

    def __init__(
        self,
        num_tokens=8192,
        embed_dim=768,
        num_classes=4,
        pretrained_model='google/vit-base-patch16-224',
        freeze_backbone=False
    ):
        super().__init__()

        # Token embedding (replaces patch embedding)
        self.token_embedding = nn.Embedding(num_tokens, embed_dim)

        # Positional encoding for 32x32 = 1024 positions
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, embed_dim) * 0.02)

        # Load pretrained ViT
        try:
            from transformers import ViTModel
            print(f"Loading pretrained ViT: {pretrained_model}")
            vit = ViTModel.from_pretrained(pretrained_model)

            # Use transformer encoder blocks
            self.encoder = vit.encoder
            self.layernorm = vit.layernorm

            print(f"✓ Loaded pretrained ViT encoder")

        except ImportError:
            print("Warning: transformers not installed, using random init")
            # Fallback: simple transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=12,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)
            self.layernorm = nn.LayerNorm(embed_dim)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("✓ Frozen encoder backbone")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, tokens):
        """
        Args:
            tokens: (B, H, W) - e.g., (32, 32, 32)
        Returns:
            logits: (B, num_classes)
        """
        B, H, W = tokens.shape

        # Embed tokens
        x = self.token_embedding(tokens)  # (B, H, W, D)
        x = x.reshape(B, H * W, -1)  # (B, N, D) where N = H*W

        # Add positional encoding
        x = x + self.pos_embedding

        # Transformer encoding
        if hasattr(self.encoder, 'layer'):  # HuggingFace ViT
            x = self.encoder(x).last_hidden_state
        else:  # Plain PyTorch transformer
            x = self.encoder(x)

        x = self.layernorm(x)

        # Global average pooling
        x = x.mean(dim=1)  # (B, D)

        # Classification
        logits = self.classifier(x)  # (B, num_classes)

        return logits


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        tokens = batch['tokens'].to(device)
        labels = batch['cls_label'].to(device)

        # Forward
        optimizer.zero_grad()
        logits = model(tokens)
        loss = criterion(logits, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

        pbar.set_postfix({'loss': loss.item()})

    # Aggregate metrics
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Multi-label accuracy (exact match)
    exact_match = (all_preds == all_labels).all(axis=1).mean()

    avg_loss = total_loss / len(dataloader)

    return avg_loss, exact_match


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc='Validating')
    for batch in pbar:
        tokens = batch['tokens'].to(device)
        labels = batch['cls_label'].to(device)

        # Forward
        logits = model(tokens)
        loss = criterion(logits, labels)

        # Metrics
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    # Aggregate
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    exact_match = (all_preds == all_labels).all(axis=1).mean()

    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    avg_loss = total_loss / len(dataloader)

    return {
        'loss': avg_loss,
        'exact_match': exact_match,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("LOADING DATASETS")
    print(f"{'='*60}")

    # Build datasets
    train_dataset = build_token_dataset(
        split='train',
        token_dir=args.train_token_dir
    )

    val_dataset = build_token_dataset(
        split='valid',
        token_dir=args.val_token_dir
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

    # Build model
    print(f"\n{'='*60}")
    print("BUILDING MODEL")
    print(f"{'='*60}")

    model = TokenViT(
        num_tokens=args.num_tokens,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        pretrained_model=args.pretrained_model,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    best_val_f1 = 0

    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['exact_match']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f} | Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['exact_match'])
        history['val_f1'].append(val_metrics['f1'])

        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                output_dir / 'best_model.pth'
            )
            print(f"✓ Saved best model (F1: {best_val_f1:.4f})")

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
    print(f"Best Val F1: {best_val_f1:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train token classifier using pretrained ViT',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data
    parser.add_argument('--train-token-dir', type=str, required=True,
                        help='Directory with training token maps (with labels in filename)')
    parser.add_argument('--val-token-dir', type=str, required=True,
                        help='Directory with validation token maps (with labels in filename)')

    # Model
    parser.add_argument('--num-tokens', type=int, default=8192,
                        help='Codebook size')
    parser.add_argument('--embed-dim', type=int, default=768,
                        help='Embedding dimension (768 for ViT-Base)')
    parser.add_argument('--num-classes', type=int, default=4,
                        help='Number of classes')
    parser.add_argument('--pretrained-model', type=str,
                        default='google/vit-base-patch16-224',
                        help='Pretrained ViT model from HuggingFace')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze encoder backbone')

    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU')

    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/token_vit',
                        help='Output directory')

    args = parser.parse_args()
    main(args)

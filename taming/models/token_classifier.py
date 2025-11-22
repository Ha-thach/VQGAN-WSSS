"""
Token Classification Models
Multiple architectures for classifying images based on VQ-VAE token indices
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


class TokenEmbedding(nn.Module):
    """Embedding layer for discrete tokens"""

    def __init__(self, num_tokens, embed_dim, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Token indices, shape (B, H, W)
        Returns:
            Token embeddings, shape (B, H, W, D)
        """
        x = self.embedding(x)
        x = self.dropout(x)
        return x


class TransformerTokenClassifier(pl.LightningModule):
    """
    Transformer-based token classifier
    Uses Vision Transformer (ViT) style architecture
    """

    def __init__(
        self,
        num_tokens=8192,
        embed_dim=256,
        num_classes=4,
        num_layers=4,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        multi_label=True,
        learning_rate=1e-4,
        weight_decay=0.01
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.multi_label = multi_label

        # Token embedding
        self.token_embedding = TokenEmbedding(num_tokens, embed_dim, dropout)

        # Positional encoding (learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, 32 * 32, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Loss function
        if multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, tokens):
        """
        Args:
            tokens: Token indices, shape (B, H, W)
        Returns:
            logits: Class logits, shape (B, num_classes)
        """
        B, H, W = tokens.shape

        # Embed tokens
        x = self.token_embedding(tokens)  # (B, H, W, D)
        x = x.reshape(B, H * W, -1)  # (B, N, D)

        # Add positional encoding
        x = x + self.pos_embedding

        # Transformer encoding
        x = self.transformer(x)  # (B, N, D)

        # Global average pooling
        x = x.mean(dim=1)  # (B, D)

        # Classification
        x = self.norm(x)
        logits = self.classifier(x)  # (B, num_classes)

        return logits

    def training_step(self, batch, batch_idx):
        tokens = batch['tokens']
        labels = batch['labels']

        logits = self(tokens)
        loss = self.criterion(logits, labels)

        # Compute metrics
        with torch.no_grad():
            if self.multi_label:
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc = (preds == labels).float().mean()
            else:
                preds = torch.argmax(logits, dim=1)
                acc = (preds == labels).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        tokens = batch['tokens']
        labels = batch['labels']

        logits = self(tokens)
        loss = self.criterion(logits, labels)

        # Compute metrics
        if self.multi_label:
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == labels).float().mean()
        else:
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


class CNNTokenClassifier(pl.LightningModule):
    """
    CNN-based token classifier
    Applies convolutions on token embeddings
    """

    def __init__(
        self,
        num_tokens=8192,
        embed_dim=256,
        num_classes=4,
        hidden_dims=[512, 512, 512],
        dropout=0.1,
        multi_label=True,
        learning_rate=1e-4,
        weight_decay=0.01
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.multi_label = multi_label

        # Token embedding
        self.token_embedding = TokenEmbedding(num_tokens, embed_dim, dropout)

        # CNN layers
        layers = []
        in_channels = embed_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ])
            in_channels = hidden_dim

        self.conv_layers = nn.Sequential(*layers)

        # Global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )

        # Loss function
        if multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, tokens):
        """
        Args:
            tokens: Token indices, shape (B, H, W)
        Returns:
            logits: Class logits, shape (B, num_classes)
        """
        # Embed tokens
        x = self.token_embedding(tokens)  # (B, H, W, D)
        x = x.permute(0, 3, 1, 2)  # (B, D, H, W)

        # CNN encoding
        x = self.conv_layers(x)  # (B, C, H, W)

        # Global pooling
        x = self.gap(x)  # (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)

        # Classification
        logits = self.classifier(x)  # (B, num_classes)

        return logits

    def training_step(self, batch, batch_idx):
        tokens = batch['tokens']
        labels = batch['labels']

        logits = self(tokens)
        loss = self.criterion(logits, labels)

        # Compute metrics
        with torch.no_grad():
            if self.multi_label:
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc = (preds == labels).float().mean()
            else:
                preds = torch.argmax(logits, dim=1)
                acc = (preds == labels).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        tokens = batch['tokens']
        labels = batch['labels']

        logits = self(tokens)
        loss = self.criterion(logits, labels)

        # Compute metrics
        if self.multi_label:
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == labels).float().mean()
        else:
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


class CodebookAwareTokenClassifier(pl.LightningModule):
    """
    Codebook-aware classifier
    Uses pre-computed token-to-class mappings as features
    """

    def __init__(
        self,
        num_tokens=8192,
        embed_dim=256,
        num_classes=4,
        token_class_count_path=None,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        multi_label=True,
        learning_rate=1e-4,
        weight_decay=0.01
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.multi_label = multi_label

        # Load token-to-class statistics
        if token_class_count_path and os.path.exists(token_class_count_path):
            token_stats = np.load(token_class_count_path)  # (num_tokens, num_classes)
            # Normalize to get probabilities
            token_stats = token_stats / (token_stats.sum(axis=1, keepdims=True) + 1e-8)
            self.register_buffer('token_stats', torch.from_numpy(token_stats).float())
        else:
            self.register_buffer('token_stats', None)

        # Token embedding (learnable)
        self.token_embedding = TokenEmbedding(num_tokens, embed_dim, dropout)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 32 * 32, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Feature fusion
        if self.token_stats is not None:
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim + num_classes, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        else:
            self.fusion = None

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Loss function
        if multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, tokens):
        """
        Args:
            tokens: Token indices, shape (B, H, W)
        Returns:
            logits: Class logits, shape (B, num_classes)
        """
        B, H, W = tokens.shape

        # Embed tokens
        x = self.token_embedding(tokens)  # (B, H, W, D)
        x = x.reshape(B, H * W, -1)  # (B, N, D)

        # Add statistical features if available
        if self.token_stats is not None:
            tokens_flat = tokens.reshape(B, H * W)  # (B, N)
            stats_features = self.token_stats[tokens_flat]  # (B, N, num_classes)
            x = torch.cat([x, stats_features], dim=-1)  # (B, N, D + num_classes)
            x = self.fusion(x)  # (B, N, D)

        # Add positional encoding
        x = x + self.pos_embedding

        # Transformer encoding
        x = self.transformer(x)  # (B, N, D)

        # Global average pooling
        x = x.mean(dim=1)  # (B, D)

        # Classification
        x = self.norm(x)
        logits = self.classifier(x)  # (B, num_classes)

        return logits

    def training_step(self, batch, batch_idx):
        tokens = batch['tokens']
        labels = batch['labels']

        logits = self(tokens)
        loss = self.criterion(logits, labels)

        # Compute metrics
        with torch.no_grad():
            if self.multi_label:
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc = (preds == labels).float().mean()
            else:
                preds = torch.argmax(logits, dim=1)
                acc = (preds == labels).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        tokens = batch['tokens']
        labels = batch['labels']

        logits = self(tokens)
        loss = self.criterion(logits, labels)

        # Compute metrics
        if self.multi_label:
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == labels).float().mean()
        else:
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


def build_token_classifier(config):
    """
    Factory function to build classifier from config

    Args:
        config: Configuration dict with keys:
            - model_type: 'transformer', 'cnn', or 'codebook_aware'
            - num_tokens: Codebook size
            - embed_dim: Embedding dimension
            - num_classes: Number of output classes
            - multi_label: Multi-label or single-label
            - ... (model-specific params)
    """
    model_type = config.get('model_type', 'transformer')

    if model_type == 'transformer':
        return TransformerTokenClassifier(**config)
    elif model_type == 'cnn':
        return CNNTokenClassifier(**config)
    elif model_type == 'codebook_aware':
        return CodebookAwareTokenClassifier(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

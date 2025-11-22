"""
Token-Level Classifier for Segmentation
Predicts class probabilities for each token position (32x32)
Then upsamples to pixel-level segmentation (256x256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenViTSegmentation(nn.Module):
    """
    Vision Transformer for token-level segmentation

    Input: Token map [B, H, W] (e.g., 32x32)
    Output: Segmentation map [B, num_classes, H*8, W*8] (e.g., 256x256)
    """

    def __init__(
        self,
        num_tokens=8192,
        embed_dim=768,
        num_classes=4,
        pretrained_model='google/vit-base-patch16-224',
        freeze_backbone=False,
        output_stride=8  # Upsample factor (32 -> 256 = 8x)
    ):
        super().__init__()

        self.output_stride = output_stride

        # Token embedding
        self.token_embedding = nn.Embedding(num_tokens, embed_dim)

        # Positional encoding for 32x32 = 1024 positions
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, embed_dim) * 0.02)

        # Load pretrained ViT
        try:
            from transformers import ViTModel
            print(f"Loading pretrained ViT: {pretrained_model}")
            vit = ViTModel.from_pretrained(pretrained_model)

            self.encoder = vit.encoder
            self.layernorm = vit.layernorm

            print(f"✓ Loaded pretrained ViT encoder")

        except ImportError:
            print("Warning: transformers not installed, using random init")
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

        # Per-token classification head (no pooling!)
        self.token_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # Upsampling to pixel-level
        # 32x32 -> 256x256 = 8x upsampling
        self.upsample = nn.Sequential(
            # 32 -> 64
            nn.ConvTranspose2d(num_classes, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 64 -> 128
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 128 -> 256
            nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, tokens):
        """
        Args:
            tokens: (B, H, W) - e.g., (32, 32, 32)

        Returns:
            seg_logits: (B, num_classes, H*stride, W*stride) - e.g., (32, 4, 256, 256)
            token_logits: (B, H, W, num_classes) - per-token predictions
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

        x = self.layernorm(x)  # (B, N, D)

        # Per-token classification (NO POOLING!)
        token_logits = self.token_classifier(x)  # (B, N, num_classes)
        token_logits = token_logits.reshape(B, H, W, -1)  # (B, H, W, num_classes)

        # Reshape for upsampling: (B, num_classes, H, W)
        token_logits_spatial = token_logits.permute(0, 3, 1, 2)  # (B, num_classes, H, W)

        # Upsample to pixel-level
        seg_logits = self.upsample(token_logits_spatial)  # (B, num_classes, H*8, W*8)

        return seg_logits, token_logits


class TokenViTWithImageClassification(nn.Module):
    """
    Dual-head model: Both image-level classification AND segmentation
    Useful for weakly-supervised training
    """

    def __init__(
        self,
        num_tokens=8192,
        embed_dim=768,
        num_classes=4,
        pretrained_model='google/vit-base-patch16-224',
        freeze_backbone=False,
        output_stride=8
    ):
        super().__init__()

        # Shared backbone
        self.token_embedding = nn.Embedding(num_tokens, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, embed_dim) * 0.02)

        try:
            from transformers import ViTModel
            vit = ViTModel.from_pretrained(pretrained_model)
            self.encoder = vit.encoder
            self.layernorm = vit.layernorm
        except ImportError:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=12,
                dim_feedforward=embed_dim * 4,
                dropout=0.1, activation='gelu',
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)
            self.layernorm = nn.LayerNorm(embed_dim)

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Image-level classification head (with pooling)
        self.image_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # Token-level classification head (no pooling)
        self.token_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # Upsampling
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(num_classes, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, tokens):
        """
        Returns:
            image_logits: (B, num_classes) - image-level classification
            seg_logits: (B, num_classes, 256, 256) - pixel-level segmentation
        """
        B, H, W = tokens.shape

        # Embed
        x = self.token_embedding(tokens)
        x = x.reshape(B, H * W, -1)
        x = x + self.pos_embedding

        # Encode
        if hasattr(self.encoder, 'layer'):
            x = self.encoder(x).last_hidden_state
        else:
            x = self.encoder(x)
        x = self.layernorm(x)  # (B, N, D)

        # Image-level classification (with pooling)
        pooled = x.mean(dim=1)  # (B, D)
        image_logits = self.image_classifier(pooled)  # (B, num_classes)

        # Token-level classification (no pooling)
        token_logits = self.token_classifier(x)  # (B, N, num_classes)
        token_logits = token_logits.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # Upsample
        seg_logits = self.upsample(token_logits)  # (B, num_classes, 256, 256)

        return image_logits, seg_logits


def generate_pseudo_labels_from_image_classifier(
    model,
    tokens,
    threshold=0.5,
    cam_method='simple-cam'
):
    """
    Generate pseudo segmentation labels from image-level classifier
    using Class Activation Mapping (CAM)

    Args:
        model: Trained TokenViT (image classifier)
        tokens: (B, H, W) token map
        threshold: Probability threshold
        cam_method: 'simple-cam' or 'grad-cam'

    Returns:
        pseudo_mask: (B, H, W) with class indices
        class_probs: (B, num_classes) image-level probabilities
    """
    B, H, W = tokens.shape
    device = tokens.device

    # Get image-level predictions
    model.eval()

    # Embed tokens
    x = model.token_embedding(tokens)
    x = x.reshape(B, H * W, -1)
    x = x + model.pos_embedding

    # Encode
    if hasattr(model.encoder, 'layer'):
        features = model.encoder(x).last_hidden_state
    else:
        features = model.encoder(x)
    features = model.layernorm(features)  # (B, N, D)

    # Classify
    pooled = features.mean(dim=1)
    logits = model.classifier(pooled)
    probs = torch.sigmoid(logits)  # (B, num_classes)

    if cam_method == 'simple-cam':
        # Compute per-token class scores (before pooling)
        # This shows which tokens contribute to which classes

        # Apply classifier to each token position
        token_logits = model.classifier(features)  # (B, N, num_classes)
        token_probs = torch.sigmoid(token_logits)  # (B, N, num_classes)

        # Reshape to spatial
        token_probs = token_probs.reshape(B, H, W, -1)  # (B, H, W, num_classes)

        # Generate pseudo mask
        # Assign class if: (1) predicted at image-level AND (2) high token probability
        pseudo_mask = torch.zeros(B, H, W, dtype=torch.long, device=device)

        for b in range(B):
            # Get all token probabilities for this batch
            batch_token_probs = token_probs[b]  # (H, W, num_classes)

            # For each position, assign class with highest probability
            # Only if image-level prediction is positive for that class
            for i in range(H):
                for j in range(W):
                    max_prob = threshold  # Use threshold as minimum
                    max_class = 0  # Background by default

                    for c in range(probs.shape[1]):
                        # Only consider classes predicted at image level
                        if probs[b, c] > threshold:
                            token_prob = batch_token_probs[i, j, c].item()

                            if token_prob > max_prob:
                                max_prob = token_prob
                                max_class = c + 1  # +1 because 0 is background

                    pseudo_mask[b, i, j] = max_class

        return pseudo_mask, probs, token_probs

    else:
        raise ValueError(f"Unknown CAM method: {cam_method}. Only 'simple-cam' is supported.")

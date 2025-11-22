# Converting Classification to Segmentation - Visual Guide

## 🎯 The Problem

You have a **token classifier** that predicts image-level labels:

```
Input:  Image → Token Map [32×32] → TokenViT → [TUM: 0.9, STR: 0.7, LYM: 0.1, NEC: 0.0]
Output: "This image contains TUM and STR"
```

But you want **pixel-level segmentation**:

```
Input:  Image [256×256×3]
Output: Segmentation Mask [256×256] with each pixel labeled as TUM, STR, LYM, NEC, or Background
```

---

## 🔑 Key Insight

The token map **preserves spatial information**:

```
Original Image [256×256]
    ↓ VQ-VAE Encoder
Token Map [32×32]  ← Each token represents an 8×8 patch of the original image
    ↓
Each position (i,j) in the token map corresponds to patch (i,j) in the image
```

So if we can classify **each token position** instead of the **whole image**, we get segmentation!

---

## 📊 Three Methods Explained

### Method 1: Class Activation Mapping (CAM)

**Idea:** Find which tokens the classifier paid attention to for each class

```
Token Map [32×32]
    ↓
TokenViT with attention tracking
    ↓
For each predicted class:
    - Find tokens with high activation
    - Those tokens likely contain that class
    ↓
Activation Map [32×32] for each class
    ↓
Upsample to [256×256]
    ↓
Segmentation Mask
```

**Pseudocode:**
```python
# 1. Forward pass through classifier
features = encoder(token_map)  # [B, 1024, 768]
pooled = features.mean(dim=1)  # [B, 768] ← This loses spatial info!
class_scores = classifier(pooled)  # [B, 4]

# 2. But we can compute per-token scores BEFORE pooling
classifier_weights = classifier.weight  # [4, 768]
per_token_scores = features @ classifier_weights.T  # [B, 1024, 4]
per_token_scores = per_token_scores.reshape(B, 32, 32, 4)

# 3. For each predicted class, find high-scoring tokens
for class_id in predicted_classes:
    activation_map = per_token_scores[:, :, :, class_id]
    # High values = this class is present here

# 4. Upsample 32×32 → 256×256
segmentation = upsample(activation_map)
```

**Pros:** No retraining!
**Cons:** Only highlights "discriminative" regions (parts that convinced the classifier), not all regions

---

### Method 2: Token-Level Prediction

**Idea:** Modify the model to predict class for EACH token position

```
Standard Classifier:
Token Map [32×32]
    ↓
Embed → [32×32×768]
    ↓
Flatten → [1024×768]
    ↓
Transformer
    ↓
*** Global Average Pooling *** ← This loses spatial structure!
    ↓
[768]
    ↓
Classifier → [4] image-level scores


Segmentation Model:
Token Map [32×32]
    ↓
Embed → [32×32×768]
    ↓
Flatten → [1024×768]
    ↓
Transformer
    ↓
*** NO POOLING! *** ← Keep all 1024 token features
    ↓
[1024×768]
    ↓
Reshape → [32×32×768]
    ↓
Per-Token Classifier → [32×32×4]  ← Each token gets 4 class scores
    ↓
Upsample → [256×256×4]
    ↓
Argmax → [256×256] segmentation
```

**Code:**
```python
class TokenViTSegmentation(nn.Module):
    def forward(self, tokens):
        # Embed
        x = self.token_embedding(tokens)  # [B, H, W, D]
        x = x.reshape(B, H*W, D)  # [B, 1024, 768]

        # Encode
        x = self.transformer(x)  # [B, 1024, 768]

        # DON'T POOL - keep spatial structure!
        # x = x.mean(dim=1)  # ← This would lose spatial info

        # Classify each token
        x = self.classifier(x)  # [B, 1024, 4]
        x = x.reshape(B, H, W, 4)  # [B, 32, 32, 4]

        # Upsample
        x = self.upsample(x)  # [B, 256, 256, 4]

        return x
```

**Pros:** Most accurate, end-to-end trainable
**Cons:** Requires pixel-level masks for training

---

### Method 3: Dual-Head Model

**Idea:** Train BOTH image classification AND segmentation heads together

```
Token Map [32×32]
    ↓
Shared Transformer Encoder
    ↓
    ├── Branch 1: Global Pool → Image Classification [4]
    └── Branch 2: Per-Token → Segmentation [32×32×4] → Upsample → [256×256×4]
```

**Why?**
- Branch 1 learns "what classes are present" (image-level supervision)
- Branch 2 learns "where they are located" (pixel-level supervision)
- They share features, so Branch 2 benefits from Branch 1's discriminative learning

**Loss:**
```python
loss = alpha * classification_loss + (1 - alpha) * segmentation_loss
```

**Pros:** Best of both worlds, can use weak supervision
**Cons:** More complex training

---

## 🧠 Understanding the Spatial Mapping

**Critical concept:** Each token position maps to a spatial region

```
Original Image [256×256]
+-------+-------+-------+-------+
| (0,0) | (0,1) | (0,2) | (0,3) |  ← Each cell is 8×8 pixels
+-------+-------+-------+-------+
| (1,0) | (1,1) | (1,2) | (1,3) |
+-------+-------+-------+-------+
...
+-------+-------+-------+-------+
|(31,0) |(31,1) |(31,2) |(31,3) |
+-------+-------+-------+-------+

        ↓ VQ-VAE Encoder (8× downsampling)

Token Map [32×32]
+---+---+---+---+
| 0 | 1 | 2 | 3 |  ← Each token index represents one 8×8 patch
+---+---+---+---+
| 4 | 5 | 6 | 7 |
+---+---+---+---+
...
+---+---+---+---+
|120|121|122|123|
+---+---+---+---+

        ↓ Segmentation Model

Segmentation [32×32]
+-----+-----+-----+-----+
| TUM | TUM | STR | STR |  ← Each position gets a class label
+-----+-----+-----+-----+
| TUM | LYM | LYM | STR |
+-----+-----+-----+-----+
...

        ↓ Upsample (8×)

Final Segmentation [256×256]
Each 8×8 block in output corresponds to one token position
```

---

## 📈 Which Method to Use?

### Use Method 1 (CAM) if:
- ✅ You already have a trained image classifier
- ✅ You need quick results without retraining
- ✅ You only need approximate localization
- ✅ You want to generate pseudo-labels for further training

### Use Method 2 (Token-Level) if:
- ✅ You have pixel-level annotations
- ✅ You need the highest segmentation accuracy
- ✅ You have GPU resources for training
- ✅ This is your final production model

### Use Method 3 (Dual-Head) if:
- ✅ You have both image-level and pixel-level labels
- ✅ You want to leverage weak supervision
- ✅ You need both classification and segmentation outputs
- ✅ You're doing research on weakly-supervised learning

---

## 💡 Common Pitfall

**Wrong approach:**
```python
# This loses spatial information!
x = transformer(tokens)  # [B, 1024, 768]
x = x.mean(dim=1)  # [B, 768] ← SPATIAL INFO LOST!
x = classifier(x)  # [B, 4]
# Now you can't get segmentation because you threw away the spatial structure
```

**Correct approach:**
```python
# Keep spatial information
x = transformer(tokens)  # [B, 1024, 768]
# DON'T pool yet!
x = classifier(x)  # [B, 1024, 4] ← Each of 1024 positions gets 4 class scores
x = x.reshape(B, 32, 32, 4)  # Restore spatial dimensions
x = upsample(x)  # [B, 256, 256, 4]
```

---

## 🎓 Mathematical Formulation

### Image Classification (Standard):
```
f: R^(H×W) → R^C
where H×W = spatial dimensions (e.g., 32×32)
      C = number of classes (e.g., 4)

Output: p ∈ R^C (probability for each class)
```

### Token-Level Segmentation:
```
f: R^(H×W) → R^(H×W×C)

Output: P ∈ R^(H×W×C) (probability for each class at each position)

Then upsample to pixel-level:
g: R^(H×W×C) → R^(sH×sW×C) where s = 8 (upsampling factor)
```

---

## 🚀 Quick Start

### Already have trained classifier? Try CAM immediately:
```bash
python scripts/infer_segmentation_from_classifier.py \
  --method cam \
  --checkpoint outputs/token_vit/best_model.pth \
  --token-map-path token_maps/test/sample[0101].npy \
  --output-dir outputs/segmentation
```

### Want best results? Train segmentation model:
```bash
# Step 1: Train
python token_classification/train_segmentation_model.py \
  --train-token-dir token_maps/train \
  --train-mask-dir data/sub_BCSS_WSSS/training/mask \
  --val-token-dir token_maps/val \
  --val-mask-dir data/sub_BCSS_WSSS/valid/mask \
  --output-dir outputs/token_seg

# Step 2: Infer
python scripts/infer_segmentation_from_classifier.py \
  --method token-level \
  --checkpoint outputs/token_seg/best_model.pth \
  --token-map-path token_maps/test/sample[0101].npy \
  --output-dir outputs/segmentation
```

---

## 📚 Further Reading

- **Class Activation Mapping:** Zhou et al. "Learning Deep Features for Discriminative Localization" (CVPR 2016)
- **Grad-CAM:** Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks" (ICCV 2017)
- **Weakly-Supervised Segmentation:** Papandreou et al. "Weakly- and Semi-Supervised Learning of a DCNN for Semantic Segmentation" (ICCV 2015)
- **Token-Based Segmentation:** Segmenter (ICCV 2021), Mask2Former (CVPR 2022)

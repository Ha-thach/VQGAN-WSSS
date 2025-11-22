# Token Classification for BCSS-WSSS

Multi-label tissue classification (TUM, STR, LYM, NEC) using VQ-VAE token maps and Vision Transformer.

---

## 📋 Quick Start

### Step 1: Generate Token Maps

#### 1.1 Training Data
```bash
python token_classification/generate_token_map.py \
  --config logs/vqgan_gumbel_f8/configs/model.yaml \
  --checkpoint logs/vqgan_gumbel_f8/checkpoints/last.ckpt \
  --img-dir data/sub_BCSS_WSSS/training \
  --output-dir token_maps/train
```

#### 1.2 Validation Data
```bash
python token_classification/generate_token_map_for_valid_test.py \
  --config logs/vqgan_gumbel_f8/configs/model.yaml \
  --checkpoint logs/vqgan_gumbel_f8/checkpoints/last.ckpt \
  --img-dir data/sub_BCSS_WSSS/valid/img \
  --mask-dir data/sub_BCSS_WSSS/valid/mask \
  --output-dir token_maps/val
```

#### 1.3 Test Data
```bash
python token_classification/generate_token_map_for_valid_test.py \
  --config logs/vqgan_gumbel_f8/configs/model.yaml \
  --checkpoint logs/vqgan_gumbel_f8/checkpoints/last.ckpt \
  --img-dir data/sub_BCSS_WSSS/test/img \
  --mask-dir data/sub_BCSS_WSSS/test/mask \
  --output-dir token_maps/test
```

**Output:** Token maps saved as `image_name[0101].npy` where `[0101]` = TUM, STR, LYM, NEC labels

---

### Step 2: Train Classifier

#### 2.1 Basic Training
```bash
python token_classification/train_token_classifier.py \
  --train-token-dir token_maps/train \
  --val-token-dir token_maps/val \
  --output-dir outputs/token_vit
```

#### 2.2 Full Training (all parameters)
```bash
python token_classification/train_token_classifier.py \
  --train-token-dir token_maps/train \
  --val-token-dir token_maps/val \
  --num-tokens 8192 \
  --embed-dim 768 \
  --num-classes 4 \
  --pretrained-model google/vit-base-patch16-224 \
  --batch-size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --num-workers 4 \
  --output-dir outputs/token_vit
```

#### 2.3 Frozen Backbone (faster)
```bash
python token_classification/train_token_classifier.py \
  --train-token-dir token_maps/train \
  --val-token-dir token_maps/val \
  --freeze-backbone \
  --epochs 30 \
  --output-dir outputs/token_vit_frozen
```

#### 2.4 Force CPU
```bash
python token_classification/train_token_classifier.py \
  --train-token-dir token_maps/train \
  --val-token-dir token_maps/val \
  --cpu \
  --output-dir outputs/token_vit_cpu
```

**Output:**
- `best_model.pth`: Best checkpoint based on validation F1
- `last_model.pth`: Latest checkpoint
- `history.json`: Training history

---

### Step 3: Evaluate Model

```bash
python token_classification/evaluate_token_classifier.py \
  --test-token-dir token_maps/test \
  --checkpoint outputs/token_vit/best_model.pth \
  --output-dir outputs/token_vit/test_results
```

**Output:**
- `test_metrics.json`: Detailed metrics (precision, recall, F1, IoU)
- `predictions.csv`: Per-image predictions with probabilities
- `predictions.npy`, `labels.npy`, `probabilities.npy`: Raw numpy arrays

---

## 🎯 From Classification to Segmentation

The token classifier predicts **image-level labels** (which classes exist in the image). To get **pixel-level segmentation**, use one of these methods:

### Method 1: CAM (Class Activation Mapping) ⭐ Recommended for Quick Start

**Use your existing trained image classifier** - no retraining needed!

```bash
python scripts/infer_segmentation_from_classifier.py \
  --method cam \
  --checkpoint outputs/token_vit/best_model.pth \
  --token-map-path token_maps/test/sample[0101].npy \
  --output-dir outputs/segmentation
```

**Pros:**
- No retraining required
- Works with existing image classifier
- Fast inference

**Cons:**
- Only highlights discriminative regions
- Lower accuracy than full segmentation

**How it works:**
1. Use classifier's attention weights to identify important spatial regions
2. For each predicted class, find which tokens contributed most
3. Upsample token-level activations to pixel-level

---

### Method 2: Token-Level Segmentation (Best Accuracy)

Train a **per-token classifier** that predicts class for each 32×32 token position, then upsample to 256×256 pixels.

#### Step 1: Train Segmentation Model

```bash
python token_classification/train_segmentation_model.py \
  --train-token-dir token_maps/train \
  --train-mask-dir data/sub_BCSS_WSSS/training/mask \
  --val-token-dir token_maps/val \
  --val-mask-dir data/sub_BCSS_WSSS/valid/mask \
  --output-dir outputs/token_seg
```

#### Step 2: Generate Segmentation

```bash
python scripts/infer_segmentation_from_classifier.py \
  --method token-level \
  --checkpoint outputs/token_seg/best_model.pth \
  --token-map-path token_maps/test/sample[0101].npy \
  --output-dir outputs/segmentation
```

**Pros:**
- Most accurate segmentation
- End-to-end trainable
- Preserves spatial structure

**Cons:**
- Requires pixel-level masks for training
- More training time

**Architecture:**
```
Token Map [32, 32]
    ↓
Token Embedding [32, 32, 768]
    ↓
ViT Encoder [32×32, 768]
    ↓
Per-Token Classifier [32, 32, 4]  ← No pooling!
    ↓
Transposed Conv Upsampling
    ↓
Segmentation [256, 256, 4]
```

---

### Method 3: Batch Processing

Process entire test set:

```bash
python scripts/infer_segmentation_from_classifier.py \
  --method batch \
  --checkpoint outputs/token_vit/best_model.pth \
  --token-dir token_maps/test \
  --output-dir outputs/segmentation_batch
```

**Output:**
- `*_seg.png`: RGB colored segmentation
- `*_seg.npy`: Class indices array

**Color Scheme:**
- Red: TUM (Tumor)
- Green: STR (Stroma)
- Blue: LYM (Lymphocytic infiltrate)
- Yellow: NEC (Necrosis)
- Black: Background

---

## 🏗️ Architecture

### TokenViT Model
```
Token Indices [B, 32, 32]
    ↓
Token Embedding [B, 1024, 768]
    ↓
+ Positional Encoding [1, 1024, 768]
    ↓
Pretrained ViT Encoder (12 layers)
    ↓
LayerNorm
    ↓
Global Average Pooling
    ↓
MLP (768 → 384 → 4)
    ↓
Logits [B, 4]
```

---

## 📊 Arguments Reference

### Data Arguments
| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--train-token-dir` | str | ✅ | - | Training token maps directory |
| `--val-token-dir` | str | ✅ | - | Validation token maps directory |

### Model Arguments
| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--num-tokens` | int | ❌ | 8192 | Codebook size |
| `--embed-dim` | int | ❌ | 768 | Embedding dimension (768 for ViT-Base) |
| `--num-classes` | int | ❌ | 4 | Number of classes |
| `--pretrained-model` | str | ❌ | google/vit-base-patch16-224 | HuggingFace ViT model |
| `--freeze-backbone` | flag | ❌ | False | Freeze encoder, only train classifier |

### Training Arguments
| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--batch-size` | int | ❌ | 32 | Batch size |
| `--epochs` | int | ❌ | 50 | Number of epochs |
| `--lr` | float | ❌ | 1e-4 | Learning rate |
| `--weight-decay` | float | ❌ | 0.01 | Weight decay |
| `--num-workers` | int | ❌ | 4 | DataLoader workers |
| `--cpu` | flag | ❌ | False | Force CPU (auto-detects CUDA otherwise) |

### Output Arguments
| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--output-dir` | str | ❌ | outputs/token_vit | Output directory |

---

## 📁 Directory Structure

```
token_maps/
├── train/
│   ├── TCGA-XXX[1010].npy
│   └── ...
├── val/
│   ├── TCGA-XXX[0110].npy
│   └── ...
└── test/
    ├── TCGA-XXX[1001].npy
    └── ...

outputs/token_vit/
├── best_model.pth
├── last_model.pth
└── history.json
```

---

## 🎯 Label Format

### Filename Convention
```
image_name[abcd].npy
           ↓↓↓↓
           TUM STR LYM NEC

Examples:
- [1000].npy → only TUM
- [0110].npy → STR + LYM
- [1111].npy → all 4 classes
- [0000].npy → no classes (background only)
```

### Palette Indices (for masks)
```
P mode palette indices:
0 = TUM (Tumor)
1 = STR (Stroma)
2 = LYM (Lymphocytic infiltrate)
3 = NEC (Necrosis)
4 = Background
```

---

## 🔧 Implementation Details

### Loss Function
- `BCEWithLogitsLoss` for multi-label classification

### Optimizer
- `AdamW` with weight decay

### Scheduler
- `CosineAnnealingLR` (T_max=epochs, eta_min=1e-6)

### Metrics
- **Training**: Loss, Exact Match Accuracy
- **Validation**: Loss, Exact Match, Precision, Recall, F1 (macro-averaged)

### Early Stopping
- Saves best model based on validation F1 score

---

## 💡 Tips

### Memory Issues
```bash
# Reduce batch size
--batch-size 16

# Freeze backbone
--freeze-backbone

# Use CPU
--cpu
```

### Speed Up Training
```bash
# Freeze backbone (only train classifier head)
--freeze-backbone --epochs 30

# Reduce workers if I/O bound
--num-workers 2
```

### Better Performance
```bash
# More epochs
--epochs 100

# Lower learning rate
--lr 5e-5

# Larger batch size (if memory allows)
--batch-size 64
```

---

## 📝 Scripts Overview

| Script | Purpose |
|--------|---------|
| **Token Map Generation** | |
| `generate_token_map.py` | Generate token maps from images (for training data) |
| `generate_token_map_for_valid_test.py` | Generate token maps with labels from masks (for val/test) |
| **Image Classification** | |
| `train_token_classifier.py` | Train TokenViT image classifier |
| `evaluate_token_classifier.py` | Evaluate classifier on test set |
| `dataset.py` | Dataset classes for loading token maps |
| **Segmentation** | |
| `train_segmentation_model.py` | Train token-level segmentation model |
| `infer_segmentation_from_classifier.py` | Convert classification to segmentation (3 methods) |
| **Models** | |
| `taming/models/token_classifier.py` | Image classification models |
| `taming/models/token_classifier_segmentation.py` | Segmentation models (TokenViTSegmentation) |

---

## 🔄 Comparison: Classification vs Segmentation

| Aspect | Image Classification | Token-Level Segmentation | CAM-Based |
|--------|---------------------|-------------------------|-----------|
| **Output** | Image-level labels | Pixel-level masks | Pseudo pixel-level |
| **Training Data** | Token maps + labels | Token maps + pixel masks | Token maps + labels |
| **Accuracy** | High for presence/absence | Highest for localization | Moderate |
| **Training Time** | Fast (30-50 epochs) | Moderate (50-100 epochs) | None (uses existing) |
| **Inference Speed** | Very fast | Fast | Fast |
| **Use Case** | "Does image contain TUM?" | "Where exactly is TUM?" | Quick localization |
| **Best For** | Initial screening | Final segmentation | Prototyping, pseudo-labels |

---

## 🎓 Class Definitions

```python
CLASSES = ["TUM", "STR", "LYM", "NEC"]

MASK_CLASSES = {
    0: "TUM",      # Tumor
    1: "STR",      # Stroma
    2: "LYM",      # Lymphocytic infiltrate
    3: "NEC",      # Necrosis
    4: "BACKGROUND"
}
```

---

## ✅ Checklist

Before training:
- [ ] VQ-VAE model checkpoint exists
- [ ] Token maps generated for train/val/test
- [ ] All token files have labels in filename `[xxxx]`
- [ ] Class distribution checked (not too imbalanced)
- [ ] GPU available (or use `--cpu` flag)

During training:
- [ ] Training loss decreasing
- [ ] Validation metrics improving
- [ ] No NaN losses
- [ ] Best model being saved

After training:
- [ ] Check `history.json` for convergence
- [ ] Validate best F1 score is reasonable
- [ ] Test on held-out test set

---

## 🚀 Next Steps

- [x] ✅ Implement evaluation script (`evaluate_token_classifier.py`)
- [x] ✅ Add inference pipeline for segmentation (CAM and token-level methods)
- [x] ✅ Decode classification activations to segmentation
- [ ] Visualize attention maps from transformer layers
- [ ] Analyze misclassified samples in detail
- [ ] Try ensemble methods (multiple models/checkpoints)
- [ ] Implement data augmentation for segmentation training
- [ ] Add multi-scale inference for better segmentation
- [ ] Export to ONNX for production deployment

---

## 📊 Typical Workflow

### For Image-Level Classification Only:
```bash
# 1. Generate token maps
python token_classification/generate_token_map_for_valid_test.py ...

# 2. Train classifier
python token_classification/train_token_classifier.py ...

# 3. Evaluate
python token_classification/evaluate_token_classifier.py ...
```

### For Pixel-Level Segmentation (Method 1 - Quick):
```bash
# 1-3. Same as above

# 4. Generate segmentation using CAM
python scripts/infer_segmentation_from_classifier.py --method cam ...
```

### For Pixel-Level Segmentation (Method 2 - Best Quality):
```bash
# 1. Generate token maps

# 2. Train segmentation model
python token_classification/train_segmentation_model.py ...

# 3. Generate segmentation
python scripts/infer_segmentation_from_classifier.py --method token-level ...
```

---

## 🤝 Contributing

If you find bugs or have suggestions:
1. Check existing issues
2. Create detailed bug report with:
   - Command used
   - Error message
   - Python/PyTorch versions
   - Expected vs actual behavior

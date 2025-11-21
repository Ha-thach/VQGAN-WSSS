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
# TODO: Add evaluation script
```

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
| `generate_token_map.py` | Generate token maps from images (for training data) |
| `generate_token_map_for_valid_test.py` | Generate token maps with labels from masks (for val/test) |
| `train_token_classifier.py` | Train TokenViT classifier |
| `dataset.py` | Dataset classes for loading token maps |

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

1. Implement evaluation script
2. Add inference pipeline for new images
3. Visualize attention maps
4. Analyze misclassified samples
5. Try ensemble methods

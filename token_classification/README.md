# Token Classification Module

Token-based classification cho BCSS-WSSS dataset.

## 📁 Structure

```
token_classification/
├── dataset.py                    # Dataset classes
├── generate_token_mask.py        # Generate token maps từ images
├── test_dataset.py              # Test dataset loading
└── README.md                     # This file
```

---

## 🔄 Workflow

### Step 1: Generate Token Maps

Chuyển images thành token maps (.npy) bằng pretrained VQ-VAE:

```bash
# Generate tokens cho training data
python token_classification/generate_token_mask.py \
    --config logs/vqgan_gumbel_f8/configs/model.yaml \
    --checkpoint logs/vqgan_gumbel_f8/checkpoints/last.ckpt \
    --img-dir data/sub_BCSS_WSSS/training \
    --output-dir token_maps/train

# Generate tokens cho validation data
python token_classification/generate_token_mask.py \
    --config logs/vqgan_gumbel_f8/configs/model.yaml \
    --checkpoint logs/vqgan_gumbel_f8/checkpoints/last.ckpt \
    --img-dir data/sub_BCSS_WSSS/validation/img \
    --output-dir token_maps/valid

# Generate tokens cho test data
python token_classification/generate_token_mask.py \
    --config logs/vqgan_gumbel_f8/configs/model.yaml \
    --checkpoint logs/vqgan_gumbel_f8/checkpoints/last.ckpt \
    --img-dir data/sub_BCSS_WSSS/test/img \
    --output-dir token_maps/test
```

**Output:**
```
token_maps/
├── train/
│   ├── TCGA-A1-A0SK[0101].npy
│   ├── TCGA-A2-A0SV[1010].npy
│   └── ...
├── valid/
│   ├── image1.npy
│   └── ...
└── test/
    ├── image1.npy
    └── ...
```

---

### Step 2: Test Dataset Loading

Verify token maps được load đúng:

```bash
# Test train dataset
python token_classification/test_dataset.py \
    --token-dir token_maps/train \
    --split train

# Test validation dataset (cần data-root cho masks)
python token_classification/test_dataset.py \
    --token-dir token_maps/valid \
    --data-root data/sub_BCSS_WSSS \
    --split valid

# Test test dataset
python token_classification/test_dataset.py \
    --token-dir token_maps/test \
    --data-root data/sub_BCSS_WSSS \
    --split test
```

---

### Step 3: Use in Training

```python
from token_classification.dataset import build_token_dataset
from torch.utils.data import DataLoader

# Build datasets
train_ds = build_token_dataset(
    split='train',
    token_dir='token_maps/train'
)

val_ds = build_token_dataset(
    split='valid',
    token_dir='token_maps/valid',
    data_root='data/sub_BCSS_WSSS'
)

# Create dataloaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# Train model
for batch in train_loader:
    tokens = batch['tokens']       # (B, H, W) - e.g., (32, 32, 32)
    labels = batch['cls_label']    # (B, 4) - multi-label binary

    # Your training code here
    # logits = model(tokens)
    # loss = criterion(logits, labels)
```

---

## 📊 Dataset Classes

### `TokenMapDatasetBase`
Base class cho tất cả token map datasets.

**Input:**
- `token_dir`: Folder chứa token .npy files
- `data_root`: Original BCSS-WSSS data (cho masks nếu cần)
- `has_mask`: Whether to use masks
- `extract_label_from_mask`: Extract labels từ mask hay filename

**Output (per sample):**
```python
{
    'tokens': Tensor[H, W],        # Token indices (e.g., 32×32)
    'cls_label': Tensor[4],        # Multi-label binary [TUM, STR, LYM, NEC]
    'cls_names': List[str],        # Active class names
    'file_path_': str              # Filename
}
```

---

### `TokenMapTrainDataset`
Training dataset - labels từ filename `[0101].npy`

```python
train_ds = TokenMapTrainDataset(
    token_dir='token_maps/train'
)
```

---

### `TokenMapValidationDataset`
Validation dataset - labels từ mask files

```python
val_ds = TokenMapValidationDataset(
    token_dir='token_maps/valid',
    data_root='data/sub_BCSS_WSSS/validation'
)
```

---

### `TokenMapTestDataset`
Test dataset - labels từ mask files

```python
test_ds = TokenMapTestDataset(
    token_dir='token_maps/test',
    data_root='data/sub_BCSS_WSSS/test'
)
```

---

## 🎯 Key Differences vs Image Dataset

| Aspect | Image Dataset | Token Dataset |
|--------|---------------|---------------|
| Input | Images (.png) | Token maps (.npy) |
| Size | ~500KB/image | ~4KB/token map |
| Shape | (256, 256, 3) | (32, 32) |
| Type | uint8 RGB | int64 indices |
| Range | [0, 255] | [0, 8191] |
| Transforms | Albumentations | None (already encoded) |
| Speed | Slower (decode) | Faster (direct load) |

---

## ⚡ Performance Tips

### 1. Pre-generate ALL token maps
```bash
# Batch generate tất cả splits
for split in training validation/img test/img; do
    python token_classification/generate_token_mask.py \
        --config logs/vqgan_gumbel_f8/configs/model.yaml \
        --checkpoint logs/vqgan_gumbel_f8/checkpoints/last.ckpt \
        --img-dir data/sub_BCSS_WSSS/$split \
        --output-dir token_maps/$(basename $split)
done
```

### 2. Use multiple workers
```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,      # ← Parallel loading
    pin_memory=True     # ← Faster GPU transfer
)
```

### 3. Cache in RAM (small datasets)
```python
# Load all token maps into memory
class CachedTokenDataset(TokenMapDatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = [np.load(p) for p in self.token_paths]

    def __getitem__(self, idx):
        tokens = torch.from_numpy(self.cache[idx]).long()
        # ... rest of code
```

---

## 🔍 Troubleshooting

### Issue: "No token .npy files found"
**Solution:** Generate token maps first với `generate_token_mask.py`

### Issue: "Filename missing class label"
**Solution:**
- Train data: Filenames phải có pattern `[0101]`
- Valid/Test: Cần provide `data_root` để extract từ masks

### Issue: Token shape mismatch
**Solution:** Check token maps đều có cùng shape (32×32 for Gumbel-8)

```python
# Verify token shapes
import numpy as np
from pathlib import Path

token_dir = Path('token_maps/train')
for npy_file in token_dir.glob('*.npy'):
    tokens = np.load(npy_file)
    if tokens.shape != (32, 32):
        print(f"Wrong shape: {npy_file.name} -> {tokens.shape}")
```

---

## 📝 Notes

1. **Token maps nhỏ hơn 100x so với images** → Faster loading
2. **Không cần augmentation** → Token indices đã là discrete representation
3. **Labels giống hệt image dataset** → Drop-in replacement
4. **Compatible với existing training code** → Chỉ cần thay dataset

---

## 🎓 Example: Complete Pipeline

```bash
# 1. Generate token maps
python token_classification/generate_token_mask.py \
    --config logs/vqgan_gumbel_f8/configs/model.yaml \
    --checkpoint logs/vqgan_gumbel_f8/checkpoints/last.ckpt \
    --img-dir data/sub_BCSS_WSSS/training \
    --output-dir token_maps/train

# 2. Test dataset
python token_classification/test_dataset.py \
    --token-dir token_maps/train \
    --split train

# 3. Train classifier (sẽ implement sau)
# python train_token_classifier.py \
#     --token-dir token_maps/train \
#     --config configs/token_classifier.yaml
```

---

## 🚀 Next Steps

Sau khi có token maps và dataset:
1. Implement token classifier models (Transformer/CNN)
2. Training pipeline
3. Evaluation metrics
4. Visualization tools

Happy coding! 🎉

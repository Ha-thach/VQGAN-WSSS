# VQGAN Training on BCSS-WSSS Dataset

Quick guide to train/finetune VQGAN on BCSS-WSSS dataset.

---

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n taming python=3.10
conda activate taming

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your BCSS-WSSS data in:
```
data/BCSS_WSSS/
├── training/
│   └── *.png (with labels [0101] in filename)
├── valid/
│   ├── img/*.png
│   └── mask/*.png
└── test/
    ├── img/*.png
    └── mask/*.png
```

### 3. Edit Config
1. Download the checkpoint of pre-trained model 
- Go to scripts/download_pretrained.ipynb -> download
- Recheck the ckpt path in the yaml file: model/params/ckpt_path

2. Update data paths at "img_root" in the yaml file `configsbcss_wsss_vqgan_finetune_gumbel.yaml`in case you wanna finetune with Gumbel (freeze encoder)

```yaml
data:
  params:
    train:
      params:
        img_root: /path/to/data/BCSS_WSSS/training
    validation:
      params:
        img_root: /path/to/data/BCSS_WSSS/valid
    test:
      params:
        img_root: /path/to/data/BCSS_WSSS/test
```

### 4. Train

**Option A: Train from scratch**
```bash
python main.py -t --base configs/bcss_wsss_vqgan_finetune_gumbel.yaml
```
### 5. Check logging
```bash
tensorboard --logdir logs/bcss_wsss_vqgan_finetune_gumbel_freeze_encoder
```
---

## References

- Original VQGAN: [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)
- Paper: [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841)

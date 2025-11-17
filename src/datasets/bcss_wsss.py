# taming/data/bcss_wsss.py
import os
import re
import glob
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ------------------------------------------------------------------
# Shared transforms (you can tweak these later)
# ------------------------------------------------------------------
def get_train_transform(size=256):
    return A.Compose([
        A.RandomCrop(width=size, height=size),           # Random crop (for training diversity)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # → [-1, 1] for VQGAN
        ToTensorV2(),
    ])


def get_val_transform(size=256):
    return A.Compose([
        A.CenterCrop(width=size, height=size),           # Fixed center crop for val/test
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


# # ------------------------------------------------------------------
# # 1. Training Dataset (image + image-level label only)
# # ------------------------------------------------------------------
# class BCSSTrainingDataset(Dataset):
#     CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]   # 5 classes

#     def __init__(self, img_root, size=256, transform=None):
#         """
#         img_root: /project/hnguyen2/mvu9/datasets/weakly_sup_segmentation_datasets/BCSS-WSSS/training
#         """
#         self.img_root = img_root
#         self.size = size
#         self.transform = transform or get_train_transform(size)

#         self.samples = self._collect_samples()

#     def _collect_samples(self):
#         img_paths = sorted(glob.glob(os.path.join(self.img_root, "img", "*.png")))
#         samples = []
#         for p in img_paths:
#             # Extract class presence from filename like: ...[1,0,1,0,0].png
#             match = re.search(r"\[([0-9,]+)\]", p)
#             if not match:
#                 continue
#             cls_str = match.group(1)
#             cls_label = np.array([int(x) for x in cls_str.split(",")], dtype=np.float32)
#             samples.append((p, cls_label))
#         return samples

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         img_path, cls_label = self.samples[idx]
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         transformed = self.transform(image=img)
#         img = transformed["image"]                     # [C, H, W] in [-1, 1]

#         return {
#             "image": img,
#             "cls_label": torch.from_numpy(cls_label),  # [5]
#             "path": os.path.basename(img_path)
#         }

class BCSSTrainingDataset(Dataset):
    CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]

    def __init__(self, img_root, size=256, transform=None):
        self.img_root = img_root
        self.size = size
        self.transform = transform or get_train_transform(size)
        self.samples = self._collect_samples()

    def _collect_samples(self):
        img_paths = sorted(glob.glob(os.path.join(self.img_root, "*.png")))
        print(f"Found {len(img_paths)} image files")

        samples = []
        for p in img_paths:
            filename = os.path.basename(p)
            match = re.search(r'\+\d+\[(\d{4})\]', filename)
            if not match:
                continue
            digits = match.group(1)           # "1101"
            labels = [int(c) for c in digits] # [1,1,0,1]
            labels = labels + [0]             # → [1,1,0,1,0]
            samples.append((p, np.array(labels, dtype=np.float32)))
            
        print(f"Loaded {len(samples)} valid samples")
        return samples 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_label = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0

        return {
            "image": img,
            "cls_label": torch.from_numpy(cls_label),
            "path": os.path.basename(img_path)
        } 
# ------------------------------------------------------------------
# 2. Validation / Test Dataset (image + pixel-wise mask)
# ------------------------------------------------------------------
class BCSSValidationDataset(Dataset):
    CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]

    def __init__(self, data_root, split="valid", size=256, transform=None):
        """
        data_root: /project/hnguyen2/mvu9/datasets/weakly_sup_segmentation_datasets/BCSS-WSSS/
        split: "valid" or "test"
        """
        assert split in {"valid", "test"}
        self.img_dir = os.path.join(data_root, split, "img")
        self.mask_dir = os.path.join(data_root, split, "mask")
        self.size = size
        self.transform = transform or get_val_transform(size)

        self.samples = self._collect_samples()

    def _collect_samples(self):
        mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, "*.png")))
        samples = []
        for mask_path in mask_paths:
            img_name = os.path.basename(mask_path)
            img_path = os.path.join(self.img_dir, img_name)
            if not os.path.exists(img_path):
                continue
            samples.append((img_path, mask_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.array(Image.open(mask_path))          # values 0–4

        # Compute image-level label from mask (for consistency with training)
        unique = np.unique(mask)
        if 4 in unique:  # sometimes background=4
            unique = unique[unique != 4]
        cls_label = np.zeros(5, dtype=np.float32)
        cls_label[unique] = 1.0

        transformed = self.transform(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"].long()               # [H, W]

        return {
            "image": img,
            "mask": mask,
            "cls_label": torch.from_numpy(cls_label),
            "path": os.path.basename(img_path)
        }


# ------------------------------------------------------------------
# 3. Optional: Weakly-supervised dataset (pseudo labels)
# ------------------------------------------------------------------
class BCSSWSSSDataset(Dataset):
    """For later weakly-supervised training using pseudo masks"""
    CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]

    def __init__(self, img_root, pseudo_dir="pseudo_label", size=256, transform=None):
        self.img_dir = os.path.join(img_root, "img")
        self.pseudo_dir = os.path.join(img_root, pseudo_dir)
        self.size = size
        self.transform = transform or get_train_transform(size)

        self.samples = self._collect_samples()

    def _collect_samples(self):
        img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
        samples = []
        for p in img_paths:
            name = os.path.basename(p)
            pseudo_path = os.path.join(self.pseudo_dir, name)
            if not os.path.exists(pseudo_path):
                continue
            # Extract class label from filename (same as training set)
            match = re.search(r"\[([0-9,]+)\]", p)
            if not match:
                continue
            cls_label = np.array([int(x) for x in match.group(1).split(",")], dtype=np.float32)
            samples.append((p, pseudo_path, cls_label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pseudo_path, cls_label = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.array(Image.open(pseudo_path))

        transformed = self.transform(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"].long()

        return {
            "image": img,
            "mask": mask,
            "cls_label": torch.from_numpy(cls_label),
            "path": os.path.basename(img_path)
        } 
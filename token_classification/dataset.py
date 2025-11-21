"""
Token Map Dataset for BCSS-WSSS Token Classification
Input: Token maps (.npy files with discrete indices)
Output: Token indices + class labels
"""

import os
import re
import glob
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image


CLASSES = ["TUM", "STR", "LYM", "NEC"]

MASK_CLASSES = {
    0: "BACKGROUND",
    1: "TUM",
    2: "STR",
    3: "LYM",
    4: "NEC"
}


def get_class_labels_from_filename(filename):
    """
    Extract class labels from filename: *[0101].png or *[0101].npy

    Args:
        filename: String like "TCGA-XX-XXXX[0101].png"

    Returns:
        cls_label: numpy array [4] with binary values
        cls_names: list of active class names
    """
    term_split = re.split(r"\[|\]", filename)

    if len(term_split) < 2:
        raise ValueError(f"Filename missing class label [xxxx]: {filename}")

    class_bits = term_split[1]

    if len(class_bits) != 4:
        raise ValueError(f"Expected 4 class bits, got {len(class_bits)}: {filename}")

    cls_label = np.array([int(x) for x in class_bits], dtype=np.float32)
    cls_names = [CLASSES[i] for i, v in enumerate(cls_label) if v == 1]

    return cls_label, cls_names


def get_class_labels_from_mask(mask):
    """
    Extract class labels from segmentation mask

    Args:
        mask: numpy array [H, W] with class indices

    Returns:
        cls_label: numpy array [4] with binary values
        cls_names: list of active class names
    """
    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes > 0]

    cls_label = np.zeros(4, dtype=np.float32)
    for class_id in unique_classes:
        if 1 <= class_id <= 4:
            cls_label[class_id - 1] = 1

    cls_names = [CLASSES[i] for i, v in enumerate(cls_label) if v == 1]

    return cls_label, cls_names


class TokenMapDatasetBase(Dataset):
    """
    Base dataset for token map classification
    Loads pre-generated token maps (.npy) and their labels
    """

    def __init__(
        self,
        token_dir,
        data_root=None,
        has_mask=False,
        extract_label_from_mask=False,
        return_raw_labels=False
    ):
        """
        Args:
            token_dir: Directory containing token .npy files
            data_root: Original data directory (for masks if needed)
            has_mask: Whether to use masks for labels
            extract_label_from_mask: Extract labels from mask instead of filename
            return_raw_labels: If True, return raw filename for manual parsing
        """
        self.token_dir = Path(token_dir)
        self.data_root = Path(data_root) if data_root else None
        self.has_mask = has_mask
        self.extract_label_from_mask = extract_label_from_mask
        self.return_raw_labels = return_raw_labels

        # Load data
        self._load_data()

    def _load_data(self):
        """Load token map paths and extract labels"""

        # Get all .npy files
        all_token_paths = sorted(list(self.token_dir.glob("*.npy")))

        # Filter out non-token files
        all_token_paths = [
            p for p in all_token_paths
            if not p.name.startswith('codebook')
            and not p.name.startswith('token_histogram')
        ]

        if len(all_token_paths) == 0:
            raise RuntimeError(f"No token .npy files found in {self.token_dir}")

        print(f"Found {len(all_token_paths)} token files in {self.token_dir}")

        # Validate and extract labels
        valid_token_paths = []
        self.cls_labels = []
        self.cls_names = []

        # First pass: validate token shapes
        print("  Validating token shapes...")
        token_shapes = []
        for token_path in all_token_paths[:min(10, len(all_token_paths))]:
            try:
                token_map = np.load(token_path)
                token_shapes.append(token_map.shape)
            except:
                continue

        if not token_shapes:
            raise RuntimeError("Could not load any token maps")

        # Expected shape (most common)
        expected_shape = max(set(token_shapes), key=token_shapes.count)
        print(f"  Expected token shape: {expected_shape}")

        if self.extract_label_from_mask:
            # Extract from mask files
            print("  Extracting labels from masks...")

            if not self.data_root:
                raise ValueError("data_root required when extract_label_from_mask=True")

            # Setup mask root
            if self.has_mask:
                self.mask_root = self.data_root / "mask"
            else:
                raise ValueError("has_mask must be True when extracting from mask")

            for token_path in all_token_paths:
                try:
                    # Validate token shape
                    token_map = np.load(token_path)
                    if token_map.shape != expected_shape:
                        print(f"Warning: Shape mismatch {token_path.name}: {token_map.shape}")
                        continue

                    # Match token file to mask file
                    token_name = token_path.stem

                    # Try to find matching mask
                    mask_path = self.mask_root / f"{token_name}.png"

                    if not mask_path.exists():
                        # Try without extension variations
                        possible_masks = list(self.mask_root.glob(f"{token_name}.*"))
                        if possible_masks:
                            mask_path = possible_masks[0]
                        else:
                            print(f"Warning: No mask found for {token_name}, skipping")
                            continue

                    # Load mask
                    mask = Image.open(mask_path).convert("L")
                    mask = np.array(mask, dtype=np.int64)

                    # Extract labels
                    cls_label, cls_names = get_class_labels_from_mask(mask)

                    # Add to valid list
                    valid_token_paths.append(token_path)
                    self.cls_labels.append(cls_label)
                    self.cls_names.append(cls_names)

                except Exception as e:
                    print(f"Warning: Error processing {token_path.name}: {e}")
                    continue

        else:
            # Extract from filename
            print("  Extracting labels from filenames...")

            for token_path in all_token_paths:
                try:
                    # Validate token shape
                    token_map = np.load(token_path)
                    if token_map.shape != expected_shape:
                        print(f"Warning: Shape mismatch {token_path.name}: {token_map.shape}")
                        continue

                    # Extract label from filename
                    cls_label, cls_names = get_class_labels_from_filename(token_path.name)

                    # Add to valid list
                    valid_token_paths.append(token_path)
                    self.cls_labels.append(cls_label)
                    self.cls_names.append(cls_names)

                except ValueError as e:
                    if self.return_raw_labels:
                        # For unlabeled data, return zeros
                        valid_token_paths.append(token_path)
                        self.cls_labels.append(np.zeros(4, dtype=np.float32))
                        self.cls_names.append([])
                    else:
                        print(f"Warning: {e}")
                        continue
                except Exception as e:
                    print(f"Warning: Error processing {token_path.name}: {e}")
                    continue

        # Set valid paths
        self.token_paths = valid_token_paths

        if len(self.token_paths) == 0:
            raise RuntimeError("No valid token maps found")

        print(f"  Loaded {len(self.token_paths)} valid samples")

        # Sanity check
        assert len(self.token_paths) == len(self.cls_labels), \
            f"Mismatch: {len(self.token_paths)} paths != {len(self.cls_labels)} labels"

        # Print class distribution
        if self.cls_labels:
            cls_counts = np.array(self.cls_labels).sum(axis=0).astype(int)
            for cls_name, count in zip(CLASSES, cls_counts):
                print(f"  {cls_name}: {count} samples")

    def __len__(self):
        return len(self.token_paths)

    def __getitem__(self, index):
        """
        Returns:
            dict with keys:
                - tokens: Tensor [H, W] with token indices (typically 32x32)
                - cls_label: Tensor [4] binary multi-label
                - cls_names: List[str] active class names
                - file_path_: str filename
        """
        token_path = self.token_paths[index]

        # Load token map
        token_map = np.load(token_path)
        tokens = torch.from_numpy(token_map).long()

        # Get labels
        cls_label = torch.tensor(self.cls_labels[index], dtype=torch.float32)
        cls_names = self.cls_names[index]

        return {
            "tokens": tokens,           # [H, W] - e.g., (32, 32)
            "cls_label": cls_label,     # [4] - binary multi-label
            "cls_names": cls_names,     # List[str]
            "file_path_": token_path.name
        }


class TokenMapTrainDataset(TokenMapDatasetBase):
    """
    Training dataset - token maps with labels in filename
    Example: TCGA-XX-XXXX[0101].npy
    """

    def __init__(self, token_dir, **kwargs):
        super().__init__(
            token_dir=token_dir,
            has_mask=False,
            extract_label_from_mask=False,  # Extract from filename
            **kwargs
        )


class TokenMapValidationDataset(TokenMapDatasetBase):
    """
    Validation dataset - token maps with labels in filename
    Example: TCGA-XX-XXXX[0101].npy
    """

    def __init__(self, token_dir, **kwargs):
        super().__init__(
            token_dir=token_dir,
            has_mask=False,
            extract_label_from_mask=False,  # Extract from filename
            **kwargs
        )


class TokenMapTestDataset(TokenMapDatasetBase):
    """
    Test dataset - token maps with labels in filename
    Example: TCGA-XX-XXXX[0101].npy
    """

    def __init__(self, token_dir, **kwargs):
        super().__init__(
            token_dir=token_dir,
            has_mask=False,
            extract_label_from_mask=False,  # Extract from filename
            **kwargs
        )


# Factory functions

def build_token_dataset(split='train', token_dir=None, **kwargs):
    """
    Factory function to build dataset

    Args:
        split: 'train', 'valid', or 'test'
        token_dir: Directory containing token .npy files with labels in filename
        **kwargs: Additional arguments

    Returns:
        Dataset instance
    """
    if token_dir is None:
        raise ValueError("token_dir is required")

    if split == 'train':
        return TokenMapTrainDataset(token_dir=token_dir, **kwargs)

    elif split == 'valid' or split == 'validation':
        return TokenMapValidationDataset(token_dir=token_dir, **kwargs)

    elif split == 'test':
        return TokenMapTestDataset(token_dir=token_dir, **kwargs)

    else:
        raise ValueError(f"Unknown split: {split}")


if __name__ == '__main__':
    # Example usage

    print("="*60)
    print("Testing TokenMapDataset")
    print("="*60)

    # Test train dataset
    try:
        train_ds = build_token_dataset(
            split='train',
            token_dir='token_maps'
        )

        print(f"\nTrain dataset: {len(train_ds)} samples")
        sample = train_ds[0]
        print(f"  Tokens shape: {sample['tokens'].shape}")
        print(f"  Label shape: {sample['cls_label'].shape}")
        print(f"  Classes: {sample['cls_names']}")
        print(f"  File: {sample['file_path_']}")

    except Exception as e:
        print(f"Train dataset error: {e}")

    # Test validation dataset
    try:
        val_ds = build_token_dataset(
            split='valid',
            token_dir='token_maps/val'
        )

        print(f"\nValidation dataset: {len(val_ds)} samples")
        sample = val_ds[0]
        print(f"  Tokens shape: {sample['tokens'].shape}")
        print(f"  Label shape: {sample['cls_label'].shape}")
        print(f"  Classes: {sample['cls_names']}")

    except Exception as e:
        print(f"Validation dataset error: {e}")

    print("\n" + "="*60)

"""
Debug segmentation output - show class distribution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Check a few segmentation results
seg_dir = Path('outputs/segmentation_batch')

seg_files = sorted(list(seg_dir.glob('*_seg.npy')))[:5]

print(f"Checking {len(seg_files)} segmentation files...\n")

CLASSES = {
    0: "Background",
    1: "TUM (Tumor)",
    2: "STR (Stroma)",
    3: "LYM (Lymphocytic)",
    4: "NEC (Necrosis)"
}

for seg_file in seg_files:
    seg = np.load(seg_file)

    print(f"File: {seg_file.name}")
    print(f"  Shape: {seg.shape}")
    print(f"  Unique classes: {np.unique(seg)}")

    total_pixels = seg.size

    for class_id in np.unique(seg):
        count = (seg == class_id).sum()
        percent = count / total_pixels * 100
        class_name = CLASSES.get(class_id, f"Unknown_{class_id}")
        print(f"    {class_name}: {count:,} pixels ({percent:.1f}%)")

    print()

print("\n" + "="*60)
print("If you see:")
print("- Mostly background (0): Threshold too high or model not predicting well")
print("- One class dominates: Model bias or normalization issue")
print("- Good mix of classes: Segmentation working correctly")
print("="*60)

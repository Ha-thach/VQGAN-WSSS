"""
Test script for TokenMapDataset
Quick verification of dataset loading
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from dataset import build_token_dataset


def test_dataset(split, token_dir, data_root=None):
    """Test dataset loading"""

    print(f"\n{'='*60}")
    print(f"Testing {split.upper()} dataset")
    print(f"{'='*60}")

    try:
        # Build dataset
        dataset = build_token_dataset(
            split=split,
            token_dir=token_dir,
            data_root=data_root
        )

        print(f"\nDataset size: {len(dataset)}")

        # Test single sample
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  tokens shape:  {sample['tokens'].shape}")
        print(f"  tokens dtype:  {sample['tokens'].dtype}")
        print(f"  tokens range:  [{sample['tokens'].min()}, {sample['tokens'].max()}]")
        print(f"  label shape:   {sample['cls_label'].shape}")
        print(f"  label:         {sample['cls_label']}")
        print(f"  classes:       {sample['cls_names']}")
        print(f"  file:          {sample['file_path_']}")

        # Test dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0
        )

        batch = next(iter(dataloader))
        print(f"\nBatch test:")
        print(f"  tokens shape:  {batch['tokens'].shape}")
        print(f"  labels shape:  {batch['cls_label'].shape}")

        # Class distribution
        all_labels = torch.stack([dataset[i]['cls_label'] for i in range(len(dataset))])
        class_counts = all_labels.sum(dim=0).int()
        class_names = ["TUM", "STR", "LYM", "NEC"]

        print(f"\nClass distribution:")
        for name, count in zip(class_names, class_counts):
            print(f"  {name}: {count.item()}/{len(dataset)} ({count.item()/len(dataset)*100:.1f}%)")

        print(f"\n✓ {split} dataset OK!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test TokenMapDataset')
    parser.add_argument('--token-dir', type=str, required=True,
                        help='Directory with token .npy files')
    parser.add_argument('--data-root', type=str, default=None,
                        help='BCSS-WSSS data root (for validation/test masks)')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'valid', 'test'],
                        help='Dataset split to test')

    args = parser.parse_args()

    # Test specified split
    test_dataset(args.split, args.token_dir, args.data_root)

    print("\n" + "="*60)
    print("Done!")
    print("="*60)

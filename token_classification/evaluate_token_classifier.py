"""
Evaluate Token Classifier on Test Set
Simple evaluation script matching train_token_classifier.py architecture
"""

import os
import sys
import argparse
import json
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from token_classification.dataset import build_token_dataset
from token_classification.train_token_classifier import TokenViT, collate_fn


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate model on dataset

    Returns:
        dict with detailed metrics
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_file_paths = []

    print("\nEvaluating...")
    for batch in tqdm(dataloader, desc='Testing'):
        tokens = batch['tokens'].to(device)
        labels = batch['cls_label'].to(device)

        # Forward
        logits = model(tokens)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_file_paths.extend(batch['file_path_'])

    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)

    # Compute metrics
    metrics = {}

    # Exact match accuracy
    exact_match = (all_preds == all_labels).all(axis=1).mean()
    metrics['exact_match'] = float(exact_match)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    # Micro averages
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0
    )

    metrics['per_class'] = {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'support': support.tolist()
    }

    metrics['macro'] = {
        'precision': float(precision_macro),
        'recall': float(recall_macro),
        'f1': float(f1_macro)
    }

    metrics['micro'] = {
        'precision': float(precision_micro),
        'recall': float(recall_micro),
        'f1': float(f1_micro)
    }

    # Sample accuracy (at least one correct class per sample)
    sample_acc = (all_preds == all_labels).any(axis=1).mean()
    metrics['sample_accuracy'] = float(sample_acc)

    # Hamming loss
    hamming_loss = (all_preds != all_labels).mean()
    metrics['hamming_loss'] = float(hamming_loss)

    return metrics, all_preds, all_labels, all_probs, all_file_paths


def print_metrics(metrics, class_names=['TUM', 'STR', 'LYM', 'NEC']):
    """Print metrics in readable format"""

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")

    print(f"\nOverall Metrics:")
    print(f"  Exact Match Accuracy: {metrics['exact_match']:.4f}")
    print(f"  Sample Accuracy:      {metrics['sample_accuracy']:.4f}")
    print(f"  Hamming Loss:         {metrics['hamming_loss']:.4f}")

    print(f"\nMacro-averaged Metrics:")
    print(f"  Precision: {metrics['macro']['precision']:.4f}")
    print(f"  Recall:    {metrics['macro']['recall']:.4f}")
    print(f"  F1-score:  {metrics['macro']['f1']:.4f}")

    print(f"\nMicro-averaged Metrics:")
    print(f"  Precision: {metrics['micro']['precision']:.4f}")
    print(f"  Recall:    {metrics['micro']['recall']:.4f}")
    print(f"  F1-score:  {metrics['micro']['f1']:.4f}")

    print(f"\nPer-class Metrics:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
    print("-" * 60)

    for i, class_name in enumerate(class_names):
        prec = metrics['per_class']['precision'][i]
        rec = metrics['per_class']['recall'][i]
        f1 = metrics['per_class']['f1'][i]
        sup = int(metrics['per_class']['support'][i])

        print(f"{class_name:<10} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {sup:<12}")

    print(f"{'='*60}\n")


def save_predictions_csv(all_preds, all_labels, all_probs, file_paths, output_path, class_names=['TUM', 'STR', 'LYM', 'NEC']):
    """Save predictions as CSV for easy reading"""

    csv_path = output_path / 'predictions.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['Image', 'True_Labels', 'Pred_Labels', 'Correct', 'Exact_Match']
        header.extend([f'{cls}_Prob' for cls in class_names])
        header.extend([f'{cls}_True' for cls in class_names])
        header.extend([f'{cls}_Pred' for cls in class_names])
        writer.writerow(header)

        # Data rows
        for i in range(len(all_preds)):
            true_label = all_labels[i]
            pred_label = all_preds[i]
            probs = all_probs[i]
            img_name = file_paths[i] if i < len(file_paths) else f"sample_{i}"

            # True and predicted class names
            true_classes = ','.join([class_names[j] for j in range(len(class_names)) if true_label[j] == 1])
            pred_classes = ','.join([class_names[j] for j in range(len(class_names)) if pred_label[j] == 1])

            # Check correctness
            exact_match = np.array_equal(true_label, pred_label)
            partial_correct = np.any(true_label == pred_label)

            # Build row
            row = [
                img_name,
                true_classes if true_classes else 'None',
                pred_classes if pred_classes else 'None',
                'Yes' if partial_correct else 'No',
                'Yes' if exact_match else 'No'
            ]

            # Add probabilities
            row.extend([f'{probs[j]:.4f}' for j in range(len(class_names))])

            # Add true labels (0/1)
            row.extend([int(true_label[j]) for j in range(len(class_names))])

            # Add predicted labels (0/1)
            row.extend([int(pred_label[j]) for j in range(len(class_names))])

            writer.writerow(row)

    print(f"✓ Saved predictions CSV to: {csv_path}")


def save_results(metrics, output_path, all_preds=None, all_labels=None, all_probs=None, file_paths=None):
    """Save evaluation results to file"""

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metrics as JSON
    with open(output_path / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Saved metrics to: {output_path / 'test_metrics.json'}")

    # Save predictions if provided
    if all_preds is not None and all_labels is not None:
        np.save(output_path / 'predictions.npy', all_preds)
        np.save(output_path / 'labels.npy', all_labels)
        np.save(output_path / 'probabilities.npy', all_probs)
        print(f"✓ Saved numpy arrays to: {output_path}")

        # Save CSV
        if all_probs is not None and file_paths is not None:
            save_predictions_csv(all_preds, all_labels, all_probs, file_paths, output_path)


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")

    # Load test dataset
    print(f"\n{'='*60}")
    print("LOADING TEST DATASET")
    print(f"{'='*60}")

    test_dataset = build_token_dataset(
        split='test',
        token_dir=args.test_token_dir
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    print(f"Test samples: {len(test_dataset)}")

    # Build model
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")

    model = TokenViT(
        num_tokens=args.num_tokens,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        pretrained_model=args.pretrained_model,
        freeze_backbone=False
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch'] + 1}")
    if 'metrics' in checkpoint:
        val_f1 = checkpoint['metrics'].get('f1', 'N/A')
        if isinstance(val_f1, (int, float)):
            print(f"Checkpoint validation F1: {val_f1:.4f}")
        else:
            print(f"Checkpoint validation F1: {val_f1}")

    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*60}")

    metrics, all_preds, all_labels, all_probs, all_file_paths = evaluate(
        model, test_loader, device
    )

    # Print results
    print_metrics(metrics)

    # Save results
    if args.output_dir:
        save_results(metrics, args.output_dir, all_preds, all_labels, all_probs, all_file_paths)
        print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate trained token classifier on test set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python token_classification/evaluate_token_classifier.py \\
      --test-token-dir token_maps/test \\
      --checkpoint outputs/token_vit/best_model.pth \\
      --output-dir outputs/token_vit/test_results
        """
    )

    # Data
    parser.add_argument('--test-token-dir', type=str, required=True,
                        help='Directory with test token maps (with labels in filename)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')

    # Model
    parser.add_argument('--num-tokens', type=int, default=8192,
                        help='Codebook size')
    parser.add_argument('--embed-dim', type=int, default=768,
                        help='Embedding dimension (768 for ViT-Base)')
    parser.add_argument('--num-classes', type=int, default=4,
                        help='Number of classes')
    parser.add_argument('--pretrained-model', type=str,
                        default='google/vit-base-patch16-224',
                        help='Pretrained ViT model from HuggingFace')

    # Evaluation
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results (optional)')

    args = parser.parse_args()
    main(args)

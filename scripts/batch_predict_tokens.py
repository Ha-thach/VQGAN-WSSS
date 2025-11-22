"""
Batch Prediction Script
Predict classes for multiple token maps and save results
"""

import argparse
import yaml
from pathlib import Path
import json

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

from taming.models.token_classifier import build_token_classifier


def load_model(config_path, checkpoint_path):
    """Load model from checkpoint"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = build_token_classifier(config['model'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    multi_label = config['model'].get('multi_label', True)

    return model, multi_label


def batch_predict(model, token_files, batch_size=32, multi_label=True, device='cpu'):
    """
    Batch prediction on multiple token files

    Args:
        model: Trained classifier
        token_files: List of paths to token .npy files
        batch_size: Batch size for inference
        multi_label: Multi-label or single-label
        device: Device to run on

    Returns:
        results: List of (filename, predictions, probabilities) tuples
    """
    model = model.to(device)
    model.eval()

    results = []

    # Process in batches
    for i in tqdm(range(0, len(token_files), batch_size), desc='Predicting'):
        batch_files = token_files[i:i + batch_size]

        # Load batch
        batch_tokens = []
        for token_file in batch_files:
            token_map = np.load(token_file)
            batch_tokens.append(torch.from_numpy(token_map).long())

        # Stack into batch
        batch_tokens = torch.stack(batch_tokens).to(device)

        # Predict
        with torch.no_grad():
            logits = model(batch_tokens)

            if multi_label:
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

        # Store results
        for j, token_file in enumerate(batch_files):
            results.append({
                'filename': Path(token_file).name,
                'filepath': str(token_file),
                'predictions': preds[j],
                'probabilities': probs[j]
            })

    return results


def save_results(results, output_dir, class_names, multi_label=True):
    """
    Save prediction results in multiple formats

    Args:
        results: List of prediction results
        output_dir: Directory to save results
        class_names: List of class names
        multi_label: Multi-label or single-label
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save as JSON
    json_results = []
    for r in results:
        entry = {
            'filename': r['filename'],
            'filepath': r['filepath'],
        }

        if multi_label:
            entry['predicted_classes'] = [
                class_names[i] for i in range(len(class_names))
                if r['predictions'][i] == 1
            ]
            entry['probabilities'] = {
                class_names[i]: float(r['probabilities'][i])
                for i in range(len(class_names))
            }
        else:
            entry['predicted_class'] = class_names[int(r['predictions'])]
            entry['probabilities'] = {
                class_names[i]: float(r['probabilities'][i])
                for i in range(len(class_names))
            }

        json_results.append(entry)

    json_path = output_dir / 'predictions.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON results saved to: {json_path}")

    # 2. Save as CSV
    csv_data = []
    for r in results:
        row = {'filename': r['filename']}

        if multi_label:
            for i, class_name in enumerate(class_names):
                row[f'{class_name}_pred'] = int(r['predictions'][i])
                row[f'{class_name}_prob'] = float(r['probabilities'][i])
        else:
            row['predicted_class'] = class_names[int(r['predictions'])]
            for i, class_name in enumerate(class_names):
                row[f'{class_name}_prob'] = float(r['probabilities'][i])

        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    csv_path = output_dir / 'predictions.csv'
    df.to_csv(csv_path, index=False)
    print(f"CSV results saved to: {csv_path}")

    # 3. Save summary statistics
    summary = {
        'total_samples': len(results),
        'class_distribution': {}
    }

    if multi_label:
        # Count predictions per class
        for i, class_name in enumerate(class_names):
            count = sum(r['predictions'][i] == 1 for r in results)
            summary['class_distribution'][class_name] = count
    else:
        # Count predictions per class
        for i, class_name in enumerate(class_names):
            count = sum(r['predictions'] == i for r in results)
            summary['class_distribution'][class_name] = count

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"Total samples: {summary['total_samples']}")
    print("\nClass Distribution:")
    for class_name, count in summary['class_distribution'].items():
        percentage = count / summary['total_samples'] * 100
        print(f"  {class_name:5s}: {count:5d} ({percentage:5.1f}%)")
    print("="*60)


def main(args):
    print("="*60)
    print("BATCH TOKEN CLASSIFICATION")
    print("="*60)

    # Load model
    print(f"\n[1/3] Loading model...")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")

    model, multi_label = load_model(args.config, args.checkpoint)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"  Device: {device}")
    print(f"  Multi-label: {multi_label}")

    # Find token files
    print(f"\n[2/3] Finding token files...")
    token_dir = Path(args.token_dir)

    if args.pattern:
        token_files = list(token_dir.rglob(args.pattern))
    else:
        token_files = list(token_dir.rglob('*_tokens.npy'))

    token_files = [str(f) for f in sorted(token_files)]
    print(f"  Found {len(token_files)} token files")

    if len(token_files) == 0:
        print("  ERROR: No token files found!")
        return

    # Predict
    print(f"\n[3/3] Running batch prediction...")
    results = batch_predict(
        model,
        token_files,
        batch_size=args.batch_size,
        multi_label=multi_label,
        device=device
    )

    # Save results
    class_names = ['TUM', 'STR', 'LYM', 'NEC']
    save_results(results, args.output_dir, class_names, multi_label)

    print("\n✓ Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch prediction on token maps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on all token files in a directory
  python scripts/batch_predict_tokens.py \\
      --config configs/token_classifier_transformer.yaml \\
      --checkpoint logs/token_classifier_transformer/checkpoints/best.ckpt \\
      --token-dir codebook_analysis_bcss_gumbel8 \\
      --output-dir batch_predictions

  # Predict on specific pattern
  python scripts/batch_predict_tokens.py \\
      --config configs/token_classifier_transformer.yaml \\
      --checkpoint logs/token_classifier_transformer/checkpoints/best.ckpt \\
      --token-dir codebook_analysis_bcss_gumbel8/TUM \\
      --pattern "TCGA*_tokens.npy" \\
      --batch-size 64 \\
      --output-dir batch_predictions/tum_only
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--token-dir',
        type=str,
        required=True,
        help='Directory containing token .npy files'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default=None,
        help='File pattern to match (default: *_tokens.npy)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='batch_predictions',
        help='Directory to save results'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU even if GPU is available'
    )

    args = parser.parse_args()
    main(args)

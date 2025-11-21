"""
Evaluation and Inference script for Token Classifier
"""

import os
import argparse
import yaml
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    hamming_loss
)
import matplotlib.pyplot as plt
import seaborn as sns

from taming.data.token_classification import build_token_classification_dataset
from taming.models.token_classifier import build_token_classifier


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config, checkpoint_path):
    """Load model from checkpoint"""
    model = build_token_classifier(config['model'])

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model


def evaluate_multi_label(model, dataloader, device, class_names):
    """Evaluate multi-label classification model"""
    model = model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    all_image_names = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            tokens = batch['tokens'].to(device)
            labels = batch['labels']

            # Forward pass
            logits = model(tokens)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            all_image_names.extend(batch['image_name'])

    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    # Calculate metrics
    metrics = {}

    # Overall metrics
    metrics['hamming_loss'] = hamming_loss(all_labels, all_preds)
    metrics['exact_match_ratio'] = accuracy_score(all_labels, all_preds)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    # Calculate AUC for each class
    auc_scores = []
    for i in range(len(class_names)):
        try:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            auc_scores.append(auc)
        except:
            auc_scores.append(0.0)

    # Store per-class metrics
    for i, class_name in enumerate(class_names):
        metrics[f'{class_name}_precision'] = precision[i]
        metrics[f'{class_name}_recall'] = recall[i]
        metrics[f'{class_name}_f1'] = f1[i]
        metrics[f'{class_name}_auc'] = auc_scores[i]
        metrics[f'{class_name}_support'] = support[i]

    # Macro and micro averages
    metrics['macro_precision'] = np.mean(precision)
    metrics['macro_recall'] = np.mean(recall)
    metrics['macro_f1'] = np.mean(f1)
    metrics['macro_auc'] = np.mean(auc_scores)

    # Micro average
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0
    )
    metrics['micro_precision'] = precision_micro
    metrics['micro_recall'] = recall_micro
    metrics['micro_f1'] = f1_micro

    return metrics, all_preds, all_probs, all_labels, all_image_names


def evaluate_single_label(model, dataloader, device, class_names):
    """Evaluate single-label classification model"""
    model = model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    all_image_names = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            tokens = batch['tokens'].to(device)
            labels = batch['labels']

            # Forward pass
            logits = model(tokens)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            all_image_names.extend(batch['image_name'])

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)

    # Calculate metrics
    metrics = {}

    # Overall accuracy
    metrics['accuracy'] = accuracy_score(all_labels, all_preds)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    # Calculate AUC (one-vs-rest)
    auc_scores = []
    for i in range(len(class_names)):
        try:
            binary_labels = (all_labels == i).astype(int)
            auc = roc_auc_score(binary_labels, all_probs[:, i])
            auc_scores.append(auc)
        except:
            auc_scores.append(0.0)

    # Store per-class metrics
    for i, class_name in enumerate(class_names):
        metrics[f'{class_name}_precision'] = precision[i]
        metrics[f'{class_name}_recall'] = recall[i]
        metrics[f'{class_name}_f1'] = f1[i]
        metrics[f'{class_name}_auc'] = auc_scores[i]
        metrics[f'{class_name}_support'] = support[i]

    # Macro and weighted averages
    metrics['macro_precision'] = np.mean(precision)
    metrics['macro_recall'] = np.mean(recall)
    metrics['macro_f1'] = np.mean(f1)
    metrics['macro_auc'] = np.mean(auc_scores)

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    metrics['weighted_precision'] = precision_weighted
    metrics['weighted_recall'] = recall_weighted
    metrics['weighted_f1'] = f1_weighted

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return metrics, all_preds, all_probs, all_labels, all_image_names, cm


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def save_predictions(predictions, labels, probs, image_names, class_names, output_path):
    """Save predictions to text file"""
    with open(output_path, 'w') as f:
        f.write("Image Name\tTrue Labels\tPredicted Labels\tProbabilities\n")
        for i in range(len(image_names)):
            img_name = image_names[i]
            true_label = labels[i]
            pred_label = predictions[i]
            prob = probs[i]

            if len(true_label.shape) > 0:  # Multi-label
                true_str = ','.join([class_names[j] for j in range(len(class_names)) if true_label[j] == 1])
                pred_str = ','.join([class_names[j] for j in range(len(class_names)) if pred_label[j] == 1])
                prob_str = ','.join([f"{class_names[j]}:{prob[j]:.4f}" for j in range(len(class_names))])
            else:  # Single-label
                true_str = class_names[int(true_label)]
                pred_str = class_names[int(pred_label)]
                prob_str = ','.join([f"{class_names[j]}:{prob[j]:.4f}" for j in range(len(class_names))])

            f.write(f"{img_name}\t{true_str}\t{pred_str}\t{prob_str}\n")

    print(f"Predictions saved to: {output_path}")


def main(args):
    # Load configuration
    config = load_config(args.config)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(config, args.checkpoint)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    split = args.split
    dataset = build_token_classification_dataset(config['data'], split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Evaluating on {split} set: {len(dataset)} samples")

    # Class names
    class_names = ['TUM', 'STR', 'LYM', 'NEC']

    # Evaluate
    multi_label = config['model'].get('multi_label', True)

    if multi_label:
        metrics, preds, probs, labels, image_names = evaluate_multi_label(
            model, dataloader, device, class_names
        )
        cm = None
    else:
        metrics, preds, probs, labels, image_names, cm = evaluate_single_label(
            model, dataloader, device, class_names
        )

    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            print(f"{key:30s}: {value}")
        else:
            print(f"{key:30s}: {value:.4f}")
    print("="*60)

    # Save metrics to file
    metrics_path = output_dir / f"{split}_metrics.txt"
    with open(metrics_path, 'w') as f:
        for key, value in metrics.items():
            if isinstance(value, (int, np.integer)):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value:.4f}\n")
    print(f"\nMetrics saved to: {metrics_path}")

    # Save predictions
    predictions_path = output_dir / f"{split}_predictions.txt"
    save_predictions(preds, labels, probs, image_names, class_names, predictions_path)

    # Plot confusion matrix (for single-label only)
    if not multi_label and cm is not None:
        cm_path = output_dir / f"{split}_confusion_matrix.png"
        plot_confusion_matrix(cm, class_names, cm_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate token classifier')
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
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'valid', 'test'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Directory to save results'
    )

    args = parser.parse_args()
    main(args)

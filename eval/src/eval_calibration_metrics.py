#!/usr/bin/env python3

import pickle as pkl
import re
import sys
import os
import collections
import numpy as np
import torch
from sklearn.metrics import roc_auc_score as roc_auc_score_original
from torchmetrics.classification import BinaryCalibrationError
from scipy import stats


def parse_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def get_confidence(text):
    pattern = r"<confidence>(.*?)</confidence>"
    result = re.search(pattern, text, re.DOTALL)
    if result:
        number = result.group(1).strip().replace('%', '')
        number = parse_float(number)
        if number is not None:
            if 0 <= number <= 1:
                return number
            elif 1 < number <= 100:
                return number / 100.0
    return None


def roc_auc_score(y_true, y_scores):
    # Filter out None values
    filtered = [(y_t, y_s) for y_t, y_s in zip(y_true, y_scores) if y_s is not None]
    if not filtered:
        return None
    y_true, y_scores = zip(*filtered)
    return roc_auc_score_original(y_true, y_scores)


def compute_wilson_score_interval(accuracy, count, confidence=0.95):
    if count == 0:
        return 0.0, 0.0

    # Wilson score interval
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    phat = accuracy

    denominator = 1 + z**2 / count
    center = (phat + z**2 / (2 * count)) / denominator
    margin = z * np.sqrt((phat * (1 - phat) / count + z**2 / (4 * count**2))) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return lower, upper


def bin_confidence_data(is_correct_tensor, conf_tensor, num_bins=10, min_bin_fraction=0.02):
    binned_data = collections.defaultdict(list)

    for idx in range(len(is_correct_tensor)):
        binned_s = round(conf_tensor[idx].item() * num_bins) / num_bins
        binned_data[binned_s].append(is_correct_tensor[idx].item())

    total_samples = len(is_correct_tensor)

    # Filter bins with less than min_bin_fraction of samples
    calibration_data = {
        'mean_confidence': [],
        'accuracy': [],
        'count': [],
        'error_lower': [],
        'error_upper': []
    }

    for bin_conf, bin_correct in sorted(binned_data.items()):
        bin_count = len(bin_correct)
        bin_fraction = bin_count / total_samples

        # Only include bins with at least min_bin_fraction of samples
        if bin_fraction >= min_bin_fraction:
            accuracy = sum(bin_correct) / bin_count

            # Compute Wilson score confidence interval
            lower, upper = compute_wilson_score_interval(accuracy, bin_count)

            calibration_data['mean_confidence'].append(bin_conf)
            calibration_data['accuracy'].append(accuracy)
            calibration_data['count'].append(bin_count)
            calibration_data['error_lower'].append(accuracy - lower)  # Distance from accuracy to lower bound
            calibration_data['error_upper'].append(upper - accuracy)  # Distance from accuracy to upper bound

    return calibration_data


def plot_calibration(result_file: str, auroc: float, ece: float,
                     is_correct: list, confidences: list,
                     num_samples: int, num_valid: int):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed. Skipping plot generation.")
        return

    # Extract model name and dataset from file path or config
    with open(result_file, 'rb') as f:
        data = pkl.load(f)

    config = data.get('config', {})
    model_name = config.get('model_path', 'Unknown Model')
    dataset_name = config.get('dataset', 'Unknown Dataset')

    # Get the filename from the path
    file_name = os.path.basename(result_file)

    # Filter out None values for plotting
    valid_pairs = [(ic, conf) for ic, conf in zip(is_correct, confidences) if conf is not None]
    if not valid_pairs:
        print("No valid confidence values for plotting")
        return

    valid_is_correct, valid_confidences = zip(*valid_pairs)

    # Convert to tensors for binning
    is_correct_tensor = torch.tensor(valid_is_correct, dtype=torch.long)
    confidence_tensor = torch.tensor(valid_confidences, dtype=torch.float32)

    # Bin the data with filtering (remove bins with < 2% of samples)
    calibration = bin_confidence_data(is_correct_tensor, confidence_tensor,
                                      num_bins=10, min_bin_fraction=0.02)

    # Calculate accuracy
    accuracy = np.mean(valid_is_correct)

    # Create reliability diagram
    fig, ax = plt.subplots(figsize=(8, 8))

    if len(calibration['mean_confidence']) > 0:
        mean_confidence_values = np.array(calibration['mean_confidence'])
        accuracies = np.array(calibration['accuracy'])
        counts = np.array(calibration['count'])
        error_lower = np.array(calibration['error_lower'])
        error_upper = np.array(calibration['error_upper'])

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5, linewidth=2)

        # Error bars (Wilson score 95% CI)
        ax.errorbar(mean_confidence_values, accuracies,
                    yerr=[error_lower, error_upper],
                    fmt='none', ecolor='steelblue', alpha=0.3, capsize=4, capthick=1.5,
                    label='95% CI (Wilson)')

        # Actual calibration with scatter (size proportional to bin count)
        ax.scatter(mean_confidence_values, accuracies, alpha=0.6, color='steelblue', zorder=3)

        # Connect points with line
        ax.plot(mean_confidence_values, accuracies, '-', alpha=0.5, color='steelblue', linewidth=1.5)

        ax.set_xlabel('Confidence', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)

        title = f'Reliability Diagram with Error Bars (n={num_valid})\n'
        if auroc is not None:
            title += f'ECE: {ece:.4f} | AUROC: {auroc:.4f}'
        else:
            title += f'ECE: {ece:.4f}' if ece is not None else 'Metrics unavailable'

        ax.set_title(title, fontsize=16)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        # Add info text
        info_text = f"Model: {model_name}\n"
        info_text += f"Dataset: {dataset_name}\n"
        info_text += f"Accuracy: {accuracy*100:.2f}%\n"
        info_text += f"Valid samples: {num_valid}/{num_samples}\n"
        info_text += f"Bins shown: {len(calibration['mean_confidence'])} (≥2% samples)"

        ax.text(0.5, -0.1, info_text, ha='center', va='top', fontsize=10,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    else:
        ax.text(0.5, 0.5, 'No bins with ≥2% of samples',
                ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Save plot to same directory as result file
    result_dir = os.path.dirname(result_file)
    plot_filename = file_name.replace('.pkl', '_calibration_plot_errorbar.png')
    plot_path = os.path.join(result_dir, plot_filename)

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    print(f"Bins included in plot: {len(calibration['mean_confidence'])} (bins with ≥2% of samples)")
    print("Error bars show 95% Wilson score confidence intervals")
    plt.close()


def calculate_calibration_metrics(result_file: str, plot: bool = False):
    print(f"Loading {result_file}...")
    with open(result_file, 'rb') as f:
        data = pkl.load(f)

    results = data['results']
    print(f"Found {len(results)} samples\n")

    # Extract confidence and correctness
    conf_results = []
    skipped = 0

    for sample in results:
        # Extract confidence from result text
        confidence = get_confidence(sample['result_text'])

        if confidence is None:
            print(f"Sample {sample['sample_idx']}: No confidence found, setting to None")
            skipped += 1

        conf_results.append({
            'is_correct': sample['is_correct'],
            'confidence': confidence
        })

    print(f"Skipped {skipped} samples with missing confidence (will be filtered out)\n")

    # Prepare data for metrics
    is_correct = [1 if r['is_correct'] else 0 for r in conf_results]
    confidences = [r['confidence'] for r in conf_results]

    # Calculate AUROC
    print("=" * 80)
    print("CALIBRATION METRICS")
    print("=" * 80)

    auroc = roc_auc_score(is_correct, confidences)
    if auroc is not None:
        print(f"AUROC: {auroc:.4f}")
    else:
        print("AUROC: Could not compute (no valid confidence values)")

    # Calculate ECE - filter out None values first
    valid_pairs = [(ic, conf) for ic, conf in zip(is_correct, confidences) if conf is not None]
    if valid_pairs:
        valid_is_correct, valid_confidences_for_ece = zip(*valid_pairs)
        bce = BinaryCalibrationError(n_bins=10, norm='l1')
        is_correct_tensor = torch.tensor(valid_is_correct, dtype=torch.long)
        confidence_tensor = torch.tensor(valid_confidences_for_ece, dtype=torch.float32)
        ece = bce(confidence_tensor, is_correct_tensor).item()
        print(f"ECE: {ece:.4f}")
    else:
        ece = None
        print("ECE: Could not compute (no valid confidence values)")

    # Calculate statistics
    valid_confidences = [c for c in confidences if c is not None]
    print(f"\nConfidence Statistics:")
    print(f"  Samples with valid confidence: {len(valid_confidences)}/{len(confidences)}")
    if valid_confidences:
        print(f"  Mean confidence: {np.mean(valid_confidences):.4f}")
        print(f"  Median confidence: {np.median(valid_confidences):.4f}")
        print(f"  Min confidence: {np.min(valid_confidences):.4f}")
        print(f"  Max confidence: {np.max(valid_confidences):.4f}")

    accuracy = np.mean(is_correct)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("=" * 80)

    metrics = {
        'auroc': auroc,
        'ece': ece,
        'num_samples': len(conf_results),
        'num_valid_confidence': len(valid_confidences),
        'accuracy': accuracy
    }

    # Generate plot if requested
    if plot:
        plot_calibration(
            result_file=result_file,
            auroc=auroc,
            ece=ece,
            is_correct=is_correct,
            confidences=confidences,
            num_samples=len(conf_results),
            num_valid=len(valid_confidences)
        )

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate calibration metrics (AUROC and ECE) for model predictions with error bars.'
    )
    parser.add_argument('result_file', help='Path to the evaluated .pkl file')
    parser.add_argument('--plot', action='store_true',
                        help='Generate and save calibration plot with error bars')

    args = parser.parse_args()

    try:
        metrics = calculate_calibration_metrics(args.result_file, plot=args.plot)
        print(f"\nCalibration metrics computed successfully!")

    except FileNotFoundError:
        print(f"Error: {args.result_file} not found!")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing field in data: {e}")
        print("Make sure the file has been evaluated with eval_gsm8k.py or eval_svamp.py first!")
        sys.exit(1)


if __name__ == "__main__":
    main()

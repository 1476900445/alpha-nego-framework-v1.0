#!/usr/bin/env python3
"""
Results Analysis Script
Analyze and visualize training results

Usage:
    python scripts/analyze_results.py --log-dir logs/
    python scripts/analyze_results.py --log-dir logs/ --compare baseline,alpha-nego
    python scripts/analyze_results.py --history logs/history.json --output plots/

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import argparse
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from utils.visualization import (
    plot_training_curves,
    create_training_dashboard,
    plot_metric_comparison,
    plot_utility_distribution,
)
from utils.metrics import NegotiationMetrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze training results')
    
    # Input
    parser.add_argument('--log-dir', type=str, default='logs/',
                        help='Directory containing logs')
    parser.add_argument('--history', type=str, default=None,
                        help='Path to history.json file')
    parser.add_argument('--metrics', type=str, default=None,
                        help='Path to metrics.json file')
    
    # Analysis
    parser.add_argument('--compare', type=str, default=None,
                        help='Comma-separated list of runs to compare')
    parser.add_argument('--window', type=int, default=10,
                        help='Smoothing window size')
    
    # Output
    parser.add_argument('--output', type=str, default='plots/',
                        help='Output directory for plots')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Output format')
    
    return parser.parse_args()


def load_history(history_path: str) -> dict:
    """Load training history"""
    with open(history_path, 'r') as f:
        data = json.load(f)
    
    # Convert to dict of lists
    history = {}
    
    if isinstance(data, list):
        # List of epoch entries
        for entry in data:
            metrics = entry.get('metrics', {})
            for key, value in metrics.items():
                if key not in history:
                    history[key] = []
                history[key].append(value)
    elif isinstance(data, dict):
        # Already in correct format
        if 'metrics' in data:
            history = data['metrics']
        else:
            history = data
    
    return history


def smooth_curve(values: list, window: int = 10) -> list:
    """Smooth curve with moving average"""
    if len(values) < window:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        smoothed.append(np.mean(values[start:end]))
    
    return smoothed


def analyze_convergence(history: dict) -> dict:
    """Analyze convergence"""
    analysis = {}
    
    # Check if metrics exist
    metric_names = ['train_loss', 'val_loss', 'agreement_rate', 'avg_utility']
    
    for metric in metric_names:
        if metric not in history:
            continue
        
        values = history[metric]
        
        if len(values) < 10:
            continue
        
        # Final value
        final_value = values[-1]
        
        # Best value
        if 'loss' in metric:
            best_value = min(values)
            best_epoch = values.index(best_value)
        else:
            best_value = max(values)
            best_epoch = values.index(best_value)
        
        # Stability (std of last 20%)
        n_stable = max(10, len(values) // 5)
        stability = np.std(values[-n_stable:])
        
        # Convergence speed (epochs to 90% of best)
        if 'loss' in metric:
            target = best_value * 1.1
            converged = [i for i, v in enumerate(values) if v <= target]
        else:
            target = best_value * 0.9
            converged = [i for i, v in enumerate(values) if v >= target]
        
        convergence_epoch = converged[0] if converged else len(values)
        
        analysis[metric] = {
            'final_value': final_value,
            'best_value': best_value,
            'best_epoch': best_epoch,
            'stability': stability,
            'convergence_epoch': convergence_epoch,
        }
    
    return analysis


def plot_convergence_analysis(history: dict, output_dir: Path, fmt: str):
    """Plot convergence analysis"""
    analysis = analyze_convergence(history)
    
    if not analysis:
        print("[Warning] No metrics available for convergence analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = list(analysis.keys())[:4]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        if metric not in history:
            ax.axis('off')
            continue
        
        values = history[metric]
        epochs = range(1, len(values) + 1)
        
        # Plot original
        ax.plot(epochs, values, alpha=0.3, label='Original')
        
        # Plot smoothed
        smoothed = smooth_curve(values, window=10)
        ax.plot(epochs, smoothed, linewidth=2, label='Smoothed')
        
        # Mark best
        info = analysis[metric]
        ax.axvline(info['best_epoch'], color='red', linestyle='--', 
                   alpha=0.5, label=f"Best (epoch {info['best_epoch']})")
        ax.axhline(info['best_value'], color='green', linestyle='--',
                   alpha=0.5, label=f"Best value: {info['best_value']:.4f}")
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / f'convergence_analysis.{fmt}'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Saved] Convergence analysis: {save_path}")


def plot_comparison(runs: dict, output_dir: Path, fmt: str):
    """Plot comparison of multiple runs"""
    # Find common metrics
    all_metrics = set()
    for history in runs.values():
        all_metrics.update(history.keys())
    
    common_metrics = []
    for metric in all_metrics:
        if all(metric in h for h in runs.values()):
            common_metrics.append(metric)
    
    if not common_metrics:
        print("[Warning] No common metrics for comparison")
        return
    
    # Plot top 4 metrics
    metrics_to_plot = common_metrics[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        for run_name, history in runs.items():
            values = history[metric]
            epochs = range(1, len(values) + 1)
            
            # Smooth
            smoothed = smooth_curve(values, window=10)
            
            ax.plot(epochs, smoothed, linewidth=2, label=run_name)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / f'comparison.{fmt}'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Saved] Comparison: {save_path}")


def print_summary(history: dict, analysis: dict):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    print(f"\nTotal epochs: {len(history.get('train_loss', []))}")
    
    for metric, info in analysis.items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Final: {info['final_value']:.4f}")
        print(f"  Best: {info['best_value']:.4f} (epoch {info['best_epoch']})")
        print(f"  Stability (std): {info['stability']:.4f}")
        print(f"  Convergence: epoch {info['convergence_epoch']}")
    
    print("\n" + "="*70)


def main():
    """Main analysis function"""
    args = parse_args()
    
    print("="*70)
    print("α-NEGO RESULTS ANALYSIS")
    print("="*70)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load history
    if args.history:
        history_path = args.history
    else:
        history_path = Path(args.log_dir) / 'history.json'
    
    if not Path(history_path).exists():
        print(f"[Error] History file not found: {history_path}")
        return
    
    print(f"\n[Loading] History: {history_path}")
    history = load_history(str(history_path))
    
    print(f"[Loaded] {len(history)} metrics")
    print(f"  Metrics: {', '.join(history.keys())}")
    
    # Convergence analysis
    print("\n[Analyzing] Convergence...")
    analysis = analyze_convergence(history)
    
    # Print summary
    print_summary(history, analysis)
    
    # Generate plots
    print("\n[Plotting] Generating visualizations...")
    
    # Training curves
    plot_training_curves(
        history,
        save_path=str(output_dir / f'training_curves.{args.format}'),
        show=False
    )
    print(f"  ✓ Training curves")
    
    # Dashboard
    create_training_dashboard(
        history,
        save_path=str(output_dir / f'dashboard.{args.format}'),
        show=False
    )
    print(f"  ✓ Dashboard")
    
    # Convergence analysis
    plot_convergence_analysis(history, output_dir, args.format)
    print(f"  ✓ Convergence analysis")
    
    # Comparison (if specified)
    if args.compare:
        print("\n[Comparing] Multiple runs...")
        run_names = [n.strip() for n in args.compare.split(',')]
        
        runs = {}
        for run_name in run_names:
            run_history_path = Path(args.log_dir) / run_name / 'history.json'
            if run_history_path.exists():
                runs[run_name] = load_history(str(run_history_path))
                print(f"  ✓ Loaded: {run_name}")
            else:
                print(f"  ✗ Not found: {run_name}")
        
        if len(runs) > 1:
            plot_comparison(runs, output_dir, args.format)
            print(f"  ✓ Comparison plot")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETED")
    print("="*70)
    print(f"Plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
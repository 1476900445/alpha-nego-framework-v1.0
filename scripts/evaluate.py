#!/usr/bin/env python3
"""
Evaluation Script for α-Nego
Evaluate trained negotiation agents

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --opponents all
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --test-styles

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from configs.craigslist_config import get_craigslist_config
from configs.dealornodeal_config import get_dealornodeal_config
from environment.negotiation_env import make_negotiation_env
from training.evaluation import (
    NegotiationEvaluator, 
    StyleEvaluator,
    evaluate_against_baselines,
    cross_evaluate
)
from utils.metrics import NegotiationMetrics
from utils.visualization import (
    plot_metric_comparison,
    plot_style_comparison,
    plot_utility_distribution,
    plot_cross_evaluation_heatmap,
)
from utils.checkpoint import ModelSaver
from algorithms.dsac import create_dsac_agent


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate α-Nego agent')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint to evaluate')
    parser.add_argument('--dataset', type=str, default='craigslistbargain',
                        choices=['craigslistbargain', 'dealornodeal'],
                        help='Dataset')
    
    # Evaluation
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--opponents', type=str, default='single',
                        choices=['single', 'all', 'baseline'],
                        help='Opponents to evaluate against')
    parser.add_argument('--test-styles', action='store_true',
                        help='Test different negotiation styles')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='evaluation_results/',
                        help='Directory for evaluation results')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save visualization plots')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic policy')
    
    return parser.parse_args()


def load_agent(checkpoint_path: str, config, device: str):
    """Load agent from checkpoint"""
    print(f"\n[Loading] Checkpoint: {checkpoint_path}")
    
    # Create agent
    agent = create_dsac_agent(config, device=device)
    
    # Load checkpoint
    ModelSaver.load_agent(agent, checkpoint_path, device)
    
    print("[Loading] Agent loaded successfully")
    
    return agent


def evaluate_single_opponent(args, agent, env, config):
    """Evaluate against single opponent"""
    print("\n" + "="*70)
    print("EVALUATION: SINGLE OPPONENT")
    print("="*70)
    
    evaluator = NegotiationEvaluator(
        env=env,
        num_episodes=args.num_episodes,
        verbose=True,
    )
    
    results = evaluator.evaluate(agent, deterministic=args.deterministic)
    
    # Save results
    if args.save_plots:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot utility distribution
        utilities = [e['agent_utility'] for e in results['by_opponent'].values()]
        utilities_flat = []
        for opp_results in results['by_opponent'].values():
            utilities_flat.extend([e['agent_utility'] for e in opp_results.get('episodes', [])])
        
        if utilities_flat:
            from utils.visualization import plot_utility_distribution
            plot_utility_distribution(
                utilities_flat,
                save_path=str(output_dir / 'utility_distribution.png'),
                show=False
            )
    
    return results


def evaluate_all_baselines(args, agent, env, config):
    """Evaluate against all baseline opponents"""
    print("\n" + "="*70)
    print("EVALUATION: ALL BASELINE OPPONENTS")
    print("="*70)
    
    results = evaluate_against_baselines(
        agent=agent,
        env=env,
        num_episodes=args.num_episodes,
    )
    
    # Print detailed results
    print("\nPer-Opponent Results:")
    for opp_name, metrics in results['by_opponent'].items():
        print(f"\n  {opp_name}:")
        print(f"    Agreement Rate: {metrics['agreement_rate']:.2%}")
        print(f"    Avg Utility: {metrics['avg_agent_utility']:.4f}")
        print(f"    Avg Dialogue Length: {metrics['avg_dialogue_length']:.2f}")
        if 'negotiation_score' in metrics:
            print(f"    Negotiation Score: {metrics['negotiation_score']:.4f}")
    
    # Save plots
    if args.save_plots:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Comparison plot
        plot_metric_comparison(
            results['by_opponent'],
            'agreement_rate',
            save_path=str(output_dir / 'baseline_comparison_agreement.png'),
            show=False
        )
        
        plot_metric_comparison(
            results['by_opponent'],
            'avg_agent_utility',
            save_path=str(output_dir / 'baseline_comparison_utility.png'),
            show=False
        )
    
    return results


def evaluate_styles(args, agent, env, config):
    """Evaluate different negotiation styles"""
    print("\n" + "="*70)
    print("EVALUATION: NEGOTIATION STYLES")
    print("="*70)
    
    style_evaluator = StyleEvaluator(
        env=env,
        num_episodes=args.num_episodes,
    )
    
    style_results = style_evaluator.evaluate_styles(
        agent=agent,
        styles=['neutral', 'aggressive', 'conservative'],
    )
    
    # Save plots
    if args.save_plots:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_style_comparison(
            style_results,
            save_path=str(output_dir / 'style_comparison.png'),
            show=False
        )
    
    return style_results


def save_evaluation_report(args, all_results):
    """Save comprehensive evaluation report"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'evaluation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("α-NEGO EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Episodes: {args.num_episodes}\n")
        f.write(f"Deterministic: {args.deterministic}\n")
        f.write("\n" + "="*70 + "\n\n")
        
        # Overall results
        if 'overall' in all_results:
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-"*70 + "\n")
            overall = all_results['overall']
            f.write(f"Agreement Rate: {overall['agreement_rate']:.2%}\n")
            f.write(f"Avg Agent Utility: {overall['avg_agent_utility']:.4f}\n")
            f.write(f"Avg Opponent Utility: {overall['avg_opponent_utility']:.4f}\n")
            f.write(f"Avg Social Welfare: {overall['avg_social_welfare']:.4f}\n")
            f.write(f"Avg Dialogue Length: {overall['avg_dialogue_length']:.2f}\n")
            if 'negotiation_score' in overall:
                f.write(f"Negotiation Score: {overall['negotiation_score']:.4f}\n")
            f.write("\n")
        
        # Per-opponent results
        if 'by_opponent' in all_results:
            f.write("PER-OPPONENT PERFORMANCE:\n")
            f.write("-"*70 + "\n")
            for opp_name, metrics in all_results['by_opponent'].items():
                f.write(f"\n{opp_name}:\n")
                f.write(f"  Agreement Rate: {metrics['agreement_rate']:.2%}\n")
                f.write(f"  Avg Utility: {metrics['avg_agent_utility']:.4f}\n")
                f.write(f"  Avg Dialogue Length: {metrics['avg_dialogue_length']:.2f}\n")
                if 'negotiation_score' in metrics:
                    f.write(f"  Negotiation Score: {metrics['negotiation_score']:.4f}\n")
            f.write("\n")
        
        # Style results
        if 'styles' in all_results:
            f.write("STYLE PERFORMANCE:\n")
            f.write("-"*70 + "\n")
            for style, metrics in all_results['styles'].items():
                f.write(f"\n{style.capitalize()}:\n")
                f.write(f"  Agreement Rate: {metrics['agreement_rate']:.2%}\n")
                f.write(f"  Avg Utility: {metrics['avg_agent_utility']:.4f}\n")
                f.write(f"  Avg Dialogue Length: {metrics['avg_dialogue_length']:.2f}\n")
            f.write("\n")
        
        f.write("="*70 + "\n")
    
    print(f"\n[Report] Saved to: {report_path}")


def main():
    """Main evaluation function"""
    args = parse_args()
    
    print("="*70)
    print("α-NEGO EVALUATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Num episodes: {args.num_episodes}")
    print("="*70)
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"\n[Device] Using: {device}")
    
    # Get config
    if args.dataset == 'craigslistbargain':
        config = get_craigslist_config()
    else:
        config = get_dealornodeal_config()
    
    # Load agent
    agent = load_agent(args.checkpoint, config, device)
    
    # Create environment
    env = make_negotiation_env(
        dataset=args.dataset,
        opponent_type='rule_based',
        max_turns=config.dataset.max_turns,
    )
    
    # Run evaluations
    all_results = {}
    
    if args.opponents == 'single':
        results = evaluate_single_opponent(args, agent, env, config)
        all_results.update(results)
    
    elif args.opponents == 'all' or args.opponents == 'baseline':
        results = evaluate_all_baselines(args, agent, env, config)
        all_results.update(results)
    
    if args.test_styles:
        style_results = evaluate_styles(args, agent, env, config)
        all_results['styles'] = style_results
    
    # Save report
    save_evaluation_report(args, all_results)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETED")
    print("="*70)
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
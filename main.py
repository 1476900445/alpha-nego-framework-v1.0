#!/usr/bin/env python3
"""
α-Nego Framework - Unified Entry Point
Main script for training, evaluation, interactive negotiation, and analysis

Usage:
    # Training
    python main.py --mode train --dataset craigslistbargain --epochs 1100
    python main.py --mode train --quick-test
    
    # Evaluation
    python main.py --mode evaluate --checkpoint checkpoints/best_model.pt
    
    # Interactive negotiation
    python main.py --mode interactive --checkpoint checkpoints/best_model.pt
    
    # Results analysis
    python main.py --mode analyze --log-dir logs/
    
    # Show help
    python main.py --help

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import argparse
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

import train
import evaluate
import interactive
import analyze_results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='α-Nego Framework - Unified Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python main.py --mode train --quick-test
  
  # Full training
  python main.py --mode train --dataset craigslistbargain --epochs 1100
  
  # Training with TensorBoard
  python main.py --mode train --use-tensorboard
  
  # Training with Wandb
  python main.py --mode train --use-wandb --wandb-project my-project
  
  # Evaluation
  python main.py --mode evaluate --checkpoint checkpoints/best_model.pt
  
  # Evaluation with all baselines
  python main.py --mode evaluate --checkpoint checkpoints/best_model.pt --opponents all
  
  # Interactive negotiation
  python main.py --mode interactive --checkpoint checkpoints/best_model.pt
  
  # Interactive with style
  python main.py --mode interactive --checkpoint checkpoints/best_model.pt --style aggressive
  
  # Analyze results
  python main.py --mode analyze --log-dir logs/
  
  # Compare runs
  python main.py --mode analyze --compare baseline,alpha-nego
        """
    )
    
    # Main mode
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'interactive', 'analyze'],
                        help='Operation mode')
    
    # ==================== Common Arguments ====================
    parser.add_argument('--dataset', type=str, default='craigslistbargain',
                        choices=['craigslistbargain', 'dealornodeal'],
                        help='Dataset to use')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # ==================== Training Arguments ====================
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument('--epochs', type=int, default=1100,
                             help='Number of training epochs')
    train_group.add_argument('--episodes-per-epoch', type=int, default=10,
                             help='Episodes per epoch')
    train_group.add_argument('--batch-size', type=int, default=32,
                             help='Batch size')
    train_group.add_argument('--learning-rate', type=float, default=1e-4,
                             help='Learning rate')
    train_group.add_argument('--style', type=str, default='neutral',
                             choices=['neutral', 'aggressive', 'conservative'],
                             help='Negotiation style')
    train_group.add_argument('--skip-warmstart', action='store_true',
                             help='Skip supervised learning warm-start')
    train_group.add_argument('--warmstart-epochs', type=int, default=10,
                             help='Epochs for warm-start training')
    train_group.add_argument('--warmstart-path', type=str, default=None,
                             help='Path to pre-trained warm-start model')
    train_group.add_argument('--resume', type=str, default=None,
                             help='Resume from checkpoint')
    train_group.add_argument('--quick-test', action='store_true',
                             help='Quick test mode (10 epochs)')
    
    # ==================== Checkpoint & Logging ====================
    log_group = parser.add_argument_group('Logging & Checkpointing')
    log_group.add_argument('--save-dir', type=str, default='checkpoints/',
                          help='Directory to save checkpoints')
    log_group.add_argument('--log-dir', type=str, default='logs/',
                          help='Directory for logs')
    log_group.add_argument('--save-frequency', type=int, default=50,
                          help='Save checkpoint every N epochs')
    log_group.add_argument('--use-tensorboard', action='store_true',
                          help='Use TensorBoard logging')
    log_group.add_argument('--use-wandb', action='store_true',
                          help='Use Weights & Biases logging')
    log_group.add_argument('--wandb-project', type=str, default='alpha-nego',
                          help='Wandb project name')
    
    # ==================== Evaluation Arguments ====================
    eval_group = parser.add_argument_group('Evaluation Options')
    eval_group.add_argument('--checkpoint', type=str, default=None,
                           help='Path to checkpoint to evaluate')
    eval_group.add_argument('--num-episodes', type=int, default=100,
                           help='Number of evaluation episodes')
    eval_group.add_argument('--opponents', type=str, default='single',
                           choices=['single', 'all', 'baseline'],
                           help='Opponents to evaluate against')
    eval_group.add_argument('--test-styles', action='store_true',
                           help='Test different negotiation styles')
    eval_group.add_argument('--save-plots', action='store_true',
                           help='Save visualization plots')
    eval_group.add_argument('--output-dir', type=str, default='evaluation_results/',
                           help='Directory for evaluation results')
    eval_group.add_argument('--deterministic', action='store_true',
                           help='Use deterministic policy')
    
    # ==================== Interactive Arguments ====================
    interactive_group = parser.add_argument_group('Interactive Options')
    interactive_group.add_argument('--max-turns', type=int, default=20,
                                  help='Maximum dialogue turns')
    interactive_group.add_argument('--listing-price', type=float, default=None,
                                  help='Listing price for negotiation')
    
    # ==================== Analysis Arguments ====================
    analyze_group = parser.add_argument_group('Analysis Options')
    analyze_group.add_argument('--history', type=str, default=None,
                              help='Path to history.json file')
    analyze_group.add_argument('--compare', type=str, default=None,
                              help='Comma-separated list of runs to compare')
    analyze_group.add_argument('--window', type=int, default=10,
                              help='Smoothing window size')
    analyze_group.add_argument('--format', type=str, default='png',
                              choices=['png', 'pdf', 'svg'],
                              help='Output format for plots')
    
    return parser.parse_args()


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║                    α-Nego Framework v1.0                      ║
    ║                                                               ║
    ║         Self-play Deep RL for Negotiation Dialogues          ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def validate_args(args):
    """Validate arguments"""
    if args.mode in ['evaluate', 'interactive']:
        if not args.checkpoint:
            print("Error: --checkpoint is required for evaluate/interactive mode")
            sys.exit(1)
        
        if not Path(args.checkpoint).exists():
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
    
    if args.mode == 'analyze':
        if not args.log_dir and not args.history:
            print("Error: --log-dir or --history is required for analyze mode")
            sys.exit(1)


def run_training(args):
    """Run training"""
    print(f"\n{'='*70}")
    print("MODE: TRAINING")
    print(f"{'='*70}\n")
    
    # Build training arguments
    train_args = argparse.Namespace(
        dataset=args.dataset,
        epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        batch_size=args.batch_size,
        skip_warmstart=args.skip_warmstart,
        warmstart_epochs=args.warmstart_epochs,
        warmstart_path=args.warmstart_path,
        save_dir=args.save_dir,
        resume=args.resume,
        save_frequency=args.save_frequency,
        log_dir=args.log_dir,
        use_tensorboard=args.use_tensorboard,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        device=args.device,
        seed=args.seed,
        quick_test=args.quick_test,
    )
    
    # Override train module's sys.argv
    sys.argv = ['train.py']
    
    # Run training
    train.main_with_args(train_args)


def run_evaluation(args):
    """Run evaluation"""
    print(f"\n{'='*70}")
    print("MODE: EVALUATION")
    print(f"{'='*70}\n")
    
    # Build evaluation arguments
    eval_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        num_episodes=args.num_episodes,
        opponents=args.opponents,
        test_styles=args.test_styles,
        output_dir=args.output_dir,
        save_plots=args.save_plots,
        device=args.device,
        deterministic=args.deterministic,
    )
    
    # Run evaluation
    evaluate.main_with_args(eval_args)


def run_interactive(args):
    """Run interactive negotiation"""
    print(f"\n{'='*70}")
    print("MODE: INTERACTIVE NEGOTIATION")
    print(f"{'='*70}\n")
    
    # Build interactive arguments
    interactive_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        style=args.style,
        max_turns=args.max_turns,
        listing_price=args.listing_price,
        device=args.device,
    )
    
    # Run interactive
    interactive.main_with_args(interactive_args)


def run_analysis(args):
    """Run results analysis"""
    print(f"\n{'='*70}")
    print("MODE: RESULTS ANALYSIS")
    print(f"{'='*70}\n")
    
    # Build analysis arguments
    analyze_args = argparse.Namespace(
        log_dir=args.log_dir,
        history=args.history,
        metrics=None,
        compare=args.compare,
        window=args.window,
        output=args.output_dir,
        format=args.format,
    )
    
    # Run analysis
    analyze_results.main_with_args(analyze_args)


def main():
    """Main entry point"""
    # Print banner
    print_banner()
    
    # Parse arguments
    args = parse_args()
    
    # Validate arguments
    validate_args(args)
    
    # Route to appropriate function
    try:
        if args.mode == 'train':
            run_training(args)
        elif args.mode == 'evaluate':
            run_evaluation(args)
        elif args.mode == 'interactive':
            run_interactive(args)
        elif args.mode == 'analyze':
            run_analysis(args)
        else:
            print(f"Error: Unknown mode: {args.mode}")
            sys.exit(1)
        
        print(f"\n{'='*70}")
        print(f"✓ {args.mode.upper()} COMPLETED SUCCESSFULLY")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Operation cancelled by user")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n\n[Error] Operation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
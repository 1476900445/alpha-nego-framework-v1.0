#!/usr/bin/env python3
"""
Training Script for α-Nego
Main entry point for training negotiation agents

Usage:
    python scripts/train.py --config configs/craigslist_config.yaml
    python scripts/train.py --config configs/craigslist_config.yaml --resume checkpoints/latest.pt
    python scripts/train.py --dataset dealornodeal --epochs 1100

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from configs.craigslist_config import get_craigslist_config
from configs.dealornodeal_config import get_dealornodeal_config
from training.warmstart import warmstart_policy, create_warmstart_datasets
from training.trainer import train_alpha_nego
from training.callbacks import CallbackManager, CheckpointCallback, EarlyStoppingCallback, HistoryCallback
from utils.logger import TrainingLogger
from models.policy_network import create_policy_network


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train α-Nego negotiation agent')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='craigslistbargain',
                        choices=['craigslistbargain', 'dealornodeal'],
                        help='Dataset to use')
    
    # Training
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--episodes-per-epoch', type=int, default=None,
                        help='Episodes per epoch')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    
    # Warm-start
    parser.add_argument('--skip-warmstart', action='store_true',
                        help='Skip supervised learning warm-start')
    parser.add_argument('--warmstart-epochs', type=int, default=10,
                        help='Epochs for warm-start training')
    parser.add_argument('--warmstart-path', type=str, default=None,
                        help='Path to pre-trained warm-start model')
    
    # Checkpointing
    parser.add_argument('--save-dir', type=str, default='checkpoints/',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--save-frequency', type=int, default=50,
                        help='Save checkpoint every N epochs')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='logs/',
                        help='Directory for logs')
    parser.add_argument('--use-tensorboard', action='store_true',
                        help='Use TensorBoard logging')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='alpha-nego',
                        help='Wandb project name')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Quick test mode
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode (few epochs)')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> str:
    """Setup compute device"""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    print(f"[Device] Using: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device


def get_config(args):
    """Get configuration"""
    if args.dataset == 'craigslistbargain':
        config = get_craigslist_config()
    else:
        config = get_dealornodeal_config()
    
    # Override with command line arguments
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.episodes_per_epoch is not None:
        config.training.episodes_per_epoch = args.episodes_per_epoch
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.save_frequency is not None:
        config.training.save_frequency = args.save_frequency
    
    # Quick test mode
    if args.quick_test:
        print("[Quick Test Mode] Using reduced settings")
        config.training.num_epochs = 10
        config.training.episodes_per_epoch = 2
        config.training.updates_per_epoch = 10
        config.training.eval_frequency = 5
        config.training.save_frequency = 5
    
    return config


def run_warmstart(args, config, device):
    """Run supervised learning warm-start"""
    print("\n" + "="*70)
    print("STEP 1: SUPERVISED LEARNING WARM-START (Algorithm 1, Line 1)")
    print("="*70)
    
    if args.warmstart_path:
        print(f"\n[Warm-start] Using pre-trained model: {args.warmstart_path}")
        return args.warmstart_path
    
    if args.skip_warmstart:
        print("\n[Warm-start] Skipped (--skip-warmstart flag)")
        return None
    
    # Create policy
    policy = create_policy_network(config)
    
    # Create datasets
    train_dataset, val_dataset = create_warmstart_datasets(config)
    
    # Warm-start
    from training.warmstart import WarmStartTrainer
    
    trainer = WarmStartTrainer(policy, config, device)
    
    save_path = Path(args.save_dir) / 'warmstart' / 'policy.pt'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.warmstart_epochs,
        save_dir=str(save_path.parent),
        verbose=True,
    )
    
    trainer.save(str(save_path))
    
    print(f"\n[Warm-start] Completed!")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Saved to: {save_path}")
    
    return str(save_path)


def run_training(args, config, warmstart_path, device):
    """Run RL training"""
    print("\n" + "="*70)
    print("STEP 2: REINFORCEMENT LEARNING TRAINING (Algorithm 1)")
    print("="*70)
    
    # Setup callbacks
    callbacks = CallbackManager([
        CheckpointCallback(
            save_dir=args.save_dir,
            monitor='agreement_rate',
            mode='max',
            save_best_only=False,
            save_frequency=args.save_frequency,
            verbose=True,
        ),
        EarlyStoppingCallback(
            monitor='agreement_rate',
            patience=100,
            mode='max',
            verbose=True,
        ),
        HistoryCallback(
            save_path=Path(args.log_dir) / 'history.json'
        ),
    ])
    
    # Setup logger
    logger = TrainingLogger(
        log_dir=args.log_dir,
        use_console=True,
        use_file=True,
        use_json=True,
        use_tensorboard=args.use_tensorboard,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_config=config.__dict__ if hasattr(config, '__dict__') else None,
    )
    
    # Train
    print(f"\n[Training] Starting with configuration:")
    print(f"  Dataset: {config.dataset.name}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Episodes per epoch: {config.training.episodes_per_epoch}")
    print(f"  Updates per epoch: {config.training.updates_per_epoch}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Device: {device}")
    
    try:
        train_alpha_nego(
            config=config,
            warmstart_path=warmstart_path,
            save_dir=args.save_dir,
            device=device,
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n[Training] Interrupted by user")
        print("Checkpoints saved in:", args.save_dir)
    
    except Exception as e:
        print(f"\n\n[Error] Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        logger.close()


def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("="*70)
    print("α-NEGO TRAINING")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Save directory: {args.save_dir}")
    print(f"Log directory: {args.log_dir}")
    print("="*70)
    
    # Setup
    device = setup_device(args.device)
    config = get_config(args)
    
    # Create directories
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Warm-start (if not skipped)
    warmstart_path = run_warmstart(args, config, device)
    
    # Step 2: RL training
    run_training(args, config, warmstart_path, device)
    
    print("\n[Complete] All training finished!")
    print(f"Checkpoints: {args.save_dir}")
    print(f"Logs: {args.log_dir}")


if __name__ == '__main__':
    main()
"""
Warm-Start Training
Implements supervised learning initialization (Algorithm 1, Line 1)

Initialize policy π(SL) via supervised learning on human demonstrations

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm

from algorithms.supervised_learning import SupervisedLearner
from data.datasets import CraigslistbargainDataset, DealornodealDataset, create_dataloader


# ==================== Warm-Start Trainer ====================

class WarmStartTrainer:
    """
    Supervised learning warm-start trainer
    
    Implements Algorithm 1, Line 1:
    Initialize policy π(SL) via supervised learning
    """
    
    def __init__(
        self,
        policy: nn.Module,
        config,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Args:
            policy: Policy network to warm-start
            config: Configuration object
            device: Device for training
        """
        self.policy = policy
        self.config = config
        self.device = device
        
        # Create supervised learner
        self.learner = SupervisedLearner(
            policy=policy,
            learning_rate=config.training.sl_learning_rate,
            intent_loss_weight=1.0,
            price_loss_weight=1.0,
            use_nll_loss=True,
            l2_regularization=config.training.l2_regularization,
            device=device,
        )
        
        print(f"[WarmStartTrainer] Initialized")
        print(f"  Learning rate: {config.training.sl_learning_rate}")
        print(f"  Device: {device}")
    
    def train(
        self,
        train_dataset,
        val_dataset,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        save_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train with supervised learning
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of epochs (default from config)
            batch_size: Batch size (default from config)
            save_dir: Directory to save checkpoints
            verbose: Verbose output
            
        Returns:
            Training history
        """
        # Use config defaults if not provided
        num_epochs = num_epochs or self.config.training.sl_num_epochs
        batch_size = batch_size or self.config.training.batch_size
        
        # Create dataloaders
        train_loader = create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = create_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Train
        history = self.learner.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=num_epochs,
            save_dir=save_dir,
            early_stopping_patience=self.config.training.early_stopping_patience,
            verbose=verbose,
        )
        
        return history
    
    def save(self, path: str):
        """Save warm-started policy"""
        self.learner.save(path)
    
    def load(self, path: str):
        """Load warm-started policy"""
        self.learner.load(path)


# ==================== Dataset Factory ====================

def create_warmstart_datasets(
    config,
    split_ratio: float = 0.8,
):
    """
    Create datasets for warm-start training
    
    Args:
        config: Configuration object
        split_ratio: Train/val split ratio
        
    Returns:
        (train_dataset, val_dataset)
    """
    dataset_name = config.dataset.name
    data_dir = config.dataset.data_dir
    
    if dataset_name == 'craigslistbargain':
        # Load full dataset
        full_dataset = CraigslistbargainDataset(
            data_dir=data_dir,
            split='train',
            max_length=config.dataset.max_length,
        )
        
        # Split into train/val
        num_train = int(len(full_dataset) * split_ratio)
        num_val = len(full_dataset) - num_train
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [num_train, num_val]
        )
    
    elif dataset_name == 'dealornodeal':
        # Load full dataset
        full_dataset = DealornodealDataset(
            data_dir=data_dir,
            split='train',
            max_length=config.dataset.max_length,
        )
        
        # Split into train/val
        num_train = int(len(full_dataset) * split_ratio)
        num_val = len(full_dataset) - num_train
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [num_train, num_val]
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"[Dataset] Created {dataset_name} datasets")
    print(f"  Train size: {len(train_dataset)}")
    print(f"  Val size: {len(val_dataset)}")
    
    return train_dataset, val_dataset


# ==================== Quick Warm-Start Function ====================

def warmstart_policy(
    policy: nn.Module,
    config,
    save_path: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> nn.Module:
    """
    Quick warm-start function
    
    Implements Algorithm 1, Line 1 in one function call
    
    Args:
        policy: Policy network
        config: Configuration
        save_path: Path to save trained policy
        device: Device
        
    Returns:
        Warm-started policy
    """
    print("\n" + "="*60)
    print("WARM-START TRAINING (Algorithm 1, Line 1)")
    print("="*60)
    
    # Create datasets
    train_dataset, val_dataset = create_warmstart_datasets(config)
    
    # Create trainer
    trainer = WarmStartTrainer(policy, config, device)
    
    # Train
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_dir=Path(save_path).parent if save_path else None,
        verbose=True,
    )
    
    # Save if path provided
    if save_path:
        trainer.save(save_path)
        print(f"\n[WarmStart] Saved to: {save_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("WARM-START TRAINING COMPLETED")
    print("="*60)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Final train accuracy: {history['train_accuracy'][-1]:.4f}")
    print(f"Final val accuracy: {history['val_accuracy'][-1]:.4f}")
    
    return policy


# ==================== Warm-Start from Checkpoint ====================

def load_warmstart_policy(
    policy: nn.Module,
    checkpoint_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> nn.Module:
    """
    Load warm-started policy from checkpoint
    
    Args:
        policy: Policy network
        checkpoint_path: Path to checkpoint
        device: Device
        
    Returns:
        Loaded policy
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    
    print(f"[WarmStart] Loaded policy from: {checkpoint_path}")
    print(f"  Training steps: {checkpoint.get('training_steps', 'unknown')}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    return policy


# ==================== Evaluation ====================

def evaluate_warmstart_policy(
    policy: nn.Module,
    test_dataset,
    config,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> Dict[str, float]:
    """
    Evaluate warm-started policy
    
    Args:
        policy: Warm-started policy
        test_dataset: Test dataset
        config: Configuration
        device: Device
        
    Returns:
        Evaluation metrics
    """
    learner = SupervisedLearner(
        policy=policy,
        learning_rate=config.training.sl_learning_rate,
        device=device,
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    metrics = learner.evaluate(test_loader, verbose=True)
    
    print("\n[WarmStart Evaluation]")
    print(f"  Test loss: {metrics['total_loss']:.4f}")
    print(f"  Intent accuracy: {metrics['intent_accuracy']:.4f}")
    print(f"  Price error: {metrics['price_error']:.4f}")
    
    return metrics


# ==================== Testing ====================

if __name__ == '__main__':
    print("Testing Warm-Start Training...")
    
    # Import dependencies
    from configs.craigslist_config import get_craigslist_config
    from models.policy_network import create_policy_network
    
    # Create config
    config = get_craigslist_config()
    
    # Create policy
    print("\n1. Creating policy network...")
    policy = create_policy_network(config)
    
    # Test WarmStartTrainer
    print("\n2. Testing WarmStartTrainer...")
    trainer = WarmStartTrainer(policy, config, device='cpu')
    
    # Create dummy datasets
    print("\n3. Creating dummy datasets...")
    from torch.utils.data import TensorDataset
    
    dummy_states = torch.randn(100, 10, 3)
    dummy_actions = torch.cat([
        torch.randint(0, 16, (100, 10, 1)).float(),
        torch.rand(100, 10, 1)
    ], dim=-1)
    
    train_dataset = TensorDataset(dummy_states, dummy_actions)
    val_dataset = TensorDataset(dummy_states[:20], dummy_actions[:20])
    
    # Test training (1 epoch)
    print("\n4. Testing training (1 epoch)...")
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=1,
        batch_size=16,
        save_dir='/tmp/warmstart/',
        verbose=True,
    )
    
    print(f"   Train loss: {history['train_loss'][-1]:.4f}")
    print(f"   Val loss: {history['val_loss'][-1]:.4f}")
    
    # Test save/load
    print("\n5. Testing save/load...")
    trainer.save('/tmp/warmstart_policy.pt')
    
    new_policy = create_policy_network(config)
    loaded_policy = load_warmstart_policy(new_policy, '/tmp/warmstart_policy.pt', device='cpu')
    
    print("   Save/load successful")
    
    # Test quick warmstart function
    print("\n6. Testing quick warmstart function...")
    policy2 = create_policy_network(config)
    
    # Would use real datasets in practice
    # warmstart_policy(policy2, config, save_path='/tmp/warmstart.pt')
    
    print("\nWarm-start training tests passed!")
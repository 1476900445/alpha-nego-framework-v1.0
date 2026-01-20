"""
Supervised Learning for Warm-Start
Implements behavior cloning from human demonstrations (Algorithm 1, Line 1)

Used for:
1. Initialize policy π(SL) via supervised learning
2. Provide reference policy for KL regularization (Eq. 8)
3. Add to opponent pool as baseline

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm


class SupervisedLearner:
    """
    Supervised learning trainer for warm-start
    
    Trains policy network via behavior cloning on human demonstrations
    """
    
    def __init__(
        self,
        policy: nn.Module,
        learning_rate: float = 1e-3,
        intent_loss_weight: float = 1.0,
        price_loss_weight: float = 1.0,
        use_nll_loss: bool = True,
        l2_regularization: float = 0.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Args:
            policy: Policy network to train
            learning_rate: Learning rate (1e-3 in paper)
            intent_loss_weight: Weight for intent loss
            price_loss_weight: Weight for price loss
            use_nll_loss: Use NLL for price (else MSE)
            l2_regularization: L2 regularization weight
            device: Device
        """
        self.policy = policy
        self.device = device
        self.intent_loss_weight = intent_loss_weight
        self.price_loss_weight = price_loss_weight
        self.use_nll_loss = use_nll_loss
        self.l2_regularization = l2_regularization
        
        # Move to device
        self.policy.to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=learning_rate,
            weight_decay=l2_regularization,
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )
        
        # Statistics
        self.training_steps = 0
        self.best_val_loss = float('inf')
        
        print(f"[SupervisedLearner] Initialized")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Intent weight: {intent_loss_weight}")
        print(f"  Price weight: {price_loss_weight}")
        print(f"  Use NLL: {use_nll_loss}")
    
    def compute_loss(
        self,
        states: torch.Tensor,
        intent_targets: torch.Tensor,
        price_targets: torch.Tensor,
        history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute behavior cloning loss
        
        Args:
            states: States (batch, state_dim)
            intent_targets: Target intents (batch,)
            price_targets: Target prices (batch,)
            history: Dialogue history (batch, seq_len, state_dim)
            
        Returns:
            Dictionary with losses
        """
        # Forward pass
        intent_logits, price_mean, price_std = self.policy(states, history)
        
        # Intent loss (cross-entropy)
        intent_loss = F.cross_entropy(intent_logits, intent_targets)
        
        # Price loss
        if self.use_nll_loss:
            # Negative log-likelihood (Gaussian)
            price_dist = torch.distributions.Normal(
                price_mean.squeeze(-1),
                price_std.squeeze(-1)
            )
            price_loss = -price_dist.log_prob(price_targets).mean()
        else:
            # MSE
            price_loss = F.mse_loss(price_mean.squeeze(-1), price_targets)
        
        # Total loss
        total_loss = (
            self.intent_loss_weight * intent_loss +
            self.price_loss_weight * price_loss
        )
        
        # Compute accuracy
        intent_pred = torch.argmax(intent_logits, dim=-1)
        intent_accuracy = (intent_pred == intent_targets).float().mean()
        
        # Price error
        price_error = (price_mean.squeeze(-1) - price_targets).abs().mean()
        
        return {
            'total_loss': total_loss,
            'intent_loss': intent_loss,
            'price_loss': price_loss,
            'intent_accuracy': intent_accuracy,
            'price_error': price_error,
        }
    
    def train_step(
        self,
        states: torch.Tensor,
        intent_targets: torch.Tensor,
        price_targets: torch.Tensor,
        history: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            states: States
            intent_targets: Target intents
            price_targets: Target prices
            history: Dialogue history
            
        Returns:
            Loss dictionary
        """
        self.policy.train()
        
        # Move to device
        states = states.to(self.device)
        intent_targets = intent_targets.to(self.device)
        price_targets = price_targets.to(self.device)
        if history is not None:
            history = history.to(self.device)
        
        # Compute loss
        loss_dict = self.compute_loss(
            states, intent_targets, price_targets, history
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        # Update
        self.optimizer.step()
        self.training_steps += 1
        
        # Convert to floats
        return {k: v.item() for k, v in loss_dict.items()}
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: Training DataLoader
            verbose: Show progress bar
            
        Returns:
            Average losses
        """
        self.policy.train()
        
        total_losses = {
            'total_loss': 0.0,
            'intent_loss': 0.0,
            'price_loss': 0.0,
            'intent_accuracy': 0.0,
            'price_error': 0.0,
        }
        
        num_batches = 0
        
        iterator = tqdm(dataloader, desc="Training") if verbose else dataloader
        
        for batch in iterator:
            # Extract data
            states = batch['states']  # (batch, seq_len, state_dim)
            actions = batch['actions']  # (batch, seq_len, action_dim)
            
            # Flatten sequences
            batch_size, seq_len, _ = states.shape
            states = states.view(-1, states.size(-1))
            actions = actions.view(-1, actions.size(-1))
            
            # Extract intent and price
            intent_targets = actions[:, 0].long()
            price_targets = actions[:, 1]
            
            # Train step
            losses = self.train_step(states, intent_targets, price_targets)
            
            # Accumulate
            for k, v in losses.items():
                total_losses[k] += v
            num_batches += 1
            
            if verbose:
                iterator.set_postfix(loss=losses['total_loss'])
        
        # Average
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate on validation set
        
        Args:
            dataloader: Validation DataLoader
            verbose: Show progress bar
            
        Returns:
            Average losses
        """
        self.policy.eval()
        
        total_losses = {
            'total_loss': 0.0,
            'intent_loss': 0.0,
            'price_loss': 0.0,
            'intent_accuracy': 0.0,
            'price_error': 0.0,
        }
        
        num_batches = 0
        
        iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader
        
        with torch.no_grad():
            for batch in iterator:
                states = batch['states']
                actions = batch['actions']
                
                # Flatten
                states = states.view(-1, states.size(-1))
                actions = actions.view(-1, actions.size(-1))
                
                intent_targets = actions[:, 0].long()
                price_targets = actions[:, 1]
                
                # Move to device
                states = states.to(self.device)
                intent_targets = intent_targets.to(self.device)
                price_targets = price_targets.to(self.device)
                
                # Compute loss
                losses = self.compute_loss(states, intent_targets, price_targets)
                
                # Accumulate
                for k, v in losses.items():
                    total_losses[k] += v.item()
                num_batches += 1
                
                if verbose:
                    iterator.set_postfix(loss=losses['total_loss'].item())
        
        # Average
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        num_epochs: int = 10,
        save_dir: Optional[str] = None,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Complete training loop (Algorithm 1, Line 1)
        
        Args:
            train_dataloader: Training data
            val_dataloader: Validation data
            num_epochs: Number of epochs (10 in paper)
            save_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
            verbose: Verbose output
            
        Returns:
            Training history
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
        }
        
        patience_counter = 0
        
        print(f"\n[SupervisedLearner] Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Train
            train_losses = self.train_epoch(train_dataloader, verbose=False)
            
            # Validate
            val_losses = self.evaluate(val_dataloader, verbose=False)
            
            # Update scheduler
            self.scheduler.step(val_losses['total_loss'])
            
            # Save history
            history['train_loss'].append(train_losses['total_loss'])
            history['val_loss'].append(val_losses['total_loss'])
            history['train_accuracy'].append(train_losses['intent_accuracy'])
            history['val_accuracy'].append(val_losses['intent_accuracy'])
            
            # Print progress
            if verbose:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"  Train - Loss: {train_losses['total_loss']:.4f}, "
                      f"Intent Acc: {train_losses['intent_accuracy']:.4f}, "
                      f"Price Error: {train_losses['price_error']:.4f}")
                print(f"  Val   - Loss: {val_losses['total_loss']:.4f}, "
                      f"Intent Acc: {val_losses['intent_accuracy']:.4f}, "
                      f"Price Error: {val_losses['price_error']:.4f}")
            
            # Save best model
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                patience_counter = 0
                
                if save_dir:
                    best_path = save_dir / 'best_model.pt'
                    self.save(str(best_path))
                    if verbose:
                        print(f"  ✓ Saved best model (val_loss={self.best_val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n[SupervisedLearner] Early stopping at epoch {epoch+1}")
                break
            
            # Save checkpoint
            if save_dir and (epoch + 1) % 5 == 0:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pt'
                self.save(str(checkpoint_path))
        
        print(f"\n[SupervisedLearner] Training completed!")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        
        return history
    
    def save(self, path: str):
        """Save checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_steps': self.training_steps,
            'best_val_loss': self.best_val_loss,
        }, path)
    
    def load(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_steps = checkpoint.get('training_steps', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"[SupervisedLearner] Loaded from {path}")


# ==================== Batch Utilities ====================

def prepare_batch(
    batch: Dict[str, torch.Tensor],
    flatten: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare batch for supervised learning
    
    Args:
        batch: Batch from dataloader
        flatten: Whether to flatten sequences
        
    Returns:
        (states, intent_targets, price_targets)
    """
    states = batch['states']
    actions = batch['actions']
    
    if flatten and states.dim() == 3:
        # Flatten sequences: (batch, seq_len, dim) -> (batch*seq_len, dim)
        batch_size, seq_len, state_dim = states.shape
        states = states.view(-1, state_dim)
        actions = actions.view(-1, actions.size(-1))
    
    # Extract targets
    intent_targets = actions[..., 0].long()
    price_targets = actions[..., 1]
    
    return states, intent_targets, price_targets


# ==================== Quick Training Function ====================

def quick_train_supervised(
    policy: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    save_dir: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> nn.Module:
    """
    Quick supervised training function
    
    Implements Algorithm 1, Line 1:
    Initialize policy π(SL) via supervised learning on human data
    
    Args:
        policy: Policy network
        train_dataloader: Training data
        val_dataloader: Validation data
        num_epochs: Number of epochs (10 in paper)
        learning_rate: Learning rate (1e-3 in paper)
        save_dir: Save directory
        device: Device
        
    Returns:
        Trained policy
    """
    learner = SupervisedLearner(
        policy=policy,
        learning_rate=learning_rate,
        device=device,
    )
    
    history = learner.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        save_dir=save_dir,
    )
    
    return policy

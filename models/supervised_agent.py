"""
Supervised Learning Agent for Warm-Start
Implements behavior cloning from human demonstrations

Used in Algorithm 1 (Lines 1-3):
1. Initialize policy π(SL) via supervised learning on human data
2. Initialize opponent pool with π(SL) and rule-based agents
3. Pretrain critics with π(SL) policy

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# ==================== Supervised Agent ====================

class SupervisedAgent(nn.Module):
    """
    Supervised learning agent trained via behavior cloning
    
    Architecture: Same as PolicyNetwork
    Training: Cross-entropy for intent, MSE for price
    
    Used as:
    1. Initial policy (warm-start)
    2. Reference policy for KL regularization (Eq. 8)
    3. Opponent in pool
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_layers: List[int] = [256, 256],
        output_intent_dim: int = 16,
        output_price_dim: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: State dimension
            hidden_layers: Hidden layer sizes
            output_intent_dim: Number of dialogue acts
            output_price_dim: Price distribution params (mean, std)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_intent_dim = output_intent_dim
        self.output_price_dim = output_price_dim
        
        # Encoder
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Intent head
        self.intent_head = nn.Linear(prev_dim, output_intent_dim)
        
        # Price head
        self.price_head = nn.Linear(prev_dim, output_price_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier uniform initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: States (batch, input_dim)
            
        Returns:
            intent_logits: (batch, output_intent_dim)
            price_mean: (batch, 1)
            price_std: (batch, 1)
        """
        # Encode state
        x = self.encoder(state)
        
        # Intent output
        intent_logits = self.intent_head(x)
        
        # Price output
        price_params = self.price_head(x)
        price_mean = torch.sigmoid(price_params[:, 0:1])
        price_std = F.softplus(price_params[:, 1:2]) + 1e-6
        
        return intent_logits, price_mean, price_std
    
    def predict(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict action
        
        Args:
            state: States
            deterministic: Use argmax/mean
            
        Returns:
            intent: (batch,)
            price: (batch,)
        """
        intent_logits, price_mean, price_std = self.forward(state)
        
        # Intent
        if deterministic:
            intent = torch.argmax(intent_logits, dim=-1)
        else:
            intent_probs = F.softmax(intent_logits, dim=-1)
            intent = torch.multinomial(intent_probs, 1).squeeze(-1)
        
        # Price
        if deterministic:
            price = price_mean.squeeze(-1)
        else:
            price_dist = torch.distributions.Normal(price_mean, price_std)
            price = price_dist.sample().squeeze(-1)
            price = torch.clamp(price, 0.0, 1.0)
        
        return intent, price


# ==================== Behavior Cloning Trainer ====================

class BehaviorCloningTrainer:
    """
    Train supervised agent via behavior cloning
    
    Loss:
    - Intent: Cross-entropy
    - Price: MSE or Gaussian NLL
    
    Used in Algorithm 1, Line 1:
    Initialize policy π(SL) via supervised learning
    """
    
    def __init__(
        self,
        agent: SupervisedAgent,
        learning_rate: float = 1e-3,
        intent_loss_weight: float = 1.0,
        price_loss_weight: float = 1.0,
        use_nll_loss: bool = True,
        device: str = 'cpu',
    ):
        """
        Args:
            agent: SupervisedAgent to train
            learning_rate: Learning rate (1e-3 in paper)
            intent_loss_weight: Weight for intent loss
            price_loss_weight: Weight for price loss
            use_nll_loss: Use negative log-likelihood for price
            device: Device
        """
        self.agent = agent
        self.device = device
        self.intent_loss_weight = intent_loss_weight
        self.price_loss_weight = price_loss_weight
        self.use_nll_loss = use_nll_loss
        
        # Optimizer
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
        
        # Move to device
        self.agent.to(device)
    
    def compute_loss(
        self,
        states: torch.Tensor,
        intent_targets: torch.Tensor,
        price_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute behavior cloning loss
        
        Args:
            states: States (batch, input_dim)
            intent_targets: Target intents (batch,)
            price_targets: Target prices (batch,)
            
        Returns:
            Dictionary with losses
        """
        # Forward pass
        intent_logits, price_mean, price_std = self.agent(states)
        
        # Intent loss (cross-entropy)
        intent_loss = F.cross_entropy(intent_logits, intent_targets)
        
        # Price loss
        if self.use_nll_loss:
            # Negative log-likelihood (Gaussian)
            price_dist = torch.distributions.Normal(price_mean.squeeze(-1), price_std.squeeze(-1))
            price_loss = -price_dist.log_prob(price_targets).mean()
        else:
            # MSE
            price_loss = F.mse_loss(price_mean.squeeze(-1), price_targets)
        
        # Total loss
        total_loss = (
            self.intent_loss_weight * intent_loss +
            self.price_loss_weight * price_loss
        )
        
        # Accuracy
        intent_pred = torch.argmax(intent_logits, dim=-1)
        intent_accuracy = (intent_pred == intent_targets).float().mean()
        
        return {
            'total_loss': total_loss,
            'intent_loss': intent_loss,
            'price_loss': price_loss,
            'intent_accuracy': intent_accuracy,
        }
    
    def train_step(
        self,
        states: torch.Tensor,
        intent_targets: torch.Tensor,
        price_targets: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            states: States
            intent_targets: Target intents
            price_targets: Target prices
            
        Returns:
            Loss dictionary
        """
        self.agent.train()
        
        # Move to device
        states = states.to(self.device)
        intent_targets = intent_targets.to(self.device)
        price_targets = price_targets.to(self.device)
        
        # Compute loss
        loss_dict = self.compute_loss(states, intent_targets, price_targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
        
        # Update
        self.optimizer.step()
        
        # Convert to floats
        return {k: v.item() for k, v in loss_dict.items()}
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: DataLoader
            
        Returns:
            Average losses
        """
        total_losses = {
            'total_loss': 0.0,
            'intent_loss': 0.0,
            'price_loss': 0.0,
            'intent_accuracy': 0.0,
        }
        
        num_batches = 0
        
        for batch in dataloader:
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
        
        # Average
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate on validation set
        
        Args:
            dataloader: DataLoader
            
        Returns:
            Average losses
        """
        self.agent.eval()
        
        total_losses = {
            'total_loss': 0.0,
            'intent_loss': 0.0,
            'price_loss': 0.0,
            'intent_accuracy': 0.0,
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
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
        
        # Average
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Saved supervised agent to {path}")
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded supervised agent from {path}")


# ==================== Training Function ====================

def train_supervised_agent(
    agent: SupervisedAgent,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    save_path: Optional[str] = None,
    device: str = 'cpu',
) -> SupervisedAgent:
    """
    Train supervised agent via behavior cloning
    
    Implements Algorithm 1, Line 1:
    Initialize policy π(SL) via supervised learning on human data
    
    Args:
        agent: SupervisedAgent
        train_dataloader: Training data
        val_dataloader: Validation data
        num_epochs: Number of epochs (10 in paper)
        learning_rate: Learning rate (1e-3 in paper)
        save_path: Path to save best model
        device: Device
        
    Returns:
        Trained agent
    """
    print(f"Training supervised agent for {num_epochs} epochs...")
    
    # Create trainer
    trainer = BehaviorCloningTrainer(
        agent=agent,
        learning_rate=learning_rate,
        device=device,
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        train_losses = trainer.train_epoch(train_dataloader)
        
        # Validate
        val_losses = trainer.evaluate(val_dataloader)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_losses['total_loss']:.4f}, "
              f"Intent Acc: {train_losses['intent_accuracy']:.4f}")
        print(f"  Val   - Loss: {val_losses['total_loss']:.4f}, "
              f"Intent Acc: {val_losses['intent_accuracy']:.4f}")
        
        # Save best model
        if save_path and val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            trainer.save(save_path)
            print(f"  Saved best model (val_loss={best_val_loss:.4f})")
    
    return agent


# ==================== Helper Functions ====================

def create_supervised_agent(config) -> SupervisedAgent:
    """
    Create supervised agent from config
    
    Args:
        config: Configuration object
        
    Returns:
        SupervisedAgent instance
    """
    return SupervisedAgent(
        input_dim=config.network.policy_input_dim,
        hidden_layers=config.network.policy_hidden_layers,
        output_intent_dim=config.network.policy_output_intent_dim,
        output_price_dim=config.network.policy_output_price_dim,
        dropout=config.network.policy_dropout,
    )


def load_supervised_agent(
    path: str,
    config,
    device: str = 'cpu',
) -> SupervisedAgent:
    """
    Load pretrained supervised agent
    
    Args:
        path: Path to checkpoint
        config: Configuration
        device: Device
        
    Returns:
        Loaded agent
    """
    agent = create_supervised_agent(config)
    checkpoint = torch.load(path, map_location=device)
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.to(device)
    agent.eval()
    
    print(f"Loaded supervised agent from {path}")
    return agent


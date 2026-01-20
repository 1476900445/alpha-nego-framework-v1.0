"""
Distributional Soft Actor-Critic (DSAC)
Implements Algorithm 2 from the paper

Algorithm 2: DSAC Training Procedure
- Step 1: Update distributional critic with quantile regression
- Step 2: Compute style-specific Q-values
- Step 3: Update policy with entropy regularization and KL penalty

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy


class DSACAgent:
    """
    DSAC Agent implementing Algorithm 2
    
    Combines:
    - Distributional critic with quantile regression (Eq. 4-5)
    - Soft actor-critic with entropy regularization (Eq. 11)
    - KL regularization with SL policy (Eq. 8)
    - Style-specific Q-value computation (Eq. 12-14)
    """
    
    def __init__(
        self,
        policy: nn.Module,
        critic: nn.Module,
        config,
        reference_policy: Optional[nn.Module] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Args:
            policy: Policy network
            critic: Distributional critic network
            config: Configuration object
            reference_policy: Reference policy for KL regularization (SL agent)
            device: Device for computation
        """
        self.policy = policy
        self.critic = critic
        self.reference_policy = reference_policy
        self.config = config
        self.device = device
        
        # Create target networks
        self.target_critic = copy.deepcopy(critic)
        self.target_critic.eval()
        
        # Freeze target network
        for param in self.target_critic.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config.training.rl_policy_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=config.training.rl_critic_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Hyperparameters
        self.gamma = config.training.gamma
        self.tau = config.training.soft_update_tau
        self.alpha_entropy_intent = config.training.alpha_entropy_intent
        self.beta_entropy_price = config.training.beta_entropy_price
        self.alpha_kl = config.training.alpha_kl
        self.gradient_clip_norm = config.training.gradient_clip_norm
        self.num_quantiles = config.training.num_quantiles
        self.huber_kappa = config.training.huber_kappa
        
        # Style
        self.style = config.style_control.active_style
        
        # Statistics
        self.training_steps = 0
        self.policy_updates = 0
        self.critic_updates = 0
        
        print(f"[DSACAgent] Initialized")
        print(f"  Policy LR: {config.training.rl_policy_lr}")
        print(f"  Critic LR: {config.training.rl_critic_lr}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Tau: {self.tau}")
        print(f"  Alpha KL: {self.alpha_kl}")
        print(f"  Style: {self.style}")
    
    def update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update distributional critic (Algorithm 2, Step 1)
        
        Implements quantile regression loss (Eq. 4-5):
        L_QR(θ) = E[Σ_i Σ_j ρ^κ_τ̂_i(δ_ij)]
        
        Args:
            states: States (batch, state_dim)
            actions: Actions (batch, action_dim)
            rewards: Rewards (batch,)
            next_states: Next states (batch, state_dim)
            dones: Done flags (batch,)
            
        Returns:
            Dictionary with loss values
        """
        # Get current quantiles from both critics
        current_quantiles_list = self.critic.get_all_quantiles(states, actions)
        
        # Compute target quantiles
        with torch.no_grad():
            # Sample next actions from current policy
            next_intent_logits, next_price_mean, next_price_std = self.policy(next_states)
            next_intent_probs = F.softmax(next_intent_logits, dim=-1)
            next_intent = torch.multinomial(next_intent_probs, 1).squeeze(-1)
            
            # Sample next price
            next_price_dist = torch.distributions.Normal(next_price_mean, next_price_std)
            next_price = next_price_dist.sample().squeeze(-1)
            next_price = torch.clamp(next_price, 0.0, 1.0)
            
            next_actions = torch.stack([next_intent.float(), next_price], dim=-1)
            
            # Get target quantiles (use target network)
            target_quantiles_list = self.target_critic.get_all_quantiles(next_states, next_actions)
            
            # Take minimum over both target critics
            target_quantiles = torch.min(
                torch.stack(target_quantiles_list, dim=0),
                dim=0
            )[0]
            
            # Compute Bellman target: r + γ * Z(s', a')
            # Shape: (batch, num_quantiles)
            rewards_expanded = rewards.unsqueeze(-1).expand_as(target_quantiles)
            dones_expanded = dones.unsqueeze(-1).expand_as(target_quantiles)
            
            target_quantiles = rewards_expanded + (1 - dones_expanded.float()) * self.gamma * target_quantiles
        
        # Compute quantile regression loss for both critics
        total_loss = 0.0
        losses = []
        
        for i, current_quantiles in enumerate(current_quantiles_list):
            loss = self._quantile_huber_loss(
                current_quantiles,
                target_quantiles,
                self.critic.tau,
                self.huber_kappa
            )
            losses.append(loss)
            total_loss += loss
        
        # Backward pass
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.gradient_clip_norm
        )
        
        self.critic_optimizer.step()
        self.critic_updates += 1
        
        return {
            'critic_loss': total_loss.item(),
            'q1_loss': losses[0].item(),
            'q2_loss': losses[1].item() if len(losses) > 1 else 0.0,
        }
    
    def _quantile_huber_loss(
        self,
        quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
        tau: torch.Tensor,
        kappa: float = 1.0,
    ) -> torch.Tensor:
        """
        Quantile Huber loss (Eq. 4-5)
        
        Args:
            quantiles: Current quantiles (batch, num_quantiles)
            target_quantiles: Target quantiles (batch, num_quantiles)
            tau: Quantile fractions (num_quantiles,)
            kappa: Huber threshold
            
        Returns:
            Loss scalar
        """
        batch_size = quantiles.size(0)
        num_quantiles = quantiles.size(1)
        
        # Expand dimensions for broadcasting
        quantiles = quantiles.unsqueeze(2)  # (batch, num_quantiles, 1)
        target_quantiles = target_quantiles.unsqueeze(1)  # (batch, 1, num_quantiles)
        
        # TD errors: δ_ij = target - prediction
        td_errors = target_quantiles - quantiles  # (batch, num_quantiles, num_quantiles)
        
        # Huber loss
        huber_loss = torch.where(
            td_errors.abs() <= kappa,
            0.5 * td_errors ** 2,
            kappa * (td_errors.abs() - 0.5 * kappa)
        )
        
        # Quantile regression weight: |τ - I{δ<0}|
        tau = tau.view(1, -1, 1)  # (1, num_quantiles, 1)
        quantile_weight = torch.abs(tau - (td_errors < 0).float())
        
        # Weighted Huber loss
        loss = (quantile_weight * huber_loss).mean()
        
        return loss
    
    def update_policy(
        self,
        states: torch.Tensor,
        history: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Update policy (Algorithm 2, Step 2-3)
        
        Implements:
        - SAC objective (Eq. 11): J_π = E[α log π(a_i|s) + β log π(a_p|s) - Q(s,a)]
        - KL regularization (Eq. 8): + α_KL · KL(π||π_SL)
        
        Args:
            states: States (batch, state_dim)
            history: Dialogue history (batch, seq_len, state_dim)
            
        Returns:
            Dictionary with loss values
        """
        # Forward pass
        intent_logits, price_mean, price_std = self.policy(states, history)
        
        # Sample actions
        intent_probs = F.softmax(intent_logits, dim=-1)
        intent = torch.multinomial(intent_probs, 1).squeeze(-1)
        
        price_dist = torch.distributions.Normal(price_mean, price_std)
        price = price_dist.sample().squeeze(-1)
        price = torch.clamp(price, 0.0, 1.0)
        
        actions = torch.stack([intent.float(), price], dim=-1)
        
        # Evaluate log probabilities and entropy
        eval_result = self.policy.evaluate_action(states, intent, price, history)
        log_prob_intent = eval_result['log_prob_intent']
        log_prob_price = eval_result['log_prob_price']
        entropy_intent = eval_result['entropy_intent']
        entropy_price = eval_result['entropy_price']
        
        # Get Q-values with current style (Step 2)
        q_value, _ = self.critic(states, actions, style=self.style)
        
        # SAC loss (Eq. 11)
        policy_loss = (
            q_value
            - self.alpha_entropy_intent * log_prob_intent
            - self.beta_entropy_price * log_prob_price
        ).mean()
        
        # KL regularization (Eq. 8)
        kl_loss = 0.0
        if self.reference_policy is not None and self.alpha_kl > 0:
            kl_divergence = self.policy.get_kl_divergence(
                states,
                self.reference_policy,
                history
            )
            kl_loss = self.alpha_kl * kl_divergence.mean()
            policy_loss += kl_loss
        
        # Backward pass
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.gradient_clip_norm
        )
        
        self.policy_optimizer.step()
        self.policy_updates += 1
        
        return {
            'policy_loss': policy_loss.item(),
            'q_values': q_value.mean().item(),
            'entropy_intent': entropy_intent.mean().item(),
            'entropy_price': entropy_price.mean().item(),
            'kl_divergence': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
        }
    
    def soft_update_targets(self):
        """
        Soft update target networks (Polyak averaging)
        
        θ̄ ← τθ + (1-τ)θ̄
        """
        for param, target_param in zip(
            self.critic.parameters(),
            self.target_critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def update(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Complete DSAC update (Algorithm 2)
        
        Args:
            batch: Batch with states, actions, rewards, next_states, dones
            
        Returns:
            Combined loss dictionary
        """
        self.training_steps += 1
        
        # Extract batch
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Step 1: Update critic
        critic_losses = self.update_critic(
            states, actions, rewards, next_states, dones
        )
        
        # Step 2-3: Update policy (every N steps)
        policy_losses = {}
        if self.training_steps % self.config.training.policy_update_frequency == 0:
            history = batch.get('history', None)
            policy_losses = self.update_policy(states, history)
        
        # Soft update target networks
        self.soft_update_targets()
        
        # Combine losses
        losses = {**critic_losses, **policy_losses}
        losses['training_steps'] = self.training_steps
        
        return losses
    
    def set_style(self, style: str):
        """
        Set negotiation style
        
        Args:
            style: 'neutral', 'aggressive', or 'conservative'
        """
        assert style in ['neutral', 'aggressive', 'conservative']
        self.style = style
        print(f"[DSACAgent] Style changed to: {style}")
    
    def save(self, path: str):
        """Save agent checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_steps': self.training_steps,
            'policy_updates': self.policy_updates,
            'critic_updates': self.critic_updates,
            'style': self.style,
        }, path)
        print(f"[DSACAgent] Saved to {path}")
    
    def load(self, path: str):
        """Load agent checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_steps = checkpoint.get('training_steps', 0)
        self.policy_updates = checkpoint.get('policy_updates', 0)
        self.critic_updates = checkpoint.get('critic_updates', 0)
        self.style = checkpoint.get('style', 'neutral')
        print(f"[DSACAgent] Loaded from {path}")


# ==================== Helper Functions ====================

def create_dsac_agent(
    config,
    supervised_agent: Optional[nn.Module] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> DSACAgent:
    """
    Factory function to create DSAC agent
    
    Args:
        config: Configuration object
        supervised_agent: Optional SL agent for KL regularization
        device: Device
        
    Returns:
        DSACAgent instance
    """
    from models.policy_network import create_policy_network
    from models.critic_network import create_distributional_critic
    
    # Create networks
    policy = create_policy_network(config)
    critic = create_distributional_critic(config)
    
    # Initialize policy from supervised agent if provided
    if supervised_agent is not None:
        print("[Factory] Initializing policy from supervised agent...")
        policy.load_state_dict(supervised_agent.state_dict(), strict=False)
    
    # Create DSAC agent
    agent = DSACAgent(
        policy=policy,
        critic=critic,
        config=config,
        reference_policy=supervised_agent,
        device=device,
    )
    
    return agent


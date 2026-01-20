"""
Distributional Critic Network for α-Nego
Implements quantile regression and style-specific Q-value computation

Architecture (Paper Section 5.1):
- Input: State (3) + Action (2) = 5 dims
- Hidden: [256, 256]
- Output: 51 quantiles for value distribution

Implements:
- Eq. 4-5: Quantile regression with Huber loss
- Eq. 12: Neutral style (mean of quantiles)
- Eq. 13: Aggressive style (mean + variance bonus)
- Eq. 14: Conservative style (CVaR lower tail)

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


# ==================== Quantile Network ====================

class QuantileNetwork(nn.Module):
    """
    Network that outputs value distribution as quantiles
    
    Implements distributional RL with 51 quantiles (Paper Section 5.1)
    """
    
    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 2,
        hidden_layers: List[int] = [256, 256],
        num_quantiles: int = 51,
        layer_norm: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_layers: Hidden layer sizes
            num_quantiles: Number of quantiles (51 in paper)
            layer_norm: Use layer normalization
            dropout: Dropout rate
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        
        # State-action encoder
        input_dim = state_dim + action_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Quantile output
        self.quantile_head = nn.Linear(prev_dim, num_quantiles)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier uniform initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: States (batch, state_dim)
            action: Actions (batch, action_dim)
            
        Returns:
            Quantiles (batch, num_quantiles)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        # Encode
        x = self.encoder(x)
        
        # Output quantiles
        quantiles = self.quantile_head(x)
        
        return quantiles


# ==================== Distributional Critic ====================

class DistributionalCritic(nn.Module):
    """
    Distributional critic with style-specific Q-value computation
    
    Uses dual Q-networks (θ1, θ2) for stability
    Implements three styles:
    - Neutral (Eq. 12): Q = E[Z]
    - Aggressive (Eq. 13): Q = E[Z_upper] + α_agg * Var[Z_upper]
    - Conservative (Eq. 14): Q = CVaR_α[Z]
    """
    
    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 2,
        hidden_layers: List[int] = [256, 256],
        num_quantiles: int = 51,
        num_critics: int = 2,
        layer_norm: bool = True,
        dropout: float = 0.1,
        alpha_agg: float = 1.0,
        alpha_con: float = 0.2,
    ):
        """
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_layers: Hidden layer sizes
            num_quantiles: Number of quantiles (51)
            num_critics: Number of Q-networks (2 for double Q)
            layer_norm: Use layer normalization
            dropout: Dropout rate
            alpha_agg: Variance bonus weight (Eq. 13)
            alpha_con: CVaR parameter (Eq. 14)
        """
        super().__init__()
        
        self.num_quantiles = num_quantiles
        self.num_critics = num_critics
        self.alpha_agg = alpha_agg
        self.alpha_con = alpha_con
        
        # Create quantile fractions (τ_i)
        # τ_i = (i + 0.5) / N for i in [0, N-1]
        tau = torch.arange(0, num_quantiles, dtype=torch.float32)
        tau = (tau + 0.5) / num_quantiles
        self.register_buffer('tau', tau)
        
        # Create dual Q-networks
        self.critics = nn.ModuleList([
            QuantileNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_layers=hidden_layers,
                num_quantiles=num_quantiles,
                layer_norm=layer_norm,
                dropout=dropout,
            )
            for _ in range(num_critics)
        ])
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        style: str = 'neutral',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with style-specific Q-value computation
        
        Args:
            state: States (batch, state_dim)
            action: Actions (batch, action_dim)
            style: 'neutral', 'aggressive', or 'conservative'
            
        Returns:
            q_value: Q-value (batch,)
            quantiles: Full distribution (batch, num_quantiles)
        """
        # Get quantiles from both critics
        quantiles_list = [critic(state, action) for critic in self.critics]
        
        # Compute Q-value based on style
        if style == 'neutral':
            q_value = self._compute_neutral_q(quantiles_list)
        elif style == 'aggressive':
            q_value = self._compute_aggressive_q(quantiles_list)
        elif style == 'conservative':
            q_value = self._compute_conservative_q(quantiles_list)
        else:
            raise ValueError(f"Unknown style: {style}")
        
        # Return min Q-value and quantiles
        quantiles = torch.min(torch.stack(quantiles_list, dim=0), dim=0)[0]
        
        return q_value, quantiles
    
    def _compute_neutral_q(self, quantiles_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Neutral style: Mean of all quantiles (Eq. 12)
        
        Q(s,a) = min_{k=1,2} E_{τ_i∼[0,1]} Z_{τ_i}(s,a;θ_k)
        
        Args:
            quantiles_list: List of quantile tensors from each critic
            
        Returns:
            Q-value (batch,)
        """
        q_values = []
        
        for quantiles in quantiles_list:
            # Mean of all quantiles
            q = quantiles.mean(dim=-1)
            q_values.append(q)
        
        # Take minimum (double Q-learning)
        q_value = torch.min(torch.stack(q_values, dim=0), dim=0)[0]
        
        return q_value
    
    def _compute_aggressive_q(self, quantiles_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggressive style: Mean + left truncated variance (Eq. 13)
        
        Q(s,a) = min_{k=1,2} [E_{τ_i∼[0,1]} Z_{τ_i} + α_agg · σ²_+(s,a;θ_k)]
        
        σ²_+ is left truncated variance (upper tail focus)
        
        Args:
            quantiles_list: List of quantile tensors
            
        Returns:
            Q-value (batch,)
        """
        q_values = []
        
        for quantiles in quantiles_list:
            # Split into lower and upper half
            mid_idx = self.num_quantiles // 2
            upper_quantiles = quantiles[:, mid_idx:]
            
            # Mean of upper quantiles
            mean_upper = upper_quantiles.mean(dim=-1)
            
            # Variance of upper quantiles
            var_upper = upper_quantiles.var(dim=-1)
            
            # Q = mean + α_agg * variance
            q = mean_upper + self.alpha_agg * var_upper
            q_values.append(q)
        
        # Take minimum
        q_value = torch.min(torch.stack(q_values, dim=0), dim=0)[0]
        
        return q_value
    
    def _compute_conservative_q(self, quantiles_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Conservative style: CVaR of lower tail (Eq. 14)
        
        Q(s,a) = min_{k=1,2} E_{τ_i∼[0,α_con]} Z_{τ_i}(s,a;θ_k)
        
        CVaR focuses on worst-case scenarios (risk-averse)
        
        Args:
            quantiles_list: List of quantile tensors
            
        Returns:
            Q-value (batch,)
        """
        q_values = []
        
        # Number of quantiles to consider (lower α_con fraction)
        num_lower = max(1, int(self.num_quantiles * self.alpha_con))
        
        for quantiles in quantiles_list:
            # Take lower tail quantiles
            lower_quantiles = quantiles[:, :num_lower]
            
            # CVaR: mean of lower tail
            q = lower_quantiles.mean(dim=-1)
            q_values.append(q)
        
        # Take minimum
        q_value = torch.min(torch.stack(q_values, dim=0), dim=0)[0]
        
        return q_value
    
    def get_all_quantiles(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Get quantiles from all critics
        
        Args:
            state: States
            action: Actions
            
        Returns:
            List of quantile tensors
        """
        return [critic(state, action) for critic in self.critics]


# ==================== Quantile Huber Loss ====================

def quantile_huber_loss(
    quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    tau: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """
    Quantile Huber loss for distributional RL
    
    Implements Eq. 4-5 from paper:
    
    L_QR(θ) = E[Σ_i Σ_j ρ^κ_τ̂_i(δ_ij)]
    
    where:
    - δ_ij = r + γZ_{τ_i}(s',a';θ̄) - Z_{τ_j}(s,a;θ)
    - ρ^κ_τ(u) = |τ - I{u<0}| * L^κ_δ(u)
    - L^κ_δ(u) = Huber loss with threshold κ
    
    Args:
        quantiles: Current quantiles (batch, num_quantiles)
        target_quantiles: Target quantiles (batch, num_quantiles)
        tau: Quantile fractions (num_quantiles,)
        kappa: Huber loss threshold (1.0 in paper)
        
    Returns:
        Loss scalar
    """
    batch_size = quantiles.size(0)
    num_quantiles = quantiles.size(1)
    
    # Expand dimensions for broadcasting
    # quantiles: (batch, num_quantiles, 1)
    # target_quantiles: (batch, 1, num_quantiles)
    quantiles = quantiles.unsqueeze(2)
    target_quantiles = target_quantiles.unsqueeze(1)
    
    # Compute TD error: δ_ij = target - prediction
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


# ==================== Helper Functions ====================

def create_distributional_critic(config) -> DistributionalCritic:
    """
    Create distributional critic from config
    
    Args:
        config: Configuration object
        
    Returns:
        DistributionalCritic instance
    """
    return DistributionalCritic(
        state_dim=config.network.critic_input_dim - config.network.policy_output_intent_dim - 1,
        action_dim=config.network.policy_output_intent_dim + 1,
        hidden_layers=config.network.critic_hidden_layers,
        num_quantiles=config.network.critic_num_quantiles,
        num_critics=config.training.num_critics,
        layer_norm=config.network.critic_layer_norm,
        dropout=config.network.critic_dropout,
        alpha_agg=config.style_control.alpha_agg,
        alpha_con=config.style_control.alpha_con,
    )


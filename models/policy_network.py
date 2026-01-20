"""
Policy Network for α-Nego
Implements dual-head architecture with attention mechanism

Architecture (Paper Section 5.1):
- Input: State encoding (3 dims for Craigslist, variable for Dealornodeal)
- Hidden: 2 layers × 256 units
- Output: Intent logits (16 for Craigslist) + Price distribution (mean, std)

Enhancements:
- Multi-head attention for dialogue history
- Residual connections with layer normalization
- Positional encoding for temporal information

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


# ==================== Attention Module ====================

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for dialogue history
    
    Enhanced feature beyond paper specifications
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len)
            
        Returns:
            Output: (batch, seq_len, d_model)
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, V)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output


# ==================== Positional Encoding ====================

class PositionalEncoding(nn.Module):
    """
    Positional encoding for dialogue turns
    """
    
    def __init__(self, d_model: int = 256, max_len: int = 20):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


# ==================== Residual Block ====================

class ResidualBlock(nn.Module):
    """
    Residual block with layer normalization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
            
        Returns:
            Output with residual connection
        """
        residual = x
        
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        x = x + residual
        x = self.norm2(x)
        
        return x


# ==================== State Encoder ====================

class StateEncoder(nn.Module):
    """
    Encode state with attention mechanism
    
    Enhanced beyond paper specifications
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_heads: int = 4,
        use_attention: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: State dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            use_attention: Whether to use attention
            dropout: Dropout rate
        """
        super().__init__()
        
        self.use_attention = use_attention
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Attention mechanism
        if use_attention:
            self.positional_encoding = PositionalEncoding(hidden_dim)
            self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
            self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        state: torch.Tensor,
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            state: Current state (batch, input_dim)
            history: Dialogue history (batch, seq_len, input_dim)
            
        Returns:
            Encoded state (batch, hidden_dim)
        """
        # Project state
        x = self.input_proj(state)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Apply attention to history if available
        if self.use_attention and history is not None:
            # Project history
            h = self.input_proj(history)
            h = self.positional_encoding(h)
            
            # Apply attention
            x_expanded = x.unsqueeze(1)  # (batch, 1, hidden_dim)
            attn_out = self.attention(x_expanded, h, h)
            x = x + attn_out.squeeze(1)
            x = self.attention_norm(x)
        
        # Final projection
        x = self.output_proj(x)
        x = F.relu(x)
        
        return x


# ==================== Policy Network ====================

class PolicyNetwork(nn.Module):
    """
    Policy network for negotiation
    
    Architecture (Paper Section 5.1):
    - Input: State (3 dims for Craigslist)
    - Hidden: [256, 256]
    - Output: Intent logits (16) + Price distribution (mean, std)
    
    Implements Eq. 11:
    J_π(φ) = E[α log π(a_i|s) + β log π(a_p|s) - Q(s,a)]
    
    With KL regularization (Eq. 8):
    J^α-Nego_π = J_π + α_KL · KL(π||π_SL)
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_layers: List[int] = [256, 256],
        output_intent_dim: int = 16,
        output_price_dim: int = 2,  # mean and std
        use_attention: bool = True,
        attention_heads: int = 4,
        use_residual: bool = True,
        layer_norm: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: State dimension (3 for Craigslist)
            hidden_layers: Hidden layer sizes [256, 256]
            output_intent_dim: Number of dialogue acts (16 for Craigslist)
            output_price_dim: Price distribution params (2: mean, std)
            use_attention: Use attention mechanism
            attention_heads: Number of attention heads
            use_residual: Use residual connections
            layer_norm: Use layer normalization
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_intent_dim = output_intent_dim
        self.output_price_dim = output_price_dim
        self.use_residual = use_residual
        
        # State encoder with attention
        self.state_encoder = StateEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_layers[0],
            num_heads=attention_heads,
            use_attention=use_attention,
            dropout=dropout,
        )
        
        # Hidden layers
        self.hidden = nn.ModuleList()
        prev_dim = hidden_layers[0]
        
        for hidden_dim in hidden_layers[1:]:
            if use_residual and prev_dim == hidden_dim:
                self.hidden.append(ResidualBlock(prev_dim, hidden_dim, dropout))
            else:
                layer = nn.Sequential(
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.hidden.append(layer)
            prev_dim = hidden_dim
        
        # Intent head (discrete action)
        self.intent_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prev_dim // 2, output_intent_dim),
        )
        
        # Price head (continuous action)
        self.price_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prev_dim // 2, output_price_dim),
        )
        
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
        history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: Current state (batch, input_dim)
            history: Dialogue history (batch, seq_len, input_dim)
            
        Returns:
            intent_logits: (batch, output_intent_dim)
            price_mean: (batch, 1)
            price_std: (batch, 1)
        """
        # Encode state
        x = self.state_encoder(state, history)
        
        # Hidden layers
        for layer in self.hidden:
            x = layer(x)
        
        # Intent output
        intent_logits = self.intent_head(x)
        
        # Price output
        price_params = self.price_head(x)
        price_mean = torch.sigmoid(price_params[:, 0:1])  # [0, 1]
        price_std = F.softplus(price_params[:, 1:2]) + 1e-6  # > 0
        
        return intent_logits, price_mean, price_std
    
    def sample_action(
        self,
        state: torch.Tensor,
        history: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample action from policy
        
        Args:
            state: Current state
            history: Dialogue history
            deterministic: Use argmax/mean instead of sampling
            
        Returns:
            intent: Sampled intent (batch,)
            price: Sampled price (batch,)
            info: Dict with logits, probs, etc.
        """
        intent_logits, price_mean, price_std = self.forward(state, history)
        
        # Sample intent (discrete)
        intent_probs = F.softmax(intent_logits, dim=-1)
        if deterministic:
            intent = torch.argmax(intent_probs, dim=-1)
        else:
            intent = torch.multinomial(intent_probs, num_samples=1).squeeze(-1)
        
        # Sample price (continuous)
        if deterministic:
            price = price_mean.squeeze(-1)
        else:
            # Gaussian sampling
            price_dist = torch.distributions.Normal(price_mean, price_std)
            price = price_dist.sample().squeeze(-1)
            price = torch.clamp(price, 0.0, 1.0)  # Clip to [0, 1]
        
        info = {
            'intent_logits': intent_logits,
            'intent_probs': intent_probs,
            'price_mean': price_mean,
            'price_std': price_std,
        }
        
        return intent, price, info
    
    def evaluate_action(
        self,
        state: torch.Tensor,
        intent: torch.Tensor,
        price: torch.Tensor,
        history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate action log probabilities and entropy
        
        Used in policy gradient (Eq. 11)
        
        Args:
            state: States (batch, input_dim)
            intent: Intents (batch,)
            price: Prices (batch,)
            history: Dialogue history
            
        Returns:
            Dict with log_prob_intent, log_prob_price, entropy
        """
        intent_logits, price_mean, price_std = self.forward(state, history)
        
        # Intent log probability
        intent_log_probs = F.log_softmax(intent_logits, dim=-1)
        log_prob_intent = intent_log_probs.gather(1, intent.unsqueeze(-1)).squeeze(-1)
        
        # Intent entropy (for exploration bonus)
        intent_entropy = -(intent_log_probs * torch.exp(intent_log_probs)).sum(dim=-1)
        
        # Price log probability
        price_dist = torch.distributions.Normal(price_mean.squeeze(-1), price_std.squeeze(-1))
        log_prob_price = price_dist.log_prob(price)
        
        # Price entropy
        price_entropy = 0.5 * torch.log(2 * np.pi * np.e * price_std.squeeze(-1) ** 2)
        
        return {
            'log_prob_intent': log_prob_intent,
            'log_prob_price': log_prob_price,
            'entropy_intent': intent_entropy,
            'entropy_price': price_entropy,
            'intent_logits': intent_logits,
            'price_mean': price_mean,
            'price_std': price_std,
        }
    
    def get_kl_divergence(
        self,
        state: torch.Tensor,
        reference_policy: 'PolicyNetwork',
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute KL divergence with reference policy (SL agent)
        
        Implements Eq. 8:
        KL(π(φ)||π(SL)) for KL regularization
        
        Args:
            state: States
            reference_policy: Supervised learning policy
            history: Dialogue history
            
        Returns:
            KL divergence (batch,)
        """
        # Current policy
        intent_logits, price_mean, price_std = self.forward(state, history)
        intent_probs = F.softmax(intent_logits, dim=-1)
        
        # Reference policy
        with torch.no_grad():
            ref_intent_logits, ref_price_mean, ref_price_std = reference_policy.forward(state, history)
            ref_intent_probs = F.softmax(ref_intent_logits, dim=-1)
        
        # KL for intent (discrete)
        kl_intent = (intent_probs * (torch.log(intent_probs + 1e-10) - torch.log(ref_intent_probs + 1e-10))).sum(dim=-1)
        
        # KL for price (continuous Gaussian)
        # KL(N(μ1,σ1)||N(μ2,σ2)) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
        kl_price = (
            torch.log(ref_price_std / price_std) +
            (price_std ** 2 + (price_mean - ref_price_mean) ** 2) / (2 * ref_price_std ** 2) -
            0.5
        ).squeeze(-1)
        
        # Total KL
        kl = kl_intent + kl_price
        
        return kl


# ==================== Helper Functions ====================

def create_policy_network(config) -> PolicyNetwork:
    """
    Create policy network from config
    
    Args:
        config: Configuration object
        
    Returns:
        PolicyNetwork instance
    """
    return PolicyNetwork(
        input_dim=config.network.policy_input_dim,
        hidden_layers=config.network.policy_hidden_layers,
        output_intent_dim=config.network.policy_output_intent_dim,
        output_price_dim=config.network.policy_output_price_dim,
        use_attention=config.network.use_attention,
        attention_heads=config.network.attention_heads,
        use_residual=config.network.policy_use_residual,
        layer_norm=config.network.policy_layer_norm,
        dropout=config.network.policy_dropout,
    )


"""
Encoders for α-Nego
Various encoding methods for states, actions, and dialogue history

Author: α-Nego Implementation  
Reference: Chen et al., 2025, Electronics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


# ==================== Dialogue History Encoder ====================

class DialogueHistoryEncoder(nn.Module):
    """
    Encode dialogue history using RNN/LSTM/GRU
    
    Alternative to attention mechanism for temporal encoding
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 2,
        rnn_type: str = 'lstm',
        bidirectional: bool = False,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Input dimension per timestep
            hidden_dim: Hidden state dimension
            num_layers: Number of RNN layers
            rnn_type: 'lstm', 'gru', or 'rnn'
            bidirectional: Use bidirectional RNN
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # RNN module
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
            )
        else:
            self.rnn = nn.RNN(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
            )
        
        # Output dimension
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
    
    def forward(
        self,
        history: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode dialogue history
        
        Args:
            history: (batch, seq_len, input_dim)
            lengths: Actual sequence lengths (batch,)
            
        Returns:
            output: (batch, seq_len, hidden_dim * num_directions)
            hidden: Final hidden state (batch, hidden_dim * num_directions)
        """
        batch_size = history.size(0)
        
        # Pack padded sequence if lengths provided
        if lengths is not None:
            history = nn.utils.rnn.pack_padded_sequence(
                history, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # Forward pass
        output, hidden = self.rnn(history)
        
        # Unpack sequence
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # Extract final hidden state
        if isinstance(hidden, tuple):  # LSTM
            hidden = hidden[0]  # (num_layers * num_directions, batch, hidden_dim)
        
        # Get last layer
        if self.bidirectional:
            # Concatenate forward and backward
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden = hidden[-1]
        
        return output, hidden


# ==================== Intent Encoder ====================

class IntentEncoder(nn.Module):
    """
    Encode dialogue acts/intents as embeddings
    """
    
    def __init__(
        self,
        num_intents: int = 16,
        embedding_dim: int = 64,
    ):
        """
        Args:
            num_intents: Number of dialogue acts
            embedding_dim: Embedding dimension
        """
        super().__init__()
        
        self.num_intents = num_intents
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(num_intents, embedding_dim)
    
    def forward(self, intent_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            intent_ids: (batch,) or (batch, seq_len)
            
        Returns:
            Embeddings: (batch, embedding_dim) or (batch, seq_len, embedding_dim)
        """
        return self.embedding(intent_ids)


# ==================== Negotiation State Encoder ====================

class NegotiationStateEncoder(nn.Module):
    """
    Encode negotiation state with structured features
    
    For Craigslistbargain:
    - Current price
    - Opponent's last price
    - Turn number
    - Additional features (price history, concession rate, etc.)
    """
    
    def __init__(
        self,
        base_dim: int = 3,
        use_price_history: bool = True,
        history_length: int = 5,
        output_dim: int = 256,
    ):
        """
        Args:
            base_dim: Base state dimension (3 for Craigslist)
            use_price_history: Include price history features
            history_length: Length of price history
            output_dim: Output encoding dimension
        """
        super().__init__()
        
        self.base_dim = base_dim
        self.use_price_history = use_price_history
        self.history_length = history_length
        
        # Calculate input dimension
        input_dim = base_dim
        if use_price_history:
            input_dim += history_length * 2  # My prices + opponent prices
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        base_state: torch.Tensor,
        price_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            base_state: (batch, base_dim)
            price_history: (batch, history_length, 2)
            
        Returns:
            Encoded state: (batch, output_dim)
        """
        if self.use_price_history and price_history is not None:
            # Flatten price history
            batch_size = base_state.size(0)
            price_history_flat = price_history.view(batch_size, -1)
            state = torch.cat([base_state, price_history_flat], dim=-1)
        else:
            state = base_state
        
        return self.encoder(state)


# ==================== Multi-Issue Encoder ====================

class MultiIssueEncoder(nn.Module):
    """
    Encode multi-issue negotiation state (Dealornodeal)
    
    Encodes:
    - Item values
    - Item counts
    - Current proposals
    - Opponent modeling estimates
    """
    
    def __init__(
        self,
        num_items: int = 3,
        max_count: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ):
        """
        Args:
            num_items: Number of negotiable items
            max_count: Maximum count per item
            hidden_dim: Hidden dimension
            output_dim: Output dimension
        """
        super().__init__()
        
        self.num_items = num_items
        self.max_count = max_count
        
        # Each item: value, count, proposal, opp_value_est
        input_dim = num_items * 4
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        values: torch.Tensor,
        counts: torch.Tensor,
        proposals: torch.Tensor,
        opponent_values_est: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            values: (batch, num_items)
            counts: (batch, num_items)
            proposals: (batch, num_items)
            opponent_values_est: (batch, num_items)
            
        Returns:
            Encoded state: (batch, output_dim)
        """
        # Normalize to [0, 1]
        values_norm = values / self.max_count
        counts_norm = counts / self.max_count
        proposals_norm = proposals / self.max_count
        opp_values_norm = opponent_values_est / self.max_count
        
        # Concatenate all features
        x = torch.cat([values_norm, counts_norm, proposals_norm, opp_values_norm], dim=-1)
        
        return self.encoder(x)


# ==================== Utterance Encoder ====================

class UtteranceEncoder(nn.Module):
    """
    Encode natural language utterances
    
    Uses pretrained embeddings or learns from scratch
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
    ):
        """
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            pretrained_embeddings: Optional pretrained embeddings
            freeze_embeddings: Freeze embedding weights
        """
        super().__init__()
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        self.output_dim = hidden_dim * 2
    
    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len)
            lengths: Actual lengths (batch,)
            
        Returns:
            Encoded utterance: (batch, output_dim)
        """
        # Embed tokens
        embedded = self.embedding(token_ids)
        
        # Pack sequence
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # LSTM encoding
        _, (hidden, _) = self.lstm(embedded)
        
        # Concatenate forward and backward
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        
        return hidden


# ==================== Context Encoder ====================

class ContextEncoder(nn.Module):
    """
    Encode negotiation context (product info, scenario, etc.)
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ):
        """
        Args:
            input_dim: Context feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (batch, input_dim)
            
        Returns:
            Encoded context: (batch, output_dim)
        """
        return self.encoder(context)


# ==================== Fusion Encoder ====================

class FusionEncoder(nn.Module):
    """
    Fuse multiple encoded representations
    
    Combines state, history, context, etc. into unified representation
    """
    
    def __init__(
        self,
        input_dims: List[int],
        fusion_method: str = 'concat',
        output_dim: int = 256,
    ):
        """
        Args:
            input_dims: List of input dimensions to fuse
            fusion_method: 'concat', 'add', or 'attention'
            output_dim: Output dimension
        """
        super().__init__()
        
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            total_dim = sum(input_dims)
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, output_dim),
                nn.ReLU(),
                nn.LayerNorm(output_dim),
            )
        elif fusion_method == 'add':
            # Project all to same dimension then add
            self.projections = nn.ModuleList([
                nn.Linear(dim, output_dim) for dim in input_dims
            ])
        elif fusion_method == 'attention':
            # Attention-based fusion
            self.projections = nn.ModuleList([
                nn.Linear(dim, output_dim) for dim in input_dims
            ])
            self.attention = nn.Linear(output_dim, 1)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            *inputs: Variable number of input tensors
            
        Returns:
            Fused representation: (batch, output_dim)
        """
        if self.fusion_method == 'concat':
            # Concatenate and project
            x = torch.cat(inputs, dim=-1)
            return self.fusion(x)
        
        elif self.fusion_method == 'add':
            # Project and add
            projected = [proj(inp) for proj, inp in zip(self.projections, inputs)]
            return torch.stack(projected, dim=0).sum(dim=0)
        
        elif self.fusion_method == 'attention':
            # Project
            projected = [proj(inp) for proj, inp in zip(self.projections, inputs)]
            projected = torch.stack(projected, dim=1)  # (batch, num_inputs, output_dim)
            
            # Attention weights
            attn_scores = self.attention(projected).squeeze(-1)  # (batch, num_inputs)
            attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (batch, num_inputs, 1)
            
            # Weighted sum
            fused = (projected * attn_weights).sum(dim=1)  # (batch, output_dim)
            return fused


# ==================== Helper Functions ====================

def create_dialogue_encoder(
    encoder_type: str = 'lstm',
    input_dim: int = 3,
    hidden_dim: int = 256,
    **kwargs
) -> nn.Module:
    """
    Factory function to create dialogue encoder
    
    Args:
        encoder_type: 'lstm', 'gru', 'rnn'
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        **kwargs: Additional arguments
        
    Returns:
        Encoder module
    """
    return DialogueHistoryEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        rnn_type=encoder_type,
        **kwargs
    )


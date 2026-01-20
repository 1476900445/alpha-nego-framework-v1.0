"""
State Representation for Negotiation
Encodes negotiation state for RL agents

State Components:
- Price information
- Turn information
- Negotiation history
- Opponent modeling

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ==================== State Components ====================

@dataclass
class StateComponents:
    """
    Components of negotiation state
    """
    # Price information (Craigslistbargain)
    current_price: float = 0.0
    opponent_last_price: float = 0.0
    listing_price: float = 100.0
    agent_target: float = 70.0
    opponent_target: float = 90.0
    
    # Turn information
    current_turn: int = 0
    max_turns: int = 20
    
    # Price history
    agent_price_history: List[float] = None
    opponent_price_history: List[float] = None
    
    # Item information (Dealornodeal)
    agent_values: Optional[Dict[str, int]] = None
    opponent_values_estimate: Optional[Dict[str, int]] = None
    current_allocation: Optional[Dict[str, int]] = None
    
    def __post_init__(self):
        if self.agent_price_history is None:
            self.agent_price_history = []
        if self.opponent_price_history is None:
            self.opponent_price_history = []


# ==================== State Encoder ====================

class StateEncoder:
    """
    Encode state into vector representation
    """
    
    def __init__(
        self,
        dataset: str = 'craigslistbargain',
        state_dim: int = 3,
        use_history: bool = False,
        history_length: int = 5,
    ):
        """
        Args:
            dataset: 'craigslistbargain' or 'dealornodeal'
            state_dim: State dimension
            use_history: Include price history
            history_length: Length of history to include
        """
        self.dataset = dataset
        self.state_dim = state_dim
        self.use_history = use_history
        self.history_length = history_length
    
    def encode_craigslist_state(
        self,
        components: StateComponents,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode Craigslistbargain state
        
        Base state (3 dims):
        - current_price: Agent's current/last price
        - opponent_last_price: Opponent's last price
        - turn_ratio: Current turn / max turns
        
        Args:
            components: State components
            normalize: Normalize prices
            
        Returns:
            State vector (3,) or (3 + history_length*2,)
        """
        # Base state
        current_price = components.current_price
        opponent_price = components.opponent_last_price
        turn_ratio = components.current_turn / max(components.max_turns, 1)
        
        # Normalize prices
        if normalize and components.listing_price > 0:
            current_price /= components.listing_price
            opponent_price /= components.listing_price
        
        state = np.array([current_price, opponent_price, turn_ratio], dtype=np.float32)
        
        # Add history if enabled
        if self.use_history:
            # Pad price history
            agent_hist = self._pad_history(
                components.agent_price_history,
                self.history_length
            )
            opponent_hist = self._pad_history(
                components.opponent_price_history,
                self.history_length
            )
            
            # Normalize history
            if normalize and components.listing_price > 0:
                agent_hist = agent_hist / components.listing_price
                opponent_hist = opponent_hist / components.listing_price
            
            state = np.concatenate([state, agent_hist, opponent_hist])
        
        return state
    
    def encode_dealornodeal_state(
        self,
        components: StateComponents,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode Dealornodeal state
        
        State (variable dims):
        - agent_utility_estimate: Estimated utility
        - opponent_utility_estimate: Opponent's estimated utility
        - turn_ratio: Current turn / max turns
        - item_features: For each item (hats, books, balls):
            - agent_value: Agent's value for item
            - opponent_value_estimate: Estimated opponent value
            - current_allocation: Current proposed allocation
        
        Args:
            components: State components
            normalize: Normalize values
            
        Returns:
            State vector
        """
        turn_ratio = components.current_turn / max(components.max_turns, 1)
        
        # Compute utilities
        agent_util = 0.5  # Placeholder
        opponent_util = 0.5  # Placeholder
        
        state_list = [agent_util, opponent_util, turn_ratio]
        
        # Add item features
        items = ['hats', 'books', 'balls']
        max_value = 3.0 if normalize else 1.0
        max_count = 3.0 if normalize else 1.0
        
        for item in items:
            agent_val = components.agent_values.get(item, 0) if components.agent_values else 0
            opp_val_est = components.opponent_values_estimate.get(item, 0) if components.opponent_values_estimate else 0
            allocation = components.current_allocation.get(item, 0) if components.current_allocation else 0
            
            if normalize:
                agent_val /= max_value
                opp_val_est /= max_value
                allocation /= max_count
            
            state_list.extend([agent_val, opp_val_est, allocation])
        
        return np.array(state_list, dtype=np.float32)
    
    def encode(self, components: StateComponents) -> np.ndarray:
        """
        Encode state based on dataset
        
        Args:
            components: State components
            
        Returns:
            State vector
        """
        if self.dataset == 'craigslistbargain':
            return self.encode_craigslist_state(components)
        elif self.dataset == 'dealornodeal':
            return self.encode_dealornodeal_state(components)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
    
    def _pad_history(self, history: List[float], length: int) -> np.ndarray:
        """
        Pad history to fixed length
        
        Args:
            history: Price history
            length: Target length
            
        Returns:
            Padded array
        """
        if len(history) >= length:
            return np.array(history[-length:], dtype=np.float32)
        else:
            # Pad with zeros
            padded = np.zeros(length, dtype=np.float32)
            if len(history) > 0:
                padded[-len(history):] = history
            return padded


# ==================== State Normalizer ====================

class StateNormalizer:
    """
    Normalize state values
    """
    
    def __init__(
        self,
        method: str = 'minmax',
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Args:
            method: 'minmax' or 'standard'
            feature_ranges: Min/max for each feature
        """
        self.method = method
        self.feature_ranges = feature_ranges or {}
        
        # Statistics for standardization
        self.mean = None
        self.std = None
    
    def fit(self, states: np.ndarray):
        """
        Fit normalizer to data
        
        Args:
            states: Array of states (num_samples, state_dim)
        """
        if self.method == 'standard':
            self.mean = np.mean(states, axis=0)
            self.std = np.std(states, axis=0) + 1e-8
    
    def normalize(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state
        
        Args:
            state: State vector
            
        Returns:
            Normalized state
        """
        if self.method == 'minmax':
            # Already normalized in encoder typically
            return state
        elif self.method == 'standard':
            if self.mean is not None and self.std is not None:
                return (state - self.mean) / self.std
            return state
        else:
            return state
    
    def denormalize(self, state: np.ndarray) -> np.ndarray:
        """
        Denormalize state
        
        Args:
            state: Normalized state
            
        Returns:
            Original state
        """
        if self.method == 'standard':
            if self.mean is not None and self.std is not None:
                return state * self.std + self.mean
        return state


# ==================== Opponent Modeler ====================

class OpponentModeler:
    """
    Model opponent's preferences and strategy
    """
    
    def __init__(self, dataset: str = 'craigslistbargain'):
        """
        Args:
            dataset: Dataset name
        """
        self.dataset = dataset
        
        # Price history
        self.opponent_prices = []
        
        # Estimated opponent parameters
        self.estimated_target = None
        self.estimated_reservation = None
        self.concession_rate = None
        
        # Item values (Dealornodeal)
        self.estimated_item_values = {}
    
    def update(
        self,
        opponent_price: Optional[float] = None,
        opponent_items: Optional[Dict[str, int]] = None,
    ):
        """
        Update opponent model
        
        Args:
            opponent_price: Opponent's latest price
            opponent_items: Opponent's latest item proposal
        """
        if opponent_price is not None:
            self.opponent_prices.append(opponent_price)
            self._estimate_price_parameters()
        
        if opponent_items is not None:
            self._estimate_item_values(opponent_items)
    
    def _estimate_price_parameters(self):
        """Estimate opponent's target price and concession rate"""
        if len(self.opponent_prices) < 2:
            return
        
        # Estimate target as first price (often close to target)
        self.estimated_target = self.opponent_prices[0]
        
        # Estimate reservation as last price
        self.estimated_reservation = self.opponent_prices[-1]
        
        # Estimate concession rate
        if len(self.opponent_prices) >= 2:
            concessions = []
            for i in range(1, len(self.opponent_prices)):
                concession = abs(self.opponent_prices[i] - self.opponent_prices[i-1])
                concessions.append(concession)
            self.concession_rate = np.mean(concessions) if concessions else 0.0
    
    def _estimate_item_values(self, items: Dict[str, int]):
        """Estimate opponent's item values"""
        # Simple heuristic: items they propose more often are likely more valuable
        for item, count in items.items():
            if item not in self.estimated_item_values:
                self.estimated_item_values[item] = 0
            self.estimated_item_values[item] += count
    
    def get_estimated_state(self) -> Dict:
        """
        Get estimated opponent state
        
        Returns:
            Dictionary with estimates
        """
        return {
            'target': self.estimated_target,
            'reservation': self.estimated_reservation,
            'concession_rate': self.concession_rate,
            'item_values': self.estimated_item_values,
        }


# ==================== State Augmenter ====================

class StateAugmenter:
    """
    Augment state with additional features
    """
    
    def __init__(self, dataset: str = 'craigslistbargain'):
        """
        Args:
            dataset: Dataset name
        """
        self.dataset = dataset
    
    def augment(
        self,
        base_state: np.ndarray,
        components: StateComponents,
    ) -> np.ndarray:
        """
        Augment state with additional features
        
        Args:
            base_state: Base state vector
            components: State components
            
        Returns:
            Augmented state
        """
        additional_features = []
        
        if self.dataset == 'craigslistbargain':
            # Price gap
            if components.current_price > 0 and components.opponent_last_price > 0:
                price_gap = abs(components.current_price - components.opponent_last_price)
                price_gap_normalized = price_gap / components.listing_price if components.listing_price > 0 else 0
                additional_features.append(price_gap_normalized)
            
            # Distance to target
            if components.agent_target > 0:
                distance_to_target = abs(components.current_price - components.agent_target)
                distance_normalized = distance_to_target / components.listing_price if components.listing_price > 0 else 0
                additional_features.append(distance_normalized)
            
            # Concession trend
            if len(components.agent_price_history) >= 2:
                recent_concession = abs(
                    components.agent_price_history[-1] - components.agent_price_history[-2]
                )
                recent_concession_normalized = recent_concession / components.listing_price if components.listing_price > 0 else 0
                additional_features.append(recent_concession_normalized)
        
        if additional_features:
            augmented = np.concatenate([base_state, np.array(additional_features, dtype=np.float32)])
            return augmented
        
        return base_state


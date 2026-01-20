"""
Data preprocessing and feature extraction
Handles text processing, state encoding, and data normalization

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import torch


# ==================== Text Preprocessing ====================

class TextPreprocessor:
    """
    Text preprocessing for dialogue utterances
    Handles tokenization, normalization, and cleaning
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        max_length: int = 50,
    ):
        """
        Args:
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation marks
            remove_numbers: Remove numbers
            max_length: Maximum sequence length
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.max_length = max_length
        
        # Price-related patterns
        self.price_pattern = re.compile(r'\$?\d+\.?\d*')
        self.number_pattern = re.compile(r'\d+')
    
    def clean_text(self, text: str) -> str:
        """
        Clean text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Basic cleaning
        text = text.strip()
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation (but keep price markers)
        if self.remove_punctuation:
            # Keep $ and . for prices
            text = re.sub(r'[^\w\s\$\.]', '', text)
        
        # Remove numbers (except in prices)
        if self.remove_numbers:
            # Extract prices first
            prices = self.price_pattern.findall(text)
            # Remove all numbers
            text = self.number_pattern.sub('<NUM>', text)
            # Restore prices
            for price in prices:
                text = text.replace('<NUM>', price, 1)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        text = self.clean_text(text)
        tokens = text.split()
        
        # Truncate to max length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        return tokens
    
    def extract_price(self, text: str) -> Optional[float]:
        """
        Extract price from text
        
        Args:
            text: Input text
            
        Returns:
            Extracted price or None
        """
        matches = self.price_pattern.findall(text)
        
        if matches:
            # Take the last price mentioned
            price_str = matches[-1].replace('$', '')
            try:
                return float(price_str)
            except ValueError:
                return None
        
        return None
    
    def extract_intent_features(self, text: str) -> Dict[str, float]:
        """
        Extract intent-related features from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of features
        """
        text = text.lower()
        
        features = {
            # Greeting indicators
            'has_greeting': float(any(w in text for w in ['hi', 'hello', 'hey', 'good'])),
            
            # Question indicators
            'has_question': float('?' in text or any(w in text for w in ['what', 'when', 'where', 'who', 'why', 'how'])),
            
            # Agreement indicators
            'has_agreement': float(any(w in text for w in ['ok', 'okay', 'yes', 'sure', 'deal', 'agree', 'accept'])),
            
            # Rejection indicators
            'has_rejection': float(any(w in text for w in ['no', 'not', 'cant', "can't", 'wont', "won't", 'reject'])),
            
            # Price indicators
            'has_price': float('$' in text or self.extract_price(text) is not None),
            
            # Concession indicators
            'has_concession': float(any(w in text for w in ['lower', 'reduce', 'down', 'less', 'more', 'higher', 'up'])),
            
            # Final offer indicators
            'has_finality': float(any(w in text for w in ['final', 'last', 'best', 'firm', 'absolute'])),
            
            # Positive sentiment
            'has_positive': float(any(w in text for w in ['great', 'good', 'excellent', 'perfect', 'nice', 'love'])),
            
            # Negative sentiment
            'has_negative': float(any(w in text for w in ['bad', 'poor', 'terrible', 'awful', 'hate', 'disappointed'])),
            
            # Uncertainty
            'has_uncertainty': float(any(w in text for w in ['maybe', 'perhaps', 'might', 'could', 'possibly', 'unsure'])),
        }
        
        return features


# ==================== State Encoder ====================

class StateEncoder:
    """
    Encode negotiation state for neural networks
    Handles Craigslistbargain and Dealornodeal states
    """
    
    def __init__(self, dataset: str = 'craigslistbargain'):
        """
        Args:
            dataset: 'craigslistbargain' or 'dealornodeal'
        """
        self.dataset = dataset
    
    def encode_craigslist_state(
        self,
        current_price: float,
        opponent_last_price: float,
        turn_number: int,
        max_turns: int = 20,
        listing_price: float = 1.0,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode Craigslistbargain state
        
        State: [current_price, opponent_last_price, turn_number]
        
        Args:
            current_price: Current negotiation price
            opponent_last_price: Opponent's last price
            turn_number: Current turn (0-indexed)
            max_turns: Maximum turns
            listing_price: Original listing price
            normalize: Whether to normalize prices
            
        Returns:
            State vector (3,)
        """
        if normalize and listing_price > 0:
            current_price /= listing_price
            opponent_last_price /= listing_price
        
        turn_ratio = turn_number / max(max_turns, 1)
        
        state = np.array([
            current_price,
            opponent_last_price,
            turn_ratio
        ], dtype=np.float32)
        
        return state
    
    def encode_dealornodeal_state(
        self,
        agent_values: Dict[str, int],
        opponent_values_estimate: Dict[str, int],
        current_proposal: Dict[str, int],
        turn_number: int,
        max_turns: int = 20,
        items: List[str] = ['hats', 'books', 'balls'],
    ) -> np.ndarray:
        """
        Encode Dealornodeal state
        
        State: [utility_estimate, opponent_utility_estimate, turn_ratio, item_features...]
        
        Args:
            agent_values: Agent's item values
            opponent_values_estimate: Estimated opponent values
            current_proposal: Current item proposal
            turn_number: Current turn
            max_turns: Maximum turns
            items: List of items
            
        Returns:
            State vector
        """
        # Calculate utilities
        agent_utility = sum(agent_values.get(item, 0) * current_proposal.get(item, 0) for item in items)
        max_agent_utility = sum(agent_values.get(item, 0) * 3 for item in items)  # Max 3 of each item
        normalized_agent_utility = agent_utility / max(max_agent_utility, 1)
        
        opponent_utility = sum(opponent_values_estimate.get(item, 0) * (3 - current_proposal.get(item, 0)) for item in items)
        max_opponent_utility = sum(opponent_values_estimate.get(item, 0) * 3 for item in items)
        normalized_opponent_utility = opponent_utility / max(max_opponent_utility, 1)
        
        turn_ratio = turn_number / max(max_turns, 1)
        
        # Item-specific features
        item_features = []
        for item in items:
            my_value = agent_values.get(item, 0) / 3.0  # Normalize to [0, 1]
            opp_value_est = opponent_values_estimate.get(item, 0) / 3.0
            my_proposal = current_proposal.get(item, 0) / 3.0
            
            item_features.extend([my_value, opp_value_est, my_proposal])
        
        state = np.array([
            normalized_agent_utility,
            normalized_opponent_utility,
            turn_ratio,
            *item_features
        ], dtype=np.float32)
        
        return state
    
    def encode_state(self, **kwargs) -> np.ndarray:
        """
        Encode state based on dataset
        
        Args:
            **kwargs: Dataset-specific arguments
            
        Returns:
            Encoded state
        """
        if self.dataset == 'craigslistbargain':
            return self.encode_craigslist_state(**kwargs)
        elif self.dataset == 'dealornodeal':
            return self.encode_dealornodeal_state(**kwargs)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")


# ==================== Action Encoder ====================

class ActionEncoder:
    """
    Encode negotiation actions
    Handles both Craigslistbargain (intent + price) and Dealornodeal (intent + items)
    """
    
    # Craigslistbargain intents (16 acts)
    CRAIGSLIST_INTENTS = [
        'greet', 'inquire', 'inform',
        'init-price', 'insist-price', 'agree-price',
        'concede-price', 'final-price', 'counter-no-price',
        'hesitant', 'positive', 'negative',
        'offer', 'accept', 'reject', 'quit'
    ]
    
    # Dealornodeal intents (6 acts)
    DEALORNODEAL_INTENTS = [
        'greet', 'disagree', 'agree',
        'insist', 'inquire', 'propose'
    ]
    
    def __init__(self, dataset: str = 'craigslistbargain'):
        """
        Args:
            dataset: 'craigslistbargain' or 'dealornodeal'
        """
        self.dataset = dataset
        
        if dataset == 'craigslistbargain':
            self.intents = self.CRAIGSLIST_INTENTS
            self.intent_to_id = {intent: i for i, intent in enumerate(self.intents)}
        else:
            self.intents = self.DEALORNODEAL_INTENTS
            self.intent_to_id = {intent: i for i, intent in enumerate(self.intents)}
    
    def encode_intent(self, intent: str) -> int:
        """
        Encode intent to ID
        
        Args:
            intent: Intent string
            
        Returns:
            Intent ID
        """
        return self.intent_to_id.get(intent, 0)  # Default to first intent
    
    def decode_intent(self, intent_id: int) -> str:
        """
        Decode intent ID to string
        
        Args:
            intent_id: Intent ID
            
        Returns:
            Intent string
        """
        if 0 <= intent_id < len(self.intents):
            return self.intents[intent_id]
        return self.intents[0]
    
    def encode_craigslist_action(
        self,
        intent: str,
        price: float,
        listing_price: float = 1.0,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode Craigslistbargain action
        
        Action: [intent_id, price]
        
        Args:
            intent: Dialogue act
            price: Proposed price
            listing_price: Original listing price
            normalize: Whether to normalize price
            
        Returns:
            Action vector (2,)
        """
        intent_id = self.encode_intent(intent)
        
        if normalize and listing_price > 0:
            price /= listing_price
        
        action = np.array([intent_id, price], dtype=np.float32)
        return action
    
    def encode_dealornodeal_action(
        self,
        intent: str,
        proposal: Dict[str, int],
        items: List[str] = ['hats', 'books', 'balls'],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode Dealornodeal action
        
        Action: [intent_id, item1_count, item2_count, item3_count]
        
        Args:
            intent: Dialogue act
            proposal: Item allocation {item: count}
            items: List of items
            normalize: Whether to normalize counts
            
        Returns:
            Action vector (1 + num_items,)
        """
        intent_id = self.encode_intent(intent)
        
        item_counts = [proposal.get(item, 0) for item in items]
        
        if normalize:
            item_counts = [count / 3.0 for count in item_counts]  # Max 3 of each
        
        action = np.array([intent_id] + item_counts, dtype=np.float32)
        return action


# ==================== Data Normalizer ====================

class DataNormalizer:
    """
    Normalize and denormalize data
    Tracks statistics for consistent normalization
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
    
    def fit(self, data: np.ndarray, method: str = 'standardize'):
        """
        Fit normalizer to data
        
        Args:
            data: Data array (n_samples, n_features)
            method: 'standardize' (z-score) or 'minmax'
        """
        self.method = method
        
        if method == 'standardize':
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0) + 1e-8
        elif method == 'minmax':
            self.min = np.min(data, axis=0)
            self.max = np.max(data, axis=0)
            self.max = np.where(self.max == self.min, self.min + 1, self.max)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data
        
        Args:
            data: Data to normalize
            
        Returns:
            Normalized data
        """
        if self.method == 'standardize':
            return (data - self.mean) / self.std
        elif self.method == 'minmax':
            return (data - self.min) / (self.max - self.min)
        else:
            return data
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Denormalize data
        
        Args:
            data: Normalized data
            
        Returns:
            Original scale data
        """
        if self.method == 'standardize':
            return data * self.std + self.mean
        elif self.method == 'minmax':
            return data * (self.max - self.min) + self.min
        else:
            return data
    
    def fit_transform(self, data: np.ndarray, method: str = 'standardize') -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(data, method)
        return self.transform(data)


# ==================== Reward Shaper ====================

class RewardShaper:
    """
    Shape rewards for better learning
    Implements various reward shaping techniques from paper
    """
    
    def __init__(
        self,
        length_penalty: float = -0.01,
        fairness_bonus: float = 0.05,
        no_deal_penalty: float = -0.5,
    ):
        """
        Args:
            length_penalty: Penalty per turn
            fairness_bonus: Bonus for fair deals
            no_deal_penalty: Penalty for no agreement
        """
        self.length_penalty = length_penalty
        self.fairness_bonus = fairness_bonus
        self.no_deal_penalty = no_deal_penalty
    
    def shape_reward(
        self,
        base_reward: float,
        dialogue_length: int,
        agent_utility: float,
        opponent_utility: float,
        agreement: bool,
    ) -> float:
        """
        Apply reward shaping
        
        Args:
            base_reward: Base reward (utility)
            dialogue_length: Number of turns
            agent_utility: Agent's utility
            opponent_utility: Opponent's utility
            agreement: Whether deal reached
            
        Returns:
            Shaped reward
        """
        reward = base_reward
        
        # Length penalty
        reward += self.length_penalty * dialogue_length
        
        # Fairness bonus
        if agreement:
            fairness = 1.0 - abs(agent_utility - opponent_utility)
            reward += self.fairness_bonus * fairness
        
        # No deal penalty
        if not agreement:
            reward += self.no_deal_penalty
        
        return reward


# ==================== Utility Functions ====================

def calculate_craigslist_utility(
    final_price: float,
    agent_target: float,
    opponent_target: float,
    agent_role: str = 'buyer',
) -> Tuple[float, float]:
    """
    Calculate utilities for Craigslistbargain
    
    Args:
        final_price: Agreed price
        agent_target: Agent's target price
        opponent_target: Opponent's target price
        agent_role: 'buyer' or 'seller'
        
    Returns:
        (agent_utility, opponent_utility)
    """
    if agent_role == 'buyer':
        # Buyer: lower price is better
        agent_utility = (opponent_target - final_price) / max(opponent_target - agent_target, 1e-6)
    else:
        # Seller: higher price is better
        agent_utility = (final_price - opponent_target) / max(agent_target - opponent_target, 1e-6)
    
    agent_utility = np.clip(agent_utility, 0.0, 1.0)
    opponent_utility = 1.0 - agent_utility
    
    return agent_utility, opponent_utility


def calculate_dealornodeal_utility(
    allocation: Dict[str, int],
    values: Dict[str, int],
    items: List[str] = ['hats', 'books', 'balls'],
) -> float:
    """
    Calculate utility for Dealornodeal
    
    U(ω) = Σ(w_j · V_j(v_jk))
    
    Args:
        allocation: Item allocation {item: count}
        values: Item values {item: value}
        items: List of items
        
    Returns:
        Utility
    """
    utility = sum(values.get(item, 0) * allocation.get(item, 0) for item in items)
    max_utility = sum(values.get(item, 0) * 3 for item in items)  # Max 3 of each
    
    normalized_utility = utility / max(max_utility, 1)
    return normalized_utility



"""
Baseline Negotiation Agents
Implements various baseline strategies for comparison and opponent pool

Baseline types (Paper Section 6):
1. Time-dependent agents (8 variants)
   - Conceder, Boulware, Linear, Tit-For-Tat
2. Behavior-dependent agents (2 variants)
   - Reciprocal Concession, Tit-For-Tat with Decay
3. Rule-based agents
   - Fixed strategy agents
4. Supervised learning agent
   - Behavior cloning from human data

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod


# ==================== Base Agent ====================

class BaseAgent(ABC):
    """
    Abstract base class for negotiation agents
    
    All agents must implement:
    - select_action(state, history) -> (intent, price)
    - reset() -> None
    """
    
    def __init__(self, name: str = "BaseAgent"):
        """
        Args:
            name: Agent name
        """
        self.name = name
        self.reset()
    
    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        history: Optional[List] = None,
    ) -> Tuple[int, float]:
        """
        Select action based on state
        
        Args:
            state: Current state
            history: Dialogue history
            
        Returns:
            intent: Dialogue act
            price: Price/proposal
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset agent state"""
        pass
    
    def __str__(self):
        return self.name


# ==================== Time-Dependent Agents ====================

class TimeDependentAgent(BaseAgent):
    """
    Base class for time-dependent negotiation strategies
    
    Concession function: P(t) = P_min + (P_max - P_min) * f(t)
    where f(t) depends on the strategy
    """
    
    def __init__(
        self,
        name: str,
        role: str = 'buyer',
        min_price: float = 0.5,
        max_price: float = 1.0,
        max_turns: int = 20,
    ):
        """
        Args:
            name: Agent name
            role: 'buyer' or 'seller'
            min_price: Minimum acceptable price
            max_price: Maximum acceptable price
            max_turns: Maximum negotiation turns
        """
        self.role = role
        self.min_price = min_price
        self.max_price = max_price
        self.max_turns = max_turns
        
        super().__init__(name)
    
    def reset(self):
        """Reset agent state"""
        self.current_turn = 0
        self.last_opponent_price = None
    
    def select_action(
        self,
        state: np.ndarray,
        history: Optional[List] = None,
    ) -> Tuple[int, float]:
        """Select action based on time-dependent strategy"""
        self.current_turn += 1
        
        # Extract opponent's last price from state
        if len(state) >= 2:
            self.last_opponent_price = state[1]
        
        # Compute price based on concession function
        t = self.current_turn / self.max_turns
        price = self._compute_price(t)
        
        # Select intent
        intent = self._select_intent(price, self.last_opponent_price)
        
        return intent, price
    
    @abstractmethod
    def _compute_price(self, t: float) -> float:
        """
        Compute price based on normalized time
        
        Args:
            t: Normalized time [0, 1]
            
        Returns:
            Price
        """
        pass
    
    def _select_intent(self, my_price: float, opponent_price: Optional[float]) -> int:
        """
        Select dialogue act based on prices
        
        Args:
            my_price: My proposed price
            opponent_price: Opponent's last price
            
        Returns:
            Intent ID
        """
        if self.current_turn == 1:
            return 3  # init-price
        
        if opponent_price is None:
            return 3  # init-price
        
        # Check if we can accept
        if self.role == 'buyer':
            if opponent_price <= my_price:
                return 13  # accept
        else:
            if opponent_price >= my_price:
                return 13  # accept
        
        # Check if we should quit
        if self.current_turn >= self.max_turns:
            return 15  # quit
        
        # Otherwise, make counteroffer
        if abs(my_price - opponent_price) < 0.05:
            return 7  # final-price
        else:
            return 6  # concede-price


class ConcederAgent(TimeDependentAgent):
    """
    Conceder: Makes quick concessions early
    
    f(t) = t^β where β < 1 (e.g., β = 0.5)
    Quickly concedes towards opponent's position
    """
    
    def __init__(self, role: str = 'buyer', beta: float = 0.5, **kwargs):
        """
        Args:
            role: 'buyer' or 'seller'
            beta: Concession rate (< 1 for quick concession)
            **kwargs: Additional arguments
        """
        self.beta = beta
        super().__init__(f"Conceder(β={beta})", role, **kwargs)
    
    def _compute_price(self, t: float) -> float:
        """Conceder concession function"""
        if self.role == 'buyer':
            # Buyer: start high, concede quickly
            return self.max_price - (self.max_price - self.min_price) * (t ** self.beta)
        else:
            # Seller: start low, concede quickly
            return self.min_price + (self.max_price - self.min_price) * (t ** self.beta)


class BoulwareAgent(TimeDependentAgent):
    """
    Boulware: Maintains position until late in negotiation
    
    f(t) = t^β where β > 1 (e.g., β = 2.0)
    Holds firm then concedes rapidly near deadline
    """
    
    def __init__(self, role: str = 'buyer', beta: float = 2.0, **kwargs):
        """
        Args:
            role: 'buyer' or 'seller'
            beta: Concession rate (> 1 for late concession)
            **kwargs: Additional arguments
        """
        self.beta = beta
        super().__init__(f"Boulware(β={beta})", role, **kwargs)
    
    def _compute_price(self, t: float) -> float:
        """Boulware concession function"""
        if self.role == 'buyer':
            return self.max_price - (self.max_price - self.min_price) * (t ** self.beta)
        else:
            return self.min_price + (self.max_price - self.min_price) * (t ** self.beta)


class LinearAgent(TimeDependentAgent):
    """
    Linear: Makes linear concessions
    
    f(t) = t
    Constant concession rate
    """
    
    def __init__(self, role: str = 'buyer', **kwargs):
        super().__init__("Linear", role, **kwargs)
    
    def _compute_price(self, t: float) -> float:
        """Linear concession function"""
        if self.role == 'buyer':
            return self.max_price - (self.max_price - self.min_price) * t
        else:
            return self.min_price + (self.max_price - self.min_price) * t


# ==================== Behavior-Dependent Agents ====================

class TitForTatAgent(BaseAgent):
    """
    Tit-For-Tat: Mirrors opponent's behavior
    
    Matches opponent's concessions
    """
    
    def __init__(
        self,
        role: str = 'buyer',
        initial_price: float = 0.75,
        **kwargs
    ):
        """
        Args:
            role: 'buyer' or 'seller'
            initial_price: Initial offer
        """
        self.role = role
        self.initial_price = initial_price
        super().__init__("TitForTat")
    
    def reset(self):
        """Reset state"""
        self.current_turn = 0
        self.my_last_price = self.initial_price
        self.opponent_last_price = None
        self.opponent_prev_price = None
    
    def select_action(
        self,
        state: np.ndarray,
        history: Optional[List] = None,
    ) -> Tuple[int, float]:
        """Select action mirroring opponent"""
        self.current_turn += 1
        
        # Extract prices from state
        current_price = state[0] if len(state) > 0 else self.initial_price
        opponent_price = state[1] if len(state) > 1 else None
        
        # Update opponent price history
        if opponent_price is not None:
            self.opponent_prev_price = self.opponent_last_price
            self.opponent_last_price = opponent_price
        
        # First turn
        if self.current_turn == 1:
            self.my_last_price = self.initial_price
            return 3, self.initial_price  # init-price
        
        # Calculate opponent's concession
        if self.opponent_prev_price is not None and self.opponent_last_price is not None:
            opponent_concession = abs(self.opponent_last_price - self.opponent_prev_price)
            
            # Mirror the concession
            if self.role == 'buyer':
                new_price = self.my_last_price - opponent_concession
            else:
                new_price = self.my_last_price + opponent_concession
            
            # Clip to valid range
            new_price = np.clip(new_price, 0.0, 1.0)
        else:
            new_price = self.my_last_price
        
        # Check acceptance
        if opponent_price is not None:
            if self.role == 'buyer' and opponent_price <= new_price:
                return 13, opponent_price  # accept
            elif self.role == 'seller' and opponent_price >= new_price:
                return 13, opponent_price  # accept
        
        # Update and return
        self.my_last_price = new_price
        
        # Select intent
        if abs(new_price - (opponent_price or 0)) < 0.05:
            intent = 7  # final-price
        else:
            intent = 6  # concede-price
        
        return intent, new_price


class ReciprocalConcessionAgent(BaseAgent):
    """
    Reciprocal Concession: Matches opponent's concession magnitude
    
    If opponent concedes X, this agent also concedes X
    """
    
    def __init__(
        self,
        role: str = 'buyer',
        initial_price: float = 0.75,
        reciprocity_factor: float = 1.0,
    ):
        """
        Args:
            role: 'buyer' or 'seller'
            initial_price: Initial offer
            reciprocity_factor: Multiplier for opponent's concession
        """
        self.role = role
        self.initial_price = initial_price
        self.reciprocity_factor = reciprocity_factor
        super().__init__(f"ReciprocalConcession(α={reciprocity_factor})")
    
    def reset(self):
        """Reset state"""
        self.current_turn = 0
        self.my_prices = [self.initial_price]
        self.opponent_prices = []
    
    def select_action(
        self,
        state: np.ndarray,
        history: Optional[List] = None,
    ) -> Tuple[int, float]:
        """Select action with reciprocal concession"""
        self.current_turn += 1
        
        # Extract opponent price
        opponent_price = state[1] if len(state) > 1 else None
        
        # First turn
        if self.current_turn == 1:
            return 3, self.initial_price  # init-price
        
        # Update opponent prices
        if opponent_price is not None:
            self.opponent_prices.append(opponent_price)
        
        # Calculate reciprocal concession
        if len(self.opponent_prices) >= 2:
            opponent_concession = abs(self.opponent_prices[-1] - self.opponent_prices[-2])
            my_concession = opponent_concession * self.reciprocity_factor
            
            if self.role == 'buyer':
                new_price = self.my_prices[-1] - my_concession
            else:
                new_price = self.my_prices[-1] + my_concession
            
            new_price = np.clip(new_price, 0.0, 1.0)
        else:
            new_price = self.my_prices[-1]
        
        # Check acceptance
        if opponent_price is not None:
            if self.role == 'buyer' and opponent_price <= new_price:
                return 13, opponent_price  # accept
            elif self.role == 'seller' and opponent_price >= new_price:
                return 13, opponent_price  # accept
        
        # Update
        self.my_prices.append(new_price)
        
        # Select intent
        intent = 6  # concede-price
        if abs(new_price - (opponent_price or 0)) < 0.05:
            intent = 7  # final-price
        
        return intent, new_price


# ==================== Rule-Based Agents ====================

class FixedPriceAgent(BaseAgent):
    """
    Fixed Price: Always proposes the same price
    """
    
    def __init__(
        self,
        role: str = 'buyer',
        fixed_price: float = 0.5,
    ):
        """
        Args:
            role: 'buyer' or 'seller'
            fixed_price: The fixed price to propose
        """
        self.role = role
        self.fixed_price = fixed_price
        super().__init__(f"FixedPrice({fixed_price:.2f})")
    
    def reset(self):
        """Reset state"""
        self.current_turn = 0
    
    def select_action(
        self,
        state: np.ndarray,
        history: Optional[List] = None,
    ) -> Tuple[int, float]:
        """Select action with fixed price"""
        self.current_turn += 1
        
        # Extract opponent price
        opponent_price = state[1] if len(state) > 1 else None
        
        # Check acceptance
        if opponent_price is not None:
            if self.role == 'buyer' and opponent_price <= self.fixed_price:
                return 13, opponent_price  # accept
            elif self.role == 'seller' and opponent_price >= self.fixed_price:
                return 13, opponent_price  # accept
        
        # Insist on fixed price
        if self.current_turn == 1:
            return 3, self.fixed_price  # init-price
        else:
            return 4, self.fixed_price  # insist-price


class RandomAgent(BaseAgent):
    """
    Random: Makes random proposals
    """
    
    def __init__(
        self,
        role: str = 'buyer',
        price_range: Tuple[float, float] = (0.3, 0.9),
    ):
        """
        Args:
            role: 'buyer' or 'seller'
            price_range: (min, max) for random prices
        """
        self.role = role
        self.price_range = price_range
        super().__init__("Random")
    
    def reset(self):
        """Reset state"""
        self.current_turn = 0
    
    def select_action(
        self,
        state: np.ndarray,
        history: Optional[List] = None,
    ) -> Tuple[int, float]:
        """Select random action"""
        self.current_turn += 1
        
        # Random price
        price = np.random.uniform(*self.price_range)
        
        # Extract opponent price
        opponent_price = state[1] if len(state) > 1 else None
        
        # Sometimes accept opponent's offer
        if opponent_price is not None and np.random.random() < 0.2:
            return 13, opponent_price  # accept
        
        # Random intent (weighted towards offers)
        intent = np.random.choice([3, 6, 7, 12], p=[0.2, 0.4, 0.2, 0.2])
        
        return intent, price


# ==================== Agent Factory ====================

def create_baseline_agent(
    agent_type: str,
    role: str = 'buyer',
    **kwargs
) -> BaseAgent:
    """
    Factory function to create baseline agents
    
    Args:
        agent_type: Agent type string
        role: 'buyer' or 'seller'
        **kwargs: Additional arguments
        
    Returns:
        BaseAgent instance
    """
    agent_type = agent_type.lower()
    
    if agent_type == 'conceder':
        return ConcederAgent(role, **kwargs)
    elif agent_type == 'conceder_mild':
        return ConcederAgent(role, beta=0.7, **kwargs)
    elif agent_type == 'boulware':
        return BoulwareAgent(role, **kwargs)
    elif agent_type == 'boulware_mild':
        return BoulwareAgent(role, beta=1.5, **kwargs)
    elif agent_type == 'linear':
        return LinearAgent(role, **kwargs)
    elif agent_type == 'tit_for_tat':
        return TitForTatAgent(role, **kwargs)
    elif agent_type == 'tft_aggressive':
        return TitForTatAgent(role, initial_price=0.9 if role=='buyer' else 0.1, **kwargs)
    elif agent_type == 'tft_conservative':
        return TitForTatAgent(role, initial_price=0.6 if role=='buyer' else 0.4, **kwargs)
    elif agent_type == 'reciprocal_concession':
        return ReciprocalConcessionAgent(role, **kwargs)
    elif agent_type == 'fixed_price':
        return FixedPriceAgent(role, **kwargs)
    elif agent_type == 'random':
        return RandomAgent(role, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_all_baseline_agents(role: str = 'buyer') -> List[BaseAgent]:
    """
    Create all baseline agents for opponent pool
    
    Paper Section 6:
    - 8 time-dependent agents
    - 2 behavior-dependent agents
    
    Args:
        role: 'buyer' or 'seller'
        
    Returns:
        List of baseline agents
    """
    agents = []
    
    # Time-dependent (8 variants)
    agents.append(create_baseline_agent('conceder', role))
    agents.append(create_baseline_agent('conceder_mild', role))
    agents.append(create_baseline_agent('boulware', role))
    agents.append(create_baseline_agent('boulware_mild', role))
    agents.append(create_baseline_agent('linear', role))
    agents.append(create_baseline_agent('tit_for_tat', role))
    agents.append(create_baseline_agent('tft_aggressive', role))
    agents.append(create_baseline_agent('tft_conservative', role))
    
    # Behavior-dependent (2 variants)
    agents.append(create_baseline_agent('reciprocal_concession', role))
    agents.append(create_baseline_agent('reciprocal_concession', role, reciprocity_factor=0.5))
    
    return agents


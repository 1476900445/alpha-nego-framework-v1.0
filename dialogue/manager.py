"""
Dialogue Manager
Manages complete negotiation dialogue flow

Features:
- Turn management
- State tracking
- Action validation
- Agreement detection
- Dialogue history

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from dialogue.dialogue_acts import (
    get_intent_name, is_terminal, requires_price, requires_items
)
from dialogue.parser import DialogueParser
from dialogue.generator import DialogueGenerator


# ==================== Dialogue Turn ====================

@dataclass
class DialogueTurn:
    """
    Single turn in negotiation dialogue
    """
    turn_id: int
    speaker: str  # 'agent' or 'opponent'
    intent_id: int
    intent_name: str
    price: Optional[float] = None
    items: Optional[Dict[str, int]] = None
    utterance: Optional[str] = None
    state: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'turn_id': self.turn_id,
            'speaker': self.speaker,
            'intent_id': self.intent_id,
            'intent_name': self.intent_name,
            'price': self.price,
            'items': self.items,
            'utterance': self.utterance,
            'state': self.state.tolist() if self.state is not None else None,
        }


# ==================== Dialogue State ====================

@dataclass
class DialogueState:
    """
    Current state of negotiation
    """
    # Turn information
    current_turn: int = 0
    max_turns: int = 20
    
    # Price tracking (Craigslistbargain)
    agent_last_price: Optional[float] = None
    opponent_last_price: Optional[float] = None
    listing_price: Optional[float] = None
    agent_target: Optional[float] = None
    opponent_target: Optional[float] = None
    
    # Item tracking (Dealornodeal)
    agent_values: Optional[Dict[str, int]] = None
    opponent_values: Optional[Dict[str, int]] = None
    agent_last_proposal: Optional[Dict[str, int]] = None
    opponent_last_proposal: Optional[Dict[str, int]] = None
    
    # Negotiation outcome
    agreement: bool = False
    final_price: Optional[float] = None
    final_allocation: Optional[Dict[str, int]] = None
    
    # History
    turns: List[DialogueTurn] = field(default_factory=list)
    
    def get_state_vector(self, dataset: str = 'craigslistbargain') -> np.ndarray:
        """
        Get state as vector for agent
        
        Args:
            dataset: Dataset name
            
        Returns:
            State vector
        """
        if dataset == 'craigslistbargain':
            # State: [current_price, opponent_last_price, turn_ratio]
            turn_ratio = self.current_turn / self.max_turns
            
            current = self.agent_last_price if self.agent_last_price else 0.0
            opponent = self.opponent_last_price if self.opponent_last_price else 0.0
            
            # Normalize prices
            if self.listing_price and self.listing_price > 0:
                current /= self.listing_price
                opponent /= self.listing_price
            
            return np.array([current, opponent, turn_ratio], dtype=np.float32)
        
        elif dataset == 'dealornodeal':
            # State: [utility_estimate, opponent_estimate, turn_ratio]
            turn_ratio = self.current_turn / self.max_turns
            
            # Compute utilities (simplified)
            agent_util = 0.5  # Placeholder
            opponent_util = 0.5  # Placeholder
            
            return np.array([agent_util, opponent_util, turn_ratio], dtype=np.float32)
        
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    def is_terminal(self) -> bool:
        """Check if dialogue has ended"""
        return self.agreement or self.current_turn >= self.max_turns


# ==================== Dialogue Manager ====================

class DialogueManager:
    """
    Main dialogue manager for negotiation
    
    Handles:
    - Turn management
    - State tracking
    - Action validation
    - Agreement detection
    """
    
    def __init__(
        self,
        dataset: str = 'craigslistbargain',
        max_turns: int = 20,
        listing_price: Optional[float] = None,
        agent_target: Optional[float] = None,
        opponent_target: Optional[float] = None,
        agent_values: Optional[Dict[str, int]] = None,
        opponent_values: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            dataset: 'craigslistbargain' or 'dealornodeal'
            max_turns: Maximum dialogue turns
            listing_price: Listing price (Craigslistbargain)
            agent_target: Agent's target price
            opponent_target: Opponent's target price
            agent_values: Agent's item values (Dealornodeal)
            opponent_values: Opponent's item values (Dealornodeal)
        """
        self.dataset = dataset
        self.max_turns = max_turns
        
        # Initialize state
        self.state = DialogueState(
            max_turns=max_turns,
            listing_price=listing_price,
            agent_target=agent_target,
            opponent_target=opponent_target,
            agent_values=agent_values,
            opponent_values=opponent_values,
        )
        
        # Parsers and generators
        self.parser = DialogueParser(dataset)
        self.generator = DialogueGenerator(dataset, mode='template')
        
        print(f"[DialogueManager] Initialized for {dataset}")
        print(f"  Max turns: {max_turns}")
    
    def reset(
        self,
        listing_price: Optional[float] = None,
        agent_target: Optional[float] = None,
        opponent_target: Optional[float] = None,
    ):
        """
        Reset dialogue state
        
        Args:
            listing_price: New listing price
            agent_target: New agent target
            opponent_target: New opponent target
        """
        self.state = DialogueState(
            max_turns=self.max_turns,
            listing_price=listing_price or self.state.listing_price,
            agent_target=agent_target or self.state.agent_target,
            opponent_target=opponent_target or self.state.opponent_target,
            agent_values=self.state.agent_values,
            opponent_values=self.state.opponent_values,
        )
    
    def step(
        self,
        speaker: str,
        intent_id: int,
        price: Optional[float] = None,
        items: Optional[Dict[str, int]] = None,
        utterance: Optional[str] = None,
    ) -> Tuple[Dict, bool]:
        """
        Execute one dialogue turn
        
        Args:
            speaker: 'agent' or 'opponent'
            intent_id: Intent ID
            price: Price value
            items: Item allocation
            utterance: Generated utterance
            
        Returns:
            (info_dict, done)
        """
        # Validate action
        if not self._validate_action(intent_id, price, items):
            return {'error': 'Invalid action'}, False
        
        # Update state
        self._update_state(speaker, intent_id, price, items)
        
        # Generate utterance if not provided
        if utterance is None:
            utterance = self.generator.generate(intent_id, price, items)
        
        # Create turn
        turn = DialogueTurn(
            turn_id=self.state.current_turn,
            speaker=speaker,
            intent_id=intent_id,
            intent_name=get_intent_name(intent_id, self.dataset),
            price=price,
            items=items,
            utterance=utterance,
            state=self.state.get_state_vector(self.dataset),
        )
        
        self.state.turns.append(turn)
        self.state.current_turn += 1
        
        # Check if terminal
        done = self._check_terminal(intent_id)
        
        # Create info
        info = {
            'turn': turn.to_dict(),
            'agreement': self.state.agreement,
            'final_price': self.state.final_price,
            'final_allocation': self.state.final_allocation,
        }
        
        return info, done
    
    def _validate_action(
        self,
        intent_id: int,
        price: Optional[float],
        items: Optional[Dict[str, int]],
    ) -> bool:
        """
        Validate action
        
        Args:
            intent_id: Intent ID
            price: Price value
            items: Item allocation
            
        Returns:
            True if valid
        """
        # Check price requirement
        if requires_price(intent_id, self.dataset) and price is None:
            print(f"[Warning] Intent {intent_id} requires price but none provided")
            return False
        
        # Check items requirement
        if requires_items(intent_id, self.dataset) and items is None:
            print(f"[Warning] Intent {intent_id} requires items but none provided")
            return False
        
        # Check price range
        if price is not None:
            if price < 0 or price > (self.state.listing_price or 1000):
                print(f"[Warning] Price {price} out of valid range")
                return False
        
        return True
    
    def _update_state(
        self,
        speaker: str,
        intent_id: int,
        price: Optional[float],
        items: Optional[Dict[str, int]],
    ):
        """
        Update dialogue state
        
        Args:
            speaker: 'agent' or 'opponent'
            intent_id: Intent ID
            price: Price value
            items: Item allocation
        """
        # Update prices
        if price is not None:
            if speaker == 'agent':
                self.state.agent_last_price = price
            else:
                self.state.opponent_last_price = price
        
        # Update items
        if items is not None:
            if speaker == 'agent':
                self.state.agent_last_proposal = items
            else:
                self.state.opponent_last_proposal = items
    
    def _check_terminal(self, intent_id: int) -> bool:
        """
        Check if dialogue should end
        
        Args:
            intent_id: Intent ID
            
        Returns:
            True if terminal
        """
        # Check terminal intent
        if is_terminal(intent_id, self.dataset):
            intent_name = get_intent_name(intent_id, self.dataset)
            
            if intent_name == 'accept':
                self.state.agreement = True
                
                # Set final outcome
                if self.dataset == 'craigslistbargain':
                    # Use opponent's last price (they made the offer)
                    self.state.final_price = self.state.opponent_last_price
                elif self.dataset == 'dealornodeal':
                    self.state.final_allocation = self.state.opponent_last_proposal
            
            return True
        
        # Check max turns
        if self.state.current_turn >= self.max_turns:
            return True
        
        return False
    
    def get_history(self, max_length: int = 10) -> List[Dict]:
        """
        Get recent dialogue history
        
        Args:
            max_length: Maximum history length
            
        Returns:
            List of turn dictionaries
        """
        recent_turns = self.state.turns[-max_length:]
        return [turn.to_dict() for turn in recent_turns]
    
    def get_state(self) -> np.ndarray:
        """Get current state vector"""
        return self.state.get_state_vector(self.dataset)
    
    def get_summary(self) -> Dict:
        """
        Get dialogue summary
        
        Returns:
            Summary dictionary
        """
        return {
            'dataset': self.dataset,
            'num_turns': len(self.state.turns),
            'max_turns': self.max_turns,
            'agreement': self.state.agreement,
            'final_price': self.state.final_price,
            'final_allocation': self.state.final_allocation,
            'agent_last_price': self.state.agent_last_price,
            'opponent_last_price': self.state.opponent_last_price,
            'listing_price': self.state.listing_price,
        }
    
    def compute_utilities(self) -> Tuple[float, float]:
        """
        Compute final utilities for both parties
        
        Returns:
            (agent_utility, opponent_utility)
        """
        if not self.state.agreement:
            return 0.0, 0.0
        
        if self.dataset == 'craigslistbargain':
            # Assume agent is buyer
            final_price = self.state.final_price
            if final_price is None:
                return 0.0, 0.0
            
            agent_target = self.state.agent_target or 0.0
            opponent_target = self.state.opponent_target or 0.0
            
            # Buyer utility: (opponent_target - final_price) / (opponent_target - agent_target)
            if opponent_target > agent_target:
                agent_util = (opponent_target - final_price) / (opponent_target - agent_target)
            else:
                agent_util = 0.5
            
            # Seller utility: (final_price - agent_target) / (opponent_target - agent_target)
            if opponent_target > agent_target:
                opponent_util = (final_price - agent_target) / (opponent_target - agent_target)
            else:
                opponent_util = 0.5
            
            # Clip to [0, 1]
            agent_util = np.clip(agent_util, 0.0, 1.0)
            opponent_util = np.clip(opponent_util, 0.0, 1.0)
            
            return agent_util, opponent_util
        
        elif self.dataset == 'dealornodeal':
            # Compute utilities based on item values
            allocation = self.state.final_allocation
            if allocation is None:
                return 0.0, 0.0
            
            agent_values = self.state.agent_values or {}
            opponent_values = self.state.opponent_values or {}
            
            # Agent utility
            agent_util = sum(
                agent_values.get(item, 0) * count
                for item, count in allocation.items()
            )
            
            # Opponent utility (complement allocation)
            max_counts = {'hats': 3, 'books': 3, 'balls': 3}
            opponent_allocation = {
                item: max_counts[item] - allocation.get(item, 0)
                for item in max_counts
            }
            opponent_util = sum(
                opponent_values.get(item, 0) * count
                for item, count in opponent_allocation.items()
            )
            
            # Normalize
            max_util = max(sum(agent_values.values()), 1)
            agent_util /= max_util
            opponent_util /= max_util
            
            return agent_util, opponent_util
        
        return 0.0, 0.0


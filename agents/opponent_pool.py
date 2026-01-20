"""
Opponent Pool Management with PFSP
Implements Priority Fictitious Self-Play for opponent selection

Key Features:
- PFSP opponent selection (Eq. 10)
- Dominance checking (Definition 4.2)
- Three-source diversity sampling (Figure 2)
- Model evaluation (Eq. 9)

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import copy


# ==================== Opponent Entry ====================

@dataclass
class OpponentEntry:
    """
    Single opponent in the pool
    
    Stores:
    - Agent: The opponent agent
    - Stats: Performance statistics
    - Metadata: Additional information
    """
    agent: Any
    name: str
    agent_type: str  # 'self_play', 'sl', 'rl', 'rule'
    
    # Performance statistics
    wins: int = 0
    losses: int = 0
    draws: int = 0
    
    # Negotiation metrics
    avg_utility: float = 0.0
    avg_agreement_rate: float = 0.0
    negotiation_score: float = 0.0  # Eq. 9
    
    # Metadata
    added_at_epoch: int = 0
    num_games: int = 0
    
    def update_stats(
        self,
        won: bool,
        utility: float,
        agreement: bool,
        score: float,
    ):
        """
        Update opponent statistics
        
        Args:
            won: Whether this opponent won
            utility: Utility achieved
            agreement: Whether agreement reached
            score: Negotiation score
        """
        if won:
            self.wins += 1
        elif utility == 0.0 and not agreement:
            self.draws += 1
        else:
            self.losses += 1
        
        self.num_games += 1
        
        # Update moving averages
        alpha = 0.1
        self.avg_utility = (1 - alpha) * self.avg_utility + alpha * utility
        self.avg_agreement_rate = (1 - alpha) * self.avg_agreement_rate + alpha * float(agreement)
        self.negotiation_score = (1 - alpha) * self.negotiation_score + alpha * score
    
    def get_win_rate(self) -> float:
        """Get win rate"""
        total = self.wins + self.losses + self.draws
        return self.wins / max(total, 1)
    
    def dominates(self, other: 'OpponentEntry', threshold: float = 0.05) -> bool:
        """
        Check if this opponent dominates another (Definition 4.2)
        
        An opponent A dominates B if:
        - A's negotiation score > B's score + threshold
        
        Args:
            other: Other opponent
            threshold: Dominance threshold
            
        Returns:
            True if this dominates other
        """
        return self.negotiation_score > other.negotiation_score + threshold


# ==================== Opponent Pool ====================

class OpponentPool:
    """
    Opponent pool with PFSP sampling
    
    Implements:
    - Three-source diversity (Algorithm 1, Figure 2)
    - PFSP sampling (Eq. 10)
    - Dominance-based addition (Definition 4.2)
    - Model evaluation (Eq. 9)
    """
    
    def __init__(
        self,
        prob_self_play: float = 0.20,
        prob_sl_agents: float = 0.30,
        prob_rl_agents: float = 0.50,
        max_pool_size: int = 50,
    ):
        """
        Args:
            prob_self_play: Probability of sampling self-play opponent (20%)
            prob_sl_agents: Probability of sampling SL agents (30%)
            prob_rl_agents: Probability of sampling RL agents (50%)
            max_pool_size: Maximum pool size
        """
        assert abs(prob_self_play + prob_sl_agents + prob_rl_agents - 1.0) < 1e-6
        
        self.prob_self_play = prob_self_play
        self.prob_sl_agents = prob_sl_agents
        self.prob_rl_agents = prob_rl_agents
        self.max_pool_size = max_pool_size
        
        # Pools by type
        self.self_play_pool: List[OpponentEntry] = []
        self.sl_pool: List[OpponentEntry] = []
        self.rl_pool: List[OpponentEntry] = []
        self.rule_pool: List[OpponentEntry] = []
        
        # Current agent (for self-play)
        self.current_agent = None
        
        # Statistics
        self.total_samples = 0
        self.samples_by_type = {'self_play': 0, 'sl': 0, 'rl': 0, 'rule': 0}
        
        print(f"[OpponentPool] Initialized with max_size={max_pool_size}")
        print(f"  PFSP probabilities: self_play={prob_self_play}, "
              f"sl={prob_sl_agents}, rl={prob_rl_agents}")
    
    def add(
        self,
        agent: Any,
        name: str,
        agent_type: str,
        epoch: int = 0,
    ) -> bool:
        """
        Add opponent to pool
        
        Args:
            agent: Opponent agent
            name: Agent name
            agent_type: 'self_play', 'sl', 'rl', or 'rule'
            epoch: Current epoch
            
        Returns:
            True if added successfully
        """
        # Create entry
        entry = OpponentEntry(
            agent=agent,
            name=name,
            agent_type=agent_type,
            added_at_epoch=epoch,
        )
        
        # Add to appropriate pool
        if agent_type == 'self_play':
            self.self_play_pool.append(entry)
        elif agent_type == 'sl':
            self.sl_pool.append(entry)
        elif agent_type == 'rl':
            # Check dominance before adding
            if self._check_dominance(entry):
                self.rl_pool.append(entry)
                print(f"[OpponentPool] Added RL agent '{name}' at epoch {epoch}")
            else:
                print(f"[OpponentPool] Rejected RL agent '{name}' (not dominant)")
                return False
        elif agent_type == 'rule':
            self.rule_pool.append(entry)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Manage pool size
        self._manage_pool_size()
        
        return True
    
    def _check_dominance(self, new_entry: OpponentEntry) -> bool:
        """
        Check if new agent dominates existing agents (Definition 4.2)
        
        For a new RL agent to be added, it should dominate
        at least some existing agents
        
        Args:
            new_entry: New opponent entry
            
        Returns:
            True if agent should be added
        """
        if len(self.rl_pool) == 0:
            return True  # First agent always added
        
        # Check dominance against existing RL agents
        dominates_count = 0
        for existing in self.rl_pool:
            if new_entry.dominates(existing):
                dominates_count += 1
        
        # Add if dominates at least 1/3 of existing agents
        return dominates_count >= len(self.rl_pool) / 3
    
    def _manage_pool_size(self):
        """
        Manage pool size by removing weakest agents
        """
        # Only manage RL pool (others are fixed)
        if len(self.rl_pool) > self.max_pool_size:
            # Sort by negotiation score
            self.rl_pool.sort(key=lambda x: x.negotiation_score, reverse=True)
            # Keep top agents
            self.rl_pool = self.rl_pool[:self.max_pool_size]
            print(f"[OpponentPool] Pruned RL pool to {self.max_pool_size} agents")
    
    def set_current_agent(self, agent: Any):
        """
        Set current agent for self-play
        
        Args:
            agent: Current agent
        """
        self.current_agent = agent
    
    def sample(self, method: str = 'pfsp') -> OpponentEntry:
        """
        Sample opponent from pool
        
        Args:
            method: 'pfsp' (default), 'uniform', or 'weighted'
            
        Returns:
            Opponent entry
        """
        self.total_samples += 1
        
        if method == 'pfsp':
            return self._sample_pfsp()
        elif method == 'uniform':
            return self._sample_uniform()
        elif method == 'weighted':
            return self._sample_weighted()
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def _sample_pfsp(self) -> OpponentEntry:
        """
        Sample using Priority Fictitious Self-Play (Eq. 10)
        
        Three-source diversity:
        - 20% self-play (current agent)
        - 30% SL agents
        - 50% RL agents from pool
        
        Within each source, use PFSP probabilities
        
        Returns:
            Sampled opponent
        """
        # Choose source
        source = np.random.choice(
            ['self_play', 'sl', 'rl'],
            p=[self.prob_self_play, self.prob_sl_agents, self.prob_rl_agents]
        )
        
        self.samples_by_type[source] += 1
        
        # Sample from chosen source
        if source == 'self_play':
            # Self-play: use current agent
            if self.current_agent is not None:
                return OpponentEntry(
                    agent=self.current_agent,
                    name='self_play',
                    agent_type='self_play'
                )
            else:
                # Fallback to SL if no current agent
                source = 'sl'
        
        if source == 'sl':
            pool = self.sl_pool
        elif source == 'rl':
            pool = self.rl_pool
        else:
            pool = self.rule_pool
        
        if len(pool) == 0:
            # Fallback: sample from any available pool
            all_pools = self.sl_pool + self.rl_pool + self.rule_pool
            if len(all_pools) == 0:
                raise ValueError("No opponents in pool!")
            return np.random.choice(all_pools)
        
        # PFSP: compute probabilities based on win rate
        # p(A) = P[A dominates current agent] / Σ P[B dominates current agent]
        win_rates = np.array([entry.get_win_rate() for entry in pool])
        
        # Softmax with temperature
        temperature = 1.0
        probs = np.exp(win_rates / temperature)
        probs = probs / probs.sum()
        
        # Sample
        idx = np.random.choice(len(pool), p=probs)
        return pool[idx]
    
    def _sample_uniform(self) -> OpponentEntry:
        """
        Sample uniformly from all pools
        
        Returns:
            Sampled opponent
        """
        all_pools = self.sl_pool + self.rl_pool + self.rule_pool
        if len(all_pools) == 0:
            raise ValueError("No opponents in pool!")
        
        return np.random.choice(all_pools)
    
    def _sample_weighted(self) -> OpponentEntry:
        """
        Sample weighted by negotiation score
        
        Returns:
            Sampled opponent
        """
        all_pools = self.sl_pool + self.rl_pool + self.rule_pool
        if len(all_pools) == 0:
            raise ValueError("No opponents in pool!")
        
        # Weight by score
        scores = np.array([entry.negotiation_score + 1.0 for entry in all_pools])
        probs = scores / scores.sum()
        
        idx = np.random.choice(len(all_pools), p=probs)
        return all_pools[idx]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_agents': len(self.sl_pool) + len(self.rl_pool) + len(self.rule_pool),
            'sl_agents': len(self.sl_pool),
            'rl_agents': len(self.rl_pool),
            'rule_agents': len(self.rule_pool),
            'self_play_samples': self.samples_by_type['self_play'],
            'sl_samples': self.samples_by_type['sl'],
            'rl_samples': self.samples_by_type['rl'],
            'total_samples': self.total_samples,
        }
    
    def get_strongest_opponent(self) -> Optional[OpponentEntry]:
        """
        Get strongest opponent by negotiation score
        
        Returns:
            Strongest opponent entry
        """
        all_opponents = self.sl_pool + self.rl_pool + self.rule_pool
        if len(all_opponents) == 0:
            return None
        
        return max(all_opponents, key=lambda x: x.negotiation_score)
    
    def get_opponents_by_type(self, agent_type: str) -> List[OpponentEntry]:
        """
        Get all opponents of a specific type
        
        Args:
            agent_type: 'sl', 'rl', or 'rule'
            
        Returns:
            List of opponents
        """
        if agent_type == 'sl':
            return self.sl_pool.copy()
        elif agent_type == 'rl':
            return self.rl_pool.copy()
        elif agent_type == 'rule':
            return self.rule_pool.copy()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")


# ==================== Model Evaluation ====================

def compute_negotiation_score(
    agreement_rate: float,
    utility: float,
    dialogue_length: float,
    social_welfare: float,
    alpha_sc: float = -5e-3,
    beta_sc: float = 0.1,
    epsilon: float = 0.01,
    max_u: float = 1.0,
) -> float:
    """
    Compute negotiation score (Eq. 9)
    
    Sc(A,B) = (1 - min(Ag, 1-ε))^(-U) + α_Sc·Len + β_Sc·SW
    
    Args:
        agreement_rate: Agreement rate [0, 1]
        utility: Agent utility [0, 1]
        dialogue_length: Average dialogue length
        social_welfare: Social welfare [0, 2]
        alpha_sc: Weight for dialogue length (-5e-3)
        beta_sc: Weight for social welfare (0.1)
        epsilon: Small constant (0.01)
        max_u: Maximum utility (1.0)
        
    Returns:
        Negotiation score
    """
    # Component 1: Agreement and utility
    # (1 - min(Ag, 1-ε))^(-U)
    agreement_term = 1.0 - min(agreement_rate, 1.0 - epsilon)
    if agreement_term <= 0:
        agreement_term = epsilon
    
    utility_normalized = utility / max_u
    component1 = agreement_term ** (-utility_normalized)
    
    # Component 2: Dialogue length penalty
    component2 = alpha_sc * dialogue_length
    
    # Component 3: Social welfare bonus
    component3 = beta_sc * social_welfare
    
    # Total score
    score = component1 + component2 + component3
    
    return score


def check_dominance(
    agent_a_score: float,
    agent_b_score: float,
    threshold: float = 0.05,
) -> bool:
    """
    Check if agent A dominates agent B (Definition 4.2)
    
    Agent A dominates B if:
    Sc(A,M) > Sc(B,M) + threshold
    
    Args:
        agent_a_score: Agent A's negotiation score
        agent_b_score: Agent B's negotiation score
        threshold: Dominance threshold (0.05)
        
    Returns:
        True if A dominates B
    """
    return agent_a_score > agent_b_score + threshold


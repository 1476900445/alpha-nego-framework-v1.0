"""
Reward Functions for Negotiation
Implements various reward shaping strategies

Reward Components:
- Agreement reward
- Utility reward
- Length penalty
- Fairness bonus
- No-deal penalty

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import numpy as np
from typing import Dict, Optional, Callable


# ==================== Base Reward Functions ====================

def agreement_reward(
    agreement: bool,
    base_reward: float = 1.0,
) -> float:
    """
    Reward for reaching agreement
    
    Args:
        agreement: Whether agreement was reached
        base_reward: Base reward value
        
    Returns:
        Reward
    """
    return base_reward if agreement else 0.0


def utility_reward(
    utility: float,
    max_utility: float = 1.0,
) -> float:
    """
    Reward based on achieved utility
    
    Args:
        utility: Achieved utility [0, 1]
        max_utility: Maximum possible utility
        
    Returns:
        Reward
    """
    return utility / max_utility if max_utility > 0 else 0.0


def length_penalty(
    dialogue_length: int,
    max_length: int = 20,
    penalty_weight: float = 0.01,
) -> float:
    """
    Penalty for long dialogues
    
    Encourages efficient negotiation
    
    Args:
        dialogue_length: Number of turns
        max_length: Maximum turns
        penalty_weight: Penalty per turn
        
    Returns:
        Penalty (negative)
    """
    return -penalty_weight * dialogue_length


def fairness_bonus(
    agent_utility: float,
    opponent_utility: float,
    bonus_weight: float = 0.05,
) -> float:
    """
    Bonus for fair outcomes
    
    Rewards outcomes where both parties get good utility
    
    Args:
        agent_utility: Agent's utility
        opponent_utility: Opponent's utility
        bonus_weight: Bonus weight
        
    Returns:
        Bonus
    """
    # Bonus inversely proportional to utility gap
    utility_gap = abs(agent_utility - opponent_utility)
    bonus = bonus_weight * (1.0 - utility_gap)
    return bonus


def no_deal_penalty(
    agreement: bool,
    penalty: float = 0.5,
) -> float:
    """
    Penalty for failing to reach agreement
    
    Args:
        agreement: Whether agreement was reached
        penalty: Penalty value
        
    Returns:
        Penalty (negative if no deal)
    """
    return 0.0 if agreement else -penalty


# ==================== Reward Shaper ====================

class RewardShaper:
    """
    Shape rewards with multiple components
    
    Combines:
    - Base reward (utility or agreement)
    - Length penalty
    - Fairness bonus
    - No-deal penalty
    """
    
    def __init__(
        self,
        base_reward_type: str = 'utility',
        use_length_penalty: bool = True,
        use_fairness_bonus: bool = True,
        use_no_deal_penalty: bool = True,
        length_penalty_weight: float = 0.01,
        fairness_bonus_weight: float = 0.05,
        no_deal_penalty_value: float = 0.5,
    ):
        """
        Args:
            base_reward_type: 'utility' or 'agreement'
            use_length_penalty: Include length penalty
            use_fairness_bonus: Include fairness bonus
            use_no_deal_penalty: Include no-deal penalty
            length_penalty_weight: Weight for length penalty
            fairness_bonus_weight: Weight for fairness bonus
            no_deal_penalty_value: No-deal penalty value
        """
        self.base_reward_type = base_reward_type
        self.use_length_penalty = use_length_penalty
        self.use_fairness_bonus = use_fairness_bonus
        self.use_no_deal_penalty = use_no_deal_penalty
        self.length_penalty_weight = length_penalty_weight
        self.fairness_bonus_weight = fairness_bonus_weight
        self.no_deal_penalty_value = no_deal_penalty_value
    
    def compute_reward(
        self,
        agreement: bool,
        agent_utility: float,
        opponent_utility: float,
        dialogue_length: int,
        max_length: int = 20,
    ) -> float:
        """
        Compute shaped reward
        
        Args:
            agreement: Whether agreement reached
            agent_utility: Agent's utility
            opponent_utility: Opponent's utility
            dialogue_length: Number of turns
            max_length: Maximum turns
            
        Returns:
            Shaped reward
        """
        # Base reward
        if self.base_reward_type == 'utility':
            reward = utility_reward(agent_utility)
        elif self.base_reward_type == 'agreement':
            reward = agreement_reward(agreement)
        else:
            reward = 0.0
        
        # Add length penalty
        if self.use_length_penalty:
            reward += length_penalty(
                dialogue_length,
                max_length,
                self.length_penalty_weight
            )
        
        # Add fairness bonus
        if self.use_fairness_bonus and agreement:
            reward += fairness_bonus(
                agent_utility,
                opponent_utility,
                self.fairness_bonus_weight
            )
        
        # Add no-deal penalty
        if self.use_no_deal_penalty:
            reward += no_deal_penalty(agreement, self.no_deal_penalty_value)
        
        return reward
    
    def compute_step_reward(
        self,
        is_terminal: bool,
        agreement: bool,
        agent_utility: float,
        opponent_utility: float,
        current_turn: int,
        max_turns: int = 20,
    ) -> float:
        """
        Compute reward for a single step
        
        Args:
            is_terminal: Whether episode ended
            agreement: Whether agreement reached
            agent_utility: Agent's utility
            opponent_utility: Opponent's utility
            current_turn: Current turn number
            max_turns: Maximum turns
            
        Returns:
            Step reward
        """
        if is_terminal:
            # Terminal reward
            return self.compute_reward(
                agreement,
                agent_utility,
                opponent_utility,
                current_turn,
                max_turns
            )
        else:
            # Step penalty (encourage efficiency)
            return -self.length_penalty_weight if self.use_length_penalty else 0.0


# ==================== Sparse Reward ====================

class SparseReward:
    """
    Sparse reward (only at end of episode)
    """
    
    def __init__(
        self,
        success_reward: float = 1.0,
        failure_reward: float = 0.0,
    ):
        """
        Args:
            success_reward: Reward for successful negotiation
            failure_reward: Reward for failed negotiation
        """
        self.success_reward = success_reward
        self.failure_reward = failure_reward
    
    def compute_reward(
        self,
        agreement: bool,
        utility: float,
    ) -> float:
        """
        Compute sparse reward
        
        Args:
            agreement: Whether agreement reached
            utility: Agent's utility
            
        Returns:
            Reward
        """
        if agreement:
            return self.success_reward * utility
        else:
            return self.failure_reward


# ==================== Dense Reward ====================

class DenseReward:
    """
    Dense reward (at every step)
    """
    
    def __init__(
        self,
        step_penalty: float = 0.01,
        progress_bonus: float = 0.05,
    ):
        """
        Args:
            step_penalty: Penalty per step
            progress_bonus: Bonus for making progress
        """
        self.step_penalty = step_penalty
        self.progress_bonus = progress_bonus
        
        self.last_price_gap = None
    
    def compute_step_reward(
        self,
        current_price: float,
        opponent_price: float,
        is_terminal: bool,
        agreement: bool,
        utility: float,
    ) -> float:
        """
        Compute dense step reward
        
        Args:
            current_price: Agent's current price
            opponent_price: Opponent's price
            is_terminal: Whether episode ended
            agreement: Whether agreement reached
            utility: Agent's utility
            
        Returns:
            Step reward
        """
        if is_terminal:
            # Terminal reward
            if agreement:
                return utility
            else:
                return -0.5
        
        # Step penalty
        reward = -self.step_penalty
        
        # Progress bonus (getting closer to opponent)
        if opponent_price > 0:
            price_gap = abs(current_price - opponent_price)
            
            if self.last_price_gap is not None:
                if price_gap < self.last_price_gap:
                    # Made progress
                    reward += self.progress_bonus
            
            self.last_price_gap = price_gap
        
        return reward
    
    def reset(self):
        """Reset for new episode"""
        self.last_price_gap = None


# ==================== Curriculum Reward ====================

class CurriculumReward:
    """
    Curriculum learning with gradually increasing difficulty
    """
    
    def __init__(
        self,
        initial_bonus: float = 0.5,
        final_bonus: float = 0.0,
        decay_episodes: int = 1000,
    ):
        """
        Args:
            initial_bonus: Initial bonus for any agreement
            final_bonus: Final bonus (only utility matters)
            decay_episodes: Episodes to decay from initial to final
        """
        self.initial_bonus = initial_bonus
        self.final_bonus = final_bonus
        self.decay_episodes = decay_episodes
        
        self.current_episode = 0
    
    def compute_reward(
        self,
        agreement: bool,
        utility: float,
    ) -> float:
        """
        Compute curriculum reward
        
        Args:
            agreement: Whether agreement reached
            utility: Agent's utility
            
        Returns:
            Reward
        """
        # Compute current bonus
        progress = min(self.current_episode / self.decay_episodes, 1.0)
        current_bonus = self.initial_bonus + (self.final_bonus - self.initial_bonus) * progress
        
        # Reward
        if agreement:
            reward = utility + current_bonus
        else:
            reward = 0.0
        
        return reward
    
    def step_episode(self):
        """Update episode counter"""
        self.current_episode += 1


# ==================== Custom Reward Function ====================

class CustomRewardFunction:
    """
    Custom reward function with user-defined logic
    """
    
    def __init__(self, reward_fn: Callable):
        """
        Args:
            reward_fn: Custom reward function
        """
        self.reward_fn = reward_fn
    
    def compute_reward(self, **kwargs) -> float:
        """
        Compute reward using custom function
        
        Args:
            **kwargs: Arguments to pass to custom function
            
        Returns:
            Reward
        """
        return self.reward_fn(**kwargs)


# ==================== Reward Analyzer ====================

class RewardAnalyzer:
    """
    Analyze and track reward statistics
    """
    
    def __init__(self):
        self.episode_rewards = []
        self.step_rewards = []
        self.agreement_rates = []
        self.utilities = []
    
    def record_episode(
        self,
        total_reward: float,
        agreement: bool,
        utility: float,
        step_rewards: Optional[list] = None,
    ):
        """
        Record episode statistics
        
        Args:
            total_reward: Total episode reward
            agreement: Whether agreement reached
            utility: Agent's utility
            step_rewards: List of step rewards
        """
        self.episode_rewards.append(total_reward)
        self.agreement_rates.append(1.0 if agreement else 0.0)
        self.utilities.append(utility)
        
        if step_rewards:
            self.step_rewards.extend(step_rewards)
    
    def get_statistics(self) -> Dict:
        """
        Get reward statistics
        
        Returns:
            Statistics dictionary
        """
        if len(self.episode_rewards) == 0:
            return {}
        
        return {
            'mean_episode_reward': np.mean(self.episode_rewards),
            'std_episode_reward': np.std(self.episode_rewards),
            'mean_agreement_rate': np.mean(self.agreement_rates),
            'mean_utility': np.mean(self.utilities),
            'max_episode_reward': np.max(self.episode_rewards),
            'min_episode_reward': np.min(self.episode_rewards),
        }
    
    def reset(self):
        """Reset statistics"""
        self.episode_rewards = []
        self.step_rewards = []
        self.agreement_rates = []
        self.utilities = []


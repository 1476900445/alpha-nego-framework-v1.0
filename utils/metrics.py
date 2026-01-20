"""
Metrics for Negotiation Evaluation
Implements various evaluation metrics including Eq. 9

Metrics:
- Negotiation Score (Eq. 9)
- Agreement rate
- Utility metrics
- Dialogue efficiency
- Fairness metrics

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ==================== Negotiation Score (Eq. 9) ====================

def compute_negotiation_score(
    agreement_rate: float,
    utility: float,
    dialogue_length: float,
    social_welfare: float,
    alpha_sc: float = -5e-3,
    beta_sc: float = 0.1,
    epsilon: float = 0.01,
) -> float:
    """
    Compute negotiation score (Eq. 9)
    
    Sc(A,B) = (1 - min(Ag, 1-ε))^(-U) + α_Sc·Len + β_Sc·SW
    
    Where:
    - Ag: Agreement rate
    - U: Average utility
    - Len: Average dialogue length
    - SW: Social welfare
    - ε: Small constant to avoid division by zero
    
    Args:
        agreement_rate: Agreement rate [0, 1]
        utility: Average utility [0, 1]
        dialogue_length: Average dialogue length
        social_welfare: Social welfare (sum of utilities)
        alpha_sc: Weight for dialogue length penalty
        beta_sc: Weight for social welfare bonus
        epsilon: Small constant
        
    Returns:
        Negotiation score
    """
    # Base score: (1 - min(Ag, 1-ε))^(-U)
    ag_term = 1 - min(agreement_rate, 1 - epsilon)
    if ag_term <= 0:
        ag_term = epsilon
    
    base_score = ag_term ** (-utility)
    
    # Length penalty
    length_penalty = alpha_sc * dialogue_length
    
    # Social welfare bonus
    welfare_bonus = beta_sc * social_welfare
    
    # Total score
    score = base_score + length_penalty + welfare_bonus
    
    return score


# ==================== Basic Metrics ====================

def compute_agreement_rate(agreements: List[bool]) -> float:
    """
    Compute agreement rate
    
    Args:
        agreements: List of agreement outcomes
        
    Returns:
        Agreement rate [0, 1]
    """
    if len(agreements) == 0:
        return 0.0
    return np.mean(agreements)


def compute_average_utility(utilities: List[float]) -> float:
    """
    Compute average utility
    
    Args:
        utilities: List of utility values
        
    Returns:
        Average utility
    """
    if len(utilities) == 0:
        return 0.0
    return np.mean(utilities)


def compute_social_welfare(
    agent_utilities: List[float],
    opponent_utilities: List[float],
) -> float:
    """
    Compute social welfare (sum of utilities)
    
    Args:
        agent_utilities: Agent utilities
        opponent_utilities: Opponent utilities
        
    Returns:
        Average social welfare
    """
    if len(agent_utilities) == 0:
        return 0.0
    
    social_welfares = [
        ag + op for ag, op in zip(agent_utilities, opponent_utilities)
    ]
    
    return np.mean(social_welfares)


def compute_average_dialogue_length(lengths: List[int]) -> float:
    """
    Compute average dialogue length
    
    Args:
        lengths: List of dialogue lengths
        
    Returns:
        Average length
    """
    if len(lengths) == 0:
        return 0.0
    return np.mean(lengths)


# ==================== Fairness Metrics ====================

def compute_fairness(
    agent_utilities: List[float],
    opponent_utilities: List[float],
) -> Dict[str, float]:
    """
    Compute fairness metrics
    
    Args:
        agent_utilities: Agent utilities
        opponent_utilities: Opponent utilities
        
    Returns:
        Fairness metrics
    """
    if len(agent_utilities) == 0:
        return {
            'utility_gap': 0.0,
            'jain_fairness': 0.0,
            'balance_rate': 0.0,
        }
    
    # Utility gap
    utility_gaps = [
        abs(ag - op) for ag, op in zip(agent_utilities, opponent_utilities)
    ]
    avg_utility_gap = np.mean(utility_gaps)
    
    # Jain's fairness index
    utilities_sum = [
        ag + op for ag, op in zip(agent_utilities, opponent_utilities)
    ]
    
    if sum(utilities_sum) > 0:
        jain_fairness = (sum(utilities_sum) ** 2) / (len(utilities_sum) * sum([u**2 for u in utilities_sum]))
    else:
        jain_fairness = 0.0
    
    # Balance rate (how often utilities are close)
    balanced = [
        abs(ag - op) < 0.1 for ag, op in zip(agent_utilities, opponent_utilities)
    ]
    balance_rate = np.mean(balanced)
    
    return {
        'utility_gap': avg_utility_gap,
        'jain_fairness': jain_fairness,
        'balance_rate': balance_rate,
    }


# ==================== Efficiency Metrics ====================

def compute_pareto_efficiency(
    agent_utilities: List[float],
    opponent_utilities: List[float],
) -> float:
    """
    Compute Pareto efficiency rate
    
    An outcome is Pareto efficient if no other outcome can improve
    one party's utility without decreasing the other's
    
    Args:
        agent_utilities: Agent utilities
        opponent_utilities: Opponent utilities
        
    Returns:
        Pareto efficiency rate
    """
    if len(agent_utilities) == 0:
        return 0.0
    
    pareto_efficient = []
    
    for i in range(len(agent_utilities)):
        ag_i, op_i = agent_utilities[i], opponent_utilities[i]
        is_efficient = True
        
        # Check if any other outcome dominates
        for j in range(len(agent_utilities)):
            if i == j:
                continue
            
            ag_j, op_j = agent_utilities[j], opponent_utilities[j]
            
            # If (ag_j, op_j) dominates (ag_i, op_i)
            if ag_j >= ag_i and op_j >= op_i and (ag_j > ag_i or op_j > op_i):
                is_efficient = False
                break
        
        pareto_efficient.append(is_efficient)
    
    return np.mean(pareto_efficient)


def compute_nash_welfare(
    agent_utilities: List[float],
    opponent_utilities: List[float],
) -> float:
    """
    Compute Nash welfare (geometric mean of utilities)
    
    Args:
        agent_utilities: Agent utilities
        opponent_utilities: Opponent utilities
        
    Returns:
        Average Nash welfare
    """
    if len(agent_utilities) == 0:
        return 0.0
    
    nash_welfares = [
        np.sqrt(max(ag, 0) * max(op, 0))
        for ag, op in zip(agent_utilities, opponent_utilities)
    ]
    
    return np.mean(nash_welfares)


# ==================== Comprehensive Metrics ====================

class NegotiationMetrics:
    """
    Compute comprehensive negotiation metrics
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.agreements = []
        self.agent_utilities = []
        self.opponent_utilities = []
        self.dialogue_lengths = []
        self.rewards = []
    
    def add_episode(
        self,
        agreement: bool,
        agent_utility: float,
        opponent_utility: float,
        dialogue_length: int,
        reward: float,
    ):
        """
        Add episode result
        
        Args:
            agreement: Whether agreement was reached
            agent_utility: Agent's utility
            opponent_utility: Opponent's utility
            dialogue_length: Dialogue length
            reward: Episode reward
        """
        self.agreements.append(agreement)
        self.agent_utilities.append(agent_utility)
        self.opponent_utilities.append(opponent_utility)
        self.dialogue_lengths.append(dialogue_length)
        self.rewards.append(reward)
    
    def compute_all(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary with all metrics
        """
        if len(self.agreements) == 0:
            return {}
        
        metrics = {}
        
        # Basic metrics
        metrics['agreement_rate'] = compute_agreement_rate(self.agreements)
        metrics['avg_agent_utility'] = compute_average_utility(self.agent_utilities)
        metrics['avg_opponent_utility'] = compute_average_utility(self.opponent_utilities)
        metrics['avg_dialogue_length'] = compute_average_dialogue_length(self.dialogue_lengths)
        metrics['avg_reward'] = np.mean(self.rewards)
        
        # Social welfare
        metrics['avg_social_welfare'] = compute_social_welfare(
            self.agent_utilities,
            self.opponent_utilities
        )
        
        # Negotiation score (Eq. 9)
        if metrics['agreement_rate'] > 0:
            metrics['negotiation_score'] = compute_negotiation_score(
                agreement_rate=metrics['agreement_rate'],
                utility=metrics['avg_agent_utility'],
                dialogue_length=metrics['avg_dialogue_length'],
                social_welfare=metrics['avg_social_welfare'],
            )
        
        # Fairness metrics
        fairness = compute_fairness(self.agent_utilities, self.opponent_utilities)
        metrics.update(fairness)
        
        # Efficiency metrics
        metrics['pareto_efficiency'] = compute_pareto_efficiency(
            self.agent_utilities,
            self.opponent_utilities
        )
        metrics['nash_welfare'] = compute_nash_welfare(
            self.agent_utilities,
            self.opponent_utilities
        )
        
        # Standard deviations
        metrics['std_agent_utility'] = np.std(self.agent_utilities)
        metrics['std_dialogue_length'] = np.std(self.dialogue_lengths)
        metrics['std_reward'] = np.std(self.rewards)
        
        return metrics
    
    def print_summary(self):
        """Print metrics summary"""
        metrics = self.compute_all()
        
        print("\n" + "="*70)
        print("NEGOTIATION METRICS SUMMARY")
        print("="*70)
        
        print("\nBasic Metrics:")
        print(f"  Episodes: {len(self.agreements)}")
        print(f"  Agreement Rate: {metrics.get('agreement_rate', 0):.2%}")
        print(f"  Avg Agent Utility: {metrics.get('avg_agent_utility', 0):.4f} ± {metrics.get('std_agent_utility', 0):.4f}")
        print(f"  Avg Opponent Utility: {metrics.get('avg_opponent_utility', 0):.4f}")
        print(f"  Avg Dialogue Length: {metrics.get('avg_dialogue_length', 0):.2f} ± {metrics.get('std_dialogue_length', 0):.2f}")
        print(f"  Avg Reward: {metrics.get('avg_reward', 0):.4f} ± {metrics.get('std_reward', 0):.4f}")
        
        print("\nSocial Welfare:")
        print(f"  Social Welfare: {metrics.get('avg_social_welfare', 0):.4f}")
        print(f"  Nash Welfare: {metrics.get('nash_welfare', 0):.4f}")
        
        print("\nFairness:")
        print(f"  Utility Gap: {metrics.get('utility_gap', 0):.4f}")
        print(f"  Jain's Fairness: {metrics.get('jain_fairness', 0):.4f}")
        print(f"  Balance Rate: {metrics.get('balance_rate', 0):.2%}")
        
        print("\nEfficiency:")
        print(f"  Pareto Efficiency: {metrics.get('pareto_efficiency', 0):.2%}")
        
        print("\nOverall Score:")
        print(f"  Negotiation Score: {metrics.get('negotiation_score', 0):.4f}")
        
        print("="*70)


# ==================== Win Rate Metrics ====================

def compute_win_rate(
    agent_utilities: List[float],
    opponent_utilities: List[float],
) -> Dict[str, float]:
    """
    Compute win rate metrics
    
    Args:
        agent_utilities: Agent utilities
        opponent_utilities: Opponent utilities
        
    Returns:
        Win rate metrics
    """
    if len(agent_utilities) == 0:
        return {
            'win_rate': 0.0,
            'tie_rate': 0.0,
            'loss_rate': 0.0,
        }
    
    wins = sum([ag > op for ag, op in zip(agent_utilities, opponent_utilities)])
    ties = sum([ag == op for ag, op in zip(agent_utilities, opponent_utilities)])
    losses = sum([ag < op for ag, op in zip(agent_utilities, opponent_utilities)])
    
    total = len(agent_utilities)
    
    return {
        'win_rate': wins / total,
        'tie_rate': ties / total,
        'loss_rate': losses / total,
    }


"""
Priority Fictitious Self-Play (PFSP)
Implements opponent selection strategy (Eq. 10)

Key Features:
- Priority-based sampling
- Diversity through three sources
- Win-rate based probabilities
- Opponent strength modeling

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class PFSPSampler:
    """
    Priority Fictitious Self-Play Sampler
    
    Implements Eq. 10:
    p(A) = P[A dominates M] / Σ_B P[B dominates M]
    
    Where M is the current model being trained
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        use_win_rate: bool = True,
        use_score: bool = False,
    ):
        """
        Args:
            temperature: Softmax temperature for probability distribution
            use_win_rate: Use win rate for priority
            use_score: Use negotiation score for priority
        """
        self.temperature = temperature
        self.use_win_rate = use_win_rate
        self.use_score = use_score
        
        # Opponent statistics
        self.opponent_stats = defaultdict(lambda: {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_games': 0,
            'scores': [],
        })
        
        print(f"[PFSPSampler] Initialized")
        print(f"  Temperature: {temperature}")
        print(f"  Use win rate: {use_win_rate}")
        print(f"  Use score: {use_score}")
    
    def update(
        self,
        opponent_id: str,
        won: bool,
        draw: bool = False,
        score: Optional[float] = None,
    ):
        """
        Update opponent statistics
        
        Args:
            opponent_id: Opponent identifier
            won: Whether opponent won against current agent
            draw: Whether it was a draw
            score: Negotiation score
        """
        stats = self.opponent_stats[opponent_id]
        
        if draw:
            stats['draws'] += 1
        elif won:
            stats['wins'] += 1
        else:
            stats['losses'] += 1
        
        stats['total_games'] += 1
        
        if score is not None:
            stats['scores'].append(score)
    
    def get_win_rate(self, opponent_id: str) -> float:
        """
        Get opponent win rate
        
        Args:
            opponent_id: Opponent identifier
            
        Returns:
            Win rate [0, 1]
        """
        stats = self.opponent_stats[opponent_id]
        total = stats['total_games']
        
        if total == 0:
            return 0.5  # Default win rate
        
        return stats['wins'] / total
    
    def get_avg_score(self, opponent_id: str) -> float:
        """
        Get opponent average negotiation score
        
        Args:
            opponent_id: Opponent identifier
            
        Returns:
            Average score
        """
        stats = self.opponent_stats[opponent_id]
        scores = stats['scores']
        
        if len(scores) == 0:
            return 1.0  # Default score
        
        return np.mean(scores)
    
    def compute_priorities(
        self,
        opponent_ids: List[str],
    ) -> np.ndarray:
        """
        Compute PFSP priorities for opponents
        
        Implements Eq. 10:
        p(A) ∝ P[A dominates current model]
        
        Args:
            opponent_ids: List of opponent identifiers
            
        Returns:
            Priority array (normalized probabilities)
        """
        if len(opponent_ids) == 0:
            return np.array([])
        
        priorities = []
        
        for opp_id in opponent_ids:
            if self.use_win_rate:
                # Use win rate as proxy for dominance probability
                win_rate = self.get_win_rate(opp_id)
                priority = win_rate
            elif self.use_score:
                # Use negotiation score
                score = self.get_avg_score(opp_id)
                priority = score
            else:
                # Uniform
                priority = 1.0
            
            priorities.append(priority)
        
        priorities = np.array(priorities)
        
        # Apply softmax with temperature
        if self.temperature > 0:
            priorities = np.exp(priorities / self.temperature)
            priorities = priorities / priorities.sum()
        else:
            # Greedy: select highest priority
            max_idx = np.argmax(priorities)
            priorities = np.zeros(len(priorities))
            priorities[max_idx] = 1.0
        
        return priorities
    
    def sample(
        self,
        opponent_ids: List[str],
    ) -> str:
        """
        Sample opponent using PFSP
        
        Args:
            opponent_ids: List of opponent identifiers
            
        Returns:
            Selected opponent ID
        """
        if len(opponent_ids) == 0:
            raise ValueError("No opponents to sample from")
        
        if len(opponent_ids) == 1:
            return opponent_ids[0]
        
        # Compute priorities
        priorities = self.compute_priorities(opponent_ids)
        
        # Sample
        selected_idx = np.random.choice(len(opponent_ids), p=priorities)
        
        return opponent_ids[selected_idx]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get sampler statistics
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'num_opponents': len(self.opponent_stats),
            'opponents': {}
        }
        
        for opp_id, opp_stats in self.opponent_stats.items():
            stats['opponents'][opp_id] = {
                'win_rate': self.get_win_rate(opp_id),
                'avg_score': self.get_avg_score(opp_id),
                'total_games': opp_stats['total_games'],
                'wins': opp_stats['wins'],
                'losses': opp_stats['losses'],
                'draws': opp_stats['draws'],
            }
        
        return stats
    
    def reset(self):
        """Reset all statistics"""
        self.opponent_stats.clear()


# ==================== Three-Source Diversity Sampler ====================

class ThreeSourceSampler:
    """
    Three-source diversity sampler (Figure 2)
    
    Samples from:
    - 20% self-play (current agent)
    - 30% SL agents
    - 50% RL agents from pool
    """
    
    def __init__(
        self,
        prob_self_play: float = 0.20,
        prob_sl_agents: float = 0.30,
        prob_rl_agents: float = 0.50,
        pfsp_temperature: float = 1.0,
    ):
        """
        Args:
            prob_self_play: Probability of self-play
            prob_sl_agents: Probability of SL agents
            prob_rl_agents: Probability of RL agents
            pfsp_temperature: Temperature for PFSP within each source
        """
        assert abs(prob_self_play + prob_sl_agents + prob_rl_agents - 1.0) < 1e-6
        
        self.prob_self_play = prob_self_play
        self.prob_sl_agents = prob_sl_agents
        self.prob_rl_agents = prob_rl_agents
        
        # PFSP samplers for each source
        self.sl_sampler = PFSPSampler(temperature=pfsp_temperature)
        self.rl_sampler = PFSPSampler(temperature=pfsp_temperature)
        
        # Statistics
        self.samples_by_source = {
            'self_play': 0,
            'sl': 0,
            'rl': 0,
        }
        
        print(f"[ThreeSourceSampler] Initialized")
        print(f"  Self-play: {prob_self_play:.1%}")
        print(f"  SL agents: {prob_sl_agents:.1%}")
        print(f"  RL agents: {prob_rl_agents:.1%}")
    
    def sample_source(self) -> str:
        """
        Sample which source to use
        
        Returns:
            Source name: 'self_play', 'sl', or 'rl'
        """
        source = np.random.choice(
            ['self_play', 'sl', 'rl'],
            p=[self.prob_self_play, self.prob_sl_agents, self.prob_rl_agents]
        )
        
        self.samples_by_source[source] += 1
        
        return source
    
    def sample_opponent(
        self,
        sl_opponent_ids: List[str],
        rl_opponent_ids: List[str],
        current_agent_id: str = 'self_play',
    ) -> Tuple[str, str]:
        """
        Sample opponent using three-source diversity
        
        Args:
            sl_opponent_ids: List of SL opponent IDs
            rl_opponent_ids: List of RL opponent IDs
            current_agent_id: Current agent ID for self-play
            
        Returns:
            (source, opponent_id)
        """
        source = self.sample_source()
        
        if source == 'self_play':
            return source, current_agent_id
        elif source == 'sl':
            if len(sl_opponent_ids) == 0:
                # Fallback to RL if no SL agents
                source = 'rl'
                return source, self.rl_sampler.sample(rl_opponent_ids)
            return source, self.sl_sampler.sample(sl_opponent_ids)
        else:  # 'rl'
            if len(rl_opponent_ids) == 0:
                # Fallback to SL if no RL agents
                source = 'sl'
                return source, self.sl_sampler.sample(sl_opponent_ids)
            return source, self.rl_sampler.sample(rl_opponent_ids)
    
    def update(
        self,
        source: str,
        opponent_id: str,
        won: bool,
        draw: bool = False,
        score: Optional[float] = None,
    ):
        """
        Update opponent statistics
        
        Args:
            source: Source name
            opponent_id: Opponent ID
            won: Whether opponent won
            draw: Whether it was a draw
            score: Negotiation score
        """
        if source == 'sl':
            self.sl_sampler.update(opponent_id, won, draw, score)
        elif source == 'rl':
            self.rl_sampler.update(opponent_id, won, draw, score)
        # No update for self_play
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sampler statistics"""
        return {
            'samples_by_source': self.samples_by_source.copy(),
            'sl_stats': self.sl_sampler.get_stats(),
            'rl_stats': self.rl_sampler.get_stats(),
        }


# ==================== Adaptive PFSP ====================

class AdaptivePFSPSampler(PFSPSampler):
    """
    Adaptive PFSP with automatic temperature adjustment
    
    Adjusts temperature based on training progress
    """
    
    def __init__(
        self,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.1,
        decay_steps: int = 100000,
    ):
        """
        Args:
            initial_temperature: Initial temperature
            final_temperature: Final temperature
            decay_steps: Steps to decay from initial to final
        """
        super().__init__(temperature=initial_temperature)
        
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.decay_steps = decay_steps
        self.current_step = 0
    
    def step(self):
        """Update temperature"""
        self.current_step += 1
        
        # Linear decay
        progress = min(self.current_step / self.decay_steps, 1.0)
        self.temperature = (
            self.initial_temperature +
            (self.final_temperature - self.initial_temperature) * progress
        )
    
    def sample(self, opponent_ids: List[str]) -> str:
        """Sample and update temperature"""
        result = super().sample(opponent_ids)
        self.step()
        return result


# ==================== Win-Rate Tracker ====================

class WinRateTracker:
    """
    Track win rates for PFSP
    
    Maintains sliding window of recent results
    """
    
    def __init__(
        self,
        window_size: int = 100,
    ):
        """
        Args:
            window_size: Number of recent games to track
        """
        self.window_size = window_size
        self.results = defaultdict(list)  # opponent_id -> list of results
    
    def update(self, opponent_id: str, won: bool):
        """
        Update results
        
        Args:
            opponent_id: Opponent ID
            won: Whether opponent won
        """
        results = self.results[opponent_id]
        results.append(int(won))
        
        # Maintain window size
        if len(results) > self.window_size:
            results.pop(0)
    
    def get_win_rate(self, opponent_id: str) -> float:
        """
        Get recent win rate
        
        Args:
            opponent_id: Opponent ID
            
        Returns:
            Win rate in recent window
        """
        results = self.results[opponent_id]
        
        if len(results) == 0:
            return 0.5  # Default
        
        return np.mean(results)
    
    def get_all_win_rates(self) -> Dict[str, float]:
        """Get win rates for all opponents"""
        return {
            opp_id: self.get_win_rate(opp_id)
            for opp_id in self.results.keys()
        }


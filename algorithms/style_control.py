"""
Style Control for α-Nego
Implements three negotiation styles with distributional Q-values

Styles (Paper Section 6.3):
1. Neutral (Eq. 12): Expected value - balanced approach
2. Aggressive (Eq. 13): Risk-seeking - pursue high-value deals
3. Conservative (Eq. 14): Risk-averse - prioritize agreement

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ==================== Style Definitions ====================

@dataclass
class NegotiationStyle:
    """
    Negotiation style specification
    """
    name: str
    description: str
    q_computation: str  # 'mean', 'mean_plus_variance', 'cvar'
    quantile_range: Tuple[float, float]  # (min_tau, max_tau)
    parameters: Dict[str, float]  # Style-specific parameters


# Predefined styles from paper
NEUTRAL_STYLE = NegotiationStyle(
    name='neutral',
    description='Balanced approach - optimize expected value',
    q_computation='mean',
    quantile_range=(0.0, 1.0),
    parameters={}
)

AGGRESSIVE_STYLE = NegotiationStyle(
    name='aggressive',
    description='Risk-seeking - pursue high-value deals',
    q_computation='mean_plus_variance',
    quantile_range=(0.5, 1.0),  # Focus on upper tail
    parameters={'alpha_agg': 1.0}
)

CONSERVATIVE_STYLE = NegotiationStyle(
    name='conservative',
    description='Risk-averse - prioritize agreement',
    q_computation='cvar',
    quantile_range=(0.0, 0.2),  # Focus on lower tail
    parameters={'alpha_con': 0.2}
)


# ==================== Style Controller ====================

class StyleController:
    """
    Controller for negotiation style
    
    Manages style switching and Q-value computation
    """
    
    def __init__(
        self,
        initial_style: str = 'neutral',
        alpha_agg: float = 1.0,
        alpha_con: float = 0.2,
    ):
        """
        Args:
            initial_style: Initial style ('neutral', 'aggressive', 'conservative')
            alpha_agg: Variance bonus weight for aggressive style (Eq. 13)
            alpha_con: CVaR parameter for conservative style (Eq. 14)
        """
        self.styles = {
            'neutral': NEUTRAL_STYLE,
            'aggressive': AGGRESSIVE_STYLE,
            'conservative': CONSERVATIVE_STYLE,
        }
        
        # Update parameters
        self.styles['aggressive'].parameters['alpha_agg'] = alpha_agg
        self.styles['conservative'].parameters['alpha_con'] = alpha_con
        
        self.current_style = initial_style
        
        # Statistics
        self.style_history = [initial_style]
        self.style_performance = {
            'neutral': {'total_utility': 0.0, 'episodes': 0},
            'aggressive': {'total_utility': 0.0, 'episodes': 0},
            'conservative': {'total_utility': 0.0, 'episodes': 0},
        }
        
        print(f"[StyleController] Initialized with style: {initial_style}")
        print(f"  Alpha_agg: {alpha_agg}")
        print(f"  Alpha_con: {alpha_con}")
    
    def get_style(self) -> NegotiationStyle:
        """Get current style"""
        return self.styles[self.current_style]
    
    def set_style(self, style: str):
        """
        Set current style
        
        Args:
            style: Style name
        """
        if style not in self.styles:
            raise ValueError(f"Unknown style: {style}")
        
        self.current_style = style
        self.style_history.append(style)
        
        print(f"[StyleController] Style changed to: {style}")
    
    def compute_q_value(
        self,
        quantiles: torch.Tensor,
        style: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute Q-value based on style
        
        Args:
            quantiles: Quantile values (batch, num_quantiles)
            style: Style name (uses current if None)
            
        Returns:
            Q-values (batch,)
        """
        if style is None:
            style = self.current_style
        
        style_obj = self.styles[style]
        
        if style_obj.q_computation == 'mean':
            return self._compute_neutral_q(quantiles, style_obj)
        elif style_obj.q_computation == 'mean_plus_variance':
            return self._compute_aggressive_q(quantiles, style_obj)
        elif style_obj.q_computation == 'cvar':
            return self._compute_conservative_q(quantiles, style_obj)
        else:
            raise ValueError(f"Unknown Q computation: {style_obj.q_computation}")
    
    def _compute_neutral_q(
        self,
        quantiles: torch.Tensor,
        style: NegotiationStyle,
    ) -> torch.Tensor:
        """
        Neutral style: Mean of all quantiles (Eq. 12)
        
        Q(s,a) = E_{τ_i∼[0,1]} Z_{τ_i}(s,a;θ)
        
        Args:
            quantiles: (batch, num_quantiles)
            style: Style object
            
        Returns:
            Q-values (batch,)
        """
        # Mean of all quantiles
        q_value = quantiles.mean(dim=-1)
        return q_value
    
    def _compute_aggressive_q(
        self,
        quantiles: torch.Tensor,
        style: NegotiationStyle,
    ) -> torch.Tensor:
        """
        Aggressive style: Mean + left truncated variance (Eq. 13)
        
        Q(s,a) = E[Z_upper] + α_agg · σ²_+(s,a)
        
        Left truncated variance focuses on upper tail
        
        Args:
            quantiles: (batch, num_quantiles)
            style: Style object
            
        Returns:
            Q-values (batch,)
        """
        alpha_agg = style.parameters['alpha_agg']
        num_quantiles = quantiles.size(1)
        
        # Split into lower and upper half
        mid_idx = num_quantiles // 2
        upper_quantiles = quantiles[:, mid_idx:]
        
        # Mean of upper quantiles
        mean_upper = upper_quantiles.mean(dim=-1)
        
        # Variance of upper quantiles (left truncated)
        var_upper = upper_quantiles.var(dim=-1)
        
        # Q = mean + alpha * variance
        q_value = mean_upper + alpha_agg * var_upper
        
        return q_value
    
    def _compute_conservative_q(
        self,
        quantiles: torch.Tensor,
        style: NegotiationStyle,
    ) -> torch.Tensor:
        """
        Conservative style: CVaR of lower tail (Eq. 14)
        
        Q(s,a) = E_{τ_i∼[0,α_con]} Z_{τ_i}(s,a)
        
        Conditional Value at Risk focuses on worst-case scenarios
        
        Args:
            quantiles: (batch, num_quantiles)
            style: Style object
            
        Returns:
            Q-values (batch,)
        """
        alpha_con = style.parameters['alpha_con']
        num_quantiles = quantiles.size(1)
        
        # Number of quantiles in lower tail
        num_lower = max(1, int(num_quantiles * alpha_con))
        
        # CVaR: mean of lower tail
        lower_quantiles = quantiles[:, :num_lower]
        q_value = lower_quantiles.mean(dim=-1)
        
        return q_value
    
    def update_performance(
        self,
        style: str,
        utility: float,
    ):
        """
        Update style performance statistics
        
        Args:
            style: Style name
            utility: Achieved utility
        """
        if style in self.style_performance:
            self.style_performance[style]['total_utility'] += utility
            self.style_performance[style]['episodes'] += 1
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics for all styles
        
        Returns:
            Dictionary with average utilities
        """
        stats = {}
        
        for style, perf in self.style_performance.items():
            if perf['episodes'] > 0:
                avg_utility = perf['total_utility'] / perf['episodes']
            else:
                avg_utility = 0.0
            
            stats[style] = {
                'avg_utility': avg_utility,
                'episodes': perf['episodes'],
            }
        
        return stats
    
    def get_best_style(self) -> str:
        """
        Get best performing style
        
        Returns:
            Style name with highest average utility
        """
        stats = self.get_performance_stats()
        
        best_style = 'neutral'
        best_utility = 0.0
        
        for style, stat in stats.items():
            if stat['avg_utility'] > best_utility:
                best_utility = stat['avg_utility']
                best_style = style
        
        return best_style


# ==================== Adaptive Style Controller ====================

class AdaptiveStyleController(StyleController):
    """
    Adaptive style controller
    
    Automatically switches styles based on performance
    """
    
    def __init__(
        self,
        initial_style: str = 'neutral',
        switch_frequency: int = 100,
        alpha_agg: float = 1.0,
        alpha_con: float = 0.2,
    ):
        """
        Args:
            initial_style: Initial style
            switch_frequency: Episodes between style switches
            alpha_agg: Variance bonus weight
            alpha_con: CVaR parameter
        """
        super().__init__(initial_style, alpha_agg, alpha_con)
        
        self.switch_frequency = switch_frequency
        self.episodes_since_switch = 0
        
        print(f"[AdaptiveStyleController] Switch frequency: {switch_frequency}")
    
    def step(self, utility: float):
        """
        Step and possibly switch style
        
        Args:
            utility: Current episode utility
        """
        # Update performance
        self.update_performance(self.current_style, utility)
        self.episodes_since_switch += 1
        
        # Check if should switch
        if self.episodes_since_switch >= self.switch_frequency:
            self._auto_switch()
            self.episodes_since_switch = 0
    
    def _auto_switch(self):
        """Automatically switch to best performing style"""
        best_style = self.get_best_style()
        
        if best_style != self.current_style:
            print(f"[AdaptiveStyleController] Auto-switching from "
                  f"{self.current_style} to {best_style}")
            self.set_style(best_style)


# ==================== Style Mixer ====================

class StyleMixer:
    """
    Mix multiple styles dynamically
    
    Combines Q-values from different styles
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            weights: Style weights (normalized automatically)
        """
        if weights is None:
            weights = {
                'neutral': 1.0,
                'aggressive': 0.0,
                'conservative': 0.0,
            }
        
        # Normalize weights
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()}
        
        self.controller = StyleController()
        
        print(f"[StyleMixer] Initialized with weights: {self.weights}")
    
    def compute_mixed_q(
        self,
        quantiles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted mixture of Q-values
        
        Args:
            quantiles: Quantile values (batch, num_quantiles)
            
        Returns:
            Mixed Q-values (batch,)
        """
        q_values = []
        
        for style, weight in self.weights.items():
            if weight > 0:
                q = self.controller.compute_q_value(quantiles, style)
                q_values.append(weight * q)
        
        # Weighted sum
        mixed_q = torch.stack(q_values, dim=0).sum(dim=0)
        
        return mixed_q
    
    def set_weights(self, weights: Dict[str, float]):
        """
        Update style weights
        
        Args:
            weights: New weights
        """
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()}
        print(f"[StyleMixer] Updated weights: {self.weights}")


# ==================== Style Analyzer ====================

class StyleAnalyzer:
    """
    Analyze and compare different styles
    """
    
    def __init__(self):
        self.controller = StyleController()
    
    def analyze_quantiles(
        self,
        quantiles: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Analyze quantile distribution for all styles
        
        Args:
            quantiles: Quantile values (batch, num_quantiles)
            
        Returns:
            Analysis results
        """
        results = {}
        
        for style in ['neutral', 'aggressive', 'conservative']:
            q_value = self.controller.compute_q_value(quantiles, style)
            
            results[style] = {
                'q_value': q_value.mean().item(),
                'q_std': q_value.std().item(),
                'q_min': q_value.min().item(),
                'q_max': q_value.max().item(),
            }
        
        return results
    
    def recommend_style(
        self,
        quantiles: torch.Tensor,
        risk_tolerance: float = 0.5,
    ) -> str:
        """
        Recommend style based on value distribution
        
        Args:
            quantiles: Quantile values
            risk_tolerance: Risk tolerance [0, 1]
                - 0: Very risk-averse (conservative)
                - 0.5: Balanced (neutral)
                - 1: Very risk-seeking (aggressive)
            
        Returns:
            Recommended style name
        """
        if risk_tolerance < 0.33:
            return 'conservative'
        elif risk_tolerance < 0.67:
            return 'neutral'
        else:
            return 'aggressive'


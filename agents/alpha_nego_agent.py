"""
α-Nego Agent - Main Negotiation Agent
Integrates policy network, critic network, and DSAC algorithm

Implements complete α-Nego framework:
- Self-play training (Algorithm 1)
- DSAC updates (Algorithm 2)
- Style-controllable negotiation (Eq. 12-14)
- KL regularization (Eq. 8)

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy


class AlphaNegoAgent:
    """
    Main α-Nego Agent
    
    Combines:
    - PolicyNetwork for action selection
    - DistributionalCritic for Q-value estimation
    - DSAC algorithm for training
    - Style control for different negotiation behaviors
    
    Training modes:
    - Warm-start: Initialize from supervised learning
    - Self-play: Train against opponent pool
    - Evaluation: Test performance
    """
    
    def __init__(
        self,
        policy_network: nn.Module,
        critic_network: nn.Module,
        config,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Args:
            policy_network: PolicyNetwork instance
            critic_network: DistributionalCritic instance
            config: Configuration object
            device: Device for computation
        """
        self.policy = policy_network
        self.critic = critic_network
        self.config = config
        self.device = device
        
        # Move to device
        self.policy.to(device)
        self.critic.to(device)
        
        # Training mode
        self.training_mode = True
        
        # Style
        self.style = config.style_control.active_style
        
        # Statistics
        self.stats = {
            'total_steps': 0,
            'episodes': 0,
            'total_reward': 0.0,
            'agreement_rate': 0.0,
            'avg_utility': 0.0,
        }
        
        print(f"[AlphaNegoAgent] Initialized on {device}")
        print(f"  Policy params: {sum(p.numel() for p in self.policy.parameters()):,}")
        print(f"  Critic params: {sum(p.numel() for p in self.critic.parameters()):,}")
        print(f"  Style: {self.style}")
    
    def set_style(self, style: str):
        """
        Set negotiation style
        
        Args:
            style: 'neutral', 'aggressive', or 'conservative'
        """
        assert style in ['neutral', 'aggressive', 'conservative']
        self.style = style
        print(f"[AlphaNegoAgent] Style changed to: {style}")
    
    def select_action(
        self,
        state: np.ndarray,
        history: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, Dict]:
        """
        Select action based on current state
        
        Args:
            state: Current state (state_dim,)
            history: Dialogue history (seq_len, state_dim)
            deterministic: Use deterministic policy (for evaluation)
            
        Returns:
            intent: Dialogue act (int)
            price: Price/proposal (float)
            info: Additional information
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if history is not None:
            history_tensor = torch.FloatTensor(history).unsqueeze(0).to(self.device)
        else:
            history_tensor = None
        
        # Select action
        with torch.no_grad():
            intent, price, info = self.policy.sample_action(
                state_tensor,
                history_tensor,
                deterministic=deterministic or not self.training_mode
            )
        
        # Convert to numpy
        intent = intent.cpu().item()
        price = price.cpu().item()
        
        return intent, price, info
    
    def evaluate_state_action(
        self,
        state: np.ndarray,
        intent: int,
        price: float,
    ) -> float:
        """
        Evaluate Q-value for state-action pair
        
        Args:
            state: State
            intent: Intent
            price: Price
            
        Returns:
            Q-value
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor([[intent, price]]).to(self.device)
        
        # Get Q-value
        with torch.no_grad():
            q_value, _ = self.critic(state_tensor, action_tensor, style=self.style)
        
        return q_value.cpu().item()
    
    def get_policy_distribution(
        self,
        state: np.ndarray,
        history: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get policy distribution for analysis
        
        Args:
            state: State
            history: Dialogue history
            
        Returns:
            Dictionary with intent probabilities and price distribution
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if history is not None:
            history_tensor = torch.FloatTensor(history).unsqueeze(0).to(self.device)
        else:
            history_tensor = None
        
        # Get distributions
        with torch.no_grad():
            intent_logits, price_mean, price_std = self.policy(state_tensor, history_tensor)
            intent_probs = F.softmax(intent_logits, dim=-1)
        
        return {
            'intent_probs': intent_probs.cpu().numpy()[0],
            'price_mean': price_mean.cpu().item(),
            'price_std': price_std.cpu().item(),
        }
    
    def train(self):
        """Set to training mode"""
        self.training_mode = True
        self.policy.train()
        self.critic.train()
    
    def eval(self):
        """Set to evaluation mode"""
        self.training_mode = False
        self.policy.eval()
        self.critic.eval()
    
    def update_stats(
        self,
        reward: float,
        agreement: bool,
        utility: float,
    ):
        """
        Update agent statistics
        
        Args:
            reward: Episode reward
            agreement: Whether agreement reached
            utility: Agent utility
        """
        self.stats['episodes'] += 1
        self.stats['total_reward'] += reward
        
        # Moving average for agreement rate and utility
        alpha = 0.01
        self.stats['agreement_rate'] = (
            (1 - alpha) * self.stats['agreement_rate'] + alpha * float(agreement)
        )
        self.stats['avg_utility'] = (
            (1 - alpha) * self.stats['avg_utility'] + alpha * utility
        )
    
    def get_stats(self) -> Dict[str, float]:
        """Get agent statistics"""
        return self.stats.copy()
    
    def save(self, path: str):
        """
        Save agent checkpoint
        
        Args:
            path: Save path
        """
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'stats': self.stats,
            'style': self.style,
        }
        torch.save(checkpoint, path)
        print(f"[AlphaNegoAgent] Saved to {path}")
    
    def load(self, path: str):
        """
        Load agent checkpoint
        
        Args:
            path: Load path
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.stats = checkpoint.get('stats', self.stats)
        self.style = checkpoint.get('style', self.style)
        print(f"[AlphaNegoAgent] Loaded from {path}")
    
    def clone(self) -> 'AlphaNegoAgent':
        """
        Create a copy of this agent
        
        Returns:
            Cloned agent
        """
        # Deep copy networks
        policy_copy = copy.deepcopy(self.policy)
        critic_copy = copy.deepcopy(self.critic)
        
        # Create new agent
        cloned_agent = AlphaNegoAgent(
            policy_network=policy_copy,
            critic_network=critic_copy,
            config=self.config,
            device=self.device,
        )
        
        cloned_agent.style = self.style
        cloned_agent.stats = self.stats.copy()
        
        return cloned_agent


# ==================== Agent Factory ====================

def create_alpha_nego_agent(
    config,
    supervised_agent: Optional[nn.Module] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> AlphaNegoAgent:
    """
    Factory function to create α-Nego agent
    
    Args:
        config: Configuration object
        supervised_agent: Optional SL agent for warm-start
        device: Device
        
    Returns:
        AlphaNegoAgent instance
    """
    from models.policy_network import create_policy_network
    from models.critic_network import create_distributional_critic
    
    # Create networks
    policy = create_policy_network(config)
    critic = create_distributional_critic(config)
    
    # Initialize policy from supervised agent if provided
    if supervised_agent is not None:
        print("[Factory] Initializing policy from supervised agent...")
        policy.load_state_dict(supervised_agent.state_dict(), strict=False)
    
    # Create agent
    agent = AlphaNegoAgent(
        policy_network=policy,
        critic_network=critic,
        config=config,
        device=device,
    )
    
    return agent


# ==================== Negotiation Interface ====================

class NegotiationInterface:
    """
    High-level interface for negotiation with α-Nego agent
    
    Handles:
    - State tracking
    - Action execution
    - Episode management
    """
    
    def __init__(
        self,
        agent: AlphaNegoAgent,
        max_turns: int = 20,
    ):
        """
        Args:
            agent: AlphaNegoAgent instance
            max_turns: Maximum dialogue turns
        """
        self.agent = agent
        self.max_turns = max_turns
        
        # Episode state
        self.reset()
    
    def reset(self):
        """Reset episode state"""
        self.current_turn = 0
        self.dialogue_history = []
        self.state_history = []
        self.action_history = []
        self.agreement = False
        self.final_price = None
    
    def step(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, bool, Dict]:
        """
        Take one negotiation step
        
        Args:
            state: Current state
            deterministic: Use deterministic policy
            
        Returns:
            intent: Dialogue act
            price: Price/proposal
            done: Whether episode ended
            info: Additional information
        """
        # Check if max turns reached
        if self.current_turn >= self.max_turns:
            return 15, 0.0, True, {'reason': 'max_turns'}  # Quit action
        
        # Get dialogue history
        if len(self.state_history) > 0:
            history = np.array(self.state_history[-10:])  # Last 10 turns
        else:
            history = None
        
        # Select action
        intent, price, info = self.agent.select_action(state, history, deterministic)
        
        # Update history
        self.current_turn += 1
        self.state_history.append(state)
        self.action_history.append((intent, price))
        
        # Check if negotiation ends
        done = False
        if intent == 13:  # accept
            done = True
            self.agreement = True
            info['reason'] = 'accept'
        elif intent == 14:  # reject
            done = True
            self.agreement = False
            info['reason'] = 'reject'
        elif intent == 15:  # quit
            done = True
            self.agreement = False
            info['reason'] = 'quit'
        
        return intent, price, done, info
    
    def get_episode_summary(self) -> Dict:
        """
        Get episode summary
        
        Returns:
            Dictionary with episode information
        """
        return {
            'num_turns': self.current_turn,
            'agreement': self.agreement,
            'final_price': self.final_price,
            'dialogue_history': self.dialogue_history,
            'state_history': self.state_history,
            'action_history': self.action_history,
        }


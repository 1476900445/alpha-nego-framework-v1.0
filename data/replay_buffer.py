"""
Experience Replay Buffer for Off-Policy RL
Implements standard replay buffer and prioritized experience replay

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import random


# ==================== Experience Structure ====================

class Experience:
    """
    Single experience tuple for RL
    (s, a, r, s', done)
    """
    
    def __init__(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[Dict] = None,
    ):
        """
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            info: Additional information
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.info = info or {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'state': self.state,
            'action': self.action,
            'reward': self.reward,
            'next_state': self.next_state,
            'done': self.done,
            'info': self.info,
        }


# ==================== Standard Replay Buffer ====================

class ReplayBuffer:
    """
    Standard experience replay buffer
    
    Used in paper Section 5.2:
    - Capacity: 1M transitions
    - Random sampling for off-policy learning
    - Stores (s, a, r, s', done) tuples
    """
    
    def __init__(
        self,
        capacity: int = 1000000,
        state_dim: int = 3,
        action_dim: int = 2,
    ):
        """
        Args:
            capacity: Maximum buffer size (paper uses 1M)
            state_dim: State dimension
            action_dim: Action dimension
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Preallocate memory for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        self.position = 0  # Current write position
        self.size = 0      # Current size
        
        print(f"[ReplayBuffer] Initialized with capacity {capacity:,}")
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add experience to buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Done flag
        """
        idx = self.position
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ):
        """
        Add batch of experiences
        
        Args:
            states: Batch of states (batch_size, state_dim)
            actions: Batch of actions (batch_size, action_dim)
            rewards: Batch of rewards (batch_size,)
            next_states: Batch of next states (batch_size, state_dim)
            dones: Batch of done flags (batch_size,)
        """
        batch_size = len(states)
        
        for i in range(batch_size):
            self.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
    
    def sample(
        self,
        batch_size: int,
        device: str = 'cpu',
    ) -> Dict[str, torch.Tensor]:
        """
        Sample random batch
        
        Args:
            batch_size: Number of samples (paper uses 32)
            device: Device for tensors
            
        Returns:
            Dictionary of tensors
        """
        if self.size < batch_size:
            batch_size = self.size
        
        # Random sampling
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # Extract batch
        batch = {
            'states': torch.FloatTensor(self.states[indices]).to(device),
            'actions': torch.FloatTensor(self.actions[indices]).to(device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(device),
            'dones': torch.BoolTensor(self.dones[indices]).to(device),
        }
        
        return batch
    
    def sample_recent(
        self,
        batch_size: int,
        recent_ratio: float = 0.5,
        device: str = 'cpu',
    ) -> Dict[str, torch.Tensor]:
        """
        Sample with bias towards recent experiences
        
        Args:
            batch_size: Number of samples
            recent_ratio: Ratio of samples from recent half
            device: Device for tensors
            
        Returns:
            Dictionary of tensors
        """
        if self.size < batch_size:
            return self.sample(batch_size, device)
        
        # Split into recent and old samples
        num_recent = int(batch_size * recent_ratio)
        num_old = batch_size - num_recent
        
        # Sample from recent half
        recent_start = max(0, self.size - self.size // 2)
        recent_indices = np.random.randint(recent_start, self.size, size=num_recent)
        
        # Sample from entire buffer
        old_indices = np.random.randint(0, self.size, size=num_old)
        
        indices = np.concatenate([recent_indices, old_indices])
        
        batch = {
            'states': torch.FloatTensor(self.states[indices]).to(device),
            'actions': torch.FloatTensor(self.actions[indices]).to(device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(device),
            'dones': torch.BoolTensor(self.dones[indices]).to(device),
        }
        
        return batch
    
    def add_episode(self, episode: List[Experience]):
        """
        Add entire episode to buffer
        
        Args:
            episode: List of experiences
        """
        for exp in episode:
            self.add(
                exp.state,
                exp.action,
                exp.reward,
                exp.next_state,
                exp.done,
            )
    
    def clear(self):
        """Clear buffer"""
        self.position = 0
        self.size = 0
    
    def __len__(self) -> int:
        """Get current size"""
        return self.size
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return self.size >= self.capacity
    
    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics"""
        if self.size == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'avg_reward': 0.0,
                'done_ratio': 0.0,
            }
        
        return {
            'size': self.size,
            'capacity': self.capacity,
            'utilization': self.size / self.capacity,
            'avg_reward': float(np.mean(self.rewards[:self.size])),
            'done_ratio': float(np.mean(self.dones[:self.size])),
        }


# ==================== Prioritized Experience Replay ====================

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER)
    
    Samples experiences based on TD error
    Reference: Schaul et al., 2016
    
    Optional enhancement for α-Nego (not in paper)
    """
    
    def __init__(
        self,
        capacity: int = 1000000,
        state_dim: int = 3,
        action_dim: int = 2,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        """
        Args:
            capacity: Buffer capacity
            state_dim: State dimension
            action_dim: Action dimension
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            beta: Importance sampling exponent
            beta_increment: Beta annealing rate
            epsilon: Small constant to prevent zero priority
        """
        super().__init__(capacity, state_dim, action_dim)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Priority tree (simple array implementation)
        self.priorities = np.ones(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
        print(f"[PrioritizedReplayBuffer] Initialized with α={alpha}, β={beta}")
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
    ):
        """
        Add experience with priority
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Done flag
            priority: Priority value (default: max priority)
        """
        idx = self.position
        
        # Add to buffer
        super().add(state, action, reward, next_state, done)
        
        # Set priority
        if priority is None:
            priority = self.max_priority
        
        self.priorities[idx] = priority
        self.max_priority = max(self.max_priority, priority)
    
    def sample(
        self,
        batch_size: int,
        device: str = 'cpu',
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        Sample batch with prioritized sampling
        
        Args:
            batch_size: Batch size
            device: Device
            
        Returns:
            (batch, indices, weights)
        """
        if self.size < batch_size:
            batch_size = self.size
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Extract batch
        batch = {
            'states': torch.FloatTensor(self.states[indices]).to(device),
            'actions': torch.FloatTensor(self.actions[indices]).to(device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(device),
            'dones': torch.BoolTensor(self.dones[indices]).to(device),
            'weights': torch.FloatTensor(weights).to(device),
        }
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled experiences
        
        Args:
            indices: Indices of experiences
            priorities: New priority values (TD errors)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
            self.max_priority = max(self.max_priority, self.priorities[idx])


# ==================== Episode Buffer ====================

class EpisodeBuffer:
    """
    Buffer for storing complete episodes
    Useful for on-policy algorithms and trajectory analysis
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Args:
            capacity: Maximum number of episodes
        """
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)
        
        print(f"[EpisodeBuffer] Initialized with capacity {capacity}")
    
    def add_episode(self, episode: List[Experience]):
        """
        Add episode
        
        Args:
            episode: List of experiences
        """
        self.episodes.append(episode)
    
    def sample_episode(self) -> List[Experience]:
        """Sample random episode"""
        return random.choice(self.episodes)
    
    def sample_episodes(self, num_episodes: int) -> List[List[Experience]]:
        """Sample multiple episodes"""
        return random.sample(self.episodes, min(num_episodes, len(self.episodes)))
    
    def get_all_transitions(self) -> List[Experience]:
        """Get all transitions from all episodes"""
        transitions = []
        for episode in self.episodes:
            transitions.extend(episode)
        return transitions
    
    def clear(self):
        """Clear buffer"""
        self.episodes.clear()
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def get_stats(self) -> Dict[str, float]:
        """Get episode statistics"""
        if len(self.episodes) == 0:
            return {
                'num_episodes': 0,
                'avg_length': 0.0,
                'avg_return': 0.0,
            }
        
        lengths = [len(ep) for ep in self.episodes]
        returns = [sum(exp.reward for exp in ep) for ep in self.episodes]
        
        return {
            'num_episodes': len(self.episodes),
            'avg_length': np.mean(lengths),
            'avg_return': np.mean(returns),
            'max_return': np.max(returns),
            'min_return': np.min(returns),
        }


# ==================== Multi-Agent Buffer ====================

class MultiAgentReplayBuffer:
    """
    Replay buffer for multi-agent scenarios
    Stores experiences from multiple agents separately
    """
    
    def __init__(
        self,
        num_agents: int,
        capacity: int = 1000000,
        state_dim: int = 3,
        action_dim: int = 2,
    ):
        """
        Args:
            num_agents: Number of agents
            capacity: Capacity per agent
            state_dim: State dimension
            action_dim: Action dimension
        """
        self.num_agents = num_agents
        
        # Create separate buffer for each agent
        self.buffers = [
            ReplayBuffer(capacity, state_dim, action_dim)
            for _ in range(num_agents)
        ]
        
        print(f"[MultiAgentReplayBuffer] Initialized for {num_agents} agents")
    
    def add(
        self,
        agent_id: int,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience for specific agent"""
        self.buffers[agent_id].add(state, action, reward, next_state, done)
    
    def sample(
        self,
        agent_id: int,
        batch_size: int,
        device: str = 'cpu',
    ) -> Dict[str, torch.Tensor]:
        """Sample from specific agent's buffer"""
        return self.buffers[agent_id].sample(batch_size, device)
    
    def sample_all(
        self,
        batch_size: int,
        device: str = 'cpu',
    ) -> List[Dict[str, torch.Tensor]]:
        """Sample from all agents"""
        return [
            self.buffers[i].sample(batch_size, device)
            for i in range(self.num_agents)
        ]
    
    def clear_all(self):
        """Clear all buffers"""
        for buffer in self.buffers:
            buffer.clear()
    
    def get_stats(self) -> Dict[int, Dict]:
        """Get stats for all agents"""
        return {
            i: buffer.get_stats()
            for i, buffer in enumerate(self.buffers)
        }


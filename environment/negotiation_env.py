"""
Negotiation Environment
Gym-compatible environment for negotiation tasks

Features:
- Compatible with Gym interface
- Support for Craigslistbargain and Dealornodeal
- Opponent integration
- Reward shaping
- State tracking

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import gym
from gym import spaces

from dialogue.dialogue_acts import get_num_intents, is_terminal
from dialogue.manager import DialogueManager
from environment.state_representation import StateComponents, StateEncoder
from environment.reward_functions import RewardShaper


# ==================== Negotiation Environment ====================

class NegotiationEnv(gym.Env):
    """
    Negotiation environment with Gym interface
    
    Observation space:
    - Craigslistbargain: [current_price, opponent_last_price, turn_ratio]
    - Dealornodeal: [agent_util, opponent_util, turn_ratio, item_features...]
    
    Action space:
    - Discrete: intent_id
    - Continuous: price (for Craigslistbargain) or item allocation (for Dealornodeal)
    - Combined: MultiDiscrete or Dict space
    """
    
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(
        self,
        dataset: str = 'craigslistbargain',
        opponent: Optional[Any] = None,
        max_turns: int = 20,
        listing_price_range: Tuple[float, float] = (50.0, 200.0),
        agent_target_range: Tuple[float, float] = (0.6, 0.8),
        opponent_target_range: Tuple[float, float] = (0.8, 1.0),
        reward_shaper: Optional[RewardShaper] = None,
        state_encoder: Optional[StateEncoder] = None,
    ):
        """
        Args:
            dataset: 'craigslistbargain' or 'dealornodeal'
            opponent: Opponent agent
            max_turns: Maximum dialogue turns
            listing_price_range: Range for listing price
            agent_target_range: Range for agent target (fraction of listing)
            opponent_target_range: Range for opponent target (fraction of listing)
            reward_shaper: Reward shaper
            state_encoder: State encoder
        """
        super().__init__()
        
        self.dataset = dataset
        self.opponent = opponent
        self.max_turns = max_turns
        self.listing_price_range = listing_price_range
        self.agent_target_range = agent_target_range
        self.opponent_target_range = opponent_target_range
        
        # Dialogue manager
        self.manager = None
        
        # Reward shaper
        self.reward_shaper = reward_shaper or RewardShaper()
        
        # State encoder
        self.state_encoder = state_encoder or StateEncoder(dataset)
        
        # Define action and observation spaces
        self._define_spaces()
        
        # Episode tracking
        self.current_episode = 0
        self.episode_rewards = []
        
        print(f"[NegotiationEnv] Initialized for {dataset}")
        print(f"  Max turns: {max_turns}")
        print(f"  Action space: {self.action_space}")
        print(f"  Observation space: {self.observation_space}")
    
    def _define_spaces(self):
        """Define action and observation spaces"""
        num_intents = get_num_intents(self.dataset)
        
        if self.dataset == 'craigslistbargain':
            # Action: [intent_id, price]
            self.action_space = spaces.Dict({
                'intent': spaces.Discrete(num_intents),
                'price': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            })
            
            # Observation: [current_price, opponent_price, turn_ratio]
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32
            )
        
        elif self.dataset == 'dealornodeal':
            # Action: [intent_id, item1, item2, item3]
            self.action_space = spaces.Dict({
                'intent': spaces.Discrete(num_intents),
                'items': spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
            })
            
            # Observation: [agent_util, opponent_util, turn_ratio, item_features]
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(12,),  # 3 + 3*3
                dtype=np.float32
            )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Reset environment for new episode
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Initial observation
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Sample scenario parameters
        if self.dataset == 'craigslistbargain':
            listing_price = np.random.uniform(*self.listing_price_range)
            agent_target = listing_price * np.random.uniform(*self.agent_target_range)
            opponent_target = listing_price * np.random.uniform(*self.opponent_target_range)
            
            # Create dialogue manager
            self.manager = DialogueManager(
                dataset=self.dataset,
                max_turns=self.max_turns,
                listing_price=listing_price,
                agent_target=agent_target,
                opponent_target=opponent_target,
            )
        
        elif self.dataset == 'dealornodeal':
            # Sample item values
            agent_values = {
                'hats': np.random.randint(0, 4),
                'books': np.random.randint(0, 4),
                'balls': np.random.randint(0, 4),
            }
            opponent_values = {
                'hats': np.random.randint(0, 4),
                'books': np.random.randint(0, 4),
                'balls': np.random.randint(0, 4),
            }
            
            self.manager = DialogueManager(
                dataset=self.dataset,
                max_turns=self.max_turns,
                agent_values=agent_values,
                opponent_values=opponent_values,
            )
        
        # Reset opponent
        if self.opponent and hasattr(self.opponent, 'reset'):
            self.opponent.reset()
        
        # Get initial observation
        obs = self.manager.get_state()
        
        self.current_episode += 1
        
        return obs
    
    def step(
        self,
        action: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step
        
        Args:
            action: Action dict with 'intent' and 'price'/'items'
            
        Returns:
            (observation, reward, done, info)
        """
        # Extract action
        intent_id = int(action['intent'])
        
        if self.dataset == 'craigslistbargain':
            price = float(action['price'][0]) * self.manager.state.listing_price
            items = None
        else:
            price = None
            items_array = action['items']
            items = {
                'hats': int(items_array[0] * 3),
                'books': int(items_array[1] * 3),
                'balls': int(items_array[2] * 3),
            }
        
        # Agent's action
        info_agent, done_agent = self.manager.step(
            speaker='agent',
            intent_id=intent_id,
            price=price,
            items=items,
        )
        
        # Check if done
        if done_agent:
            done = True
        else:
            # Opponent's turn
            if self.opponent:
                opponent_action = self._get_opponent_action()
                info_opp, done_opp = self.manager.step(
                    speaker='opponent',
                    **opponent_action
                )
                done = done_opp
            else:
                done = False
        
        # Get observation
        obs = self.manager.get_state()
        
        # Compute reward
        reward = self._compute_reward(done)
        
        # Create info
        info = {
            'agreement': self.manager.state.agreement,
            'final_price': self.manager.state.final_price,
            'dialogue_length': self.manager.state.current_turn,
            'agent_action': info_agent['turn'],
        }
        
        # Add utilities if done
        if done:
            agent_util, opponent_util = self.manager.compute_utilities()
            info['agent_utility'] = agent_util
            info['opponent_utility'] = opponent_util
            info['social_welfare'] = agent_util + opponent_util
        
        return obs, reward, done, info
    
    def _get_opponent_action(self) -> Dict:
        """
        Get opponent's action
        
        Returns:
            Action dictionary
        """
        state = self.manager.get_state()
        
        if hasattr(self.opponent, 'select_action'):
            # RL agent opponent
            intent, price, _ = self.opponent.select_action(state)
            
            if self.dataset == 'craigslistbargain':
                return {
                    'intent_id': intent,
                    'price': price * self.manager.state.listing_price if price else None,
                }
            else:
                return {
                    'intent_id': intent,
                    'items': None,  # Would need to convert
                }
        else:
            # Rule-based opponent
            intent, price = self.opponent.select_action(state)
            
            if self.dataset == 'craigslistbargain':
                return {
                    'intent_id': intent,
                    'price': price,
                }
            else:
                return {
                    'intent_id': intent,
                    'items': None,
                }
    
    def _compute_reward(self, done: bool) -> float:
        """
        Compute reward
        
        Args:
            done: Whether episode ended
            
        Returns:
            Reward value
        """
        if not done:
            # Step reward
            return self.reward_shaper.compute_step_reward(
                is_terminal=False,
                agreement=False,
                agent_utility=0.0,
                opponent_utility=0.0,
                current_turn=self.manager.state.current_turn,
                max_turns=self.max_turns,
            )
        else:
            # Terminal reward
            agent_util, opponent_util = self.manager.compute_utilities()
            
            return self.reward_shaper.compute_reward(
                agreement=self.manager.state.agreement,
                agent_utility=agent_util,
                opponent_utility=opponent_util,
                dialogue_length=self.manager.state.current_turn,
                max_length=self.max_turns,
            )
    
    def render(self, mode: str = 'human'):
        """
        Render environment
        
        Args:
            mode: Render mode
        """
        if mode == 'human' or mode == 'ansi':
            print(f"\n=== Negotiation Turn {self.manager.state.current_turn} ===")
            
            # Print recent turns
            history = self.manager.get_history(max_length=5)
            for turn in history:
                speaker = turn['speaker']
                utterance = turn['utterance']
                print(f"{speaker}: {utterance}")
            
            # Print summary
            summary = self.manager.get_summary()
            print(f"\nAgreement: {summary['agreement']}")
            if summary['final_price']:
                print(f"Final price: ${summary['final_price']:.2f}")
    
    def close(self):
        """Close environment"""
        pass
    
    def seed(self, seed: Optional[int] = None):
        """
        Set random seed
        
        Args:
            seed: Random seed
        """
        np.random.seed(seed)


# ==================== Environment Factory ====================

def make_negotiation_env(
    dataset: str = 'craigslistbargain',
    opponent_type: str = 'rule_based',
    **kwargs
) -> NegotiationEnv:
    """
    Factory function to create negotiation environment
    
    Args:
        dataset: 'craigslistbargain' or 'dealornodeal'
        opponent_type: 'rule_based', 'rl', or 'sl'
        **kwargs: Additional arguments
        
    Returns:
        NegotiationEnv instance
    """
    # Create opponent
    opponent = None
    
    if opponent_type == 'rule_based':
        from agents.baseline_agents import create_baseline_agent
        opponent = create_baseline_agent('linear', role='seller')
    elif opponent_type == 'rl':
        # Would load trained RL opponent
        pass
    elif opponent_type == 'sl':
        # Would load supervised learning opponent
        pass
    
    # Create environment
    env = NegotiationEnv(
        dataset=dataset,
        opponent=opponent,
        **kwargs
    )
    
    return env


# ==================== Vectorized Environment ====================

class VectorizedNegotiationEnv:
    """
    Vectorized environment for parallel training
    """
    
    def __init__(
        self,
        num_envs: int,
        dataset: str = 'craigslistbargain',
        **kwargs
    ):
        """
        Args:
            num_envs: Number of parallel environments
            dataset: Dataset name
            **kwargs: Arguments for each environment
        """
        self.num_envs = num_envs
        self.envs = [
            NegotiationEnv(dataset=dataset, **kwargs)
            for _ in range(num_envs)
        ]
    
    def reset(self) -> np.ndarray:
        """
        Reset all environments
        
        Returns:
            Stacked observations (num_envs, obs_dim)
        """
        obs_list = [env.reset() for env in self.envs]
        return np.stack(obs_list, axis=0)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments
        
        Args:
            actions: Actions for all environments
            
        Returns:
            (observations, rewards, dones, infos)
        """
        results = [
            env.step(action)
            for env, action in zip(self.envs, actions)
        ]
        
        obs_list, rewards, dones, infos = zip(*results)
        
        return (
            np.stack(obs_list, axis=0),
            np.array(rewards),
            np.array(dones),
            list(infos)
        )


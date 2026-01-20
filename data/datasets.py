"""
Dataset loaders for Craigslistbargain and Dealornodeal
Implements data loading, parsing, and batch generation

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


# ==================== Data Structures ====================

@dataclass
class CraigslistSample:
    """
    Single sample from Craigslistbargain dataset
    
    Reference: He et al., 2018
    """
    # Dialogue info
    dialogue_id: str
    scenario_id: str
    
    # Agent info
    agent_role: str  # 'buyer' or 'seller'
    agent_target: float  # Target price
    
    # Opponent info
    opponent_role: str
    opponent_target: float
    
    # Product info
    product_title: str
    product_description: str
    listing_price: float
    
    # Dialogue turns
    turns: List[Dict[str, any]]
    
    # Outcome
    agreement: bool
    final_price: Optional[float]
    
    # Metadata
    num_turns: int
    agent_utility: float
    opponent_utility: float


@dataclass
class DealornodealSample:
    """
    Single sample from Dealornodeal dataset
    
    Reference: Lewis et al., 2017
    """
    # Dialogue info
    dialogue_id: str
    scenario_id: str
    
    # Agent info
    agent_name: str
    agent_values: Dict[str, int]
    agent_counts: Dict[str, int]
    
    # Opponent info
    opponent_name: str
    opponent_values: Dict[str, int]
    opponent_counts: Dict[str, int]
    
    # Dialogue turns
    turns: List[Dict[str, any]]
    
    # Outcome
    agreement: bool
    final_allocation: Optional[Dict[str, Dict[str, int]]]
    
    # Metadata
    num_turns: int
    agent_utility: float
    opponent_utility: float
    social_welfare: float


# ==================== Craigslistbargain Dataset ====================

class CraigslistbargainDataset(Dataset):
    """
    Craigslistbargain dataset for price negotiation
    
    Dataset structure:
    - dialogues/
      - train.json
      - val.json
      - test.json
    
    Each dialogue contains:
    - Scenario: product info, target prices
    - Dialogue: turns with utterances and actions
    - Outcome: agreement, final price
    
    Reference: He et al., 2018
    """
    
    INTENT_TO_ID = {
        'greet': 0, 'inquire': 1, 'inform': 2,
        'init-price': 3, 'insist-price': 4, 'agree-price': 5,
        'concede-price': 6, 'final-price': 7, 'counter-no-price': 8,
        'hesitant': 9, 'positive': 10, 'negative': 11,
        'offer': 12, 'accept': 13, 'reject': 14, 'quit': 15,
    }
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        max_dialogue_length: int = 20,
        normalize_prices: bool = True,
    ):
        """
        Args:
            data_path: Path to dataset directory
            split: 'train', 'val', or 'test'
            max_dialogue_length: Maximum number of turns
            normalize_prices: Whether to normalize prices to [0, 1]
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_dialogue_length = max_dialogue_length
        self.normalize_prices = normalize_prices
        
        # Load data
        self.dialogues = self._load_dialogues()
        
        print(f"[Craigslistbargain] Loaded {len(self.dialogues)} dialogues from {split} split")
    
    def _load_dialogues(self) -> List[CraigslistSample]:
        """Load dialogues from JSON file"""
        file_path = self.data_path / f'{self.split}.json'
        
        if not file_path.exists():
            print(f"Warning: {file_path} not found. Using dummy data for demonstration.")
            return self._create_dummy_data()
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        dialogues = []
        for item in data:
            sample = self._parse_dialogue(item)
            if sample is not None:
                dialogues.append(sample)
        
        return dialogues
    
    def _parse_dialogue(self, item: Dict) -> Optional[CraigslistSample]:
        """Parse single dialogue from JSON"""
        try:
            scenario = item.get('scenario', {})
            
            # Parse turns
            turns = []
            for turn in item.get('dialogue', []):
                turns.append({
                    'agent': turn.get('agent'),
                    'utterance': turn.get('utterance', ''),
                    'intent': turn.get('intent', 'inform'),
                    'price': turn.get('price', None),
                })
            
            # Parse outcome
            outcome = item.get('outcome', {})
            agreement = outcome.get('agreement', False)
            final_price = outcome.get('price', None) if agreement else None
            
            # Calculate utilities
            agent_target = scenario.get('agent_target', 0.0)
            opponent_target = scenario.get('opponent_target', 0.0)
            
            if agreement and final_price is not None:
                agent_role = scenario.get('agent_role', 'buyer')
                if agent_role == 'buyer':
                    agent_utility = (opponent_target - final_price) / max(opponent_target - agent_target, 1e-6)
                else:
                    agent_utility = (final_price - opponent_target) / max(agent_target - opponent_target, 1e-6)
                agent_utility = np.clip(agent_utility, 0.0, 1.0)
                opponent_utility = 1.0 - agent_utility
            else:
                agent_utility = 0.0
                opponent_utility = 0.0
            
            sample = CraigslistSample(
                dialogue_id=item.get('dialogue_id', ''),
                scenario_id=scenario.get('scenario_id', ''),
                agent_role=scenario.get('agent_role', 'buyer'),
                agent_target=agent_target,
                opponent_role=scenario.get('opponent_role', 'seller'),
                opponent_target=opponent_target,
                product_title=scenario.get('product_title', ''),
                product_description=scenario.get('product_description', ''),
                listing_price=scenario.get('listing_price', 0.0),
                turns=turns,
                agreement=agreement,
                final_price=final_price,
                num_turns=len(turns),
                agent_utility=agent_utility,
                opponent_utility=opponent_utility,
            )
            
            return sample
            
        except Exception as e:
            print(f"Error parsing dialogue: {e}")
            return None
    
    def _create_dummy_data(self) -> List[CraigslistSample]:
        """Create dummy data for demonstration"""
        print("Creating dummy Craigslistbargain data...")
        
        dialogues = []
        num_samples = 100 if self.split == 'train' else 20
        
        for i in range(num_samples):
            listing_price = np.random.uniform(50, 500)
            agent_target = listing_price * np.random.uniform(0.5, 0.8)
            opponent_target = listing_price * np.random.uniform(0.8, 1.0)
            
            num_turns = np.random.randint(3, 15)
            turns = []
            for j in range(num_turns):
                turns.append({
                    'agent': 'buyer' if j % 2 == 0 else 'seller',
                    'utterance': f'Turn {j} utterance',
                    'intent': 'init-price' if j == 0 else 'concede-price',
                    'price': np.random.uniform(agent_target, opponent_target),
                })
            
            agreement = np.random.random() > 0.3
            final_price = np.random.uniform(agent_target, opponent_target) if agreement else None
            agent_utility = (opponent_target - final_price) / (opponent_target - agent_target) if agreement else 0.0
            
            dialogues.append(CraigslistSample(
                dialogue_id=f'dummy_{i}',
                scenario_id=f'scenario_{i}',
                agent_role='buyer',
                agent_target=agent_target,
                opponent_role='seller',
                opponent_target=opponent_target,
                product_title=f'Product {i}',
                product_description=f'Description {i}',
                listing_price=listing_price,
                turns=turns,
                agreement=agreement,
                final_price=final_price,
                num_turns=num_turns,
                agent_utility=agent_utility,
                opponent_utility=1.0 - agent_utility if agreement else 0.0,
            ))
        
        return dialogues
    
    def __len__(self) -> int:
        return len(self.dialogues)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single sample
        
        Returns:
            Dictionary containing:
            - states: (num_turns, 3)
            - actions: (num_turns, 2)
            - rewards: (num_turns,)
            - dones: (num_turns,)
            - info: metadata
        """
        dialogue = self.dialogues[idx]
        
        states, actions, rewards, dones = [], [], [], []
        
        for t, turn in enumerate(dialogue.turns):
            # State: [current_price, opponent_last_price, turn_number]
            if t == 0:
                current_price = dialogue.listing_price
                opponent_last_price = dialogue.listing_price
            else:
                current_price = dialogue.turns[t-1].get('price', dialogue.listing_price)
                opponent_last_price = current_price
            
            if self.normalize_prices and dialogue.listing_price > 0:
                current_price /= dialogue.listing_price
                opponent_last_price /= dialogue.listing_price
            
            state = np.array([current_price, opponent_last_price, t / self.max_dialogue_length], dtype=np.float32)
            
            # Action: [intent_id, price]
            intent_id = self.INTENT_TO_ID.get(turn['intent'], 2)
            price = turn.get('price', dialogue.listing_price)
            if self.normalize_prices and dialogue.listing_price > 0:
                price /= dialogue.listing_price
            
            action = np.array([intent_id, price], dtype=np.float32)
            
            # Reward
            if t == len(dialogue.turns) - 1:
                reward = dialogue.agent_utility if dialogue.agreement else -0.5
            else:
                reward = -0.01
            
            done = (t == len(dialogue.turns) - 1)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
        # Pad
        states = self._pad_sequence(states, self.max_dialogue_length)
        actions = self._pad_sequence(actions, self.max_dialogue_length)
        rewards = self._pad_sequence(rewards, self.max_dialogue_length, pad_value=0.0)
        dones = self._pad_sequence(dones, self.max_dialogue_length, pad_value=True)
        
        return {
            'states': torch.FloatTensor(states),
            'actions': torch.FloatTensor(actions),
            'rewards': torch.FloatTensor(rewards),
            'dones': torch.BoolTensor(dones),
            'dialogue_length': len(dialogue.turns),
            'agreement': dialogue.agreement,
            'agent_utility': dialogue.agent_utility,
        }
    
    def _pad_sequence(self, seq: List, max_len: int, pad_value: any = 0) -> np.ndarray:
        """Pad sequence to max length"""
        if isinstance(seq[0], np.ndarray):
            pad_shape = (max_len, *seq[0].shape)
            padded = np.full(pad_shape, pad_value, dtype=seq[0].dtype)
            for i, item in enumerate(seq):
                padded[i] = item
        else:
            padded = np.full(max_len, pad_value)
            for i, item in enumerate(seq):
                padded[i] = item
        return padded


# ==================== Dealornodeal Dataset ====================

class DealornodealDataset(Dataset):
    """Dealornodeal dataset for multi-issue negotiation"""
    
    INTENT_TO_ID = {
        'greet': 0, 'disagree': 1, 'agree': 2,
        'insist': 3, 'inquire': 4, 'propose': 5,
    }
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        max_dialogue_length: int = 20,
        items: List[str] = ['hats', 'books', 'balls'],
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.max_dialogue_length = max_dialogue_length
        self.items = items
        
        self.dialogues = self._load_dialogues()
        print(f"[Dealornodeal] Loaded {len(self.dialogues)} dialogues from {split} split")
    
    def _load_dialogues(self) -> List[DealornodealSample]:
        """Load dialogues"""
        file_path = self.data_path / f'{self.split}.json'
        
        if not file_path.exists():
            print(f"Warning: {file_path} not found. Using dummy data.")
            return self._create_dummy_data()
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return [self._parse_dialogue(item) for item in data if self._parse_dialogue(item)]
    
    def _parse_dialogue(self, item: Dict) -> Optional[DealornodealSample]:
        """Parse dialogue"""
        try:
            scenario = item.get('scenario', {})
            turns = [{'agent': t.get('agent'), 'utterance': t.get('utterance', ''),
                     'intent': t.get('intent', 'inform'), 'proposal': t.get('proposal', None)}
                    for t in item.get('dialogue', [])]
            
            outcome = item.get('outcome', {})
            agreement = outcome.get('agreement', False)
            final_allocation = outcome.get('allocation', None) if agreement else None
            
            agent_values = scenario.get('agent_values', {})
            opponent_values = scenario.get('opponent_values', {})
            
            if agreement and final_allocation:
                agent_alloc = final_allocation.get('agent', {})
                opponent_alloc = final_allocation.get('opponent', {})
                
                agent_utility = sum(agent_values.get(i, 0) * agent_alloc.get(i, 0) for i in self.items)
                opponent_utility = sum(opponent_values.get(i, 0) * opponent_alloc.get(i, 0) for i in self.items)
                
                max_agent = sum(agent_values.get(i, 0) * 3 for i in self.items)
                max_opponent = sum(opponent_values.get(i, 0) * 3 for i in self.items)
                
                agent_utility /= max(max_agent, 1)
                opponent_utility /= max(max_opponent, 1)
                social_welfare = agent_utility + opponent_utility
            else:
                agent_utility = opponent_utility = social_welfare = 0.0
            
            return DealornodealSample(
                dialogue_id=item.get('dialogue_id', ''),
                scenario_id=scenario.get('scenario_id', ''),
                agent_name=scenario.get('agent_name', 'agent'),
                agent_values=agent_values,
                agent_counts=scenario.get('counts', {}),
                opponent_name=scenario.get('opponent_name', 'opponent'),
                opponent_values=opponent_values,
                opponent_counts=scenario.get('counts', {}),
                turns=turns,
                agreement=agreement,
                final_allocation=final_allocation,
                num_turns=len(turns),
                agent_utility=agent_utility,
                opponent_utility=opponent_utility,
                social_welfare=social_welfare,
            )
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def _create_dummy_data(self) -> List[DealornodealSample]:
        """Create dummy data"""
        dialogues = []
        num_samples = 100 if self.split == 'train' else 20
        
        for i in range(num_samples):
            agent_values = {item: np.random.randint(0, 4) for item in self.items}
            opponent_values = {item: np.random.randint(0, 4) for item in self.items}
            counts = {item: 3 for item in self.items}
            
            num_turns = np.random.randint(3, 12)
            turns = [{'agent': 'agent' if j % 2 == 0 else 'opponent',
                     'utterance': f'Turn {j}', 'intent': 'propose',
                     'proposal': {item: np.random.randint(0, 4) for item in self.items}}
                    for j in range(num_turns)]
            
            agreement = np.random.random() > 0.3
            final_allocation = {'agent': {i: np.random.randint(0, 4) for i in self.items},
                               'opponent': {i: np.random.randint(0, 4) for i in self.items}} if agreement else None
            
            agent_utility = np.random.uniform(0.3, 0.8) if agreement else 0.0
            opponent_utility = np.random.uniform(0.3, 0.8) if agreement else 0.0
            
            dialogues.append(DealornodealSample(
                dialogue_id=f'dummy_{i}', scenario_id=f'scenario_{i}',
                agent_name='agent', agent_values=agent_values, agent_counts=counts,
                opponent_name='opponent', opponent_values=opponent_values, opponent_counts=counts,
                turns=turns, agreement=agreement, final_allocation=final_allocation,
                num_turns=num_turns, agent_utility=agent_utility,
                opponent_utility=opponent_utility, social_welfare=agent_utility+opponent_utility
            ))
        
        return dialogues
    
    def __len__(self) -> int:
        return len(self.dialogues)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample"""
        dialogue = self.dialogues[idx]
        
        states, actions, rewards, dones = [], [], [], []
        
        for t, turn in enumerate(dialogue.turns):
            state = np.array([0.5, 0.5, t / self.max_dialogue_length], dtype=np.float32)
            
            intent_id = self.INTENT_TO_ID.get(turn['intent'], 0)
            proposal = turn.get('proposal', {})
            item_alloc = [proposal.get(i, 0) / 3.0 for i in self.items]
            action = np.array([intent_id] + item_alloc, dtype=np.float32)
            
            reward = (dialogue.agent_utility if dialogue.agreement else -0.5) if t == len(dialogue.turns)-1 else -0.01
            done = (t == len(dialogue.turns) - 1)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
        states = self._pad_sequence(states, self.max_dialogue_length)
        actions = self._pad_sequence(actions, self.max_dialogue_length)
        rewards = self._pad_sequence(rewards, self.max_dialogue_length, 0.0)
        dones = self._pad_sequence(dones, self.max_dialogue_length, True)
        
        return {
            'states': torch.FloatTensor(states),
            'actions': torch.FloatTensor(actions),
            'rewards': torch.FloatTensor(rewards),
            'dones': torch.BoolTensor(dones),
            'dialogue_length': len(dialogue.turns),
            'agreement': dialogue.agreement,
            'agent_utility': dialogue.agent_utility,
            'social_welfare': dialogue.social_welfare,
        }
    
    def _pad_sequence(self, seq: List, max_len: int, pad_value: any = 0) -> np.ndarray:
        if isinstance(seq[0], np.ndarray):
            padded = np.full((max_len, *seq[0].shape), pad_value, dtype=seq[0].dtype)
            for i, item in enumerate(seq): padded[i] = item
        else:
            padded = np.full(max_len, pad_value)
            for i, item in enumerate(seq): padded[i] = item
        return padded


# ==================== Helper Functions ====================

def create_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    """Create PyTorch DataLoader"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def get_dataset_stats(dataset: Dataset) -> Dict[str, any]:
    """Calculate dataset statistics"""
    stats = {'num_samples': len(dataset), 'num_agreements': 0, 'avg_dialogue_length': 0.0,
             'avg_utility': 0.0, 'agreement_rate': 0.0}
    
    total_length = total_utility = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample['agreement']: stats['num_agreements'] += 1
        total_length += sample['dialogue_length']
        total_utility += sample['agent_utility']
    
    stats['avg_dialogue_length'] = total_length / len(dataset)
    stats['avg_utility'] = total_utility / len(dataset)
    stats['agreement_rate'] = stats['num_agreements'] / len(dataset)
    return stats


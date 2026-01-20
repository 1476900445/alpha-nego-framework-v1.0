"""
Checkpoint Management
Advanced checkpoint saving, loading, and management

Features:
- Automatic checkpointing
- Best model tracking
- Checkpoint rotation
- Resume training
- Model versioning

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import torch
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


# ==================== Checkpoint Manager ====================

class CheckpointManager:
    """
    Manage model checkpoints
    """
    
    def __init__(
        self,
        save_dir: str,
        max_checkpoints: int = 5,
        save_best: bool = True,
        monitor: str = 'val_loss',
        mode: str = 'min',
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best: Save best model separately
            monitor: Metric to monitor for best model
            mode: 'min' or 'max'
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        
        # Track checkpoints
        self.checkpoints = []
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_checkpoint = None
        
        # Metadata file
        self.metadata_file = self.save_dir / 'checkpoints_metadata.json'
        self._load_metadata()
        
        print(f"[CheckpointManager] Initialized")
        print(f"  Save directory: {save_dir}")
        print(f"  Max checkpoints: {max_checkpoints}")
        print(f"  Monitor: {monitor} ({mode})")
    
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save checkpoint
        
        Args:
            state: State dictionary to save
            epoch: Epoch number
            metrics: Metrics dictionary
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'state': state,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save regular checkpoint
        checkpoint_name = f'checkpoint_epoch_{epoch}.pt'
        checkpoint_path = self.save_dir / checkpoint_name
        
        torch.save(checkpoint, checkpoint_path)
        
        # Add to list
        self.checkpoints.append({
            'path': str(checkpoint_path),
            'epoch': epoch,
            'metrics': metrics,
        })
        
        # Save best if needed
        if is_best and self.save_best:
            best_path = self.save_dir / 'best_model.pt'
            shutil.copy(checkpoint_path, best_path)
            self.best_checkpoint = str(best_path)
            
            if metrics:
                self.best_value = metrics.get(self.monitor, self.best_value)
        
        # Rotate checkpoints
        self._rotate_checkpoints()
        
        # Save metadata
        self._save_metadata()
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"[CheckpointManager] Loaded checkpoint from: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Timestamp: {checkpoint.get('timestamp', 'unknown')}")
        
        return checkpoint
    
    def load_best(self) -> Optional[Dict[str, Any]]:
        """
        Load best checkpoint
        
        Returns:
            Best checkpoint or None
        """
        best_path = self.save_dir / 'best_model.pt'
        
        if best_path.exists():
            return self.load_checkpoint(str(best_path))
        
        return None
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """
        Load latest checkpoint
        
        Returns:
            Latest checkpoint or None
        """
        if len(self.checkpoints) == 0:
            return None
        
        latest = max(self.checkpoints, key=lambda x: x['epoch'])
        return self.load_checkpoint(latest['path'])
    
    def _rotate_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints"""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by epoch
        self.checkpoints.sort(key=lambda x: x['epoch'])
        
        # Remove oldest
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            old_path = Path(old_checkpoint['path'])
            
            if old_path.exists():
                old_path.unlink()
                print(f"[CheckpointManager] Removed old checkpoint: {old_path.name}")
    
    def _save_metadata(self):
        """Save metadata to file"""
        metadata = {
            'checkpoints': self.checkpoints,
            'best_checkpoint': self.best_checkpoint,
            'best_value': self.best_value,
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self):
        """Load metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self.checkpoints = metadata.get('checkpoints', [])
            self.best_checkpoint = metadata.get('best_checkpoint')
            self.best_value = metadata.get('best_value', 
                float('inf') if self.mode == 'min' else float('-inf'))
    
    def get_checkpoint_list(self) -> List[Dict]:
        """
        Get list of checkpoints
        
        Returns:
            List of checkpoint info
        """
        return self.checkpoints.copy()
    
    def delete_all(self):
        """Delete all checkpoints"""
        for checkpoint in self.checkpoints:
            path = Path(checkpoint['path'])
            if path.exists():
                path.unlink()
        
        self.checkpoints = []
        self.best_checkpoint = None
        self._save_metadata()
        
        print("[CheckpointManager] Deleted all checkpoints")


# ==================== Model Saver ====================

class ModelSaver:
    """
    Save model components
    """
    
    @staticmethod
    def save_agent(agent, path: str):
        """
        Save agent
        
        Args:
            agent: Agent to save
            path: Save path
        """
        state_dict = {
            'policy': agent.policy.state_dict(),
            'critic': agent.critic.state_dict(),
        }
        
        if hasattr(agent, 'target_critic'):
            state_dict['target_critic'] = agent.target_critic.state_dict()
        
        if hasattr(agent, 'policy_optimizer'):
            state_dict['policy_optimizer'] = agent.policy_optimizer.state_dict()
        
        if hasattr(agent, 'critic_optimizer'):
            state_dict['critic_optimizer'] = agent.critic_optimizer.state_dict()
        
        torch.save(state_dict, path)
        print(f"[ModelSaver] Saved agent to: {path}")
    
    @staticmethod
    def load_agent(agent, path: str, device: str = 'cpu'):
        """
        Load agent
        
        Args:
            agent: Agent to load into
            path: Load path
            device: Device
        """
        state_dict = torch.load(path, map_location=device)
        
        agent.policy.load_state_dict(state_dict['policy'])
        agent.critic.load_state_dict(state_dict['critic'])
        
        if 'target_critic' in state_dict and hasattr(agent, 'target_critic'):
            agent.target_critic.load_state_dict(state_dict['target_critic'])
        
        if 'policy_optimizer' in state_dict and hasattr(agent, 'policy_optimizer'):
            agent.policy_optimizer.load_state_dict(state_dict['policy_optimizer'])
        
        if 'critic_optimizer' in state_dict and hasattr(agent, 'critic_optimizer'):
            agent.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        
        print(f"[ModelSaver] Loaded agent from: {path}")


# ==================== Training State ====================

class TrainingState:
    """
    Save and load complete training state
    """
    
    @staticmethod
    def save_state(
        path: str,
        epoch: int,
        agent,
        optimizer,
        scheduler,
        replay_buffer,
        metrics: Dict,
        config,
    ):
        """
        Save complete training state
        
        Args:
            path: Save path
            epoch: Current epoch
            agent: Agent
            optimizer: Optimizer
            scheduler: LR scheduler
            replay_buffer: Replay buffer
            metrics: Training metrics
            config: Configuration
        """
        state = {
            'epoch': epoch,
            'agent': {
                'policy': agent.policy.state_dict(),
                'critic': agent.critic.state_dict(),
            },
            'optimizer': optimizer.state_dict() if optimizer else None,
            'scheduler': scheduler.state_dict() if scheduler else None,
            'replay_buffer': {
                'size': len(replay_buffer),
                # Could save buffer data if needed
            },
            'metrics': metrics,
            'config': config.__dict__ if hasattr(config, '__dict__') else config,
            'timestamp': datetime.now().isoformat(),
        }
        
        torch.save(state, path)
        print(f"[TrainingState] Saved training state to: {path}")
    
    @staticmethod
    def load_state(path: str, device: str = 'cpu') -> Dict:
        """
        Load complete training state
        
        Args:
            path: Load path
            device: Device
            
        Returns:
            Training state dictionary
        """
        state = torch.load(path, map_location=device)
        
        print(f"[TrainingState] Loaded training state from: {path}")
        print(f"  Epoch: {state['epoch']}")
        print(f"  Timestamp: {state['timestamp']}")
        
        return state


# ==================== Checkpoint Utilities ====================

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find latest checkpoint in directory
    
    Args:
        checkpoint_dir: Directory to search
        
    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Find all checkpoints
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
    
    if len(checkpoints) == 0:
        return None
    
    # Sort by epoch number
    def extract_epoch(path):
        try:
            return int(path.stem.split('_')[-1])
        except:
            return 0
    
    checkpoints.sort(key=extract_epoch, reverse=True)
    
    return str(checkpoints[0])


def cleanup_checkpoints(
    checkpoint_dir: str,
    keep_best: bool = True,
    keep_latest: int = 3,
):
    """
    Cleanup old checkpoints
    
    Args:
        checkpoint_dir: Directory to clean
        keep_best: Keep best_model.pt
        keep_latest: Number of latest checkpoints to keep
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return
    
    # Find all checkpoints
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
    
    # Sort by epoch
    def extract_epoch(path):
        try:
            return int(path.stem.split('_')[-1])
        except:
            return 0
    
    checkpoints.sort(key=extract_epoch, reverse=True)
    
    # Keep latest N
    to_keep = set(checkpoints[:keep_latest])
    
    # Keep best
    if keep_best:
        best_path = checkpoint_dir / 'best_model.pt'
        if best_path.exists():
            to_keep.add(best_path)
    
    # Delete others
    deleted = 0
    for checkpoint in checkpoints:
        if checkpoint not in to_keep:
            checkpoint.unlink()
            deleted += 1
    
    print(f"[Cleanup] Deleted {deleted} old checkpoints")


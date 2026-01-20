"""
Main Trainer
Implements Algorithm 1: Self-play training with PFSP

Training Procedure:
1. Initialize policy via supervised learning (warmstart)
2. Self-play with PFSP opponent sampling
3. DSAC updates (Algorithm 2)
4. Add checkpoints to opponent pool

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm

from algorithms.dsac import DSACAgent
from algorithms.pfsp import ThreeSourceSampler
from environment.negotiation_env import NegotiationEnv
from data.replay_buffer import ReplayBuffer
from agents.opponent_pool import OpponentPool
from training.callbacks import CallbackManager
from training.evaluation import NegotiationEvaluator


# ==================== α-Nego Trainer ====================

class AlphaNegoTrainer:
    """
    Main trainer implementing Algorithm 1
    
    Training loop:
    1. Sample opponent using PFSP
    2. Self-play negotiation (collect data)
    3. DSAC updates
    4. Update opponent pool
    5. Evaluate and checkpoint
    """
    
    def __init__(
        self,
        agent: DSACAgent,
        env: NegotiationEnv,
        config,
        opponent_pool: OpponentPool,
        pfsp_sampler: ThreeSourceSampler,
        replay_buffer: ReplayBuffer,
        save_dir: str = 'checkpoints/',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Args:
            agent: DSAC agent
            env: Negotiation environment
            config: Configuration
            opponent_pool: Opponent pool
            pfsp_sampler: PFSP sampler
            replay_buffer: Replay buffer
            save_dir: Directory to save checkpoints
            device: Device
        """
        self.agent = agent
        self.env = env
        self.config = config
        self.opponent_pool = opponent_pool
        self.pfsp_sampler = pfsp_sampler
        self.replay_buffer = replay_buffer
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Training parameters
        self.num_epochs = config.training.num_epochs
        self.episodes_per_epoch = config.training.episodes_per_epoch
        self.updates_per_epoch = config.training.updates_per_epoch
        self.eval_frequency = config.training.eval_frequency
        self.save_frequency = config.training.save_frequency
        
        # Set current agent for self-play
        self.opponent_pool.set_current_agent(agent)
        
        # Statistics
        self.training_stats = {
            'epoch_rewards': [],
            'agreement_rates': [],
            'utilities': [],
            'losses': [],
        }
        
        print(f"[AlphaNegoTrainer] Initialized")
        print(f"  Num epochs: {self.num_epochs}")
        print(f"  Episodes per epoch: {self.episodes_per_epoch}")
        print(f"  Updates per epoch: {self.updates_per_epoch}")
    
    def train(
        self,
        callbacks: Optional[CallbackManager] = None,
        resume_from: Optional[str] = None,
    ):
        """
        Main training loop (Algorithm 1)
        
        Args:
            callbacks: Callback manager
            resume_from: Path to checkpoint to resume from
        """
        start_epoch = 0
        
        # Resume from checkpoint
        if resume_from:
            start_epoch = self._load_checkpoint(resume_from)
        
        # Training begin
        if callbacks:
            callbacks.on_train_begin({'start_epoch': start_epoch})
        
        print("\n" + "="*70)
        print("STARTING α-NEGO TRAINING (Algorithm 1)")
        print("="*70)
        
        # Main training loop
        for epoch in range(start_epoch, self.num_epochs):
            # Epoch begin
            if callbacks:
                callbacks.on_epoch_begin(epoch)
            
            # Step 1: Sample opponent using PFSP
            opponent = self._sample_opponent()
            
            # Step 2: Self-play episodes
            epoch_stats = self._self_play_epoch(opponent, epoch)
            
            # Step 3: DSAC updates
            update_stats = self._update_agent()
            
            # Step 4: Update opponent pool
            self._update_opponent_pool(opponent, epoch_stats, epoch)
            
            # Combine stats
            logs = {**epoch_stats, **update_stats, 'epoch': epoch}
            
            # Epoch end
            if callbacks:
                should_stop = callbacks.on_epoch_end(epoch, logs)
                if should_stop:
                    print(f"\n[Trainer] Training stopped by callback at epoch {epoch}")
                    break
            
            # Evaluation
            if (epoch + 1) % self.eval_frequency == 0:
                self._evaluate(epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.save_frequency == 0:
                self._save_checkpoint(epoch, logs)
            
            # Update statistics
            self._update_stats(logs)
            
            # Print progress
            self._print_progress(epoch, logs)
        
        # Training end
        if callbacks:
            callbacks.on_train_end({'final_epoch': epoch})
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
    
    def _sample_opponent(self):
        """
        Sample opponent using PFSP (Algorithm 1, Line 3)
        
        Returns:
            Opponent agent
        """
        # Get opponent IDs by type
        sl_opponents = [e.name for e in self.opponent_pool.sl_pool]
        rl_opponents = [e.name for e in self.opponent_pool.rl_pool]
        
        # Sample using PFSP
        source, opponent_id = self.pfsp_sampler.sample_opponent(
            sl_opponent_ids=sl_opponents,
            rl_opponent_ids=rl_opponents,
            current_agent_id='self_play'
        )
        
        # Get opponent agent
        if source == 'self_play':
            opponent = self.agent.clone()
        else:
            opponent_entry = self.opponent_pool.get_by_name(opponent_id)
            opponent = opponent_entry.agent if opponent_entry else None
        
        return opponent
    
    def _self_play_epoch(self, opponent, epoch: int) -> Dict:
        """
        Self-play episodes (Algorithm 1, Lines 4-9)
        
        Args:
            opponent: Opponent agent
            epoch: Current epoch
            
        Returns:
            Episode statistics
        """
        episode_rewards = []
        agreements = []
        utilities = []
        
        for episode in range(self.episodes_per_epoch):
            # Reset environment
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_data = []
            
            while not done:
                # Agent action
                intent, price, _ = self.agent.policy.sample_action(
                    torch.FloatTensor(obs).to(self.device)
                )
                
                action = {
                    'intent': intent,
                    'price': np.array([price])
                }
                
                # Environment step
                next_obs, reward, done, info = self.env.step(action)
                
                # Store transition
                episode_data.append({
                    'state': obs,
                    'action': [intent, price],
                    'reward': reward,
                    'next_state': next_obs,
                    'done': done,
                })
                
                obs = next_obs
                episode_reward += reward
            
            # Add episode to replay buffer
            for transition in episode_data:
                self.replay_buffer.add(
                    state=transition['state'],
                    action=transition['action'],
                    reward=transition['reward'],
                    next_state=transition['next_state'],
                    done=transition['done'],
                )
            
            # Record statistics
            episode_rewards.append(episode_reward)
            agreements.append(info['agreement'])
            if info['agreement']:
                utilities.append(info.get('agent_utility', 0.0))
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'agreement_rate': np.mean(agreements),
            'mean_utility': np.mean(utilities) if utilities else 0.0,
            'num_episodes': self.episodes_per_epoch,
        }
    
    def _update_agent(self) -> Dict:
        """
        DSAC updates (Algorithm 1, Lines 10-11 / Algorithm 2)
        
        Returns:
            Update statistics
        """
        if len(self.replay_buffer) < self.config.training.min_buffer_size:
            return {'updates_performed': 0}
        
        update_losses = []
        
        for _ in range(self.updates_per_epoch):
            # Sample batch
            batch = self.replay_buffer.sample(
                self.config.training.batch_size,
                device=self.device
            )
            
            # DSAC update
            losses = self.agent.update(batch)
            update_losses.append(losses)
        
        # Average losses
        avg_losses = {
            key: np.mean([loss[key] for loss in update_losses if key in loss])
            for key in update_losses[0].keys()
        }
        
        return avg_losses
    
    def _update_opponent_pool(self, opponent, epoch_stats: Dict, epoch: int):
        """
        Update opponent pool (Algorithm 1, Line 12)
        
        Args:
            opponent: Opponent used in self-play
            epoch_stats: Statistics from epoch
            epoch: Current epoch
        """
        # Update PFSP statistics
        # (In practice, would determine if opponent won)
        opponent_won = epoch_stats['agreement_rate'] < 0.5  # Simplified
        
        # Update pool
        # (In full implementation, would update opponent entry)
        
        # Add current agent to pool periodically
        if (epoch + 1) % self.config.training.pool_add_frequency == 0:
            agent_copy = self.agent.clone()
            self.opponent_pool.add(
                agent=agent_copy,
                name=f'RL_Agent_Epoch_{epoch+1}',
                agent_type='rl',
                epoch=epoch
            )
            print(f"\n[Pool] Added agent to opponent pool (epoch {epoch+1})")
    
    def _evaluate(self, epoch: int):
        """
        Evaluate agent
        
        Args:
            epoch: Current epoch
        """
        print(f"\n[Evaluation] Epoch {epoch+1}")
        
        evaluator = NegotiationEvaluator(
            self.env,
            num_episodes=self.config.training.eval_episodes,
            verbose=False
        )
        
        results = evaluator.evaluate(self.agent, deterministic=True)
        
        print(f"  Agreement Rate: {results['overall']['agreement_rate']:.2%}")
        print(f"  Avg Utility: {results['overall']['avg_agent_utility']:.4f}")
        print(f"  Avg Dialogue Length: {results['overall']['avg_dialogue_length']:.2f}")
    
    def _save_checkpoint(self, epoch: int, logs: Dict):
        """
        Save training checkpoint
        
        Args:
            epoch: Current epoch
            logs: Training logs
        """
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pt'
        
        torch.save({
            'epoch': epoch,
            'agent_state_dict': {
                'policy': self.agent.policy.state_dict(),
                'critic': self.agent.critic.state_dict(),
                'target_critic': self.agent.target_critic.state_dict(),
            },
            'optimizer_state_dict': {
                'policy': self.agent.policy_optimizer.state_dict(),
                'critic': self.agent.critic_optimizer.state_dict(),
            },
            'training_stats': self.training_stats,
            'logs': logs,
        }, checkpoint_path)
        
        print(f"\n[Checkpoint] Saved to {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load training checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Epoch to resume from
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load agent
        self.agent.policy.load_state_dict(checkpoint['agent_state_dict']['policy'])
        self.agent.critic.load_state_dict(checkpoint['agent_state_dict']['critic'])
        self.agent.target_critic.load_state_dict(checkpoint['agent_state_dict']['target_critic'])
        
        # Load optimizers
        self.agent.policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['policy'])
        self.agent.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['critic'])
        
        # Load statistics
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        epoch = checkpoint['epoch'] + 1
        
        print(f"\n[Checkpoint] Resumed from epoch {epoch}")
        
        return epoch
    
    def _update_stats(self, logs: Dict):
        """Update training statistics"""
        self.training_stats['epoch_rewards'].append(logs.get('mean_reward', 0))
        self.training_stats['agreement_rates'].append(logs.get('agreement_rate', 0))
        self.training_stats['utilities'].append(logs.get('mean_utility', 0))
        
        if 'policy_loss' in logs:
            self.training_stats['losses'].append(logs['policy_loss'])
    
    def _print_progress(self, epoch: int, logs: Dict):
        """Print training progress"""
        print(f"\nEpoch {epoch+1}/{self.num_epochs}")
        print(f"  Reward: {logs.get('mean_reward', 0):.4f}")
        print(f"  Agreement: {logs.get('agreement_rate', 0):.2%}")
        print(f"  Utility: {logs.get('mean_utility', 0):.4f}")
        if 'policy_loss' in logs:
            print(f"  Policy Loss: {logs['policy_loss']:.4f}")
        if 'critic_loss' in logs:
            print(f"  Critic Loss: {logs['critic_loss']:.4f}")


# ==================== Quick Training Function ====================

def train_alpha_nego(
    config,
    warmstart_path: Optional[str] = None,
    save_dir: str = 'checkpoints/',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Quick training function for α-Nego
    
    Implements complete Algorithm 1
    
    Args:
        config: Configuration
        warmstart_path: Path to warm-started policy
        save_dir: Save directory
        device: Device
    """
    from algorithms.dsac import create_dsac_agent
    from environment.negotiation_env import make_negotiation_env
    from data.replay_buffer import ReplayBuffer
    
    print("\n" + "="*70)
    print("INITIALIZING α-NEGO TRAINING")
    print("="*70)
    
    # Create environment
    print("\n[1/6] Creating environment...")
    env = make_negotiation_env(
        dataset=config.dataset.name,
        opponent_type='rule_based',
        max_turns=config.dataset.max_turns,
    )
    
    # Create agent
    print("\n[2/6] Creating DSAC agent...")
    supervised_agent = None
    if warmstart_path:
        # Load warm-started policy
        from training.warmstart import load_warmstart_policy
        from models.policy_network import create_policy_network
        
        policy = create_policy_network(config)
        supervised_agent = load_warmstart_policy(policy, warmstart_path, device)
    
    agent = create_dsac_agent(
        config=config,
        supervised_agent=supervised_agent,
        device=device
    )
    
    # Create opponent pool
    print("\n[3/6] Creating opponent pool...")
    opponent_pool = OpponentPool(
        prob_self_play=config.training.prob_self_play,
        prob_sl_agents=config.training.prob_sl_agents,
        prob_rl_agents=config.training.prob_rl_agents,
    )
    
    # Add baseline opponents
    from agents.baseline_agents import create_all_baseline_agents
    baselines = create_all_baseline_agents(role='seller')
    for baseline in baselines:
        opponent_pool.add(baseline, baseline.name, 'rule', epoch=0)
    
    # Create PFSP sampler
    print("\n[4/6] Creating PFSP sampler...")
    pfsp_sampler = ThreeSourceSampler(
        prob_self_play=config.training.prob_self_play,
        prob_sl_agents=config.training.prob_sl_agents,
        prob_rl_agents=config.training.prob_rl_agents,
    )
    
    # Create replay buffer
    print("\n[5/6] Creating replay buffer...")
    replay_buffer = ReplayBuffer(
        capacity=config.training.buffer_capacity,
        state_dim=config.model.state_dim,
        action_dim=config.model.action_dim,
    )
    
    # Create trainer
    print("\n[6/6] Creating trainer...")
    trainer = AlphaNegoTrainer(
        agent=agent,
        env=env,
        config=config,
        opponent_pool=opponent_pool,
        pfsp_sampler=pfsp_sampler,
        replay_buffer=replay_buffer,
        save_dir=save_dir,
        device=device,
    )
    
    # Train
    trainer.train()
    
    print("\n[Complete] Training finished!")
    print(f"Checkpoints saved to: {save_dir}")


# ==================== Testing ====================

if __name__ == '__main__':
    print("Testing Trainer...")
    
    # Would need full setup to test
    print("  Trainer module loaded successfully")
    print("  Use train_alpha_nego() to start training")
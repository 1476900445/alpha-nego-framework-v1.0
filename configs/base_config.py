"""
Base Configuration for α-Nego Framework
Implements all hyperparameters from the paper (Section 5.2)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import torch


@dataclass
class NetworkConfig:
    """Neural network architecture configuration (Section 5.1)"""
    
    # Policy Network
    policy_input_dim: int = 3
    policy_hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    policy_activation: str = 'ReLU'
    policy_output_intent_dim: int = 16  # 16 dialogue acts
    policy_output_price_dim: int = 2  # Mean and std for continuous price
    policy_dropout: float = 0.1
    policy_layer_norm: bool = True
    
    # Critic Network (Distributional SAC)
    critic_input_dim: int = 5  # state_dim (3) + action_dim (2)
    critic_hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    critic_activation: str = 'ReLU'
    critic_num_quantiles: int = 51  # Number of quantiles for value distribution
    critic_dropout: float = 0.1
    critic_layer_norm: bool = True
    
    # State Encoder
    encoder_embedding_dim: int = 128
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    encoder_num_layers: int = 2
    encoder_dropout: float = 0.1
    
    # Attention mechanism
    use_attention: bool = True
    attention_heads: int = 4
    attention_dim: int = 128
    
    # Initialization
    init_method: str = 'xavier_uniform'
    init_gain: float = 1.0


@dataclass
class TrainingConfig:
    """Training configuration (Section 5.2)"""
    
    # Warm-start (Supervised Learning)
    warm_start_epochs: int = 10
    warm_start_learning_rate: float = 1e-3
    warm_start_batch_size: int = 32
    warm_start_optimizer: str = 'Adam'
    
    # Main RL Training
    rl_policy_lr: float = 1e-4
    rl_critic_lr: float = 1e-3
    rl_optimizer: str = 'Adam'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 1e-5
    
    batch_size: int = 32
    replay_buffer_size: int = 1000000  # 1M
    min_replay_size: int = 10000  # Start training after this many steps
    
    # Update frequencies
    target_update_frequency: int = 1  # Update target every step
    soft_update_tau: float = 0.005
    policy_update_frequency: int = 2  # Update policy every 2 critic updates
    
    # Gradient management
    gradient_clip_norm: float = 1.0
    gradient_clip_value: float = 10.0
    
    # Distributional RL (DSAC)
    num_quantiles: int = 51
    quantile_regression_loss: str = 'huber'
    huber_kappa: float = 1.0
    num_critics: int = 2  # Dual critics for stability
    use_min_critics: bool = True  # Take min Q-value
    
    # KL Regularization (Eq. 8)
    alpha_kl: float = 0.5  # Weight for KL penalty
    kl_target_policy: str = 'sl_agent'
    
    # Entropy regularization (SAC)
    alpha_entropy_intent: float = 0.2  # Entropy coefficient for dialogue acts
    beta_entropy_price: float = 0.1   # Entropy coefficient for prices
    auto_tune_entropy: bool = False  # Automatic entropy tuning
    target_entropy: Optional[float] = None  # If None, set to -dim(A)
    
    # Discount factor
    gamma: float = 0.99
    
    # Training control
    max_epochs: int = 1100
    steps_per_epoch: int = 1000
    eval_frequency: int = 10  # Evaluate every N epochs
    save_frequency: int = 50  # Save checkpoint every N epochs
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = 'cosine'  # 'cosine', 'step', 'exponential'
    lr_decay_rate: float = 0.95
    lr_decay_steps: int = 100
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 100
    min_delta: float = 1e-4


@dataclass
class SelfPlayConfig:
    """Self-play configuration (Algorithm 1 & Section 4.1)"""
    
    # Warm-start initialization
    use_supervised_warmstart: bool = True
    pretrain_critics: bool = True
    critic_pretraining_epochs: int = 5
    
    # PFSP opponent selection (Eq. 10)
    pfsp_method: str = 'Priority_Fictitious_Self_Play'
    dominance_metric: str = 'utility_based'
    
    # Opponent pool management
    initial_pool_size: int = 3  # SL agent + rules
    max_pool_size: int = 50
    addition_criterion: str = 'dominates_all_opponents'
    
    # Three-source diversity (Figure 2)
    prob_self_play: float = 0.20      # 20% self
    prob_sl_agents: float = 0.30      # 30% SL variants
    prob_rl_agents: float = 0.50      # 50% RL agents
    
    # Model evaluation (Definition 4.1)
    eval_metric: str = 'negotiation_score'
    eval_components: List[str] = field(default_factory=lambda: [
        'agreement_rate', 'utility', 'dialogue_length', 'social_welfare'
    ])
    alpha_sc: float = -5e-3  # Weight for dialogue length
    beta_sc: float = 0.1     # Weight for social welfare
    epsilon_clip: float = 0.01  # Clipping for stability
    
    # Opponent snapshot saving
    snapshot_frequency: int = 100  # Save opponent every N steps
    max_snapshots: int = 100


@dataclass
class StyleControlConfig:
    """Style control configuration (Section 4.2, Eq. 12-14)"""
    
    # Neutral style (Eq. 12)
    neutral_description: str = 'Expected value - balanced approach'
    neutral_q_computation: str = 'mean_of_all_quantiles'
    neutral_quantile_range: Tuple[float, float] = (0.0, 1.0)
    
    # Aggressive style (Eq. 13)
    aggressive_description: str = 'Risk-seeking - pursue high-value deals'
    aggressive_q_computation: str = 'mean_plus_variance_bonus'
    aggressive_quantile_range: Tuple[float, float] = (0.5, 1.0)
    alpha_agg: float = 1.0  # Left truncated variance weight
    
    # Conservative style (Eq. 14)
    conservative_description: str = 'Risk-averse - prioritize agreement'
    conservative_q_computation: str = 'CVaR_lower_tail'
    conservative_quantile_range: Tuple[float, float] = (0.0, 0.2)
    alpha_con: float = 0.2  # CVaR parameter
    
    # Active style
    active_style: str = 'neutral'  # 'neutral', 'aggressive', 'conservative'


@dataclass
class DialogueSystemConfig:
    """Dialogue system configuration (Section 3.1, Table 1)"""
    
    # Dialogue acts
    num_dialogue_acts: int = 16
    dialogue_acts: List[str] = field(default_factory=lambda: [
        'greet', 'inquire', 'inform', 'init-price', 'insist-price',
        'agree-price', 'concede-price', 'final-price', 'counter-no-price',
        'hesitant', 'positive', 'negative', 'offer', 'accept', 'reject', 'quit'
    ])
    
    # Price-related acts (require price argument)
    price_related_acts: List[str] = field(default_factory=lambda: [
        'init-price', 'insist-price', 'agree-price',
        'concede-price', 'final-price', 'offer'
    ])
    
    # Generator configuration
    generator_type: str = 'retrieval_based'
    num_template_candidates: int = 10
    use_response_diversity: bool = True
    
    # Parser configuration
    parser_type: str = 'rule_based'
    extraction_method: str = 'regex_and_if_then_rules'
    
    # Dialogue constraints
    max_dialogue_length: int = 20
    min_dialogue_length: int = 1


@dataclass
class OpponentConfig:
    """Opponent configuration (Section 5.2)"""
    
    # Time-dependent opponents (8 variants)
    num_time_dependent: int = 8
    time_dependent_types: List[str] = field(default_factory=lambda: [
        'Conceder', 'Boulware', 'Tit_For_Tat', 'Linear',
        'Conceder_Mild', 'Boulware_Mild', 'TFT_Aggressive', 'TFT_Conservative'
    ])
    
    # Behavior-dependent opponents (2 variants)
    num_behavior_dependent: int = 2
    behavior_dependent_types: List[str] = field(default_factory=lambda: [
        'Reciprocal_Concession', 'Tit_For_Tat_with_Decay'
    ])
    
    # Utterance diversity
    use_diverse_utterances: bool = True


@dataclass
class RewardConfig:
    """Reward function configuration (Section 5.2)"""
    
    # Craigslistbargain task
    craigslist_reward_type: str = 'linear_function_of_deal_price'
    craigslist_max_reward: float = 1.0
    craigslist_min_reward: float = 0.0
    craigslist_no_deal_penalty: float = -0.5
    
    # Dealornodeal task
    dealornodeal_reward_type: str = 'utility_function'
    dealornodeal_no_deal_penalty: float = -0.5
    
    # Reward shaping
    use_reward_shaping: bool = True
    shape_dialogue_length: bool = True
    shape_fairness: bool = True
    length_penalty_weight: float = -0.01
    fairness_bonus_weight: float = 0.05


@dataclass
class ExperimentConfig:
    """Experiment configuration (Section 6)"""
    
    # Dataset
    dataset: str = 'craigslistbargain'  # 'craigslistbargain' or 'dealornodeal'
    data_path: str = './data/'
    
    # Training stages
    stage_epochs: Dict[str, int] = field(default_factory=lambda: {
        'alpha_nego_1': 100,
        'alpha_nego_m': 500,
        'alpha_nego_f': 1100,
    })
    
    # Reproducibility
    random_seed: int = 42
    num_runs: int = 5
    different_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 999])
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging
    log_dir: str = './logs/'
    checkpoint_dir: str = './checkpoints/'
    result_dir: str = './results/'
    
    # Wandb logging
    use_wandb: bool = False
    wandb_project: str = 'alpha-nego'
    wandb_entity: Optional[str] = None
    
    # Tensorboard logging
    use_tensorboard: bool = True
    tensorboard_dir: str = './runs/'


@dataclass
class AlphaNegoConfig:
    """Complete α-Nego configuration"""
    
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    style_control: StyleControlConfig = field(default_factory=StyleControlConfig)
    dialogue_system: DialogueSystemConfig = field(default_factory=DialogueSystemConfig)
    opponent: OpponentConfig = field(default_factory=OpponentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'network': self.network.__dict__,
            'training': self.training.__dict__,
            'self_play': self.self_play.__dict__,
            'style_control': self.style_control.__dict__,
            'dialogue_system': self.dialogue_system.__dict__,
            'opponent': self.opponent.__dict__,
            'reward': self.reward.__dict__,
            'experiment': self.experiment.__dict__,
        }
    
    def save(self, path: str):
        """Save configuration to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                attr = getattr(config, key)
                for k, v in value.items():
                    setattr(attr, k, v)
        return config


def get_default_config(dataset: str = 'craigslistbargain') -> AlphaNegoConfig:
    """Get default configuration for a dataset"""
    config = AlphaNegoConfig()
    config.experiment.dataset = dataset
    return config
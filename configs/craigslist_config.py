"""
Craigslistbargain Dataset Configuration
Based on paper experiments (Section 6)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from configs.base_config import AlphaNegoConfig


def get_craigslist_config() -> AlphaNegoConfig:
    """
    Get configuration for Craigslistbargain dataset
    
    Dataset Description:
    - Human-human price negotiations from Craigslist.org
    - Real goods negotiation scenarios
    - Contains cheap talk, embellishment, side offers
    - 2 negotiators: buyer and seller
    
    Reference: He et al., 2018
    """
    
    config = AlphaNegoConfig()
    
    # ==================== Dataset Specific ====================
    config.experiment.dataset = 'craigslistbargain'
    config.experiment.data_path = './data/craigslistbargain/'
    
    # ==================== Dialogue System ====================
    # 16 dialogue acts specific to price negotiation (Table 1)
    config.dialogue_system.num_dialogue_acts = 16
    config.dialogue_system.dialogue_acts = [
        'greet',           # 0: Say hi/hello/hey
        'inquire',         # 1: Inquiry about product details
        'inform',          # 2: Provide information
        'init-price',      # 3: Propose first price
        'insist-price',    # 4: Insist last price
        'agree-price',     # 5: Agree on price
        'concede-price',   # 6: Give concession
        'final-price',     # 7: Final price, no more compromise
        'counter-no-price',# 8: Reject price without counteroffer
        'hesitant',        # 9: Indecisive
        'positive',        # 10: Positive attitude
        'negative',        # 11: Express dissatisfaction
        'offer',           # 12: Formal offer
        'accept',          # 13: Accept formal offer
        'reject',          # 14: Reject formal offer
        'quit',            # 15: Quit negotiation
    ]
    
    # Price-related acts (require price argument)
    config.dialogue_system.price_related_acts = [
        'init-price', 'insist-price', 'agree-price',
        'concede-price', 'final-price', 'offer'
    ]
    
    # Dialogue constraints
    config.dialogue_system.max_dialogue_length = 20
    config.dialogue_system.min_dialogue_length = 1
    
    # ==================== Network Architecture ====================
    # State encoding: [current_price, opponent_last_price, turn_number]
    config.network.policy_input_dim = 3
    config.network.critic_input_dim = 5  # state(3) + action(2)
    
    # Policy network: Two hidden layers with 256 units each
    config.network.policy_hidden_layers = [256, 256]
    config.network.policy_output_intent_dim = 16  # 16 dialogue acts
    config.network.policy_output_price_dim = 2   # mean and std
    
    # Critic network: Same architecture
    config.network.critic_hidden_layers = [256, 256]
    config.network.critic_num_quantiles = 51  # 51 quantiles for distribution
    
    # Advanced features
    config.network.use_attention = True
    config.network.attention_heads = 4
    config.network.policy_dropout = 0.1
    config.network.critic_dropout = 0.1
    config.network.policy_layer_norm = True
    config.network.critic_layer_norm = True
    
    # ==================== Training Configuration ====================
    # Warm-start (Supervised Learning)
    config.training.warm_start_epochs = 10
    config.training.warm_start_learning_rate = 1e-3
    config.training.warm_start_batch_size = 32
    
    # Main RL Training
    config.training.rl_policy_lr = 1e-4
    config.training.rl_critic_lr = 1e-3
    config.training.batch_size = 32
    config.training.replay_buffer_size = 1000000  # 1M
    
    # Update frequencies
    config.training.target_update_frequency = 1
    config.training.soft_update_tau = 0.005
    config.training.policy_update_frequency = 2
    
    # Distributional RL
    config.training.num_quantiles = 51
    config.training.huber_kappa = 1.0
    config.training.num_critics = 2
    
    # KL Regularization (Eq. 8)
    config.training.alpha_kl = 0.5  # Weight for KL penalty with SL agent
    
    # Entropy regularization (SAC)
    config.training.alpha_entropy_intent = 0.2  # For dialogue acts
    config.training.beta_entropy_price = 0.1    # For prices
    
    # Training duration
    config.training.max_epochs = 1100  # Final stage (α-Nego_f)
    config.training.steps_per_epoch = 1000
    config.training.eval_frequency = 10
    config.training.save_frequency = 50
    
    # Gradient management
    config.training.gradient_clip_norm = 1.0
    config.training.gradient_clip_value = 10.0
    
    # Discount factor
    config.training.gamma = 0.99
    
    # ==================== Self-Play Configuration ====================
    # PFSP opponent selection
    config.self_play.use_supervised_warmstart = True
    config.self_play.pretrain_critics = True
    config.self_play.critic_pretraining_epochs = 5
    
    # Three-source diversity (Figure 2, line 421-422)
    config.self_play.prob_self_play = 0.20   # 20% self-play
    config.self_play.prob_sl_agents = 0.30   # 30% SL agents
    config.self_play.prob_rl_agents = 0.50   # 50% RL agents from pool
    
    # Opponent pool management
    config.self_play.initial_pool_size = 3
    config.self_play.max_pool_size = 50
    config.self_play.addition_criterion = 'dominates_all_opponents'
    
    # Model evaluation (Definition 4.1, Eq. 9)
    config.self_play.alpha_sc = -5e-3  # Weight for dialogue length
    config.self_play.beta_sc = 0.1     # Weight for social welfare
    config.self_play.epsilon_clip = 0.01
    
    # ==================== Style Control ====================
    # Neutral style (Eq. 12): Balanced approach
    config.style_control.neutral_description = 'Expected value - balanced approach'
    config.style_control.neutral_q_computation = 'mean_of_all_quantiles'
    config.style_control.neutral_quantile_range = (0.0, 1.0)
    
    # Aggressive style (Eq. 13): Risk-seeking
    config.style_control.aggressive_description = 'Risk-seeking - pursue high-value deals'
    config.style_control.aggressive_q_computation = 'mean_plus_variance_bonus'
    config.style_control.aggressive_quantile_range = (0.5, 1.0)
    config.style_control.alpha_agg = 1.0  # Left truncated variance weight
    
    # Conservative style (Eq. 14): Risk-averse
    config.style_control.conservative_description = 'Risk-averse - prioritize agreement'
    config.style_control.conservative_q_computation = 'CVaR_lower_tail'
    config.style_control.conservative_quantile_range = (0.0, 0.2)
    config.style_control.alpha_con = 0.2  # CVaR parameter
    
    # Default style
    config.style_control.active_style = 'neutral'
    
    # ==================== Opponent Configuration ====================
    # Time-dependent opponents (8 variants)
    config.opponent.num_time_dependent = 8
    config.opponent.time_dependent_types = [
        'Conceder',              # Quickly goes to minimum price
        'Boulware',              # Maintains offered value
        'Tit_For_Tat',          # Reproduces opponent behavior
        'Linear',                # Linear concession
        'Conceder_Mild',         # Milder concession
        'Boulware_Mild',         # Milder boulware
        'TFT_Aggressive',        # Aggressive TFT
        'TFT_Conservative',      # Conservative TFT
    ]
    
    # Behavior-dependent opponents (2 variants)
    config.opponent.num_behavior_dependent = 2
    config.opponent.behavior_dependent_types = [
        'Reciprocal_Concession',     # Match opponent concessions
        'Tit_For_Tat_with_Decay',   # TFT with decay factor
    ]
    
    config.opponent.use_diverse_utterances = True
    
    # ==================== Reward Configuration ====================
    # Linear function of deal price (Section 5.2)
    config.reward.craigslist_reward_type = 'linear_function_of_deal_price'
    config.reward.craigslist_max_reward = 1.0  # At ideal price
    config.reward.craigslist_min_reward = 0.0  # At midpoint
    config.reward.craigslist_no_deal_penalty = -0.5
    
    # Reward shaping
    config.reward.use_reward_shaping = True
    config.reward.shape_dialogue_length = True
    config.reward.shape_fairness = True
    config.reward.length_penalty_weight = -0.01
    config.reward.fairness_bonus_weight = 0.05
    
    # ==================== Experiment Configuration ====================
    # Training stages (Section 6)
    config.experiment.stage_epochs = {
        'alpha_nego_1': 100,   # Early stage
        'alpha_nego_m': 500,   # Middle stage
        'alpha_nego_f': 1100,  # Final stage
    }
    
    # Reproducibility
    config.experiment.random_seed = 42
    config.experiment.num_runs = 5
    config.experiment.different_seeds = [42, 123, 456, 789, 999]
    
    # Logging
    config.experiment.log_dir = './logs/craigslistbargain/'
    config.experiment.checkpoint_dir = './checkpoints/craigslistbargain/'
    config.experiment.result_dir = './results/craigslistbargain/'
    
    return config


# ==================== Evaluation Benchmarks ====================

# Expected results from paper (Table 4)
EXPECTED_RESULTS = {
    'alpha_nego_1': {
        'agreement_rate': 0.71,
        'utility': 0.61,
        'score': 2.26,
        'epochs': 100,
    },
    'alpha_nego_m': {
        'agreement_rate': 0.73,
        'utility': 0.79,
        'score': 2.99,
        'epochs': 500,
    },
    'alpha_nego_f': {
        'agreement_rate': 0.75,
        'utility': 0.83,
        'score': 3.22,
        'epochs': 1100,
    },
}

# Style-specific results (Section 6.3, Table 6)
STYLE_RESULTS = {
    'neutral': {
        'agreement_rate': 0.75,
        'utility': 0.83,
        'dialogue_length': 8.8,
        'score': 3.22,
    },
    'aggressive': {
        'agreement_rate': 0.62,
        'utility': 0.89,
        'dialogue_length': 8.3,
        'score': 3.15,
        'characteristics': [
            'Higher utility',
            'Fewer concessions',
            'Earlier termination',
            'Lower agreement rate',
        ],
    },
    'conservative': {
        'agreement_rate': 0.85,
        'utility': 0.71,
        'dialogue_length': 8.1,
        'score': 3.08,
        'characteristics': [
            'Higher agreement rate',
            'More concessions',
            'Better deal fairness',
            'Lower individual utility',
        ],
    },
}

# Baseline comparisons (Table 4)
BASELINE_RESULTS = {
    'SL_rule': {'Ag': 0.22, 'Ut': 0.27, 'Sc': 1.17},
    'A2C': {'Ag': 0.42, 'Ut': 0.33, 'Sc': 1.31},
    'ToM(i)': {'Ag': 0.42, 'Ut': 0.48, 'Sc': 1.39},
    'ToM(e)': {'Ag': 0.44, 'Ut': 0.50, 'Sc': 1.45},
    'CHAI': {'Ag': 0.54, 'Ut': 0.50, 'Sc': 1.57},
}


# ==================== Helper Functions ====================

def get_stage_config(stage: str) -> AlphaNegoConfig:
    """
    Get configuration for specific training stage
    
    Args:
        stage: 'alpha_nego_1', 'alpha_nego_m', or 'alpha_nego_f'
    
    Returns:
        Configuration for that stage
    """
    config = get_craigslist_config()
    
    if stage == 'alpha_nego_1':
        config.training.max_epochs = 100
    elif stage == 'alpha_nego_m':
        config.training.max_epochs = 500
    elif stage == 'alpha_nego_f':
        config.training.max_epochs = 1100
    else:
        raise ValueError(f"Unknown stage: {stage}")
    
    return config


def get_style_config(style: str) -> AlphaNegoConfig:
    """
    Get configuration for specific negotiation style
    
    Args:
        style: 'neutral', 'aggressive', or 'conservative'
    
    Returns:
        Configuration for that style
    """
    config = get_craigslist_config()
    config.style_control.active_style = style
    
    if style == 'aggressive':
        # May want to adjust these for more aggressive behavior
        config.style_control.alpha_agg = 1.5  # Higher variance bonus
    elif style == 'conservative':
        # May want to adjust for more conservative behavior
        config.style_control.alpha_con = 0.15  # Focus on lower 15% tail
    
    return config


# ==================== Validation ====================

def validate_config(config: AlphaNegoConfig) -> bool:
    """
    Validate Craigslistbargain configuration
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check dataset
    assert config.experiment.dataset == 'craigslistbargain'
    
    # Check dialogue acts
    assert config.dialogue_system.num_dialogue_acts == 16
    assert len(config.dialogue_system.dialogue_acts) == 16
    
    # Check network dimensions
    assert config.network.policy_input_dim == 3
    assert config.network.policy_output_intent_dim == 16
    assert config.network.policy_output_price_dim == 2
    
    # Check training parameters (from paper)
    assert config.training.rl_policy_lr == 1e-4
    assert config.training.rl_critic_lr == 1e-3
    assert config.training.alpha_kl == 0.5
    assert config.training.gamma == 0.99
    
    # Check quantiles
    assert config.training.num_quantiles == 51
    
    # Check PFSP probabilities sum to 1
    prob_sum = (config.self_play.prob_self_play + 
                config.self_play.prob_sl_agents + 
                config.self_play.prob_rl_agents)
    assert abs(prob_sum - 1.0) < 1e-6
    
    return True


"""
Dealornodeal Dataset Configuration
Based on paper experiments (Section 6.4-6.5)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from configs.base_config import AlphaNegoConfig


def get_dealornodeal_config() -> AlphaNegoConfig:
    """
    Get configuration for Dealornodeal dataset
    
    Dataset Description:
    - Multi-issue negotiation scenarios
    - Items: hats, books, balls
    - Each item has different values for each agent
    - 2 negotiators with different preferences
    
    Reference: Lewis et al., 2017
    """
    
    config = AlphaNegoConfig()
    
    # ==================== Dataset Specific ====================
    config.experiment.dataset = 'dealornodeal'
    config.experiment.data_path = './data/dealornodeal/'
    
    # ==================== Dialogue System ====================
    # 6 dialogue acts for multi-issue negotiation (Table 2)
    config.dialogue_system.num_dialogue_acts = 6
    config.dialogue_system.dialogue_acts = [
        'greet',      # 0: Say hi, initial proposal
        'disagree',   # 1: Say no, reject
        'agree',      # 2: Accept, ok, great
        'insist',     # 3: Same offer as previous
        'inquire',    # 4: Ask question
        'propose',    # 5: Make proposal with items
    ]
    
    # Proposal-related acts (require item allocation)
    config.dialogue_system.price_related_acts = [
        'greet',    # Can include initial proposal
        'propose',  # Main proposal act
    ]
    
    # Dialogue constraints
    config.dialogue_system.max_dialogue_length = 20
    config.dialogue_system.min_dialogue_length = 1
    
    # ==================== Multi-Issue Negotiation ====================
    # Items to negotiate over
    ITEMS = ['hats', 'books', 'balls']
    MAX_ITEMS_PER_TYPE = 3  # Maximum count for each item type
    
    # State encoding: [my_utility_estimate, opp_utility_estimate, turn_number]
    config.network.policy_input_dim = 3
    config.network.critic_input_dim = 5  # state(3) + action(2)
    
    # Policy network outputs
    config.network.policy_output_intent_dim = 6    # 6 dialogue acts
    config.network.policy_output_price_dim = 9     # 3 items × 3 counts each
    # Actually represents item allocation: [hats(0-3), books(0-3), balls(0-3)]
    
    # ==================== Network Architecture ====================
    # Similar to Craigslistbargain but adjusted for multi-issue
    config.network.policy_hidden_layers = [256, 256]
    config.network.critic_hidden_layers = [256, 256]
    config.network.critic_num_quantiles = 51
    
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
    config.training.replay_buffer_size = 1000000
    
    # Update frequencies
    config.training.target_update_frequency = 1
    config.training.soft_update_tau = 0.005
    config.training.policy_update_frequency = 2
    
    # Distributional RL
    config.training.num_quantiles = 51
    config.training.huber_kappa = 1.0
    config.training.num_critics = 2
    
    # KL Regularization
    config.training.alpha_kl = 0.5
    
    # Entropy regularization (higher for multi-issue)
    config.training.alpha_entropy_intent = 0.3   # Higher for more exploration
    config.training.beta_entropy_price = 0.15    # Higher for item allocation
    
    # Training duration
    config.training.max_epochs = 1100
    config.training.steps_per_epoch = 1000
    config.training.eval_frequency = 10
    config.training.save_frequency = 50
    
    # Gradient management
    config.training.gradient_clip_norm = 1.0
    config.training.gradient_clip_value = 10.0
    
    # Discount factor
    config.training.gamma = 0.99
    
    # ==================== Self-Play Configuration ====================
    config.self_play.use_supervised_warmstart = True
    config.self_play.pretrain_critics = True
    config.self_play.critic_pretraining_epochs = 5
    
    # Three-source diversity
    config.self_play.prob_self_play = 0.20
    config.self_play.prob_sl_agents = 0.30
    config.self_play.prob_rl_agents = 0.50
    
    # Opponent pool
    config.self_play.initial_pool_size = 3
    config.self_play.max_pool_size = 50
    
    # Model evaluation
    config.self_play.alpha_sc = -5e-3
    config.self_play.beta_sc = 0.1
    config.self_play.epsilon_clip = 0.01
    
    # ==================== Style Control ====================
    # Same as Craigslistbargain
    config.style_control.neutral_quantile_range = (0.0, 1.0)
    config.style_control.aggressive_quantile_range = (0.5, 1.0)
    config.style_control.conservative_quantile_range = (0.0, 0.2)
    config.style_control.alpha_agg = 1.0
    config.style_control.alpha_con = 0.2
    config.style_control.active_style = 'neutral'
    
    # ==================== Opponent Configuration ====================
    # Similar opponent types but adapted for multi-issue
    config.opponent.num_time_dependent = 8
    config.opponent.num_behavior_dependent = 2
    config.opponent.use_diverse_utterances = True
    
    # ==================== Reward Configuration ====================
    # Utility function (Section 5.2)
    # U(ω) = Σ(w_j · V_j(v_jk))
    config.reward.dealornodeal_reward_type = 'utility_function'
    config.reward.dealornodeal_no_deal_penalty = -0.5
    
    # Components:
    # w_j: weighting preference for issue j (hats, books, balls)
    # V_j: evaluation function for issue j
    # v_jk: k-th possible choice of issue j
    
    # Reward shaping
    config.reward.use_reward_shaping = True
    config.reward.shape_dialogue_length = True
    config.reward.shape_fairness = True
    config.reward.length_penalty_weight = -0.01
    config.reward.fairness_bonus_weight = 0.05
    
    # ==================== Experiment Configuration ====================
    # Training stages
    config.experiment.stage_epochs = {
        'alpha_nego_1': 100,
        'alpha_nego_m': 500,
        'alpha_nego_f': 1100,
    }
    
    # Reproducibility
    config.experiment.random_seed = 42
    config.experiment.num_runs = 5
    config.experiment.different_seeds = [42, 123, 456, 789, 999]
    
    # Logging
    config.experiment.log_dir = './logs/dealornodeal/'
    config.experiment.checkpoint_dir = './checkpoints/dealornodeal/'
    config.experiment.result_dir = './results/dealornodeal/'
    
    # ==================== Human Evaluation (Section 6.5) ====================
    # Settings used in paper's human evaluation
    config.experiment.human_eval = {
        'num_participants': 120,
        'incentive': 'VIP account bonus for successful negotiation',
        'evaluation_dimensions': 4,  # In, Fl, Lg, Wl
        'scale': '1-5 Likert',
        'dimensions': {
            'intelligence': 'In',      # How intelligent the agent seems
            'fluency': 'Fl',          # Language fluency
            'logic': 'Lg',            # Logical coherence
            'willingness': 'Wl',      # Willingness to play again
        },
    }
    
    return config


# ==================== Evaluation Benchmarks ====================

# Expected results from paper (Table 5, Section 6.4)
EXPECTED_RESULTS = {
    'alpha_nego_f': {
        'agreement_rate': 0.74,
        'utility': 0.66,
        'social_welfare': 0.57,
        'dialogue_length': 11.0,
    },
}

# Baseline comparisons (Figure 5, Table 5)
BASELINE_RESULTS = {
    'Lewis': {
        'Ag': 0.69,
        'Ut': 0.52,
        'SW': 0.44,
        'Len': 7.3,
    },
    'RL+Rollouts': {
        'Ag': 0.69,
        'Ut': 0.52,
        'SW': 0.44,
        'Len': 7.3,
    },
    'CHAI': {
        'Ag': 0.59,
        'Ut': 0.38,
        'SW': 0.31,
        'Len': 6.8,
    },
    'ToM(e)': {
        'Ag': 0.64,
        'Ut': 0.51,
        'SW': 0.42,
        'Len': 8.2,
    },
    'A2C': {
        'Ag': 0.62,
        'Ut': 0.43,
        'SW': 0.43,
        'Len': 8.2,
    },
}

# Human evaluation results (Table 7, Section 6.5)
HUMAN_EVAL_RESULTS = {
    'human': {
        'Ag': 0.71,
        'Ut': 0.61,
        'SW': 0.51,
        'Len': 7.2,
        'Hu': 4.90,  # Human-likeness
        'In': 4.9,   # Intelligence
        'Fl': 5.0,   # Fluency
        'Lg': 4.9,   # Logic
        'Wl': 4.8,   # Willingness to play again
    },
    'alpha_nego_conservative': {
        'Ag': 0.74,
        'Ut': 0.45,
        'SW': 0.57,
        'Len': 8.1,
        'Hu': 4.43,
        'In': 4.6,
        'Fl': 4.0,
        'Lg': 4.5,
        'Wl': 4.6,
    },
    'alpha_nego_aggressive': {
        'Ag': 0.61,
        'Ut': 0.66,
        'SW': 0.41,
        'Len': 8.3,
        'Hu': 4.30,
        'In': 4.7,
        'Fl': 3.9,
        'Lg': 4.4,
        'Wl': 4.2,
    },
    'CHAI': {
        'Ag': 0.59,
        'Ut': 0.38,
        'SW': 0.31,
        'Len': 6.8,
        'Hu': 3.55,
        'In': 3.9,
        'Fl': 3.6,
        'Lg': 3.2,
        'Wl': 3.5,
    },
}

# Performance improvements (Section 6.4)
IMPROVEMENTS = {
    'vs_baselines': {
        'agreement_rate': '+20% vs mean',
        'utility': '+29.6% vs mean',
        'score': '+147% vs mean',
    },
    'vs_best_baseline': {
        'agreement_rate': '+7.2% vs Lewis',
        'utility': '+26.9% vs Lewis',
        'social_welfare': '+29.5% vs Lewis',
    },
}


# ==================== Helper Functions ====================

def get_item_config() -> Dict:
    """
    Get item configuration for Dealornodeal
    
    Returns:
        Dictionary with item specifications
    """
    return {
        'items': ['hats', 'books', 'balls'],
        'max_counts': {
            'hats': 3,
            'books': 3,
            'balls': 3,
        },
        'value_ranges': {
            'hats': [0, 1, 2, 3],      # Possible values
            'books': [0, 1, 2, 3],
            'balls': [0, 1, 2, 3],
        },
    }


def get_utility_function_example() -> Dict:
    """
    Example utility function from paper
    
    U(ω) = Σ(w_j · V_j(v_jk))
    
    Returns:
        Example utility specification
    """
    return {
        'agent_1': {
            'weights': {
                'hats': 0.5,   # w_1 = 0.5
                'books': 0.3,  # w_2 = 0.3
                'balls': 0.2,  # w_3 = 0.2
            },
            'values': {
                'hats': [0, 1, 2, 3],    # V_1(v_1k)
                'books': [0, 1, 2, 3],   # V_2(v_2k)
                'balls': [0, 1, 2, 3],   # V_3(v_3k)
            },
        },
        'agent_2': {
            'weights': {
                'hats': 0.2,
                'books': 0.5,
                'balls': 0.3,
            },
            'values': {
                'hats': [0, 1, 2, 3],
                'books': [0, 1, 2, 3],
                'balls': [0, 1, 2, 3],
            },
        },
    }


def get_style_config(style: str) -> AlphaNegoConfig:
    """
    Get configuration for specific negotiation style
    
    Args:
        style: 'neutral', 'aggressive', or 'conservative'
    
    Returns:
        Configuration for that style
    """
    config = get_dealornodeal_config()
    config.style_control.active_style = style
    
    if style == 'aggressive':
        config.style_control.alpha_agg = 1.0
        # Adjust entropy for more aggressive exploration
        config.training.alpha_entropy_intent = 0.25
        config.training.beta_entropy_price = 0.12
    elif style == 'conservative':
        config.style_control.alpha_con = 0.2
        # Adjust entropy for more cautious exploration
        config.training.alpha_entropy_intent = 0.35
        config.training.beta_entropy_price = 0.18
    
    return config


def get_human_eval_config() -> AlphaNegoConfig:
    """
    Get configuration for human evaluation experiments
    
    Returns:
        Configuration optimized for human interaction
    """
    config = get_dealornodeal_config()
    
    # Use more natural language generation
    config.dialogue_system.generator_type = 'retrieval_based'
    config.dialogue_system.num_template_candidates = 10
    config.dialogue_system.use_response_diversity = True
    
    # Adjust for human-like behavior
    config.training.alpha_kl = 0.8  # Stronger adherence to human patterns
    
    return config


# ==================== Validation ====================

def validate_config(config: AlphaNegoConfig) -> bool:
    """
    Validate Dealornodeal configuration
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check dataset
    assert config.experiment.dataset == 'dealornodeal'
    
    # Check dialogue acts
    assert config.dialogue_system.num_dialogue_acts == 6
    assert len(config.dialogue_system.dialogue_acts) == 6
    
    # Check network dimensions
    assert config.network.policy_input_dim == 3
    assert config.network.policy_output_intent_dim == 6
    assert config.network.policy_output_price_dim == 9  # 3 items × 3 counts
    
    # Check training parameters
    assert config.training.rl_policy_lr == 1e-4
    assert config.training.rl_critic_lr == 1e-3
    assert config.training.gamma == 0.99
    
    # Check quantiles
    assert config.training.num_quantiles == 51
    
    # Check PFSP probabilities
    prob_sum = (config.self_play.prob_self_play + 
                config.self_play.prob_sl_agents + 
                config.self_play.prob_rl_agents)
    assert abs(prob_sum - 1.0) < 1e-6
    
    return True


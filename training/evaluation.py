"""
Evaluation Module
Comprehensive evaluation of trained negotiation agents

Metrics:
- Agreement rate
- Average utility
- Social welfare
- Negotiation score (Eq. 9)
- Dialogue length
- Style performance

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from collections import defaultdict

from environment.negotiation_env import NegotiationEnv
from agents.opponent_pool import compute_negotiation_score


# ==================== Evaluator ====================

class NegotiationEvaluator:
    """
    Evaluate negotiation agents
    """
    
    def __init__(
        self,
        env: NegotiationEnv,
        num_episodes: int = 100,
        verbose: bool = True,
    ):
        """
        Args:
            env: Negotiation environment
            num_episodes: Number of evaluation episodes
            verbose: Print progress
        """
        self.env = env
        self.num_episodes = num_episodes
        self.verbose = verbose
        
        print(f"[NegotiationEvaluator] Initialized")
        print(f"  Num episodes: {num_episodes}")
    
    def evaluate(
        self,
        agent,
        opponents: Optional[List] = None,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate agent against opponents
        
        Args:
            agent: Agent to evaluate
            opponents: List of opponents (uses env opponent if None)
            deterministic: Use deterministic policy
            
        Returns:
            Evaluation metrics
        """
        # Use single opponent if not provided
        if opponents is None:
            opponents = [self.env.opponent] if self.env.opponent else [None]
        
        all_results = []
        
        # Evaluate against each opponent
        for opponent in opponents:
            # Temporarily set opponent
            original_opponent = self.env.opponent
            self.env.opponent = opponent
            
            results = self._evaluate_single_opponent(agent, opponent, deterministic)
            all_results.append(results)
            
            # Restore original opponent
            self.env.opponent = original_opponent
        
        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        
        if self.verbose:
            self._print_results(aggregated)
        
        return aggregated
    
    def _evaluate_single_opponent(
        self,
        agent,
        opponent,
        deterministic: bool,
    ) -> Dict:
        """
        Evaluate against single opponent
        
        Args:
            agent: Agent to evaluate
            opponent: Opponent agent
            deterministic: Use deterministic policy
            
        Returns:
            Results dictionary
        """
        episode_results = []
        
        iterator = range(self.num_episodes)
        if self.verbose:
            iterator = tqdm(iterator, desc="Evaluating")
        
        for episode in iterator:
            obs = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Agent action
                if hasattr(agent, 'select_action'):
                    intent, price, _ = agent.select_action(
                        torch.FloatTensor(obs).to(agent.device),
                        deterministic=deterministic
                    )
                    action = {
                        'intent': intent,
                        'price': np.array([price])
                    }
                else:
                    # Random agent
                    action = self.env.action_space.sample()
                
                # Step
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
            
            # Record results
            episode_results.append({
                'reward': episode_reward,
                'agreement': info['agreement'],
                'agent_utility': info.get('agent_utility', 0.0),
                'opponent_utility': info.get('opponent_utility', 0.0),
                'social_welfare': info.get('social_welfare', 0.0),
                'dialogue_length': info['dialogue_length'],
                'final_price': info.get('final_price'),
            })
        
        return {
            'opponent': opponent.__class__.__name__ if opponent else 'None',
            'episodes': episode_results,
        }
    
    def _aggregate_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate results across opponents
        
        Args:
            all_results: List of results per opponent
            
        Returns:
            Aggregated metrics
        """
        aggregated = {
            'by_opponent': {},
            'overall': {},
        }
        
        # Aggregate per opponent
        for results in all_results:
            opponent_name = results['opponent']
            episodes = results['episodes']
            
            aggregated['by_opponent'][opponent_name] = {
                'agreement_rate': np.mean([e['agreement'] for e in episodes]),
                'avg_reward': np.mean([e['reward'] for e in episodes]),
                'avg_agent_utility': np.mean([e['agent_utility'] for e in episodes]),
                'avg_opponent_utility': np.mean([e['opponent_utility'] for e in episodes]),
                'avg_social_welfare': np.mean([e['social_welfare'] for e in episodes]),
                'avg_dialogue_length': np.mean([e['dialogue_length'] for e in episodes]),
                'std_reward': np.std([e['reward'] for e in episodes]),
                'std_utility': np.std([e['agent_utility'] for e in episodes]),
            }
            
            # Compute negotiation score (Eq. 9)
            if aggregated['by_opponent'][opponent_name]['agreement_rate'] > 0:
                score = compute_negotiation_score(
                    agreement_rate=aggregated['by_opponent'][opponent_name]['agreement_rate'],
                    utility=aggregated['by_opponent'][opponent_name]['avg_agent_utility'],
                    dialogue_length=aggregated['by_opponent'][opponent_name]['avg_dialogue_length'],
                    social_welfare=aggregated['by_opponent'][opponent_name]['avg_social_welfare'],
                )
                aggregated['by_opponent'][opponent_name]['negotiation_score'] = score
        
        # Overall aggregation
        all_episodes = []
        for results in all_results:
            all_episodes.extend(results['episodes'])
        
        aggregated['overall'] = {
            'num_episodes': len(all_episodes),
            'agreement_rate': np.mean([e['agreement'] for e in all_episodes]),
            'avg_reward': np.mean([e['reward'] for e in all_episodes]),
            'avg_agent_utility': np.mean([e['agent_utility'] for e in all_episodes]),
            'avg_opponent_utility': np.mean([e['opponent_utility'] for e in all_episodes]),
            'avg_social_welfare': np.mean([e['social_welfare'] for e in all_episodes]),
            'avg_dialogue_length': np.mean([e['dialogue_length'] for e in all_episodes]),
            'std_reward': np.std([e['reward'] for e in all_episodes]),
            'std_utility': np.std([e['agent_utility'] for e in all_episodes]),
        }
        
        # Overall negotiation score
        if aggregated['overall']['agreement_rate'] > 0:
            aggregated['overall']['negotiation_score'] = compute_negotiation_score(
                agreement_rate=aggregated['overall']['agreement_rate'],
                utility=aggregated['overall']['avg_agent_utility'],
                dialogue_length=aggregated['overall']['avg_dialogue_length'],
                social_welfare=aggregated['overall']['avg_social_welfare'],
            )
        
        return aggregated
    
    def _print_results(self, results: Dict):
        """Print evaluation results"""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        # Overall results
        overall = results['overall']
        print("\nOverall Performance:")
        print(f"  Episodes: {overall['num_episodes']}")
        print(f"  Agreement Rate: {overall['agreement_rate']:.2%}")
        print(f"  Avg Reward: {overall['avg_reward']:.4f} ± {overall['std_reward']:.4f}")
        print(f"  Avg Utility: {overall['avg_agent_utility']:.4f} ± {overall['std_utility']:.4f}")
        print(f"  Avg Social Welfare: {overall['avg_social_welfare']:.4f}")
        print(f"  Avg Dialogue Length: {overall['avg_dialogue_length']:.2f}")
        if 'negotiation_score' in overall:
            print(f"  Negotiation Score: {overall['negotiation_score']:.4f}")
        
        # Per-opponent results
        if len(results['by_opponent']) > 1:
            print("\nPer-Opponent Performance:")
            for opponent_name, metrics in results['by_opponent'].items():
                print(f"\n  vs {opponent_name}:")
                print(f"    Agreement Rate: {metrics['agreement_rate']:.2%}")
                print(f"    Avg Utility: {metrics['avg_agent_utility']:.4f}")
                print(f"    Avg Dialogue Length: {metrics['avg_dialogue_length']:.2f}")
                if 'negotiation_score' in metrics:
                    print(f"    Negotiation Score: {metrics['negotiation_score']:.4f}")
        
        print("\n" + "="*70)


# ==================== Style Evaluator ====================

class StyleEvaluator:
    """
    Evaluate different negotiation styles
    """
    
    def __init__(
        self,
        env: NegotiationEnv,
        num_episodes: int = 50,
    ):
        """
        Args:
            env: Negotiation environment
            num_episodes: Episodes per style
        """
        self.env = env
        self.num_episodes = num_episodes
        self.evaluator = NegotiationEvaluator(env, num_episodes, verbose=False)
    
    def evaluate_styles(
        self,
        agent,
        styles: List[str] = ['neutral', 'aggressive', 'conservative'],
    ) -> Dict[str, Dict]:
        """
        Evaluate agent with different styles
        
        Args:
            agent: Agent to evaluate
            styles: List of styles to test
            
        Returns:
            Results per style
        """
        results = {}
        
        print(f"\n[StyleEvaluator] Evaluating {len(styles)} styles...")
        
        for style in styles:
            print(f"\nEvaluating style: {style}")
            
            # Set style
            if hasattr(agent, 'set_style'):
                agent.set_style(style)
            
            # Evaluate
            style_results = self.evaluator.evaluate(agent, deterministic=True)
            results[style] = style_results['overall']
        
        # Print comparison
        self._print_style_comparison(results)
        
        return results
    
    def _print_style_comparison(self, results: Dict[str, Dict]):
        """Print style comparison"""
        print("\n" + "="*70)
        print("STYLE COMPARISON")
        print("="*70)
        
        print(f"\n{'Style':<15} {'Agreement':<12} {'Utility':<12} {'Length':<10} {'Score':<10}")
        print("-" * 70)
        
        for style, metrics in results.items():
            agreement = metrics['agreement_rate']
            utility = metrics['avg_agent_utility']
            length = metrics['avg_dialogue_length']
            score = metrics.get('negotiation_score', 0.0)
            
            print(f"{style:<15} {agreement:>10.2%}  {utility:>10.4f}  {length:>8.2f}  {score:>8.4f}")
        
        print("="*70)


# ==================== Opponent Evaluation ====================

def evaluate_against_baselines(
    agent,
    env: NegotiationEnv,
    num_episodes: int = 50,
) -> Dict[str, Dict]:
    """
    Evaluate agent against all baseline opponents
    
    Args:
        agent: Agent to evaluate
        env: Environment
        num_episodes: Episodes per opponent
        
    Returns:
        Results per opponent
    """
    from agents.baseline_agents import create_all_baseline_agents
    
    # Create baseline opponents
    opponents = create_all_baseline_agents(role='seller')
    
    print(f"\n[Baseline Evaluation] Testing against {len(opponents)} opponents...")
    
    # Evaluate
    evaluator = NegotiationEvaluator(env, num_episodes, verbose=False)
    results = evaluator.evaluate(agent, opponents, deterministic=True)
    
    return results


# ==================== Cross Evaluation ====================

def cross_evaluate(
    agents: List,
    env: NegotiationEnv,
    num_episodes: int = 50,
) -> Dict[str, Dict[str, float]]:
    """
    Cross-evaluate multiple agents against each other
    
    Args:
        agents: List of agents
        env: Environment
        num_episodes: Episodes per matchup
        
    Returns:
        Win matrix
    """
    n = len(agents)
    win_matrix = np.zeros((n, n))
    
    evaluator = NegotiationEvaluator(env, num_episodes, verbose=False)
    
    print(f"\n[Cross Evaluation] {n}x{n} matchups...")
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            print(f"  Agent {i} vs Agent {j}")
            
            # Agent i vs Agent j
            results = evaluator.evaluate(agents[i], [agents[j]], deterministic=True)
            
            # Win if higher utility
            avg_utility = results['overall']['avg_agent_utility']
            avg_opp_utility = results['overall']['avg_opponent_utility']
            
            if avg_utility > avg_opp_utility:
                win_matrix[i, j] = 1.0
            elif avg_utility == avg_opp_utility:
                win_matrix[i, j] = 0.5
    
    # Print matrix
    print("\nWin Matrix (row vs column):")
    print(win_matrix)
    
    return win_matrix


# ==================== Testing ====================

if __name__ == '__main__':
    print("Testing Evaluation...")
    
    # Import dependencies
    from environment.negotiation_env import make_negotiation_env
    from agents.baseline_agents import create_baseline_agent
    
    # Create environment
    print("\n1. Creating environment...")
    env = make_negotiation_env(
        dataset='craigslistbargain',
        opponent_type='rule_based',
        max_turns=20
    )
    
    # Create test agent (rule-based)
    print("\n2. Creating test agent...")
    agent = create_baseline_agent('linear', role='buyer')
    
    # Test NegotiationEvaluator
    print("\n3. Testing NegotiationEvaluator...")
    evaluator = NegotiationEvaluator(env, num_episodes=10, verbose=True)
    
    results = evaluator.evaluate(agent, deterministic=True)
    
    print(f"\n   Agreement rate: {results['overall']['agreement_rate']:.2%}")
    print(f"   Avg utility: {results['overall']['avg_agent_utility']:.4f}")
    
    # Test StyleEvaluator
    print("\n4. Testing StyleEvaluator...")
    
    # Would need actual agent with style control
    # style_evaluator = StyleEvaluator(env, num_episodes=5)
    # style_results = style_evaluator.evaluate_styles(agent)
    
    # Test baseline evaluation
    print("\n5. Testing baseline evaluation...")
    baseline_results = evaluate_against_baselines(agent, env, num_episodes=5)
    
    print(f"   Tested against {len(baseline_results['by_opponent'])} opponents")
    
    print("\nEvaluation tests passed!")
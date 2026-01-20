"""
Visualization Utilities
Plot training curves, metrics, and evaluation results

Features:
- Training curves
- Metric comparison
- Heatmaps
- Distribution plots
- Negotiation visualization

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ==================== Training Curves ====================

def plot_training_curves(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot training curves
    
    Args:
        history: Training history dictionary
        metrics: Metrics to plot (None = all)
        save_path: Path to save figure
        show: Whether to show plot
    """
    if metrics is None:
        metrics = list(history.keys())
    
    # Filter existing metrics
    metrics = [m for m in metrics if m in history]
    
    if len(metrics) == 0:
        print("[Warning] No metrics to plot")
        return
    
    # Create subplots
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = history[metric]
        epochs = range(1, len(values) + 1)
        
        ax.plot(epochs, values, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Visualization] Saved training curves to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ==================== Metric Comparison ====================

def plot_metric_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot metric comparison across agents
    
    Args:
        results: Dictionary of {agent_name: {metric: value}}
        metric: Metric to compare
        save_path: Path to save figure
        show: Whether to show plot
    """
    # Extract data
    agents = list(results.keys())
    values = [results[agent].get(metric, 0) for agent in agents]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(agents, values, alpha=0.7)
    
    # Color bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(agents)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Agent')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Visualization] Saved comparison to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ==================== Heatmap ====================

def plot_cross_evaluation_heatmap(
    win_matrix: np.ndarray,
    agent_names: List[str],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot cross-evaluation heatmap
    
    Args:
        win_matrix: Win matrix (n_agents, n_agents)
        agent_names: List of agent names
        save_path: Path to save figure
        show: Whether to show plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        win_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        xticklabels=agent_names,
        yticklabels=agent_names,
        ax=ax,
        cbar_kws={'label': 'Win Rate'},
        vmin=0,
        vmax=1,
    )
    
    ax.set_xlabel('Opponent')
    ax.set_ylabel('Agent')
    ax.set_title('Cross-Evaluation Win Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Visualization] Saved heatmap to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ==================== Distribution Plot ====================

def plot_utility_distribution(
    utilities: List[float],
    bins: int = 20,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot utility distribution
    
    Args:
        utilities: List of utility values
        bins: Number of bins
        save_path: Path to save figure
        show: Whether to show plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(utilities, bins=bins, alpha=0.7, edgecolor='black')
    
    # Add mean line
    mean_utility = np.mean(utilities)
    ax.axvline(mean_utility, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_utility:.3f}')
    
    ax.set_xlabel('Utility')
    ax.set_ylabel('Frequency')
    ax.set_title('Utility Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Visualization] Saved distribution to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ==================== Scatter Plot ====================

def plot_utility_scatter(
    agent_utilities: List[float],
    opponent_utilities: List[float],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot agent vs opponent utilities
    
    Args:
        agent_utilities: Agent utilities
        opponent_utilities: Opponent utilities
        save_path: Path to save figure
        show: Whether to show plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(agent_utilities, opponent_utilities, alpha=0.5, s=50)
    
    # Add diagonal line (equal utilities)
    max_val = max(max(agent_utilities), max(opponent_utilities))
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal Utility')
    
    ax.set_xlabel('Agent Utility')
    ax.set_ylabel('Opponent Utility')
    ax.set_title('Agent vs Opponent Utilities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Visualization] Saved scatter plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ==================== Style Comparison ====================

def plot_style_comparison(
    style_results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot style comparison
    
    Args:
        style_results: Dictionary of {style: {metric: value}}
        metrics: Metrics to compare
        save_path: Path to save figure
        show: Whether to show plot
    """
    if metrics is None:
        # Use common metrics
        metrics = ['agreement_rate', 'avg_agent_utility', 'avg_dialogue_length']
    
    styles = list(style_results.keys())
    n_metrics = len(metrics)
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_metrics)
    width = 0.25
    
    for idx, style in enumerate(styles):
        values = [style_results[style].get(m, 0) for m in metrics]
        offset = (idx - len(styles)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=style.capitalize(), alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title('Style Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Visualization] Saved style comparison to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ==================== Dialogue Visualization ====================

def plot_negotiation_trajectory(
    price_history: List[Tuple[float, float]],
    listing_price: float,
    final_price: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot negotiation price trajectory
    
    Args:
        price_history: List of (agent_price, opponent_price) tuples
        listing_price: Listing price
        final_price: Final agreed price
        save_path: Path to save figure
        show: Whether to show plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if len(price_history) == 0:
        print("[Warning] No price history to plot")
        return
    
    agent_prices = [p[0] for p in price_history if p[0] is not None]
    opponent_prices = [p[1] for p in price_history if p[1] is not None]
    
    # Plot prices
    if agent_prices:
        ax.plot(range(len(agent_prices)), agent_prices, 
                marker='o', label='Agent', linewidth=2)
    
    if opponent_prices:
        ax.plot(range(len(opponent_prices)), opponent_prices,
                marker='s', label='Opponent', linewidth=2)
    
    # Add listing price line
    ax.axhline(listing_price, color='gray', linestyle='--', 
               alpha=0.5, label='Listing Price')
    
    # Add final price
    if final_price is not None:
        ax.axhline(final_price, color='green', linestyle='-',
                   alpha=0.7, label='Final Price', linewidth=2)
    
    ax.set_xlabel('Turn')
    ax.set_ylabel('Price')
    ax.set_title('Negotiation Price Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Visualization] Saved trajectory to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ==================== Comprehensive Dashboard ====================

def create_training_dashboard(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Create comprehensive training dashboard
    
    Args:
        history: Training history
        save_path: Path to save figure
        show: Whether to show plot
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Training loss
    if 'train_loss' in history:
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(history['train_loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.grid(True, alpha=0.3)
    
    # 2. Validation loss
    if 'val_loss' in history:
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(history['val_loss'])
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.grid(True, alpha=0.3)
    
    # 3. Agreement rate
    if 'agreement_rate' in history:
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(history['agreement_rate'])
        ax3.set_title('Agreement Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Rate')
        ax3.grid(True, alpha=0.3)
    
    # 4. Utility
    if 'avg_utility' in history:
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(history['avg_utility'])
        ax4.set_title('Average Utility')
        ax4.set_xlabel('Epoch')
        ax4.grid(True, alpha=0.3)
    
    # 5. Dialogue length
    if 'avg_dialogue_length' in history:
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(history['avg_dialogue_length'])
        ax5.set_title('Dialogue Length')
        ax5.set_xlabel('Epoch')
        ax5.grid(True, alpha=0.3)
    
    # 6. Reward
    if 'avg_reward' in history:
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(history['avg_reward'])
        ax6.set_title('Average Reward')
        ax6.set_xlabel('Epoch')
        ax6.grid(True, alpha=0.3)
    
    # 7. Policy loss
    if 'policy_loss' in history:
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(history['policy_loss'])
        ax7.set_title('Policy Loss')
        ax7.set_xlabel('Epoch')
        ax7.grid(True, alpha=0.3)
    
    # 8. Critic loss
    if 'critic_loss' in history:
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(history['critic_loss'])
        ax8.set_title('Critic Loss')
        ax8.set_xlabel('Epoch')
        ax8.grid(True, alpha=0.3)
    
    # 9. Summary text
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = "Training Summary\n\n"
    summary_text += f"Total Epochs: {len(history.get('train_loss', []))}\n"
    if 'agreement_rate' in history:
        summary_text += f"Final Agreement: {history['agreement_rate'][-1]:.2%}\n"
    if 'avg_utility' in history:
        summary_text += f"Final Utility: {history['avg_utility'][-1]:.4f}\n"
    
    ax9.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
    
    fig.suptitle('Training Dashboard', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Visualization] Saved dashboard to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


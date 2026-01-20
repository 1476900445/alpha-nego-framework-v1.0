# α-Nego: Self-play Deep Reinforcement Learning for Negotiation Dialogues

**Production-Ready Implementation of the α-Nego Framework**

## Overview

This repository offers a fully validated implementation of the α-Nego framework, as presented in the paper *α-Nego: Self-play Deep Reinforcement Learning for Negotiation Dialogues*.

Negotiation is an inherently complex social process that demands strategic reasoning and effective communication. Traditional negotiation models often struggle with poor generalization due to fixed opponent training, inadequate return distribution estimation (relying solely on expected Q-values), and a lack of adaptive strategies for diverse scenarios.

The α-Nego framework resolves these bottlenecks through three core innovations, delivering state-of-the-art performance across multiple negotiation tasks.

## Core Innovations

### 1. Self-Play Reinforcement Learning Framework
- **Warm-start Mechanism**: Policy initialization via supervised learning on human negotiation datasets, laying a solid foundation for subsequent reinforcement learning.
- **Priority Fictitious Self-Play (PFSP)**: Dynamically selects challenging opponents from the pool to enhance training robustness and strategy diversity.
- **Opponent Pool Management**: Continuously integrates high-performance strategies into the opponent pool, iteratively boosting training difficulty and agent adaptability.

### 2. Distributional Reinforcement Learning (DSAC)
- **Value Distribution Modeling**: Employs 51 quantiles to capture the full distribution of negotiation returns, rather than merely estimating expected values.
- **Quantile Regression Optimization**: Utilizes Huber loss (κ=1.0) to refine quantile estimation, enhancing stability and accuracy of value predictions.
- **Dual Critic Networks**: Adopts the minimum Q-value from two parallel critic networks to mitigate overestimation bias, a common pitfall in reinforcement learning.

### 3. Style-Controllable Negotiation Strategies
- **Neutral Style (α = 0.5)**: Optimizes for expected returns, striking a balance between agreement rate and utility.
- **Aggressive Style (α = 0.9)**: Focuses on the upper tail of the return distribution to pursue high utility, with a trade-off in lower agreement rates.
- **Conservative Style (α = 0.1)**: Prioritizes high agreement rates using Conditional Value at Risk (CVaR) on the lower tail of returns, ensuring reliable performance.

## Key Features

- Full-fledged implementation of the α-Nego training pipeline and DSAC algorithm, rigorously verified to align with the paper's specifications.
- Advanced neural network architecture integrated with attention mechanisms, residual connections, and layer normalization for enhanced feature extraction.
- End-to-end dialogue system, including a rule-based parser, strategy manager, and retrieval-based response generator, enabling seamless negotiation interactions.
- Native support for multiple benchmark datasets, such as Craigslistbargain and DealOrNoDeal, facilitating direct performance comparison.
- Comprehensive evaluation metrics and visualization tools to quantify agent performance and analyze negotiation dynamics.

## Quick Start

### 1. Environment Preparation

```bash
# Navigate to the local project root directory
cd alpha-nego-framework

# Create a virtual environment to isolate dependencies
python -m venv venv

# Activate the virtual environment
# For Linux/macOS
source venv/bin/activate
# For Windows
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install core required packages
pip install -r requirements.txt

# Install NLP toolkit for dialogue processing
python -m spacy download en_core_web_sm
```

## Training

### Basic Training Commands

**Note**: All command-line arguments follow Python's argparse convention, using hyphens instead of underscores for consistency.

```bash
# Train a neutral-style agent (1100 epochs, full training cycle)
python scripts/train.py --dataset craigslistbargain --epochs 1100 --batch-size 32 --learning-rate 1e-4 --style neutral

# Train an aggressive-style agent
python scripts/train.py --dataset craigslistbargain --epochs 1100 --style aggressive

# Train a conservative-style agent
python scripts/train.py --dataset craigslistbargain --epochs 1100 --style conservative
```

### Multi-Stage Training

As specified in the paper, the training process is divided into three stages to progressively refine the agent's performance:

```bash
# Stage 1: Early-stage training (100 epochs)
python scripts/train.py --epochs 100 --save-dir checkpoints/stage1/

# Stage 2: Mid-stage training (500 epochs)
python scripts/train.py --epochs 500 --save-dir checkpoints/stage2/

# Stage 3: Final-stage training (1100 epochs)
python scripts/train.py --epochs 1100 --save-dir checkpoints/stage3/
```

### Multi-GPU Training

```bash
# Specify GPUs and run training with expanded batch size
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --batch-size 128 --epochs 1100
```

## Evaluation

### Evaluate Trained Models

```bash
# Evaluate the best model with 200 test episodes
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --num-episodes 200
```

## Interactive Negotiation

Test the trained agent through real-time interactive dialogue:

```bash
python scripts/interactive.py --checkpoint checkpoints/best_model.pt
```

## Training Monitoring

```bash
# Monitor training progress with TensorBoard
tensorboard --logdir logs/

# Monitor with Weights & Biases (requires a W&B account)
python scripts/train.py --use-wandb --wandb-project alpha-nego
```

---


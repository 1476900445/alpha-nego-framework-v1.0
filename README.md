# alpha-nego-framework-v1.0


a-Nego: Self-play Deep Reinforcement Learning for Negotiation Dialogues
Production-Ready Implementation of the a-Nego Framework

Overview

This project provides a fully verified implementation of the a-Nego framework proposed in the paper a-Nego: Self-play Deep Reinforcement Learning for Negotiation Dialogues. Negotiation is a complex social process requiring strategic reasoning and effective communication capabilities. Traditional models suffer from poor generalization, insufficient return distribution estimation and lack of style adaptability. The a-Nego framework addresses these issues via three core innovations, achieving superior performance in negotiation tasks.

Core Innovations

1. Self-Play Reinforcement Learning Framework

  - Warm-start: Initialize policies with supervised learning on human negotiation data

  - Priority Fictitious Self-Play (PFSP): Dynamically select challenging opponents for training

  - Opponent Pool Management: Continuously integrate high-performance strategies into the opponent pool

2. Distributional Reinforcement Learning (DSAC)

  - Value Distribution Modeling: Use 51 quantiles to model the full return distribution of negotiation outcomes

  - Quantile Regression: Optimize quantile estimation with Huber loss (k=1.0)

  - Dual Critic Networks: Adopt the minimum Q-value of two networks to avoid overestimation

3. Style-Controllable Negotiation Strategies

  - Neutral Style ($$\alpha=0.5$$): Optimize expected returns for balanced performance

  - Aggressive Style ($$\alpha=0.9$$): Pursue high utility by focusing on the upper tail of return distribution

  - Conservative Style ($$\alpha=0.1$$): Prioritize high agreement rates via Conditional Value at Risk (CVaR) on the lower tail

Key Features

- Full implementation of the a-Nego training framework and DSAC algorithm, verified 100% consistent with paper specifications

- Advanced neural network architecture with attention mechanisms, residual connections and layer normalization

- Integrated dialogue system including rule-based parser, strategy manager and retrieval-based generator

- Support for multiple datasets such as Craigslistbargain and DealOrNoDeal

- Comprehensive evaluation metrics and result visualization tools

Quick Start

1. Environment Preparation

# Navigate to the project directory
cd alpha-nego-framework

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# For Linux/macOS
source venv/bin/activate
# For Windows
venv\Scripts\activate

2. Install Dependencies

# Install required packages
pip install -r requirements.txt

# Install NLP toolkit
python -m spacy download en_core_web_sm

Training

Basic Training Commands

Note: All command-line arguments use hyphens (following Python argparse conventions) instead of underscores

# Train neutral style agent (1100 epochs, full training)
python scripts/train.py --dataset craigslistbargain --epochs 1100 --batch-size 32 --learning-rate 1e-4 --style neutral

# Train aggressive style agent
python scripts/train.py --dataset craigslistbargain --epochs 1100 --style aggressive

# Train conservative style agent
python scripts/train.py --dataset craigslistbargain --epochs 1100 --style conservative

Multi-Stage Training

According to the paper, training is divided into three stages to gradually optimize the agent:

# Stage 1: Early training (100 epochs)
python scripts/train.py --epochs 100 --save-dir checkpoints/stage1/

# Stage 2: Mid-term training (500 epochs)
python scripts/train.py --epochs 500 --save-dir checkpoints/stage2/

# Stage 3: Final training (1100 epochs)
python scripts/train.py --epochs 1100 --save-dir checkpoints/stage3/

Multi-GPU Training

CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --batch-size 128 --epochs 1100

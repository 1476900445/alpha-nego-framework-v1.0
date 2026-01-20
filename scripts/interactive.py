#!/usr/bin/env python3
"""
Interactive Negotiation Script
Negotiate with a trained α-Nego agent interactively

Usage:
    python scripts/interactive.py --checkpoint checkpoints/best_model.pt
    python scripts/interactive.py --checkpoint checkpoints/best_model.pt --style aggressive

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from configs.craigslist_config import get_craigslist_config
from dialogue.parser import DialogueParser
from dialogue.generator import DialogueGenerator
from dialogue.manager import DialogueManager
from utils.checkpoint import ModelSaver
from algorithms.dsac import create_dsac_agent


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Interactive negotiation')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='craigslistbargain',
                        choices=['craigslistbargain', 'dealornodeal'],
                        help='Dataset')
    
    # Negotiation
    parser.add_argument('--style', type=str, default='neutral',
                        choices=['neutral', 'aggressive', 'conservative'],
                        help='Agent negotiation style')
    parser.add_argument('--max-turns', type=int, default=20,
                        help='Maximum dialogue turns')
    parser.add_argument('--listing-price', type=float, default=None,
                        help='Listing price (random if not specified)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    
    return parser.parse_args()


class InteractiveNegotiation:
    """
    Interactive negotiation session
    """
    
    def __init__(
        self,
        agent,
        config,
        style: str = 'neutral',
        max_turns: int = 20,
        listing_price: float = None,
    ):
        """
        Args:
            agent: Trained agent
            config: Configuration
            style: Negotiation style
            max_turns: Maximum turns
            listing_price: Listing price
        """
        self.agent = agent
        self.config = config
        self.style = style
        self.max_turns = max_turns
        
        # Set agent style
        if hasattr(agent, 'set_style'):
            agent.set_style(style)
        
        # Setup dialogue components
        self.parser = DialogueParser(config.dataset.name)
        self.generator = DialogueGenerator(config.dataset.name, mode='context_aware')
        
        # Random listing price if not provided
        if listing_price is None:
            listing_price = np.random.uniform(50, 200)
        
        self.listing_price = listing_price
        
        # Create dialogue manager
        self.manager = DialogueManager(
            dataset=config.dataset.name,
            max_turns=max_turns,
            listing_price=listing_price,
            agent_target=listing_price * 0.7,  # Agent wants to pay 70%
            opponent_target=listing_price * 0.9,  # User wants 90%
        )
        
        print(f"\n[Negotiation] Starting interactive session")
        print(f"  Listing price: ${listing_price:.2f}")
        print(f"  Agent style: {style}")
        print(f"  Max turns: {max_turns}")
    
    def run(self):
        """Run interactive negotiation"""
        print("\n" + "="*70)
        print("INTERACTIVE NEGOTIATION")
        print("="*70)
        print(f"\nYou are negotiating to buy an item listed at ${self.listing_price:.2f}")
        print("The agent is the seller.")
        print("\nCommands:")
        print("  - Type your message naturally")
        print("  - Type 'quit' to end negotiation")
        print("  - Type 'accept' to accept the current offer")
        print("  - Type 'reject' to reject and end negotiation")
        print("\n" + "="*70 + "\n")
        
        # Agent starts
        print("Agent: Hello! I have this item for sale.")
        
        turn = 0
        done = False
        
        while not done and turn < self.max_turns:
            # User input
            print("\n" + "-"*70)
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Check for commands
            if user_input.lower() == 'quit':
                print("\n[Negotiation] You ended the negotiation.")
                break
            
            # Parse user input
            parsed = self.parser.parse(user_input)
            
            print(f"[Debug] Detected intent: {parsed['intent_name']}")
            if parsed.get('price'):
                print(f"[Debug] Detected price: ${parsed['price']:.2f}")
            
            # User's turn
            info_user, done_user = self.manager.step(
                speaker='opponent',
                intent_id=parsed['intent_id'],
                price=parsed.get('price'),
                utterance=user_input,
            )
            
            if done_user:
                self._handle_end(info_user)
                break
            
            # Agent's turn
            state = self.manager.get_state()
            intent, price, _ = self.agent.select_action(
                torch.FloatTensor(state).to(self.agent.device),
                deterministic=True
            )
            
            # Denormalize price
            if price is not None:
                price = price * self.listing_price
            
            # Generate agent response
            agent_utterance = self.generator.generate(intent, price)
            
            # Execute agent action
            info_agent, done_agent = self.manager.step(
                speaker='agent',
                intent_id=intent,
                price=price,
                utterance=agent_utterance,
            )
            
            print(f"\nAgent: {agent_utterance}")
            
            if done_agent:
                self._handle_end(info_agent)
                break
            
            turn += 1
        
        if turn >= self.max_turns:
            print("\n[Negotiation] Maximum turns reached. No deal.")
        
        # Show summary
        self._print_summary()
    
    def _handle_end(self, info):
        """Handle negotiation end"""
        if info['agreement']:
            print("\n" + "="*70)
            print("🎉 DEAL REACHED!")
            print("="*70)
            print(f"Final price: ${info['final_price']:.2f}")
            
            # Calculate utilities
            agent_util, opponent_util = self.manager.compute_utilities()
            print(f"Your utility: {opponent_util:.4f}")
            print(f"Agent utility: {agent_util:.4f}")
            print(f"Social welfare: {agent_util + opponent_util:.4f}")
        else:
            print("\n" + "="*70)
            print("❌ NO DEAL")
            print("="*70)
            print("Negotiation ended without agreement.")
    
    def _print_summary(self):
        """Print negotiation summary"""
        summary = self.manager.get_summary()
        
        print("\n" + "="*70)
        print("NEGOTIATION SUMMARY")
        print("="*70)
        print(f"Turns: {summary['num_turns']}")
        print(f"Agreement: {'Yes' if summary['agreement'] else 'No'}")
        if summary['final_price']:
            print(f"Final price: ${summary['final_price']:.2f}")
            discount = (self.listing_price - summary['final_price']) / self.listing_price * 100
            print(f"Discount: {discount:.1f}%")
        print("="*70)


def main():
    """Main function"""
    args = parse_args()
    
    print("="*70)
    print("α-NEGO INTERACTIVE NEGOTIATION")
    print("="*70)
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Get config
    if args.dataset == 'craigslistbargain':
        config = get_craigslist_config()
    else:
        from configs.dealornodeal_config import get_dealornodeal_config
        config = get_dealornodeal_config()
    
    # Load agent
    print(f"\n[Loading] Checkpoint: {args.checkpoint}")
    agent = create_dsac_agent(config, device=device)
    ModelSaver.load_agent(agent, args.checkpoint, device)
    print("[Loading] Agent loaded successfully")
    
    # Create interactive session
    session = InteractiveNegotiation(
        agent=agent,
        config=config,
        style=args.style,
        max_turns=args.max_turns,
        listing_price=args.listing_price,
    )
    
    # Run negotiation
    try:
        session.run()
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Negotiation interrupted by user.")
    
    print("\n[Complete] Session ended.")


if __name__ == '__main__':
    main()
"""
Dialogue Generator
Generates natural language utterances from dialogue acts

Features:
- Template-based generation
- Price/item insertion
- Context-aware generation
- Multiple variations

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import random
from typing import Dict, List, Optional
from dialogue.dialogue_acts import get_intent_name, get_metadata


# ==================== Template Generator ====================

class TemplateGenerator:
    """
    Template-based natural language generation
    """
    
    # Craigslistbargain templates
    CRAIGSLIST_TEMPLATES = {
        'greet': [
            "Hello!",
            "Hi there!",
            "Hey, how are you?",
            "Greetings!",
        ],
        'inquire': [
            "Is this item still available?",
            "What's the condition of the item?",
            "Can you tell me more about it?",
            "When can I pick it up?",
            "Do you have any pictures?",
        ],
        'inform': [
            "The item is in good condition.",
            "It's been well maintained.",
            "Here's more information about it.",
            "I can provide additional details.",
        ],
        'init-price': [
            "I'm asking ${price} for this.",
            "The price is ${price}.",
            "I'm selling it for ${price}.",
            "It's listed at ${price}.",
        ],
        'insist-price': [
            "I'm firm at ${price}.",
            "My price of ${price} is non-negotiable.",
            "${price} is the lowest I can go.",
            "I can't budge on ${price}.",
        ],
        'agree-price': [
            "Sounds good at ${price}!",
            "${price} works for me.",
            "I agree to ${price}.",
            "Perfect, ${price} it is!",
        ],
        'concede-price': [
            "How about ${price}?",
            "I could do ${price}.",
            "What if we say ${price}?",
            "I'm willing to go down to ${price}.",
            "Could you meet me at ${price}?",
        ],
        'final-price': [
            "My final offer is ${price}.",
            "${price} is the best I can do.",
            "Last offer: ${price}, take it or leave it.",
            "I can't go lower than ${price}.",
        ],
        'counter-no-price': [
            "That's too high.",
            "That's too low for me.",
            "I was thinking of something lower.",
            "That's not quite what I had in mind.",
        ],
        'hesitant': [
            "Hmm, I'm not sure...",
            "Let me think about it.",
            "I need to consider this.",
            "Maybe...",
        ],
        'positive': [
            "Great!",
            "Excellent!",
            "That sounds wonderful!",
            "I'm very interested!",
            "Perfect!",
        ],
        'negative': [
            "No thanks.",
            "I'm not interested.",
            "That won't work for me.",
            "Sorry, I'll pass.",
        ],
        'offer': [
            "I'll offer you ${price}.",
            "How does ${price} sound?",
            "Would you take ${price}?",
            "My offer is ${price}.",
        ],
        'accept': [
            "I accept!",
            "Deal!",
            "You've got a deal!",
            "I'll take it!",
            "Agreed!",
        ],
        'reject': [
            "I have to decline.",
            "Sorry, I can't accept that.",
            "I'm going to pass.",
            "No deal.",
        ],
        'quit': [
            "Never mind, I'm not interested anymore.",
            "I think I'll look elsewhere.",
            "Thanks anyway, goodbye.",
            "I'll pass on this one.",
        ],
    }
    
    # Dealornodeal templates
    DEALORNODEAL_TEMPLATES = {
        'greet': [
            "Hello!",
            "Hi there!",
            "Hey!",
        ],
        'disagree': [
            "I disagree with that.",
            "That's not fair.",
            "I can't accept that allocation.",
            "No, that doesn't work for me.",
        ],
        'agree': [
            "I agree!",
            "That's a fair deal.",
            "Sounds good to me!",
            "Deal!",
        ],
        'insist': [
            "I really need {item_desc}.",
            "I must have {item_desc}.",
            "{item_desc} are important to me.",
            "I insist on {item_desc}.",
        ],
        'inquire': [
            "What do you value most?",
            "How many {item} do you want?",
            "Which items are most important to you?",
            "What's your preference?",
        ],
        'propose': [
            "How about {proposal}?",
            "What if I take {proposal}?",
            "I propose {proposal}.",
            "Could we do {proposal}?",
        ],
    }
    
    def __init__(self, dataset: str = 'craigslistbargain'):
        """
        Args:
            dataset: 'craigslistbargain' or 'dealornodeal'
        """
        self.dataset = dataset
        
        if dataset == 'craigslistbargain':
            self.templates = self.CRAIGSLIST_TEMPLATES
        elif dataset == 'dealornodeal':
            self.templates = self.DEALORNODEAL_TEMPLATES
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    def generate(
        self,
        intent_id: int,
        price: Optional[float] = None,
        items: Optional[Dict[str, int]] = None,
    ) -> str:
        """
        Generate utterance from intent
        
        Args:
            intent_id: Intent ID
            price: Price value (for Craigslistbargain)
            items: Item allocation (for Dealornodeal)
            
        Returns:
            Generated text
        """
        intent_name = get_intent_name(intent_id, self.dataset)
        
        if intent_name not in self.templates:
            return f"[Intent: {intent_name}]"
        
        # Get random template
        template = random.choice(self.templates[intent_name])
        
        # Fill in price
        if '${price}' in template:
            if price is not None:
                price_str = f"{price:.0f}" if price % 1 == 0 else f"{price:.2f}"
                template = template.replace('${price}', price_str)
            else:
                template = template.replace('${price}', '??')
        
        # Fill in items (for Dealornodeal)
        if '{item_desc}' in template or '{proposal}' in template:
            if items is not None:
                item_desc = self._format_items(items)
                template = template.replace('{item_desc}', item_desc)
                template = template.replace('{proposal}', item_desc)
            else:
                template = template.replace('{item_desc}', 'the items')
                template = template.replace('{proposal}', 'this split')
        
        if '{item}' in template:
            # Pick random item
            template = template.replace('{item}', random.choice(['hats', 'books', 'balls']))
        
        return template
    
    def _format_items(self, items: Dict[str, int]) -> str:
        """
        Format item allocation as text
        
        Args:
            items: Dict mapping item to count
            
        Returns:
            Formatted string
        """
        parts = []
        
        for item, count in items.items():
            if count > 0:
                parts.append(f"{count} {item}")
        
        if len(parts) == 0:
            return "nothing"
        elif len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        else:
            return f"{', '.join(parts[:-1])}, and {parts[-1]}"
    
    def generate_multiple(
        self,
        intent_id: int,
        price: Optional[float] = None,
        items: Optional[Dict[str, int]] = None,
        num_variations: int = 3,
    ) -> List[str]:
        """
        Generate multiple variations
        
        Args:
            intent_id: Intent ID
            price: Price value
            items: Item allocation
            num_variations: Number of variations
            
        Returns:
            List of generated texts
        """
        return [
            self.generate(intent_id, price, items)
            for _ in range(num_variations)
        ]


# ==================== Context-Aware Generator ====================

class ContextAwareGenerator:
    """
    Generate utterances considering dialogue context
    """
    
    def __init__(self, dataset: str = 'craigslistbargain'):
        """
        Args:
            dataset: 'craigslistbargain' or 'dealornodeal'
        """
        self.dataset = dataset
        self.template_generator = TemplateGenerator(dataset)
        self.history = []
    
    def reset(self):
        """Reset dialogue history"""
        self.history = []
    
    def generate(
        self,
        intent_id: int,
        price: Optional[float] = None,
        items: Optional[Dict[str, int]] = None,
        opponent_last_utterance: Optional[str] = None,
    ) -> str:
        """
        Generate utterance with context
        
        Args:
            intent_id: Intent ID
            price: Price value
            items: Item allocation
            opponent_last_utterance: Opponent's last message
            
        Returns:
            Generated text
        """
        # Generate base utterance
        utterance = self.template_generator.generate(intent_id, price, items)
        
        # Add context
        if len(self.history) == 0:
            # First utterance - add greeting
            if intent_id != 0:  # Not already a greeting
                utterance = f"Hi! {utterance}"
        
        # Add to history
        self.history.append({
            'intent_id': intent_id,
            'utterance': utterance,
            'price': price,
            'items': items,
        })
        
        return utterance
    
    def generate_response(
        self,
        intent_id: int,
        price: Optional[float] = None,
        opponent_last_price: Optional[float] = None,
    ) -> str:
        """
        Generate response considering opponent's last price
        
        Args:
            intent_id: Intent ID
            price: My price
            opponent_last_price: Opponent's last price
            
        Returns:
            Generated text
        """
        intent_name = get_intent_name(intent_id, self.dataset)
        
        # Context-aware templates
        if intent_name == 'concede-price' and opponent_last_price is not None and price is not None:
            if abs(price - opponent_last_price) < 5:
                return f"Alright, I can meet you at ${price:.0f}."
            else:
                return f"I'm willing to come down to ${price:.0f}."
        
        elif intent_name == 'counter-no-price' and opponent_last_price is not None:
            if price is not None and opponent_last_price < price:
                return f"${opponent_last_price:.0f} is too low for me."
            elif price is not None and opponent_last_price > price:
                return f"${opponent_last_price:.0f} is more than I can pay."
            else:
                return "That price doesn't work for me."
        
        # Default to template generator
        return self.template_generator.generate(intent_id, price)


# ==================== Neural Generator (Placeholder) ====================

class NeuralGenerator:
    """
    Neural generation (placeholder for future implementation)
    
    Could use GPT-2, T5, or other language models
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path to pretrained model
        """
        self.model_path = model_path
        # In practice, would load a pretrained model here
        print("[NeuralGenerator] Placeholder - using template generation")
        self.fallback = TemplateGenerator()
    
    def generate(
        self,
        intent_id: int,
        price: Optional[float] = None,
        items: Optional[Dict[str, int]] = None,
        context: Optional[List[str]] = None,
    ) -> str:
        """
        Generate with neural model
        
        Args:
            intent_id: Intent ID
            price: Price value
            items: Item allocation
            context: Previous utterances
            
        Returns:
            Generated text
        """
        # TODO: Implement neural generation
        # For now, fallback to templates
        return self.fallback.generate(intent_id, price, items)


# ==================== Dialogue Generator ====================

class DialogueGenerator:
    """
    Main dialogue generator
    
    Combines different generation strategies
    """
    
    def __init__(
        self,
        dataset: str = 'craigslistbargain',
        mode: str = 'template',
    ):
        """
        Args:
            dataset: 'craigslistbargain' or 'dealornodeal'
            mode: 'template', 'context_aware', or 'neural'
        """
        self.dataset = dataset
        self.mode = mode
        
        if mode == 'template':
            self.generator = TemplateGenerator(dataset)
        elif mode == 'context_aware':
            self.generator = ContextAwareGenerator(dataset)
        elif mode == 'neural':
            self.generator = NeuralGenerator()
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def generate(
        self,
        intent_id: int,
        price: Optional[float] = None,
        items: Optional[Dict[str, int]] = None,
        **kwargs
    ) -> str:
        """
        Generate utterance
        
        Args:
            intent_id: Intent ID
            price: Price value
            items: Item allocation
            **kwargs: Additional arguments
            
        Returns:
            Generated text
        """
        return self.generator.generate(intent_id, price, items, **kwargs)
    
    def generate_batch(
        self,
        intents: List[int],
        prices: Optional[List[float]] = None,
        items_list: Optional[List[Dict]] = None,
    ) -> List[str]:
        """
        Generate multiple utterances
        
        Args:
            intents: List of intent IDs
            prices: List of prices
            items_list: List of item allocations
            
        Returns:
            List of generated texts
        """
        results = []
        
        for i, intent_id in enumerate(intents):
            price = prices[i] if prices else None
            items = items_list[i] if items_list else None
            
            text = self.generate(intent_id, price, items)
            results.append(text)
        
        return results


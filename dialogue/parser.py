"""
Dialogue Parser
Parses natural language utterances into structured dialogue acts

Features:
- Intent classification
- Price extraction
- Item extraction
- Entity recognition

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from dialogue.dialogue_acts import (
    get_intent_id, get_intent_name, get_num_intents,
    requires_price, requires_items
)


# ==================== Price Parser ====================

class PriceParser:
    """
    Extract prices from natural language
    """
    
    # Price patterns
    PATTERNS = [
        r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',  # $100, $1,000.00
        r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*dollars?',  # 100 dollars
        r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*bucks?',  # 100 bucks
        r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*usd',  # 100 USD
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.PATTERNS]
    
    def extract_price(self, text: str) -> Optional[float]:
        """
        Extract price from text
        
        Args:
            text: Input text
            
        Returns:
            Price value or None
        """
        for pattern in self.compiled_patterns:
            match = pattern.search(text)
            if match:
                price_str = match.group(1).replace(',', '')
                try:
                    return float(price_str)
                except ValueError:
                    continue
        
        return None
    
    def extract_all_prices(self, text: str) -> List[float]:
        """
        Extract all prices from text
        
        Args:
            text: Input text
            
        Returns:
            List of prices
        """
        prices = []
        for pattern in self.compiled_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                price_str = match.group(1).replace(',', '')
                try:
                    prices.append(float(price_str))
                except ValueError:
                    continue
        
        return prices


# ==================== Intent Parser ====================

class IntentParser:
    """
    Classify intent from natural language
    
    Uses keyword-based classification
    """
    
    # Intent keywords
    CRAIGSLIST_KEYWORDS = {
        'greet': ['hello', 'hi', 'hey', 'greetings'],
        'inquire': ['what', 'when', 'where', 'why', 'how', 'is it', 'do you', 'can you', '?'],
        'inform': ['it is', 'this is', 'there are', 'here is', 'the condition'],
        'init-price': ['asking', 'want', 'price is', 'listed at', 'selling for'],
        'insist-price': ['firm', 'not budging', 'final offer', 'can\'t go', 'lowest'],
        'agree-price': ['sounds good', 'deal', 'agreed', 'perfect', 'ok'],
        'concede-price': ['how about', 'what if', 'could you', 'can do', 'willing to'],
        'final-price': ['last offer', 'final', 'best I can do', 'take it or leave'],
        'counter-no-price': ['too high', 'too low', 'not enough', 'way too'],
        'hesitant': ['hmm', 'not sure', 'maybe', 'thinking', 'considering'],
        'positive': ['great', 'awesome', 'excellent', 'wonderful', 'interested'],
        'negative': ['no', 'not interested', 'pass', 'sorry'],
        'offer': ['offer', 'propose', 'suggest', 'how does'],
        'accept': ['accept', 'yes', 'agree', 'deal', "i'll take"],
        'reject': ['reject', 'decline', 'no thanks', 'not interested'],
        'quit': ['bye', 'goodbye', 'never mind', 'forget it', 'not interested anymore'],
    }
    
    DEALORNODEAL_KEYWORDS = {
        'greet': ['hello', 'hi', 'hey', 'greetings'],
        'disagree': ['no', 'disagree', 'not fair', 'not acceptable'],
        'agree': ['yes', 'agree', 'deal', 'sounds good', 'ok'],
        'insist': ['need', 'must have', 'require', 'important'],
        'inquire': ['what', 'how many', 'which', '?'],
        'propose': ['how about', 'suggest', 'propose', 'what if', 'could we'],
    }
    
    def __init__(self, dataset: str = 'craigslistbargain'):
        """
        Args:
            dataset: 'craigslistbargain' or 'dealornodeal'
        """
        self.dataset = dataset
        
        if dataset == 'craigslistbargain':
            self.keywords = self.CRAIGSLIST_KEYWORDS
        elif dataset == 'dealornodeal':
            self.keywords = self.DEALORNODEAL_KEYWORDS
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    def classify_intent(self, text: str) -> int:
        """
        Classify intent from text
        
        Args:
            text: Input text
            
        Returns:
            Intent ID
        """
        text = text.lower()
        
        # Score each intent
        scores = {}
        
        for intent_name, keywords in self.keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[intent_name] = score
        
        if not scores:
            # Default intent
            if self.dataset == 'craigslistbargain':
                return 2  # inform
            else:
                return 5  # propose
        
        # Get intent with highest score
        best_intent = max(scores, key=scores.get)
        return get_intent_id(best_intent, self.dataset)
    
    def classify_with_confidence(self, text: str) -> Tuple[int, float]:
        """
        Classify intent with confidence score
        
        Args:
            text: Input text
            
        Returns:
            (intent_id, confidence)
        """
        text = text.lower()
        
        # Score each intent
        scores = {}
        
        for intent_name, keywords in self.keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                intent_id = get_intent_id(intent_name, self.dataset)
                scores[intent_id] = score
        
        if not scores:
            # Default intent
            if self.dataset == 'craigslistbargain':
                return 2, 0.5  # inform
            else:
                return 5, 0.5  # propose
        
        # Get intent with highest score
        best_intent = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_intent] / total_score
        
        return best_intent, confidence


# ==================== Item Parser ====================

class ItemParser:
    """
    Extract item allocations for Dealornodeal
    """
    
    ITEMS = ['hat', 'hats', 'book', 'books', 'ball', 'balls']
    
    def __init__(self):
        # Number extraction pattern
        self.number_pattern = re.compile(r'(\d+)\s*(hat|book|ball)s?', re.IGNORECASE)
    
    def extract_items(self, text: str) -> Optional[Dict[str, int]]:
        """
        Extract item allocation from text
        
        Args:
            text: Input text
            
        Returns:
            Dict mapping item to count, or None
        """
        items = {'hats': 0, 'books': 0, 'balls': 0}
        found_any = False
        
        matches = self.number_pattern.finditer(text)
        
        for match in matches:
            count = int(match.group(1))
            item = match.group(2).lower()
            
            # Normalize to plural
            if item in ['hat']:
                items['hats'] = count
            elif item in ['book']:
                items['books'] = count
            elif item in ['ball']:
                items['balls'] = count
            
            found_any = True
        
        return items if found_any else None


# ==================== Dialogue Parser ====================

class DialogueParser:
    """
    Main parser for dialogue utterances
    
    Combines intent, price, and item parsing
    """
    
    def __init__(self, dataset: str = 'craigslistbargain'):
        """
        Args:
            dataset: 'craigslistbargain' or 'dealornodeal'
        """
        self.dataset = dataset
        self.intent_parser = IntentParser(dataset)
        self.price_parser = PriceParser()
        self.item_parser = ItemParser()
    
    def parse(self, text: str) -> Dict:
        """
        Parse utterance into structured form
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with intent, price, items, etc.
        """
        # Classify intent
        intent_id, confidence = self.intent_parser.classify_with_confidence(text)
        intent_name = get_intent_name(intent_id, self.dataset)
        
        result = {
            'text': text,
            'intent_id': intent_id,
            'intent_name': intent_name,
            'confidence': confidence,
        }
        
        # Extract price if needed
        if self.dataset == 'craigslistbargain':
            if requires_price(intent_id, self.dataset):
                price = self.price_parser.extract_price(text)
                result['price'] = price
            else:
                result['price'] = None
        
        # Extract items if needed
        if self.dataset == 'dealornodeal':
            if requires_items(intent_id, self.dataset):
                items = self.item_parser.extract_items(text)
                result['items'] = items
            else:
                result['items'] = None
        
        return result
    
    def parse_batch(self, texts: List[str]) -> List[Dict]:
        """
        Parse multiple utterances
        
        Args:
            texts: List of texts
            
        Returns:
            List of parsed results
        """
        return [self.parse(text) for text in texts]


# ==================== Rule-based Parser ====================

class RuleBasedParser:
    """
    Rule-based parser with handcrafted rules
    
    More accurate than keyword-based but requires more rules
    """
    
    def __init__(self, dataset: str = 'craigslistbargain'):
        self.dataset = dataset
        self.price_parser = PriceParser()
        self.item_parser = ItemParser()
    
    def parse_craigslist(self, text: str) -> Dict:
        """
        Parse Craigslistbargain utterance
        
        Args:
            text: Input text
            
        Returns:
            Parsed result
        """
        text_lower = text.lower()
        
        # Rule 1: Accept/Reject/Quit
        if any(w in text_lower for w in ['accept', 'deal', "i'll take"]):
            return {'intent_id': 13, 'intent_name': 'accept', 'price': None}
        
        if any(w in text_lower for w in ['reject', 'no thanks', 'not interested']):
            return {'intent_id': 14, 'intent_name': 'reject', 'price': None}
        
        if any(w in text_lower for w in ['bye', 'goodbye', 'never mind']):
            return {'intent_id': 15, 'intent_name': 'quit', 'price': None}
        
        # Rule 2: Greeting
        if any(w in text_lower for w in ['hello', 'hi', 'hey']):
            return {'intent_id': 0, 'intent_name': 'greet', 'price': None}
        
        # Rule 3: Question
        if '?' in text:
            return {'intent_id': 1, 'intent_name': 'inquire', 'price': None}
        
        # Rule 4: Price negotiation
        price = self.price_parser.extract_price(text)
        
        if price is not None:
            # Has price
            if any(w in text_lower for w in ['firm', 'final', 'lowest']):
                return {'intent_id': 7, 'intent_name': 'final-price', 'price': price}
            elif any(w in text_lower for w in ['how about', 'what if', 'could you']):
                return {'intent_id': 6, 'intent_name': 'concede-price', 'price': price}
            elif any(w in text_lower for w in ['asking', 'listed', 'selling']):
                return {'intent_id': 3, 'intent_name': 'init-price', 'price': price}
            else:
                return {'intent_id': 12, 'intent_name': 'offer', 'price': price}
        
        # Rule 5: Attitude
        if any(w in text_lower for w in ['great', 'excellent', 'interested']):
            return {'intent_id': 10, 'intent_name': 'positive', 'price': None}
        
        if any(w in text_lower for w in ['too high', 'too low', 'not enough']):
            return {'intent_id': 8, 'intent_name': 'counter-no-price', 'price': None}
        
        # Default: inform
        return {'intent_id': 2, 'intent_name': 'inform', 'price': None}
    
    def parse(self, text: str) -> Dict:
        """
        Parse utterance
        
        Args:
            text: Input text
            
        Returns:
            Parsed result
        """
        if self.dataset == 'craigslistbargain':
            return self.parse_craigslist(text)
        else:
            # Use simple parser for Dealornodeal
            parser = DialogueParser(self.dataset)
            return parser.parse(text)


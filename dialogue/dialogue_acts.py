"""
Dialogue Acts for Negotiation
Defines all dialogue acts (intents) used in the framework

Paper Section 5.1:
- Craigslistbargain: 16 dialogue acts
- Dealornodeal: 6 dialogue acts

Author: α-Nego Implementation
Reference: Chen et al., 2025, Electronics
"""

from enum import IntEnum
from typing import Dict, List, Optional
from dataclasses import dataclass


# ==================== Craigslistbargain Dialogue Acts ====================

class CraigslistDialogueAct(IntEnum):
    """
    16 dialogue acts for Craigslistbargain dataset
    
    Categories:
    - Greeting: greet
    - Information: inquire, inform
    - Price negotiation: init-price, insist-price, agree-price, concede-price,
                        final-price, counter-no-price
    - Attitude: hesitant, positive, negative
    - General: offer, accept, reject, quit
    """
    GREET = 0              # Greeting
    INQUIRE = 1            # Ask for information
    INFORM = 2             # Provide information
    INIT_PRICE = 3         # Initial price offer
    INSIST_PRICE = 4       # Insist on current price
    AGREE_PRICE = 5        # Agree to a price
    CONCEDE_PRICE = 6      # Concede/make counter-offer
    FINAL_PRICE = 7        # Final offer
    COUNTER_NO_PRICE = 8   # Counter without specific price
    HESITANT = 9           # Show hesitation
    POSITIVE = 10          # Positive attitude
    NEGATIVE = 11          # Negative attitude
    OFFER = 12             # Make an offer
    ACCEPT = 13            # Accept offer/deal
    REJECT = 14            # Reject offer/deal
    QUIT = 15              # Quit negotiation


# Intent name mappings
CRAIGSLIST_INTENT_NAMES = {
    0: "greet",
    1: "inquire",
    2: "inform",
    3: "init-price",
    4: "insist-price",
    5: "agree-price",
    6: "concede-price",
    7: "final-price",
    8: "counter-no-price",
    9: "hesitant",
    10: "positive",
    11: "negative",
    12: "offer",
    13: "accept",
    14: "reject",
    15: "quit"
}

CRAIGSLIST_NAME_TO_INTENT = {v: k for k, v in CRAIGSLIST_INTENT_NAMES.items()}


# ==================== Dealornodeal Dialogue Acts ====================

class DealornodealDialogueAct(IntEnum):
    """
    6 dialogue acts for Dealornodeal dataset
    
    Categories:
    - Greeting: greet
    - Attitude: disagree, agree
    - Negotiation: insist, inquire, propose
    """
    GREET = 0      # Greeting
    DISAGREE = 1   # Disagree with proposal
    AGREE = 2      # Agree with proposal
    INSIST = 3     # Insist on position
    INQUIRE = 4    # Ask question
    PROPOSE = 5    # Make proposal


# Intent name mappings
DEALORNODEAL_INTENT_NAMES = {
    0: "greet",
    1: "disagree",
    2: "agree",
    3: "insist",
    4: "inquire",
    5: "propose"
}

DEALORNODEAL_NAME_TO_INTENT = {v: k for k, v in DEALORNODEAL_INTENT_NAMES.items()}


# ==================== Dialogue Act Metadata ====================

@dataclass
class DialogueActMetadata:
    """
    Metadata for a dialogue act
    """
    intent_id: int
    intent_name: str
    category: str
    description: str
    requires_price: bool = False
    requires_items: bool = False
    is_terminal: bool = False  # Ends negotiation
    
    def __str__(self):
        return f"{self.intent_name} ({self.category})"


# Craigslistbargain metadata
CRAIGSLIST_METADATA = {
    0: DialogueActMetadata(0, "greet", "greeting", "Greeting message"),
    1: DialogueActMetadata(1, "inquire", "information", "Ask for information"),
    2: DialogueActMetadata(2, "inform", "information", "Provide information"),
    3: DialogueActMetadata(3, "init-price", "price", "Initial price offer", requires_price=True),
    4: DialogueActMetadata(4, "insist-price", "price", "Insist on current price", requires_price=True),
    5: DialogueActMetadata(5, "agree-price", "price", "Agree to a price", requires_price=True),
    6: DialogueActMetadata(6, "concede-price", "price", "Concede/counter-offer", requires_price=True),
    7: DialogueActMetadata(7, "final-price", "price", "Final offer", requires_price=True),
    8: DialogueActMetadata(8, "counter-no-price", "price", "Counter without price"),
    9: DialogueActMetadata(9, "hesitant", "attitude", "Show hesitation"),
    10: DialogueActMetadata(10, "positive", "attitude", "Positive attitude"),
    11: DialogueActMetadata(11, "negative", "attitude", "Negative attitude"),
    12: DialogueActMetadata(12, "offer", "general", "Make an offer", requires_price=True),
    13: DialogueActMetadata(13, "accept", "general", "Accept offer/deal", is_terminal=True),
    14: DialogueActMetadata(14, "reject", "general", "Reject offer/deal", is_terminal=True),
    15: DialogueActMetadata(15, "quit", "general", "Quit negotiation", is_terminal=True),
}

# Dealornodeal metadata
DEALORNODEAL_METADATA = {
    0: DialogueActMetadata(0, "greet", "greeting", "Greeting message"),
    1: DialogueActMetadata(1, "disagree", "attitude", "Disagree with proposal"),
    2: DialogueActMetadata(2, "agree", "attitude", "Agree with proposal", is_terminal=True),
    3: DialogueActMetadata(3, "insist", "negotiation", "Insist on position", requires_items=True),
    4: DialogueActMetadata(4, "inquire", "negotiation", "Ask question"),
    5: DialogueActMetadata(5, "propose", "negotiation", "Make proposal", requires_items=True),
}


# ==================== Helper Functions ====================

def get_intent_name(intent_id: int, dataset: str = 'craigslistbargain') -> str:
    """
    Get intent name from ID
    
    Args:
        intent_id: Intent ID
        dataset: 'craigslistbargain' or 'dealornodeal'
        
    Returns:
        Intent name
    """
    if dataset == 'craigslistbargain':
        return CRAIGSLIST_INTENT_NAMES.get(intent_id, "unknown")
    elif dataset == 'dealornodeal':
        return DEALORNODEAL_INTENT_NAMES.get(intent_id, "unknown")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_intent_id(intent_name: str, dataset: str = 'craigslistbargain') -> int:
    """
    Get intent ID from name
    
    Args:
        intent_name: Intent name
        dataset: 'craigslistbargain' or 'dealornodeal'
        
    Returns:
        Intent ID
    """
    if dataset == 'craigslistbargain':
        return CRAIGSLIST_NAME_TO_INTENT.get(intent_name, -1)
    elif dataset == 'dealornodeal':
        return DEALORNODEAL_NAME_TO_INTENT.get(intent_name, -1)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_metadata(intent_id: int, dataset: str = 'craigslistbargain') -> DialogueActMetadata:
    """
    Get metadata for intent
    
    Args:
        intent_id: Intent ID
        dataset: 'craigslistbargain' or 'dealornodeal'
        
    Returns:
        Metadata object
    """
    if dataset == 'craigslistbargain':
        return CRAIGSLIST_METADATA.get(intent_id)
    elif dataset == 'dealornodeal':
        return DEALORNODEAL_METADATA.get(intent_id)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def is_terminal(intent_id: int, dataset: str = 'craigslistbargain') -> bool:
    """
    Check if intent is terminal (ends negotiation)
    
    Args:
        intent_id: Intent ID
        dataset: Dataset name
        
    Returns:
        True if terminal
    """
    metadata = get_metadata(intent_id, dataset)
    return metadata.is_terminal if metadata else False


def requires_price(intent_id: int, dataset: str = 'craigslistbargain') -> bool:
    """
    Check if intent requires price
    
    Args:
        intent_id: Intent ID
        dataset: Dataset name
        
    Returns:
        True if requires price
    """
    metadata = get_metadata(intent_id, dataset)
    return metadata.requires_price if metadata else False


def requires_items(intent_id: int, dataset: str = 'craigslistbargain') -> bool:
    """
    Check if intent requires items
    
    Args:
        intent_id: Intent ID
        dataset: Dataset name
        
    Returns:
        True if requires items
    """
    metadata = get_metadata(intent_id, dataset)
    return metadata.requires_items if metadata else False


def get_intents_by_category(category: str, dataset: str = 'craigslistbargain') -> List[int]:
    """
    Get all intents in a category
    
    Args:
        category: Category name
        dataset: Dataset name
        
    Returns:
        List of intent IDs
    """
    if dataset == 'craigslistbargain':
        metadata_dict = CRAIGSLIST_METADATA
    elif dataset == 'dealornodeal':
        metadata_dict = DEALORNODEAL_METADATA
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return [
        intent_id for intent_id, meta in metadata_dict.items()
        if meta.category == category
    ]


def get_all_intents(dataset: str = 'craigslistbargain') -> List[int]:
    """
    Get all intent IDs
    
    Args:
        dataset: Dataset name
        
    Returns:
        List of all intent IDs
    """
    if dataset == 'craigslistbargain':
        return list(CRAIGSLIST_INTENT_NAMES.keys())
    elif dataset == 'dealornodeal':
        return list(DEALORNODEAL_INTENT_NAMES.keys())
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_num_intents(dataset: str = 'craigslistbargain') -> int:
    """
    Get number of intents
    
    Args:
        dataset: Dataset name
        
    Returns:
        Number of intents
    """
    if dataset == 'craigslistbargain':
        return len(CRAIGSLIST_INTENT_NAMES)
    elif dataset == 'dealornodeal':
        return len(DEALORNODEAL_INTENT_NAMES)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ==================== Intent Validation ====================

def validate_action(
    intent_id: int,
    price: Optional[float] = None,
    items: Optional[Dict] = None,
    dataset: str = 'craigslistbargain',
) -> bool:
    """
    Validate if action is well-formed
    
    Args:
        intent_id: Intent ID
        price: Price value
        items: Item allocation
        dataset: Dataset name
        
    Returns:
        True if valid
    """
    metadata = get_metadata(intent_id, dataset)
    
    if metadata is None:
        return False
    
    # Check price requirement
    if metadata.requires_price and price is None:
        return False
    
    # Check items requirement
    if metadata.requires_items and items is None:
        return False
    
    return True

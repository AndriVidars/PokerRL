# Card Module Documentation

## Overview
The `card.py` module defines the fundamental components for representing playing cards in the PokerRL framework. It provides enumerations for card ranks and suits along with a Card class that combines them.

## Key Components

### Rank Enum
An enumeration representing the 13 possible card ranks.

```python
from poker.core.card import Rank

# Examples
two = Rank.TWO     # value = 2
jack = Rank.JACK   # value = 11
ace = Rank.ACE     # value = 14
```

The ranks are ordered from TWO (2) to ACE (14), with values corresponding to their poker strength.

### Suit Enum
An enumeration representing the 4 possible card suits.

```python
from poker.core.card import Suit

# Examples
hearts = Suit.HEART
clubs = Suit.CLUB
diamonds = Suit.DIAMOND
spades = Suit.SPADE
```

### Card Class
Represents a single playing card with a rank and suit.

#### Initialization
```python
from poker.core.card import Card, Rank, Suit

# Create a card (Ace of Spades)
card = Card(Rank.ACE, Suit.SPADE)
```

#### Properties
- `rank`: The card's rank (Rank enum)
- `suit`: The card's suit (Suit enum)

#### String Representation
The Card class implements `__str__()` and `__repr__()` methods for easy printing:

```python
card = Card(Rank.ACE, Suit.SPADE)
print(card)  # Outputs: "A♠" or similar representation
```

### RANK_ORDER
A dictionary mapping from Rank enums to their numerical values, used for hand evaluation:

```python
# Access the numerical value of a rank
rank_value = RANK_ORDER[Rank.QUEEN]  # Returns 12
```

## Usage Examples

### Creating a Deck of Cards
```python
from poker.core.card import Card, Rank, Suit

# Create a full deck of 52 cards
deck = []
for suit in Suit:
    for rank in Rank:
        deck.append(Card(rank, suit))
```

### Card Comparison
For comparing card values:

```python
card1 = Card(Rank.ACE, Suit.HEART)
card2 = Card(Rank.KING, Suit.SPADE)

# Compare ranks
if RANK_ORDER[card1.rank] > RANK_ORDER[card2.rank]:
    print("Card 1 has higher rank")
```

### Displaying Cards
For user interfaces, you might want to create a visual representation:

```python
def display_card(card):
    suit_symbols = {
        Suit.HEART: '♥',
        Suit.DIAMOND: '♦',
        Suit.CLUB: '♣',
        Suit.SPADE: '♠'
    }
    
    rank_symbols = {
        Rank.TWO: '2', Rank.THREE: '3', Rank.FOUR: '4',
        Rank.FIVE: '5', Rank.SIX: '6', Rank.SEVEN: '7',
        Rank.EIGHT: '8', Rank.NINE: '9', Rank.TEN: '10',
        Rank.JACK: 'J', Rank.QUEEN: 'Q', Rank.KING: 'K',
        Rank.ACE: 'A'
    }
    
    return f"{rank_symbols[card.rank]}{suit_symbols[card.suit]}"
```

## Integration with Game Components

The Card class is used throughout the framework:

- In `Deck` for managing the deck of cards
- In `Player` for hole cards
- In `Game` for community cards
- In `HandEvaluator` for evaluating hand strength
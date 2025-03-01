# Deck Module Documentation

## Overview
The `deck.py` module provides the Deck class that represents a standard 52-card playing deck. It handles deck creation, shuffling, and card dealing functionality for the poker game.

## Key Components

### Deck Class
Represents a standard deck of 52 playing cards.

#### Initialization
```python
from poker.core.deck import Deck

# Create a new shuffled deck
deck = Deck()
```

When initialized, the Deck class automatically creates a full set of 52 cards (one for each combination of the 13 ranks and 4 suits) and shuffles them.

#### Properties
- `deck`: A list containing all cards in the deck

#### Methods
- `shuffle()`: Randomizes the order of cards in the deck
- `deal(n)`: Deals n cards from the top of the deck and returns them

## Usage Examples

### Dealing Cards
```python
from poker.core.deck import Deck

# Create a deck
deck = Deck()

# Deal hole cards to players
player1_cards = [deck.deck.pop() for _ in range(2)]
player2_cards = [deck.deck.pop() for _ in range(2)]

# Deal community cards
flop = [deck.deck.pop() for _ in range(3)]
turn = deck.deck.pop()
river = deck.deck.pop()
```

### Resetting the Deck
To reset the deck after a hand:

```python
# Reset to a new shuffled deck
deck = Deck()
```

### Manual Deck Control (for Testing)
For deterministic testing, you can manipulate the deck directly:

```python
from poker.core.deck import Deck
from poker.core.card import Card, Rank, Suit

# Create deck but don't shuffle yet
deck = Deck(shuffle=False)

# Place specific cards on top of the deck
aces = [
    Card(Rank.ACE, Suit.SPADE),
    Card(Rank.ACE, Suit.HEART),
    Card(Rank.ACE, Suit.DIAMOND),
    Card(Rank.ACE, Suit.CLUB)
]

# Remove the aces from the deck
deck.deck = [card for card in deck.deck if card not in aces]
# Add them to the top
deck.deck.extend(aces)
```

## Integration with Game

The deck is used in the Game class to deal cards during different stages:

```python
# In preflop
self.deck = Deck()
for _ in range(2):
    for player in self.players:
        player.hand.append(self.deck.deck.pop())

# In flop
burn = self.deck.deck.pop()  # Burn card
for _ in range(3):
    self.community_cards.append(self.deck.deck.pop())

# In turn/river
burn = self.deck.deck.pop()  # Burn card
self.community_cards.append(self.deck.deck.pop())
```

## Best Practices

- Always create a new deck instance for each new game
- Use the `pop()` method to deal cards, which removes them from the deck
- Remember to burn a card before dealing each community card round (flop, turn, river)
- For realistic poker play, never look at the next cards in the deck until they are dealt
# Hand Evaluator Module Documentation

## Overview
The `hand_evaluator.py` module provides functionality for evaluating poker hands to determine their strength. It implements standard poker hand ranking rules and can handle hands from 2 to 7 cards, useful for calculating the best 5-card hand from 7 available cards (2 hole cards + 5 community cards).

## Key Components

### Hand Evaluation Functions

#### evaluate_hand(cards)
The main function for evaluating a poker hand.

```python
from poker.core.hand_evaluator import evaluate_hand
from poker.core.card import Card, Rank, Suit

# Example hand
hand = [
    Card(Rank.ACE, Suit.HEART),
    Card(Rank.KING, Suit.HEART),
    Card(Rank.QUEEN, Suit.HEART),
    Card(Rank.JACK, Suit.HEART),
    Card(Rank.TEN, Suit.HEART)
]

rank, tiebreakers = evaluate_hand(hand)
# rank = 9 (Straight Flush)
# tiebreakers = [14] (Ace high)
```

- `cards`: List of Card objects to evaluate
- Returns: Tuple of (hand_rank, tiebreakers)
  - `hand_rank`: Integer from 1-9 indicating hand strength
  - `tiebreakers`: List of values used to break ties between hands of the same rank

#### Hand Ranks
1. High Card
2. One Pair
3. Two Pair
4. Three of a Kind
5. Straight
6. Flush
7. Full House
8. Four of a Kind
9. Straight Flush

#### _find_best_hand(cards)
Helper function to find the best 5-card hand from more than 5 cards.

```python
best_rank, best_tiebreakers = _find_best_hand(cards)
```

- `cards`: List of Card objects (typically 7 cards in Texas Hold'em)
- Returns: Same as `evaluate_hand()`, but for the best possible 5-card combination

### Helper Functions

- `is_flush(suits)`: Determines if all cards have the same suit
- `is_straight(ranks)`: Determines if the cards form a straight (sequential ranks)

## Usage in Game Flow

The hand evaluator is used at the end of a hand to determine the winner:

```python
# In game.py's decide_pot method
for player in active_players:
    best_rank, best_tiebreakers = evaluate_hand(community_cards + player.hand)
    player_hand_rankings[player] = (best_rank, best_tiebreakers)

# Sort players by hand strength
sorted_players = sorted(player_hand_rankings.items(), 
                         key=lambda x: x[1], 
                         reverse=True)
```

## Comparing Hands

To compare two hands:

```python
def hand_is_better(hand1, hand2):
    rank1, tiebreakers1 = evaluate_hand(hand1)
    rank2, tiebreakers2 = evaluate_hand(hand2)
    
    if rank1 > rank2:
        return True
    elif rank1 < rank2:
        return False
    else:
        # Same rank, compare tiebreakers
        return tiebreakers1 > tiebreakers2
```

## Advanced Use Cases

### Pre-flop Hand Strength
For AI players, you might want to evaluate starting hand strength:

```python
def evaluate_preflop_strength(hole_cards):
    # Simple ranking based on card values and whether they're paired
    ranks = [card.rank for card in hole_cards]
    suited = hole_cards[0].suit == hole_cards[1].suit
    
    if ranks[0] == ranks[1]:  # Pocket pair
        return RANK_ORDER[ranks[0]] * 2
    elif suited:  # Suited cards
        return (RANK_ORDER[ranks[0]] + RANK_ORDER[ranks[1]]) * 1.1
    else:  # Offsuit
        return RANK_ORDER[ranks[0]] + RANK_ORDER[ranks[1]]
```
# Pot Module Documentation

## Overview
The `pot.py` module provides the Pot class, which is responsible for managing betting contributions, tracking eligible players, and handling the distribution of chips during a poker game. It also supports side pots created during all-in situations.

## Key Components

### Pot Class
Represents a betting pot in the poker game.

#### Initialization
```python
from poker.core.pot import Pot

# Create a new empty pot
pot = Pot()
```

The Pot is initialized empty, with no players or contributions.

#### Properties
- `contributions`: Dictionary mapping player objects to their contribution amount
- `eligible_players`: Set of players eligible to win the pot
- `total_amount`: Total amount in the pot (sum of all contributions)

#### Methods
- `add_contribution(player, amount)`: Adds a contribution from a player to the pot
- `split_pot(all_in_player)`: Creates a side pot when a player is all-in, returning the new side pot

## Usage Examples

### Basic Pot Usage
```python
from poker.core.pot import Pot
from poker.core.player import Player

# Create a pot
pot = Pot()

# Add player contributions
player1 = Player("Player 1", 1000)
player2 = Player("Player 2", 1000)
player3 = Player("Player 3", 1000)

pot.add_contribution(player1, 10)
pot.add_contribution(player2, 10)
pot.add_contribution(player3, 10)

# Check pot total
print(f"Total pot: {pot.total_amount}")  # Output: Total pot: 30
```

### Handling All-in with Side Pots
```python
# Player goes all-in with less than the current bet
all_in_player = Player("All-in Player", 50)
pot.add_contribution(all_in_player, 5)  # All they can afford

# Split the pot to create a side pot
side_pot = pot.split_pot(all_in_player)

# Continue betting in the side pot
pot.add_contribution(player1, 5)
pot.add_contribution(player2, 5)
pot.add_contribution(player3, 5)

# all_in_player is not eligible for the side pot
print(all_in_player in side_pot.eligible_players)  # Output: False
```

## Integration with Game

The Game class manages multiple pots during gameplay:

```python
# In game.py
self.pots = []  # List of active pots

# During betting
if player.all_in:
    # Create side pot
    side_pot = current_pot.split_pot(player)
    self.pots.append(side_pot)

# When handling raises
def handle_raise(self, amount):
    # Call first
    self.handle_check_call()
    
    # Then create a new pot for the raise
    pot = Pot()
    pot.add_contribution(self, amount)
    self.game.pots.append(pot)
```

## Pot Distribution

When determining winners, each pot is evaluated separately:

```python
# For each pot
for pot in self.pots:
    # Get eligible players who haven't folded
    pot_players = [p for p in pot.eligible_players if not p.folded]
    
    # Find the best hand among pot players
    winner = max(pot_players, key=lambda p: evaluate_hand(p.hand + community_cards))
    
    # Award pot to winner
    winner.stack += pot.total_amount
```

## Best Practices

- Always track which players are eligible for each pot
- Create side pots when players go all-in
- Process pots in order (main pot first, then side pots)
- Handle tied hands by splitting pots among tied players
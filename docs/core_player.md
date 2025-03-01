# Player Module Documentation

## Overview
The `player.py` module defines the abstract Player class that serves as the foundation for all player types in the PokerRL framework. It establishes the interface for player actions and handles the mechanics of betting, folding, and checking/calling.

## Key Components

### Player Class
An abstract base class for poker players.

#### Initialization
```python
player = Player(name, stack)
```

- `name`: The player's name (string)
- `stack`: Starting chip stack (integer)

#### Properties
- `name`: Player's name
- `stack`: Current chip count
- `hand`: List of hole cards
- `folded`: Boolean indicating if player has folded
- `all_in`: Boolean indicating if player is all-in
- `game`: Reference to the Game object (set by the Game when adding players)

#### Abstract Methods

- `act() -> Action`: The main action method that must be implemented by subclasses. Should return an Action enum value (FOLD, CHECK_CALL, or RAISE).

#### Implementation Methods

- `handle_fold()`: Processes a fold action, marking the player as folded and removing them from eligible pot players
- `handle_check_call()`: Processes a check/call action, adding chips to the pot based on current bet amounts
- `handle_raise(amount)`: Processes a raise action, calling `handle_check_call()` first, then creating a new pot with the raise amount

## Creating Custom Players

To create a custom player, subclass the Player class and implement the `act()` method:

```python
from poker.core.player import Player
from poker.core.action import Action

class YourCustomPlayer(Player):
    def act(self):
        # Your decision-making logic here
        # Should return Action.FOLD, Action.CHECK_CALL, or Action.RAISE
        
        # For example:
        if self.should_fold():
            self.handle_fold()
            return Action.FOLD
        elif self.should_raise():
            raise_amount = self.determine_raise_amount()
            self.handle_raise(raise_amount)
            return Action.RAISE
        else:
            self.handle_check_call()
            return Action.CHECK_CALL
```

## Player Types in the Framework

The framework includes several player implementations:

1. **PlayerIO**: A player controlled by user input through the console
2. **PlayerRandom**: A player that makes random legal moves
3. **DeepLearningPlayer**: A player that uses a trained neural network to make decisions

## Integration with Game Loop

Players interact with the game through the `Game` object. The game calls each player's `act()` method during their turn in the betting rounds:

```python
# Inside the Game's betting loop
current_player = self.players[current_idx]
action = current_player.act()
# Game processes the action...
```

## Advanced Usage

### Accessing Game State
Players can access the current game state through their `game` reference:

```python
def act(self):
    # Access community cards
    community_cards = self.game.community_cards
    
    # Check current pot
    pot_size = sum(pot.total_amount for pot in self.game.pots)
    
    # Make decisions based on game state
    # ...
```
# Action Module Documentation

## Overview
The `action.py` module defines the possible actions a player can take during their turn in a poker game. It implements an enumeration that represents the three fundamental poker actions: folding, checking/calling, and raising.

## Key Components

### Action Enum
An enumeration class that defines the possible actions in poker.

```python
from poker.core.action import Action

# Available actions
fold_action = Action.FOLD  # value = 0
check_call_action = Action.CHECK_CALL  # value = 1
raise_action = Action.RAISE  # value = 2
```

#### Action Values
- `FOLD (0)`: Player folds their hand and is no longer in the current game round
- `CHECK_CALL (1)`: Player checks (if no bet to call) or calls the current bet
- `RAISE (2)`: Player raises the current bet amount

## Usage in Player Implementation

When implementing a custom player, the `act()` method should return one of these action enum values:

```python
from poker.core.player import Player
from poker.core.action import Action

class YourCustomPlayer(Player):
    def act(self):
        # Decision logic...
        
        if should_fold:
            self.handle_fold()
            return Action.FOLD
        elif should_raise:
            self.handle_raise(raise_amount)
            return Action.RAISE
        else:
            self.handle_check_call()
            return Action.CHECK_CALL
```

## Usage in Game Logic

The Action enum is used throughout the game logic to determine how to process a player's turn:

```python
# In game.py
action = current_player.act()

if action == Action.FOLD:
    # Handle fold logic
elif action == Action.CHECK_CALL:
    # Handle check/call logic
elif action == Action.RAISE:
    # Handle raise logic
```

## Integration with UI

When building a user interface, you can use the Action enum to present options to the player:

```python
# In a command-line interface
print("Choose an action:")
print(f"0: Fold")
print(f"1: {'Check' if can_check else 'Call'}")
print(f"2: Raise")

choice = int(input("> "))
return Action(choice)  # Convert integer to Action enum
```

## Extending Actions

If you need to extend the set of actions (for example, to include a separate CHECK and CALL), you can subclass the Action enum:

```python
from enum import Enum
from poker.core.action import Action as BaseAction

class ExtendedAction(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE = 3
    
    @staticmethod
    def to_base_action(extended_action):
        if extended_action == ExtendedAction.FOLD:
            return BaseAction.FOLD
        elif extended_action in (ExtendedAction.CHECK, ExtendedAction.CALL):
            return BaseAction.CHECK_CALL
        else:
            return BaseAction.RAISE
```

## Best Practices

- Always use the Action enum rather than hardcoded integers for better code readability
- When processing actions, use the enum for comparison instead of its value
- Handle all possible actions to avoid unexpected behavior
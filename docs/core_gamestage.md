# GameStage Module Documentation

## Overview
The `gamestage.py` module defines the different stages of a Texas Hold'em poker game through the Stage enumeration. It provides a way to track and manage the current stage of gameplay.

## Key Components

### Stage Enum
An enumeration representing the four main stages of a poker hand.

```python
from poker.core.gamestage import Stage

# The four stages in order
preflop = Stage.PREFLOP  # value = 0
flop = Stage.FLOP        # value = 1
turn = Stage.TURN        # value = 2
river = Stage.RIVER      # value = 3
```

#### Stage Values
- `PREFLOP (0)`: Initial stage where players receive their hole cards and first betting round occurs
- `FLOP (1)`: Three community cards are dealt, followed by the second betting round
- `TURN (2)`: Fourth community card is dealt, followed by the third betting round
- `RIVER (3)`: Fifth and final community card is dealt, followed by the final betting round

## Usage in Game Flow

The Stage enum is used to track the current stage in the game:

```python
from poker.core.gamestage import Stage

# Initialize the game at PREFLOP
current_stage = Stage.PREFLOP

# Progress through stages
def next_stage():
    global current_stage
    current_stage = Stage((current_stage.value + 1) % len(Stage))
```

## Integration with Game Class

In the Game class, the current stage is tracked and used to determine the flow of the game:

```python
# In game.py
self.current_stage = Stage.PREFLOP

# To advance to the next stage
def next_stage(self):
    self.current_stage = Stage((self.current_stage.value + 1) % len(Stage))
    
# In gameplay_loop
while True:
    match self.current_stage:
        case Stage.PREFLOP:
            self.preflop()
        case Stage.FLOP:
            self.flop()
        case Stage.TURN:
            self.turn()
        case Stage.RIVER:
            self.river()
            break
```

## Usage in Player Logic

Players can use the game stage to inform their decision-making:

```python
def act(self):
    current_stage = self.game.current_stage
    
    if current_stage == Stage.PREFLOP:
        # Preflop strategy
        return preflop_decision()
    elif current_stage == Stage.FLOP:
        # Flop strategy
        return flop_decision()
    elif current_stage == Stage.TURN:
        # Turn strategy
        return turn_decision()
    else:  # RIVER
        # River strategy
        return river_decision()
```

## Usage in Hand Evaluation

The stage can also be important for hand strength evaluation:

```python
def evaluate_hand_strength(cards, stage):
    if stage == Stage.PREFLOP:
        # Simple preflop evaluation
        return evaluate_hole_cards(cards)
    elif stage == Stage.FLOP:
        # Evaluate with 3 community cards
        return evaluate_with_flop(cards)
    elif stage == Stage.TURN:
        # Evaluate with 4 community cards
        return evaluate_with_turn(cards)
    else:  # RIVER
        # Final hand evaluation with all 5 community cards
        return evaluate_complete_hand(cards)
```

## Best Practices

- Use the Stage enum for comparisons rather than integer values
- Use match-case (in Python 3.10+) or if-elif chains for stage-specific logic
- Always progress through stages in order (PREFLOP → FLOP → TURN → RIVER)
- When implementing custom poker variants, you might need to extend this enum
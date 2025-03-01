# Game Module Documentation

## Overview
The `game.py` module is the central component of the PokerRL framework, responsible for managing and controlling the entire Texas Hold'em poker game flow. It orchestrates the progression through game stages, player actions, pot management, and determining winners.

## Key Components

### Game Class
The main class that manages the poker game.

#### Initialization
```python
game = Game(players, big_blind, small_blind)
```

- `players`: List of Player objects participating in the game
- `big_blind`: Amount for the big blind
- `small_blind`: Amount for the small blind

#### Game Stages
The game follows the standard Texas Hold'em stages:
1. **Preflop**: Initial dealing of hole cards and first round of betting
2. **Flop**: Three community cards are dealt followed by betting
3. **Turn**: Fourth community card is dealt followed by betting
4. **River**: Fifth community card is dealt followed by betting

#### Methods

- `gameplay_loop()`: Main game loop that progresses through all stages
- `preflop()`: Handles blinds, deals hole cards, and manages preflop betting
- `flop()`: Deals three community cards and manages betting
- `turn()`: Deals the fourth community card and manages betting
- `river()`: Deals the fifth community card, manages betting and determines winners
- `betting_loop()`: Manages a round of betting at any stage
- `decide_pot()`: Determines the winner(s) and distributes the pot
- `move_blinds()`: Moves dealer button and blinds positions for the next hand

## Usage Example

```python
from poker.core.game import Game
from poker.core.player import Player

# Create players
player1 = YourPlayerImplementation("Player 1", 1000)
player2 = YourPlayerImplementation("Player 2", 1000)
player3 = YourPlayerImplementation("Player 3", 1000)

# Initialize game
game = Game([player1, player2, player3], 20, 10)

# Play a full hand
game.gameplay_loop()
```

## Integration with Other Components

The Game class integrates with:
- **Player** objects that implement the `act()` method
- **Deck** for card management
- **Pot** for tracking betting and side pots
- **Hand Evaluator** for determining winning hands

## Advanced Usage

### Handling All-in Situations
The game automatically manages all-in scenarios and creates side pots when necessary.

### Multiple Rounds
To play multiple rounds, reset player hands between rounds:

```python
for _ in range(num_rounds):
    # Reset player hands
    for player in players:
        player.hand = []
        player.folded = False
        player.all_in = False
    
    # Play a round
    game.gameplay_loop()
```
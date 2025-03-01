# Pluribus Parser Module Documentation

## Overview
The `pluribus_parser.py` module provides functionality to parse Pluribus poker game logs into structured hand histories. It extracts information about players, actions, bets, cards, and game flow, making it possible to analyze Pluribus's playing style and train imitation learning agents.

## Key Components

### PluribusParser Class
The main class for parsing Pluribus log files.

#### Initialization
```python
from poker.parsers.pluribus_parser import PluribusParser

# Create a parser for a specific log file
parser = PluribusParser('path/to/log/file.txt')
```

#### Methods
- `parse()`: Parses the log file and returns hand histories
- `parse_hand(hand_text)`: Parses a single hand's text into a structured hand history
- `parse_player_action(action_text)`: Parses a player action text string
- `parse_cards(cards_text)`: Parses card notations from text

### Hand Class
Represents a complete hand history from the parsed logs.

#### Properties
- `hand_id`: Unique identifier for the hand
- `players`: List of player names
- `blinds`: Dictionary mapping players to their blind amounts
- `stacks`: Dictionary mapping players to their stack sizes
- `preflop_actions`: List of player actions in the preflop stage
- `flop_cards`: List of community cards on the flop
- `flop_actions`: List of player actions in the flop stage
- `turn_card`: The turn card
- `turn_actions`: List of player actions in the turn stage
- `river_card`: The river card
- `river_actions`: List of player actions in the river stage
- `winners`: List of players who won the hand
- `payouts`: Dictionary mapping winners to their winnings

### Action Class
Represents a player action in the hand history.

#### Properties
- `player`: Name of the player taking the action
- `action_type`: Type of action (fold, check, call, bet, raise)
- `amount`: Bet or raise amount (if applicable)

## Usage Examples

### Parsing Log Files
```python
from poker.parsers.pluribus_parser import PluribusParser

# Parse a log file
parser = PluribusParser('pluribus_converted_logs/pluribus_100.txt')
hands = parser.parse()

# Display information about the first hand
first_hand = hands[0]
print(f"Hand ID: {first_hand.hand_id}")
print(f"Players: {first_hand.players}")
print(f"Winners: {first_hand.winners}")
```

### Analyzing Player Actions
```python
# Analyze Pluribus's actions
pluribus_actions = []

for hand in hands:
    # Collect all actions by Pluribus
    for action in hand.preflop_actions + hand.flop_actions + hand.turn_actions + hand.river_actions:
        if action.player == "Pluribus":
            pluribus_actions.append(action)

# Count action types
action_counts = {}
for action in pluribus_actions:
    if action.action_type not in action_counts:
        action_counts[action.action_type] = 0
    action_counts[action.action_type] += 1

print("Pluribus action distribution:")
for action_type, count in action_counts.items():
    print(f"{action_type}: {count}")
```

### Extracting Training Data
```python
# Extract training examples for imitation learning
training_examples = []

for hand in hands:
    # Extract card information
    hole_cards = {}  # Collect known hole cards
    community_cards = []
    
    if hand.flop_cards:
        community_cards.extend(hand.flop_cards)
        
        # Extract Pluribus's actions on the flop
        for action in hand.flop_actions:
            if action.player == "Pluribus":
                # Create a training example (state, action)
                state = {
                    'community_cards': community_cards.copy(),
                    'player_stacks': hand.stacks.copy(),
                    # Other state features...
                }
                
                training_examples.append((state, action))
    
    # Similar extraction for turn and river...

# Use training examples to train an agent
# ...
```

## Integration with Game State Retriever

The PluribusParser is used by the GameStateRetriever to create GameState objects:

```python
from poker.parsers.game_state_retriever import GameStateRetriever

# Create a retriever for a directory of logs
retriever = GameStateRetriever('./pluribus_converted_logs')

# Initialize (parses all logs)
retriever.initialize()

# Get all Pluribus decisions
pluribus_decisions = retriever.get_pluribus_decisions()

# Each decision is a tuple (player_name, stage, game_state, action, amount)
for player, stage, state, action, amount in pluribus_decisions:
    # Use for training, analysis, etc.
    # ...
```

## Advanced Usage: Custom Analysis

```python
# Analyze Pluribus's preflop strategy
def analyze_preflop_strategy(hands):
    preflop_actions = {}
    
    for hand in hands:
        for action in hand.preflop_actions:
            if action.player == "Pluribus":
                # Categorize by position
                position = get_position(hand.players, "Pluribus")
                
                if position not in preflop_actions:
                    preflop_actions[position] = []
                
                preflop_actions[position].append(action)
    
    # Analyze by position
    for position, actions in preflop_actions.items():
        print(f"Position: {position}")
        analyze_actions(actions)
```
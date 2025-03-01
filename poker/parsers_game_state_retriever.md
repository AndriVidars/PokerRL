# Game State Retriever Module Documentation

## Overview
The `game_state_retriever.py` module provides a high-level interface for extracting game states and player decisions from Pluribus log files. It processes the raw logs into structured GameState objects that can be used for training AI agents or analyzing gameplay patterns.

## Key Components

### GameStateRetriever Class
The main class for retrieving game states from Pluribus logs.

#### Initialization
```python
from poker.parsers.game_state_retriever import GameStateRetriever

# Create a retriever for a specific logs directory
retriever = GameStateRetriever('./pluribus_converted_logs')

# Or use the default logs directory
retriever = GameStateRetriever()
```

#### Methods
- `initialize(verbose=False)`: Parses all log files in the directory (must be called before other methods)
- `get_game_state(hand_id, player_name, stage)`: Gets game states for a player in a specific hand and stage
- `get_decisions(criteria={})`: Gets player decisions based on criteria like player name, stage, or action
- `get_pluribus_decisions()`: Gets all decisions made by Pluribus
- `get_hand_count()`: Gets the total number of hands parsed

## Usage Examples

### Basic Initialization and Retrieval
```python
from poker.parsers.game_state_retriever import GameStateRetriever
from poker.core.gamestage import Stage

# Create and initialize the retriever
retriever = GameStateRetriever('./pluribus_converted_logs')
retriever.initialize(verbose=True)  # Shows progress and statistics

# Get the total number of hands
hand_count = retriever.get_hand_count()
print(f"Successfully parsed {hand_count} hands")

# Get all Pluribus decisions
pluribus_decisions = retriever.get_decisions({'player_name': 'Pluribus'})
print(f"Found {len(pluribus_decisions)} Pluribus decisions")
```

### Filtering Decisions by Criteria
```python
# Get Pluribus decisions on the flop
flop_decisions = retriever.get_decisions({
    'player_name': 'Pluribus',
    'stage': Stage.FLOP
})
print(f"Found {len(flop_decisions)} Pluribus decisions on the flop")

# Get all fold actions by any player
fold_decisions = retriever.get_decisions({
    'action': Action.FOLD
})
print(f"Found {len(fold_decisions)} fold actions")
```

### Examining Game States
```python
# Get a sample decision
player, stage, state, action, amount = pluribus_decisions[0]

# Examine the game state
print(f"Player: {player}")
print(f"Stage: {stage.name}")
print(f"Action: {action.name}")
print(f"Amount: {amount}")
print(f"Community cards: {[str(card) for card in state.community_cards]}")
print(f"Hand strength: {state.hand_strength}")
print(f"Pot size: {state.pot_size}")
print(f"Min bet to continue: {state.min_bet_to_continue}")
```

## Integration with Training

The GameStateRetriever is essential for training the ImitationAgent:

```python
from poker.agents.imitation_agent import ImitationAgent

# Initialize retriever
retriever = GameStateRetriever()
retriever.initialize()

# Get Pluribus decisions for training
training_data = retriever.get_pluribus_decisions()

# Extract features and labels
features = []
action_labels = []
raise_amounts = []

for player, stage, state, action, amount in training_data:
    # Extract features from state
    feature_vec = [
        stage.value,
        state.pot_size,
        state.min_bet_to_continue,
        # ... more features ...
    ]
    features.append(feature_vec)
    
    # Convert action to label
    action_labels.append(action.value)
    
    # Record raise amount if applicable
    if action == Action.RAISE and amount is not None:
        raise_amounts.append(amount / state.pot_size)  # Normalize by pot
    else:
        raise_amounts.append(0.0)

# Train the agent
agent = ImitationAgent()
agent.train(
    features=features,
    action_labels=action_labels,
    raise_amounts=raise_amounts
)
```

## Advanced Usage: Custom Analysis

```python
# Analyze playing patterns by position
def analyze_by_position():
    all_decisions = retriever.get_pluribus_decisions()
    position_stats = {}
    
    for _, stage, state, action, _ in all_decisions:
        position = state.my_player.spots_left_bb
        if position not in position_stats:
            position_stats[position] = {'total': 0, 'actions': {}}
        
        position_stats[position]['total'] += 1
        
        if action.name not in position_stats[position]['actions']:
            position_stats[position]['actions'][action.name] = 0
        position_stats[position]['actions'][action.name] += 1
    
    # Calculate action percentages by position
    for position, stats in position_stats.items():
        print(f"Position {position} (Total: {stats['total']})")
        for action_name, count in stats['actions'].items():
            percentage = count / stats['total'] * 100
            print(f"  {action_name}: {percentage:.2f}%")
```
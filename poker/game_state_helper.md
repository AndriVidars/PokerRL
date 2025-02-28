# GameStateHelper Module Documentation

## Overview
The `game_state_helper.py` module provides utility functions to create GameState objects from the core Game class. It bridges the gap between the game engine and the AI decision-making components by extracting relevant game state information.

## Key Components

### GameStateHelper Class
A static utility class that provides methods to create and manage GameState objects.

#### Methods
- `create_game_states(game, stage)`: Creates GameState objects for all players in the game
- `create_game_state_for_player(game, player, stage)`: Creates a GameState for a specific player
- `update_player_actions(game_states, actions)`: Updates GameState objects with player actions
- `get_all_game_states_by_stage(game)`: Gets all game states organized by game stage

## Usage Examples

### Creating Game States for All Players
```python
from poker.game_state_helper import GameStateHelper
from poker.core.game import Game
from poker.core.gamestage import Stage

# Create a game
game = Game([player1, player2, player3], big_blind, small_blind)

# Run the preflop stage
game.preflop()

# Create game states for all players at the flop stage
game_states = GameStateHelper.create_game_states(game, Stage.FLOP)

# Access a specific player's state
player1_state = game_states[player1]
```

### Creating a Single Player's Game State
```python
# Create game state for just one player
player_state = GameStateHelper.create_game_state_for_player(
    game, 
    specific_player, 
    Stage.TURN
)

# Use the state for AI decision-making
action = ai_agent.predict_action(player_state)
```

### Tracking Game States Across Stages
```python
# Track all game states by stage
all_states = GameStateHelper.get_all_game_states_by_stage(game)

# Access states by stage
preflop_states = all_states[Stage.PREFLOP]
flop_states = all_states[Stage.FLOP]
turn_states = all_states[Stage.TURN]
river_states = all_states[Stage.RIVER]

# Access a specific player's state at a specific stage
player_at_flop = flop_states[player1]
```

### Updating Game States with Actions
```python
# After players have acted, update the game states
player_actions = {
    player1: (Action.RAISE, 50),
    player2: (Action.FOLD, None),
    player3: (Action.CHECK_CALL, None)
}

# Update states with actions
GameStateHelper.update_player_actions(game_states, player_actions)
```

## Integration with AI Agents

The GameStateHelper is particularly useful for training and using AI agents:

```python
from poker.agents.imitation_agent import ImitationAgent

# Load a trained agent
agent = ImitationAgent()
agent.load('./models')

# In the game loop
for current_player in game.active_players:
    # Create game state for this player
    state = GameStateHelper.create_game_state_for_player(
        game, 
        current_player, 
        game.current_stage
    )
    
    # Get action from agent
    action = agent.predict_action(state)
    
    # Apply action to the game
    # ...
```

## Feature Extraction for Training

The helper is also used to extract features for training AI models:

```python
# For collecting training data
all_states = {}

# Throughout gameplay, collect states
def collect_game_states(game):
    stage_states = GameStateHelper.create_game_states(game, game.current_stage)
    
    if game.current_stage not in all_states:
        all_states[game.current_stage] = []
    
    all_states[game.current_stage].append(stage_states)

# After collecting, extract features for training
def extract_features():
    features = []
    labels = []
    
    for stage, states_list in all_states.items():
        for states in states_list:
            for player, state in states.items():
                # Extract features from state
                state_features = extract_features_from_state(state)
                features.append(state_features)
                
                # Get the actual action taken as label
                labels.append(state.my_player_action)
    
    return features, labels
```
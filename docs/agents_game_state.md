# GameState Module Documentation

## Overview
The `game_state.py` module defines the GameState class, which provides a representation of the poker game state from a specific player's perspective. This abstraction is crucial for AI agents to make decisions based on the available information without accessing hidden game details.

## Key Components

### GameState Class
Represents the state of the game from a specific player's point of view.

#### Initialization
```python
from poker.agents.game_state import GameState, Player

# Create a game state for a specific player
game_state = GameState(
    stage,                # Current game stage (preflop, flop, etc.)
    my_player,            # Player object with public+private info
    other_players,        # List of other players with public info
    community_cards,      # List of community cards
    pot_size,             # Current pot size  
    min_bet_to_continue,  # Amount needed to call
    core_game             # Reference to the original Game object
)
```

#### Properties
- `stage`: Current stage of the game (Stage enum)
- `my_player`: Representation of the player's own state
- `other_players`: List of representations of the other players' states
- `community_cards`: List of community cards currently visible
- `pot_size`: Total size of the pot
- `min_bet_to_continue`: Minimum amount needed to call
- `core_game`: Reference to the original Game object
- `hand_strength`: Calculated strength of the player's hand
- `community_hand_strength`: Strength of just the community cards

#### Methods
- `calculate_hand_strength()`: Evaluates the strength of the player's hand with the current community cards
- `calculate_community_strength()`: Evaluates the strength of just the community cards

### Player Class (Inner Class)
Represents a player's state within the GameState.

#### Properties
- `player_id`: Unique identifier for the player
- `name`: Player's name
- `stack_size`: Player's current stack size
- `in_game`: Whether the player is still in the current hand
- `is_all_in`: Whether the player is all-in
- `spots_left_bb`: Position relative to the big blind
- `hand`: List of the player's hole cards (only available for the main player)

## Usage in AI Agents

GameState is primarily used by AI agents to make decisions:

```python
from poker.agents.game_state import GameState
from poker.core.action import Action

class YourAIAgent:
    def act(self, game_state: GameState) -> Action:
        # Analyze game state
        my_hand = game_state.my_player.hand
        community = game_state.community_cards
        hand_strength = game_state.hand_strength
        
        # Make decisions based on state
        if hand_strength > 0.8:  # Very strong hand
            return Action.RAISE
        elif hand_strength > 0.4:  # Medium hand
            return Action.CHECK_CALL
        else:  # Weak hand
            return Action.FOLD
```

## Creating GameState from Game

To create a GameState from the core Game object:

```python
from poker.game_state_helper import GameStateHelper

# Create game states for all active players
game_states = GameStateHelper.create_game_states(game, game.current_stage)

# Get game state for a specific player
player_state = game_states[specific_player]
```

## Advanced Usage

### Tracking Game History
For more advanced agents, you can track the history of game states to analyze betting patterns:

```python
class HistoryTrackingAgent:
    def __init__(self):
        self.game_state_history = []
        
    def act(self, game_state: GameState) -> Action:
        # Add current state to history
        self.game_state_history.append(game_state)
        
        # Analyze betting patterns based on history
        if self.detect_pattern_from_history():
            return self.counter_strategy()
        
        # Default decision logic
        return self.base_strategy(game_state)
```

### Simulating Future Outcomes
For agents that look ahead at possible outcomes:

```python
def simulate_outcomes(game_state: GameState, num_simulations=1000):
    """Monte Carlo simulation of possible outcomes."""
    win_count = 0
    
    for _ in range(num_simulations):
        # Copy the game state
        sim_state = copy.deepcopy(game_state)
        
        # Add random cards to complete the board
        remaining_cards = 5 - len(sim_state.community_cards)
        sim_state.community_cards.extend(draw_random_cards(remaining_cards))
        
        # Determine if we win
        if sim_state.calculate_hand_strength() > 0.7:  # Simplified win check
            win_count += 1
    
    # Return win probability
    return win_count / num_simulations
```
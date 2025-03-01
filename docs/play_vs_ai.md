# Play vs AI Module Documentation

## Overview
The `play_vs_ai.py` module provides a command-line interface for playing poker against AI opponents. It allows users to interact with trained neural network agents and test their poker skills against different AI implementations.

## Key Components

### Visualization Functions
- `colored_card(card)`: Returns a colored string representation of a card
- `display_community_cards(cards)`: Displays community cards with colors
- `display_hand(cards)`: Displays a player's hand with colors
- `display_game_state(game, human_player)`: Displays the current game state

### Main Gameplay Function
- `play_game(model_dir='./models', num_ai_players=2, starting_stack=1000, big_blind=10, small_blind=5, num_hands=3)`: Main function to play poker against AI agents

## Usage

### Basic Play from Command Line
Run the script directly to play against the AI:

```bash
python poker/play_vs_ai.py
```

This will start a game with default settings.

### Command Line Options
Customize your game with command line arguments:

```bash
python poker/play_vs_ai.py --model-dir ./my_models --ai-players 3 --stack 2000 --big-blind 20 --small-blind 10 --hands 5
```

Available options:
- `--model-dir`: Directory containing trained models
- `--ai-players`: Number of AI players (1-5)
- `--stack`: Starting stack for all players
- `--big-blind`: Big blind amount
- `--small-blind`: Small blind amount
- `--hands`: Number of hands to play

### Importing as a Module
You can also import and use the module in your own code:

```python
from poker.play_vs_ai import play_game

# Play with custom settings
play_game(
    model_dir='./my_models',
    num_ai_players=2,
    starting_stack=2000,
    big_blind=20,
    small_blind=10,
    num_hands=5
)
```

## Gameplay Example

When running the module, you'll see a display like this during gameplay:

```
============================================================
Stage: PREFLOP
Pot: $15
Community cards: 
------------------------------------------------------------
You (Human): $990
Your hand: A♥ K♠

AI Player 1: $995
AI Player 2: $990 (folded)
============================================================

Current Pot State:
Contributions: {Human: 10, AI Player 1: 5}, Eligible: {Human, AI Player 1}
Call Amount: 0 (Check)

It is Human's turn - Hand: [A♥, K♠] - Current Stack: 990
Enter action (0=Fold, 1=Check/Call, 2=Raise):
```

You can then enter your action choice:
- `0`: Fold
- `1`: Check/Call
- `2`: Raise (you'll be prompted for the amount)

## AI Implementation

The module uses the ImitationAgent that's been trained on Pluribus data:

```python
# Load the AI agent
device = 'cuda' if torch.cuda.is_available() else 'cpu'
agent = ImitationAgent(device=device)

try:
    agent.load(model_dir)
    print(f"Loaded AI model from {model_dir}")
except FileNotFoundError:
    print(f"Could not find model in {model_dir}. Using a random player instead.")
    ai_players = [PlayerRandom(f"AI Player {i+1}", starting_stack) for i in range(num_ai_players)]
else:
    # Create AI players using the trained agent
    ai_players = [DeepLearningPlayer(f"AI Player {i+1}", starting_stack, agent) 
                for i in range(num_ai_players)]
```

If the trained model can't be found, it falls back to using random players instead.

## Customizing the UI

To customize the UI display, you can modify the visualization functions:

```python
def colored_card(card: Card) -> str:
    # Customize card display
    # ...

def display_game_state(game: Game, human_player: Player) -> None:
    # Customize game state display
    # ...
```

## Advanced Usage: Custom AI

To use a different AI implementation, create a custom class that inherits from `Player`:

```python
class YourCustomAI(Player):
    def act(self):
        # Your custom AI logic
        # ...
        return action

# Use in play_game
def play_with_custom_ai():
    custom_ai = YourCustomAI("Custom AI", 1000)
    human = PlayerIO("Human", 1000)
    
    game = Game([human, custom_ai], 20, 10)
    game.gameplay_loop()
```
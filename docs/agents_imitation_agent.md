# Imitation Agent Module Documentation

## Overview
The `imitation_agent.py` module implements a neural network-based poker agent that learns to play poker by imitating the Pluribus AI's strategies from game logs. It includes models for predicting both actions (fold, check/call, raise) and raise amounts.

## Key Components

### ImitationAgent Class
The main class that implements the imitation learning agent.

#### Initialization
```python
from poker.agents.imitation_agent import ImitationAgent

# Create a new agent
agent = ImitationAgent(device='cpu')  # or 'cuda' for GPU

# Create with pre-trained models
agent = ImitationAgent(action_model, raise_model, device='cpu')
```

#### Methods
- `create_models(input_size, hidden_size=128, dropout_rate=0.3)`: Creates neural network models
- `standardize_features(features, is_training=False)`: Standardizes input features
- `prepare_features(game_states)`: Extracts features from game states
- `train(features, action_labels, raise_amounts, ...)`: Trains the models on Pluribus data
- `predict_action(game_state)`: Predicts the best action for a given game state
- `act(game_state)`: Interface method that performs the predicted action
- `save(save_dir='./models')`: Saves trained models and parameters
- `load(load_dir='./models')`: Loads trained models and parameters
- `plot_training_history()`: Visualizes training metrics

### ActionPredictionModel Class
Neural network model for predicting poker actions.

```python
model = ActionPredictionModel(input_size=18, hidden_size=128)
```

### RaiseAmountModel Class
Neural network model for predicting raise amounts when a raise action is chosen.

```python
model = RaiseAmountModel(input_size=18, hidden_size=128)
```

### DeepLearningPlayer Class
A player implementation that uses the ImitationAgent to make decisions.

```python
from poker.agents.imitation_agent import DeepLearningPlayer, ImitationAgent

# Create agent and player
agent = ImitationAgent()
agent.load('./models')
player = DeepLearningPlayer("AI Player", 1000, agent)

# Use in a game
game = Game([player, other_players...], big_blind, small_blind)
game.gameplay_loop()
```

### PokerDataset Class
PyTorch dataset for poker decision points.

## Training the Agent

```python
from poker.agents.imitation_agent import train_imitation_agent

# Train a new agent on Pluribus data
agent = train_imitation_agent(
    log_dir='./pluribus_converted_logs',
    save_dir='./models',
    device='cpu'
)
```

The training process:
1. Parses Pluribus log files
2. Extracts game states and Pluribus decisions
3. Converts them to features and labels
4. Trains neural networks to predict actions and raise amounts
5. Saves the trained models

## Evaluating the Agent

```python
from poker.agents.imitation_agent import evaluate_agent

# Evaluate on held-out data
metrics = evaluate_agent(agent, log_dir='./pluribus_converted_logs', num_samples=1000)

# View metrics
print(f"Accuracy: {metrics['accuracy']}")
print(f"Action accuracies: {metrics['action_accuracies']}")
```

## Playing Against the Agent

```python
from poker.agents.imitation_agent import ImitationAgent, DeepLearningPlayer
from poker.play_vs_ai import play_game

# Load a trained agent
agent = ImitationAgent()
agent.load('./models')

# Play against the agent
play_game(model_dir='./models', num_ai_players=2, num_hands=5)
```

## Feature Engineering

The agent uses the following features:
- Game stage (preflop, flop, turn, river)
- Pot size and minimum bet to continue
- Player's stack size and position
- Number of community cards
- Hand strength and community hand strength
- Number of active opponents
- Pot-to-stack ratio and pot odds
- Position categories (early, middle, late)

## Advanced Usage: Custom Training

```python
from poker.agents.imitation_agent import ImitationAgent
from poker.parsers.game_state_retriever import GameStateRetriever

# Get training data
retriever = GameStateRetriever('./your_logs')
retriever.initialize()
decisions = retriever.get_pluribus_decisions()

# Extract features and labels
features = []
action_labels = []
raise_amounts = []
# ... Process decisions into features and labels ...

# Create and train agent
agent = ImitationAgent()
agent.train(
    features=features,
    action_labels=action_labels,
    raise_amounts=raise_amounts,
    val_split=0.1,
    batch_size=64,
    epochs=20
)

# Save the agent
agent.save('./your_models')
```
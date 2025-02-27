# PokerRL: Deep Learning for Poker

This project implements both Deep Q-Network (DQN) reinforcement learning and imitation learning agents for Texas Hold'em poker. The agents learn optimal poker strategies through self-play, expert imitation, and evaluation.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Training Agents](#training-agents)
  - [DQN Training](#dqn-training)
  - [Imitation Learning](#imitation-learning)
- [Evaluating Agents](#evaluating-agents)
- [Adding Custom Models](#adding-custom-models)
- [Future Improvements](#future-improvements)

## Installation

### Dependencies

The project requires Python 3.11+ and PyTorch. You can set up the environment using the provided `env.yml` file:

```bash
conda env create -f env.yml
conda activate poker_rl
```

### Manual Installation

Alternatively, you can install the dependencies manually:

```bash
conda create -n poker_rl python=3.11
conda activate poker_rl
conda install -c conda-forge matplotlib numpy
conda install -c pytorch -c nvidia pytorch pytorch-cuda=11.8
```

## Project Structure

```
PokerRL/
├── env.yml                  # Conda environment file
├── poker/                   # Main package
│   ├── __init__.py
│   ├── agents/              # AI agents
│   │   ├── deep_learning_agent.py
│   │   ├── game_state.py
│   │   ├── imitation_agent.py
│   │   ├── dqn_agent.py
│   │   └── util.py
│   ├── core/                # Poker game mechanics
│   │   ├── action.py
│   │   ├── card.py
│   │   ├── deck.py
│   │   ├── game.py
│   │   ├── gamestage.py
│   │   ├── hand_evaluator.py
│   │   ├── player.py
│   │   ├── pokerhand.py
│   │   └── pot.py
│   ├── parsers/             # Data parsers
│   │   └── pluribus_parser.py
│   ├── evaluate_basic.py    # Basic evaluation script
│   ├── evaluate_dqn.py      # DQN evaluation script
│   ├── evaluate_imitation.py # Imitation learning evaluation
│   ├── player_io.py         # Human player interface
│   ├── tests.py             # Basic game tests
│   ├── train_dqn.py         # DQN training script
│   └── train_imitation.py   # Imitation learning script
├── models/                  # Saved models directory
└── pluribus_converted_logs/ # Pluribus game logs for training
```

## Core Components

### Poker Game Engine (`poker/core/`)

The core module implements the mechanics of Texas Hold'em poker:

- `Card` and `Deck`: Card representations and deck operations
- `Game`: Manages game flow, rounds, and player interactions
- `Player`: Base player class for both AI and human players
- `Action`: Enumeration of possible poker actions (fold, check, call, raise)
- `Pot`: Handles pot management and side pots
- `hand_evaluator`: Evaluates poker hand strengths

### AI Agents (`poker/agents/`)

- `DQNAgent`: Deep Q-Network implementation for reinforcement learning
- `ImitationLearningAgent`: Neural network agent that learns from expert data
- `game_state`: State representation for decision-making

### Data Parsers (`poker/parsers/`)

- `pluribus_parser.py`: Parses Pluribus poker logs for imitation learning

## Training Agents

### DQN Training

The DQN agent implementation includes:

- Neural network architecture for Q-value estimation
- Experience replay for stable learning
- Target network for reducing overestimation bias
- Epsilon-greedy exploration strategy

To train the DQN agent, run:

```bash
python -m poker.train_dqn
```

This will train the agent for 10,000 episodes and save checkpoints in the `models/` directory. Training parameters can be adjusted in the script.

### Imitation Learning

The imitation learning approach learns poker strategies by mimicking the expert decisions from Pluribus logs.

1. First, create a dataset from Pluribus logs:

```bash
python -c "from poker.parsers.pluribus_parser import create_imitation_dataset; create_imitation_dataset('pluribus_converted_logs', 'models/pluribus_dataset.npz')"
```

2. Train the imitation learning agent:

```bash
python -m poker.train_imitation --logs pluribus_converted_logs --output models --epochs 50 --eval-games 10
```

## Evaluating Agents

### Basic Evaluation

For a quick evaluation of the imitation learning agent:

```bash
python -m poker.evaluate_basic
```

### DQN Agent Evaluation

To evaluate a trained DQN agent against random players:

```bash
python -m poker.evaluate_dqn --model models/dqn_agent_final.pt --games 100
```

Command line arguments:
- `--model`: Path to saved model
- `--games`: Number of games to play (default: 100)
- `--players`: Number of players per game (default: 6)
- `--stack`: Initial stack size (default: 1000)
- `--big-blind`: Big blind amount (default: 20)
- `--small-blind`: Small blind amount (default: 10)

### Imitation Agent Evaluation

For a comprehensive evaluation of the imitation learning agent:

```bash
python -m poker.evaluate_imitation --model models/imitation_agent.pt --games 50
```

## State Representation

The agents' state representation includes:
- Pot size and minimum bet to continue
- Player stack size
- Hand strength and community card strength
- Position information and game stage
- Number of active players

## Adding Custom Models

The framework is designed to be easily extensible. Follow these steps to create a custom agent:

### 1. Extend the Base Player Class

Create a new class inheriting from `poker.core.player.Player`:

```python
from poker.core.player import Player
from poker.core.action import Action

class MyCustomAgent(Player):
    def __init__(self, name, stack_size):
        super().__init__(name, stack_size)
        
    def act(self) -> Action:
        # Implement policy
        return chosen_action
```

### 2. Implement Your Neural Network

If using a neural network, create a PyTorch model:

```python
import torch.nn as nn

class MyCustomNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)
```

### 3. Create a Training Script

Implement a training script for your agent:

```python
def train_my_agent(episodes=10000, save_path="models/my_custom_agent.pt"):
    # Setup environment, agent, etc.
    agent = MyCustomAgent("MyAgent", 1000)
    
    # Training loop
    for episode in range(episodes):
        # Setup game
        # Train agent
        # Save checkpoints
    
    # Save final model
    agent.save(save_path)
```

### 4. Create an Evaluation Script

Implement a script to evaluate your agent's performance:

```python
def evaluate_my_agent(model_path, num_games=100):
    # Load agent
    agent = MyCustomAgent("MyAgent", 1000)
    agent.load(model_path)
    
    # Run evaluation games
    # Report metrics (win rate, profit, etc.)
```
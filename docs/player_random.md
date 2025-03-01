# PlayerRandom Module Documentation

## Overview
The `player_random.py` module implements a Player subclass that makes random but valid poker decisions. It provides a baseline opponent for testing and a simple AI player for gameplay.

## Key Components

### PlayerRandom Class
A Player implementation that makes random decisions.

#### Initialization
```python
from poker.player_random import PlayerRandom

# Create a random player with a starting stack
random_player = PlayerRandom("Random Player", 1000)
```

#### Methods
- `act()`: Makes a random valid action choice
- `max_raise_amount()`: Calculates the maximum possible raise amount
- `get_raise_amt(max_raise_amount)`: Selects a random raise amount

## Usage

### Adding Random Players to a Game
```python
from poker.core.game import Game
from poker.player_random import PlayerRandom
from poker.player_io import PlayerIO

# Create players
human = PlayerIO("Human", 1000)
random1 = PlayerRandom("AI Random 1", 1000)
random2 = PlayerRandom("AI Random 2", 1000)

# Create and play a game
game = Game([human, random1, random2], 20, 10)
game.gameplay_loop()
```

### Random Decision Logic

The PlayerRandom class makes decisions based on the following logic:

1. First, it determines which actions are valid:
   - FOLD is always valid (unless forced all-in)
   - CHECK_CALL is always valid
   - RAISE is only valid if the player has enough chips to cover the minimum raise

2. In certain situations, it avoids folding:
   - If it's not preflop and there hasn't been a raise in the current round
   
3. It then randomly selects from the valid actions

4. If RAISE is selected, it uses a weighted random approach to select a raise amount:
   - Amounts closer to the minimum raise are more likely
   - Amounts closer to the maximum possible raise are less likely

## Raise Amount Selection

The raise amount selection uses a probability distribution that decreases linearly from the minimum raise to the maximum possible raise:

```python
def get_raise_amt(self, max_raise_amount):
    if max_raise_amount < self.game.min_bet:
        return 0

    min_raise = self.game.min_bet
    choices = np.arange(min_raise, max_raise_amount + 1)
    probabilities = np.linspace(1, 0.1, len(choices))
    probabilities /= probabilities.sum()

    return np.random.choice(choices, p=probabilities)
```

This means that the random player is more likely to make smaller raises than larger ones, which matches the general pattern of human play.

## Integration with Other Player Types

To create a game with mixed player types:

```python
from poker.core.game import Game
from poker.player_random import PlayerRandom
from poker.player_io import PlayerIO
from poker.agents.imitation_agent import ImitationAgent, DeepLearningPlayer

# Create players of different types
human = PlayerIO("Human", 1000)

agent = ImitationAgent()
agent.load('./models')
ai_player = DeepLearningPlayer("Pluribus AI", 1000, agent)

random_player = PlayerRandom("Random AI", 1000)

# Create and play a game with mixed player types
game = Game([human, ai_player, random_player], 20, 10)
game.gameplay_loop()
```

## Using for Evaluation

Random players can be useful for evaluating AI performance:

```python
def evaluate_against_random(ai_agent, num_games=100):
    ai_wins = 0
    
    for _ in range(num_games):
        ai_player = DeepLearningPlayer("AI", 1000, ai_agent)
        random_player = PlayerRandom("Random", 1000)
        
        # Play a game
        game = Game([ai_player, random_player], 20, 10)
        game.gameplay_loop()
        
        # Check who won (simplified)
        if ai_player.stack > 1000:
            ai_wins += 1
    
    win_rate = ai_wins / num_games
    print(f"AI win rate against random player: {win_rate:.2f}")
    return win_rate
```

## Advanced Usage: Customizing Randomness

To create a more aggressive or conservative random player, you can extend the class:

```python
class AggressiveRandom(PlayerRandom):
    def act(self):
        # Increase probability of raising
        if self.can_raise() and random.random() < 0.7:  # 70% chance to raise if possible
            self.handle_raise(self.get_raise_amt(self.max_raise_amount()))
            return Action.RAISE
        
        # Reduce probability of folding
        if random.random() < 0.2:  # Only 20% chance to fold
            self.handle_fold()
            return Action.FOLD
        
        # Otherwise call
        self.handle_check_call()
        return Action.CHECK_CALL
```
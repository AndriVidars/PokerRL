# PlayerIO Module Documentation

## Overview
The `player_io.py` module implements a Player subclass that allows human interaction through console input. It provides a way for users to play poker against AI agents by making decisions through the command line interface.

## Key Components

### PlayerIO Class
A Player implementation that uses console input for decision making.

#### Initialization
```python
from poker.player_io import PlayerIO

# Create a human player with a starting stack
human_player = PlayerIO("Human", 1000)
```

#### Methods
- `act()`: Prompts the user for an action and processes it
- `print_hand()`: Displays the player's hole cards
- `get_action()`: Gets user input for an action choice
- `raise_select()`: Handles raise amount selection

## Usage

### Playing a Game with Human Interaction
```python
from poker.core.game import Game
from poker.player_io import PlayerIO
from poker.player_random import PlayerRandom

# Create players
human = PlayerIO("Human", 1000)
ai1 = PlayerRandom("AI Player 1", 1000)
ai2 = PlayerRandom("AI Player 2", 1000)

# Create and play a game
game = Game([human, ai1, ai2], 20, 10)
game.gameplay_loop()
```

### User Input
When it's the human player's turn, they'll see a prompt with the current game state and available actions:

```
Current Pot State:
Contributions: {Human: 10, AI Player 1: 20}, Eligible: {Human, AI Player 1}
Call Amount: 10

It is Human's turn - Hand: [A♥, K♠] - Current Stack: 990
Enter action (0=Fold, 1=Check/Call, 2=Raise):
```

The user can enter:
- `0`: Fold their hand
- `1`: Check (if no bet to call) or call the current bet
- `2`: Raise, which will prompt for a raise amount

If raise is selected:
```
Enter raise amount (min 20, max 990):
```

## Integration with AI Players

To create a game with both human and AI players:

```python
from poker.core.game import Game
from poker.player_io import PlayerIO
from poker.agents.imitation_agent import ImitationAgent, DeepLearningPlayer

# Create a human player
human = PlayerIO("Human", 1000)

# Create an AI agent
agent = ImitationAgent()
agent.load('./models')
ai_player = DeepLearningPlayer("Pluribus AI", 1000, agent)

# Create another random player
random_player = PlayerRandom("Random AI", 1000)

# Create and play a game with mixed player types
game = Game([human, ai_player, random_player], 20, 10)
game.gameplay_loop()
```

## Display Enhancements

To improve the user experience, you might want to extend the PlayerIO class with better card visualization:

```python
def colored_card(card):
    """Return a colored string representation of a card"""
    suits = {
        Suit.HEART: '♥',
        Suit.DIAMOND: '♦',
        Suit.CLUB: '♣',
        Suit.SPADE: '♠'
    }
    
    colors = {
        Suit.HEART: '\033[91m',  # Red
        Suit.DIAMOND: '\033[91m',  # Red
        Suit.CLUB: '\033[0m',  # Default
        Suit.SPADE: '\033[0m'  # Default
    }
    
    rank_str = str(card.rank).split('.')[-1]
    if rank_str == 'TEN':
        rank_str = 'T'
    elif rank_str == 'JACK':
        rank_str = 'J'
    elif rank_str == 'QUEEN':
        rank_str = 'Q'
    elif rank_str == 'KING':
        rank_str = 'K'
    elif rank_str == 'ACE':
        rank_str = 'A'
    else:
        rank_str = rank_str[0]
    
    return f"{colors[card.suit]}{rank_str}{suits[card.suit]}\033[0m"

# Then in PlayerIO
def print_hand(self):
    hand_str = " ".join(colored_card(card) for card in self.hand)
    print(f"Your hand: {hand_str}")
```

## Game Loop with Multiple Hands

To play multiple hands:

```python
def play_multiple_hands(num_hands=10):
    human = PlayerIO("Human", 1000)
    ai1 = PlayerRandom("AI 1", 1000)
    ai2 = PlayerRandom("AI 2", 1000)
    
    players = [human, ai1, ai2]
    
    for i in range(num_hands):
        print(f"\n--- Hand {i+1}/{num_hands} ---\n")
        
        # Reset player hands
        for player in players:
            player.hand = []
            player.folded = False
            player.all_in = False
        
        # Create and play a new game
        game = Game(players, 20, 10)
        game.gameplay_loop()
        
        # Display current stacks
        print("\nCurrent stacks:")
        for player in players:
            print(f"{player.name}: ${player.stack}")
            
        # Ask to continue
        if i < num_hands - 1:
            choice = input("\nContinue to next hand? (y/n): ")
            if choice.lower() != 'y':
                break
```
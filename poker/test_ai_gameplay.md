# Test AI Gameplay Module Documentation

## Overview
The `test_ai_gameplay.py` module provides a framework for testing and demonstrating AI gameplay. It simulates poker games between AI players and outputs detailed information about the game progression, player decisions, and hand evaluations.

## Key Components

### Visualization Functions
- `colored_card(card)`: Returns a colored string representation of a card
- `display_cards(cards)`: Displays a list of cards with colors
- `display_game_state(game)`: Displays the current game state

### Game Analysis Functions
- `determine_winners(game)`: Determines which player(s) won each pot
- `award_pot(game, winners)`: Awards pot amounts to winning players with detailed output

### Logging System
- `log_action(player, action, amount)`: Logs player actions
- `GameLogger`: Class that wraps player action methods to log all gameplay activity

### Main Gameplay Function
- `play_test_game(num_hands=1)`: Main function to simulate poker gameplay between a trained AI and random players

## Usage

### Running a Test Game
Execute the script to run a test simulation:

```bash
python poker/test_ai_gameplay.py
```

This will simulate poker hands between a trained AI agent and random players, showing detailed game progression.

### Customizing Test Parameters
Modify the script parameters to customize the test:

```python
# Change the number of hands to simulate
play_test_game(num_hands=5)

# Change initial stacks or adjust blind levels
big_blind = 50
small_blind = 25
```

### Importing as a Module
You can also import and use the module in your own testing framework:

```python
from poker.test_ai_gameplay import play_test_game, determine_winners, award_pot

# Run a test simulation with custom parameters
play_test_game(num_hands=10)

# Or use the individual components for your own testing
game = setup_your_game()
winners = determine_winners(game)
award_pot(game, winners)
```

## Output Example

The test script produces detailed output of the game progression:

```
################################################################################
Hand 1/2
################################################################################

============================================================
Stage: PREFLOP
Pot: $30
Community cards: 
------------------------------------------------------------
AI (Pluribus): $1000 
  Hand: A♥ K♠
RandomPlayer1: $990 
  Hand: Q♥ J♣
RandomPlayer2: $980 
  Hand: 8♥ 9♦
============================================================
AI (Pluribus) -> RAISE 50

...

-------------------------
Flop
-------------------------

Community Cards After Flop:
2♣ K♦ Q♣
RandomPlayer1 -> CHECK_CALL
RandomPlayer2 -> FOLD
AI (Pluribus) -> RAISE 100

...

Pot #1 ($350) winners:
  AI (Pluribus) wins $350 with One Pair: A♥ K♠

Hand completed!
Final community cards: 2♣ K♦ Q♣ 9♠ 4♥
```

## Hand Evaluation

The module includes detailed hand evaluation output:

```
Pot #1 ($350) winners:
  AI (Pluribus) wins $350 with One Pair: A♥ K♠
```

For each pot, it shows:
- Which player won
- How much they won
- What poker hand they had (Pair, Straight, etc.)
- The cards that made up their hand

## Blind Progression

The module can simulate blind increases to speed up eliminations:

```python
# Increase blinds every 5 hands
if hand_num > 0 and hand_num % 5 == 0:
    big_blind *= 2
    small_blind = big_blind // 2
    print(f"\n*** BLINDS INCREASED TO ${small_blind}/${big_blind} ***\n")
```

This creates a more realistic tournament-style progression.

## Advanced Usage: Custom Analysis

Extend the module with your own analysis functions:

```python
def analyze_ai_decisions(all_actions):
    """Analyze the AI's decision patterns"""
    action_counts = {'FOLD': 0, 'CHECK_CALL': 0, 'RAISE': 0}
    
    for player, action, _ in all_actions:
        if player.name == "AI (Pluribus)":
            action_counts[action.name] += 1
    
    total = sum(action_counts.values())
    print("AI Decision Analysis:")
    for action, count in action_counts.items():
        print(f"{action}: {count} ({count/total:.2%})")
```
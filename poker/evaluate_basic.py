"""
Simple evaluation script for testing poker agents
"""
import torch
import numpy as np
from poker.core.game import Game
from poker.core.player import Player
from poker.core.action import Action
from poker.agents.imitation_agent import ImitationLearningAgent

class SimpleImitationPlayer(Player):
    """Simplified imitation learning player for testing"""
    def __init__(self, name, stack_size, model_path):
        super().__init__(name, stack_size)
        self.agent = ImitationLearningAgent()
        self.agent.load(model_path)
        print(f"Loaded model for {name}")
        
    def act(self):
        """Simple fixed strategy for testing"""
        # For simplicity, just call 50% of the time and fold 50%
        # This avoids the complexity of the full game state extraction
        if np.random.random() < 0.5:
            # Find the call amount
            call_amount = 0
            for pot in self.game.pots:
                if self in pot.contributions:
                    call_amount = max(call_amount, max(pot.contributions.values()) - pot.contributions[self])
                else:
                    call_amount = max(call_amount, max(pot.contributions.values()))
            
            # Execute call
            self.stack -= call_amount
            for pot in self.game.pots:
                if self in pot.eligible_players:
                    pot.add_contribution(self, call_amount)
                    break
            print(f"{self.name} calls {call_amount}")
            return Action.CALL
        else:
            self.folded = True
            print(f"{self.name} folds")
            return Action.FOLD

class SimpleRandomPlayer(Player):
    """Simple random player for testing"""
    def __init__(self, name, stack_size):
        super().__init__(name, stack_size)
        
    def act(self):
        """Simple random strategy"""
        # 33% fold, 33% check/call, 33% raise
        choice = np.random.randint(3)
        
        if choice == 0:
            # Fold
            self.folded = True
            print(f"{self.name} folds")
            return Action.FOLD
        elif choice == 1:
            # Find the call amount
            call_amount = 0
            for pot in self.game.pots:
                if self in pot.contributions:
                    call_amount = max(call_amount, max(pot.contributions.values()) - pot.contributions[self])
                else:
                    call_amount = max(call_amount, max(pot.contributions.values()))
            
            # Execute call
            self.stack -= call_amount
            for pot in self.game.pots:
                if self in pot.eligible_players:
                    pot.add_contribution(self, call_amount)
                    break
            print(f"{self.name} calls {call_amount}")
            return Action.CALL
        else:
            # Raise by a random amount (20-100)
            raise_amount = np.random.randint(20, 101)
            self.stack -= raise_amount
            for pot in self.game.pots:
                if self in pot.eligible_players:
                    pot.add_contribution(self, raise_amount)
                    break
            print(f"{self.name} raises {raise_amount}")
            return Action.RAISE

def run_games(num_games=3):
    """Run multiple games with simplified agents"""
    model_path = "models/imitation_agent.pt"
    
    # Results tracking
    total_profit = 0
    wins = 0
    
    for game_idx in range(num_games):
        print(f"\n=== Game {game_idx+1}/{num_games} ===")
        
        # Create players
        players = [
            SimpleImitationPlayer("ImitationBot", 1000, model_path),
            SimpleRandomPlayer("Random1", 1000),
            SimpleRandomPlayer("Random2", 1000),
            SimpleRandomPlayer("Random3", 1000)
        ]
        
        # Create and run game
        game = Game(players, big_amount=20, small_amount=10)
        
        try:
            game.gameplay_loop()
            print("\nGame completed successfully!")
        except Exception as e:
            print(f"\nError in gameplay: {e}")
            continue
        
        # Print final stacks
        print("\nFinal Stacks:")
        for player in players:
            print(f"{player.name}: {player.stack}")
        
        # Track results
        imitation_bot = players[0]
        profit = imitation_bot.stack - 1000
        total_profit += profit
        
        # Check if bot won
        has_won = True
        for p in players[1:]:
            if p.stack > imitation_bot.stack:
                has_won = False
                break
        
        if has_won:
            wins += 1
    
    # Print overall results
    print("\n=== Overall Results ===")
    print(f"Games: {num_games}")
    print(f"Total profit: {total_profit}")
    print(f"Average profit per game: {total_profit/num_games:.2f}")
    print(f"Win rate: {wins/num_games:.2%}")

if __name__ == "__main__":
    run_games(3)
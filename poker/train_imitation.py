import os
import argparse
import numpy as np
import torch

from poker.parsers.pluribus_parser import create_imitation_dataset
from poker.agents.imitation_agent import ImitationLearningAgent
from poker.core.player import Player
from poker.core.action import Action
from poker.core.game import Game
from poker.core.gamestage import Stage


class ImitationPlayer(Player):
    """
    Player that uses an imitation learning agent to make decisions
    """
    def __init__(self, name: str, stack_size: int, agent: ImitationLearningAgent):
        super().__init__(name, stack_size)
        self.agent = agent
        self.current_state = None
        
    def act(self) -> Action:
        # Get game state
        game_state = self._create_game_state()
        state_features = self.agent.preprocess_state(game_state)
        
        # Select action using agent
        action_idx, raise_amount = self.agent.select_action(state_features)
        
        # Convert to poker action
        action, raise_size = self.agent.convert_action_to_poker_action(
            action_idx, raise_amount, game_state
        )
        
        # Execute action in the game
        return self._execute_action(action, raise_size)
    
    def _create_game_state(self):
        """
        Create a GameState object for the agent
        """
        from poker.agents.game_state import GameState, Player as StatePlayer
        
        other_players = [p for p in self.game.players if p != self]
        
        # Calculate min bet to continue
        min_bet = 0
        for pot in self.game.pots:
            if self in pot.contributions:
                min_bet = max(min_bet, max(pot.contributions.values()) - pot.contributions.get(self, 0))
            else:
                min_bet = max(min_bet, max(pot.contributions.values())) if pot.contributions else 0
        
        # Calculate pot size
        pot_size = sum(pot.total_amount for pot in self.game.pots)
        if pot_size == 0:
            pot_size = 1  # Avoid division by zero
        
        # Create my player state
        my_player = StatePlayer(
            spots_left_bb=self.game.players.index(self),  # Position
            cards=tuple(self.hand),  # Convert to tuple
            stack_size=self.stack
        )
        
        # Create other player states
        other_player_states = []
        for p in other_players:
            player_state = StatePlayer(
                spots_left_bb=self.game.players.index(p),  # Position
                cards=tuple(p.hand) if hasattr(p, 'hand_visible') and p.hand_visible else None,
                stack_size=p.stack
            )
            # Add history to track if player has folded
            if p.folded:
                player_state.add_preflop_action(Action.FOLD, None)
            other_player_states.append(player_state)
        
        # Create GameState
        game_state = GameState(
            stage=self.game.current_stage,
            community_cards=self.game.community_cards,
            pot_size=pot_size,
            min_bet_to_continue=min_bet,
            my_player=my_player,
            other_players=other_player_states,
            my_player_action=None  # Will be filled later when we act
        )
        
        return game_state
    
    def _execute_action(self, action: Action, raise_size: int) -> Action:
        """
        Execute the chosen action in the game
        """
        if action == Action.FOLD:
            self.folded = True
            print(f"{self.name} folds")
            return Action.FOLD
            
        elif action == Action.CHECK:
            # Check if we can check
            can_check = True
            for pot in self.game.pots:
                if self in pot.contributions:
                    if pot.contributions[self] < max(pot.contributions.values()):
                        can_check = False
                else:
                    can_check = False
            
            if can_check:
                print(f"{self.name} checks")
                return Action.CHECK
            else:
                # If we can't check, we fold
                self.folded = True
                print(f"{self.name} folds (invalid check)")
                return Action.FOLD
                
        elif action == Action.CALL:
            # Calculate amount to call
            call_amount = 0
            for pot in self.game.pots:
                if self in pot.contributions:
                    call_amount = max(call_amount, max(pot.contributions.values()) - pot.contributions[self])
                else:
                    call_amount = max(call_amount, max(pot.contributions.values()))
            
            # If we can't afford to call, go all-in
            if call_amount >= self.stack:
                call_amount = self.stack
                self.all_in = True
            
            # Execute call
            self.stack -= call_amount
            for pot in self.game.pots:
                if self in pot.eligible_players:
                    pot.add_contribution(self, call_amount)
                    break
            
            print(f"{self.name} calls {call_amount}")
            return Action.CALL
            
        elif action == Action.RAISE:
            # Ensure raise is valid
            if raise_size > self.stack:
                raise_size = self.stack
                self.all_in = True
            
            self.stack -= raise_size
            for pot in self.game.pots:
                if self in pot.eligible_players:
                    pot.add_contribution(self, raise_size)
                    break
            
            print(f"{self.name} raises {raise_size}")
            return Action.RAISE
            
        return Action.FOLD  # Default
    


class RandomPlayer(Player):
    """
    Player that makes random decisions
    """
    def __init__(self, name: str, stack_size: int):
        super().__init__(name, stack_size)
    
    def act(self) -> Action:
        # Random action
        actions = [Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE]
        action = np.random.choice(actions)
        
        if action == Action.FOLD:
            self.folded = True
            return Action.FOLD
            
        elif action == Action.CHECK:
            # Check if we can check
            can_check = True
            for pot in self.game.pots:
                if self in pot.contributions:
                    if pot.contributions[self] < max(pot.contributions.values()):
                        can_check = False
                else:
                    can_check = False
            
            if can_check:
                return Action.CHECK
            else:
                # If we can't check, fold 50% of the time, call 50%
                if np.random.random() < 0.5:
                    self.folded = True
                    return Action.FOLD
                else:
                    action = Action.CALL
        
        if action == Action.CALL:
            # Calculate amount to call
            call_amount = 0
            for pot in self.game.pots:
                if self in pot.contributions:
                    call_amount = max(call_amount, max(pot.contributions.values()) - pot.contributions[self])
                else:
                    call_amount = max(call_amount, max(pot.contributions.values()))
            
            # If we can't afford to call, go all-in
            if call_amount >= self.stack:
                call_amount = self.stack
                self.all_in = True
            
            # Execute call
            self.stack -= call_amount
            for pot in self.game.pots:
                if self in pot.eligible_players:
                    pot.add_contribution(self, call_amount)
                    break
            
            return Action.CALL
            
        elif action == Action.RAISE:
            # Random raise size
            min_raise = 0
            for pot in self.game.pots:
                if self in pot.contributions:
                    min_raise = max(min_raise, max(pot.contributions.values()) - pot.contributions.get(self, 0))
                else:
                    min_raise = max(min_raise, max(pot.contributions.values())) if pot.contributions else 0
            
            # Random raise between min and stack
            raise_size = min_raise + int(np.random.random() * (self.stack - min_raise))
            if raise_size <= min_raise:
                raise_size = min_raise + 1
            
            # Ensure raise is valid
            if raise_size > self.stack:
                raise_size = self.stack
                self.all_in = True
            
            self.stack -= raise_size
            for pot in self.game.pots:
                if self in pot.eligible_players:
                    pot.add_contribution(self, raise_size)
                    break
            
            return Action.RAISE
            
        return Action.FOLD  # Default


def train_and_evaluate(logs_dir: str, output_dir: str, num_eval_games: int = 100, epochs: int = 50):
    """
    Train an imitation learning agent on the Pluribus dataset and evaluate it
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    dataset_path = os.path.join(output_dir, "pluribus_dataset.npz")
    if not os.path.exists(dataset_path):
        print(f"Creating dataset from {logs_dir}...")
        create_imitation_dataset(logs_dir, dataset_path)
    else:
        print(f"Using existing dataset at {dataset_path}")
    
    # Train agent
    model_path = os.path.join(output_dir, "imitation_agent.pt")
    agent = ImitationLearningAgent()
    agent.train(dataset_path, epochs=epochs)
    agent.save(model_path)
    
    # Evaluate agent
    print("\nEvaluating agent...")
    evaluate_agent(agent, num_games=num_eval_games)


def evaluate_agent(agent: ImitationLearningAgent, num_games: int = 100):
    """
    Evaluate the agent against random players
    """
    # Metrics
    total_profit = 0
    win_count = 0
    
    for game_idx in range(num_games):
        print(f"\nGame {game_idx+1}/{num_games}")
        
        # Create players
        initial_stack = 1000
        num_players = 6
        big_blind = 20
        small_blind = 10
        
        imitation_player = ImitationPlayer("ImitationBot", initial_stack, agent)
        other_players = [
            RandomPlayer(f"Random_{i}", initial_stack) 
            for i in range(num_players-1)
        ]
        all_players = [imitation_player] + other_players
        
        # Create game
        game = Game(all_players, big_blind, small_blind)
        
        # Play game
        try:
            game.gameplay_loop()
        except Exception as e:
            print(f"Error in game {game_idx}: {e}")
            continue
        
        # Calculate results
        profit = imitation_player.stack - initial_stack
        total_profit += profit
        
        # Check if imitation player won
        has_won = True
        for player in other_players:
            if player.stack > imitation_player.stack:
                has_won = False
                break
        
        if has_won:
            win_count += 1
        
        print(f"Game {game_idx+1} result: Profit = {profit}, Stack = {imitation_player.stack}")
    
    # Print overall results
    print("\n=== Evaluation Results ===")
    print(f"Number of games: {num_games}")
    print(f"Total profit: {total_profit}")
    print(f"Average profit per game: {total_profit / num_games:.2f}")
    print(f"Win rate: {win_count / num_games:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate an imitation learning agent")
    parser.add_argument("--logs", type=str, default="pluribus_converted_logs", help="Directory containing Pluribus log files")
    parser.add_argument("--output", type=str, default="models", help="Output directory for models and datasets")
    parser.add_argument("--eval-games", type=int, default=50, help="Number of games to play for evaluation")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    
    args = parser.parse_args()
    
    train_and_evaluate(args.logs, args.output, args.eval_games, args.epochs)
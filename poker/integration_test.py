import sys
from typing import Dict, List, Tuple

from poker.core.game import Game
from poker.core.player import Player
from poker.core.action import Action
from poker.core.card import Card, Rank, Suit
from poker.core.gamestage import Stage
from poker.core.deck import Deck
from poker.core.pot import Pot
from poker.agents.game_state import GameState
from poker.game_state_helper import GameStateHelper

class TestPlayer(Player):
    """Simple player that follows a predefined strategy for testing"""
    
    def __init__(self, name: str, stack: int, strategy: List[Action]):
        super().__init__(name, stack)
        self.strategy = strategy
        self.action_index = 0
        
    def act(self) -> Action:
        # Get the next action from strategy
        if self.action_index < len(self.strategy):
            action = self.strategy[self.action_index]
            self.action_index += 1
        else:
            # Default to check/call if no more actions
            action = Action.CHECK_CALL
        
        # Handle the action
        if action == Action.FOLD:
            self.handle_fold()
        elif action == Action.CHECK_CALL:
            self.handle_check_call()
        elif action == Action.RAISE:
            # Simple raise of min bet
            raise_amount = max(self.game.min_bet, 10)
            self.handle_raise(raise_amount)
            
        return action
        
def run_integration_test():
    """
    Run an integration test that demonstrates the connection between
    the core game and the game state helper.
    """
    # Create players with predefined strategies
    # Player 1: Aggressive, raises and reraises
    # Player 2: Tight, folds early
    # Player 3: Calls and checks
    p1_strategy = [Action.RAISE, Action.RAISE, Action.RAISE]
    p2_strategy = [Action.FOLD]
    p3_strategy = [Action.CHECK_CALL, Action.CHECK_CALL, Action.CHECK_CALL]
    
    player1 = TestPlayer("Player 1", 1000, p1_strategy)
    player2 = TestPlayer("Player 2", 1000, p2_strategy)
    player3 = TestPlayer("Player 3", 1000, p3_strategy)
    
    game = Game([player1, player2, player3], 10, 5)
    
    # Create a record of player actions by stage
    player_actions: Dict[Stage, Dict[Player, Tuple[Action, int]]] = {
        Stage.PREFLOP: {},
        Stage.FLOP: {},
        Stage.TURN: {},
        Stage.RIVER: {}
    }
    
    # Run the preflop
    print("\n--- PREFLOP ---")
    game.preflop()
    
    # Record player actions
    player_actions[Stage.PREFLOP] = {
        player1: (Action.RAISE, 10),
        player2: (Action.FOLD, None),
        player3: (Action.CHECK_CALL, None)
    }
    
    # Create game states for preflop
    preflop_states = GameStateHelper.create_game_states(game, Stage.PREFLOP)
    
    # Update states with player actions
    GameStateHelper.update_player_actions(preflop_states, player_actions[Stage.PREFLOP])
    
    # Show game states
    print(f"\nCreated {len(preflop_states)} game states for PREFLOP")
    for player, state in preflop_states.items():
        print(f"\nGameState for {player.name}:")
        print(f"Stage: {state.stage.name}")
        print(f"Pot size: {state.pot_size}")
        print(f"Min bet to continue: {state.min_bet_to_continue}")
        print(f"Community cards: {[str(card) for card in state.community_cards]}")
        print(f"Hand: {[str(card) for card in player.hand]}")
        print(f"Position: {state.my_player.spots_left_bb}")
        print(f"Stack size: {state.my_player.stack_size}")
        print(f"Action taken: {state.my_player_action}")
    
    # Only proceed to flop if at least 2 players are active
    if len(game.active_players) > 1:
        # Run the flop
        print("\n--- FLOP ---")
        game.next_stage()
        game.flop()
        
        # Record player actions
        player_actions[Stage.FLOP] = {
            player1: (Action.RAISE, 10),
            player3: (Action.CHECK_CALL, None)
        }
        
        # Create game states for flop
        flop_states = GameStateHelper.create_game_states(game, Stage.FLOP)
        
        # Update states with player actions
        GameStateHelper.update_player_actions(flop_states, player_actions[Stage.FLOP])
        
        # Show game states
        print(f"\nCreated {len(flop_states)} game states for FLOP")
        for player, state in flop_states.items():
            print(f"\nGameState for {player.name}:")
            print(f"Stage: {state.stage.name}")
            print(f"Pot size: {state.pot_size}")
            print(f"Min bet to continue: {state.min_bet_to_continue}")
            print(f"Community cards: {[str(card) for card in state.community_cards]}")
            print(f"Hand: {[str(card) for card in player.hand]}")
            print(f"Position: {state.my_player.spots_left_bb}")
            print(f"Stack size: {state.my_player.stack_size}")
            print(f"Action taken: {state.my_player_action}")
    
    # Get all game states for all players across all stages
    print("\n--- ALL GAME STATES ---")
    all_states = GameStateHelper.get_all_game_states_by_stage(game)
    
    print(f"\nTotal stages with game states: {len(all_states)}")
    for stage, stage_states in all_states.items():
        print(f"Stage {stage.name}: {len(stage_states)} player states")
    
    print("\nIntegration test completed!")

if __name__ == "__main__":
    run_integration_test()
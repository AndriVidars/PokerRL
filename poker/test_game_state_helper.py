import unittest
from poker.core.game import Game
from poker.core.player import Player
from poker.core.action import Action
from poker.core.gamestage import Stage
from poker.game_state_helper import GameStateHelper
from poker.agents.game_state import GameState

class TestPlayer(Player):
    """Simple player for testing"""
    def act(self):
        # Always check or call
        self.handle_check_call()
        return Action.CHECK_CALL

class TestGameStateHelper(unittest.TestCase):
    
    def setUp(self):
        """Set up a game with 3 players for testing"""
        self.player1 = TestPlayer("Player 1", 1000)
        self.player2 = TestPlayer("Player 2", 1000)
        self.player3 = TestPlayer("Player 3", 1000)
        self.game = Game([self.player1, self.player2, self.player3], 10, 5)
        
        # Initialize game by running preflop
        self.game.active_players = set(self.game.players)  # Ensure active_players is initialized
        self.game.handle_blinds()
        
        # Deal cards
        from poker.core.deck import Deck
        self.game.deck = Deck()
        for _ in range(2):
            for p in self.game.players:
                p.hand.append(self.game.deck.deck.pop())
                
        # Add some community cards to test later stages
        self.game.community_cards = [self.game.deck.deck.pop() for _ in range(3)]
        
    def test_create_game_states(self):
        """Test creation of game states for all players"""
        game_states = GameStateHelper.create_game_states(self.game, Stage.PREFLOP)
        
        # Should have a state for each active player
        self.assertEqual(len(game_states), len(self.game.players))
        
        # Check that each player has a valid state
        for player, state in game_states.items():
            self.assertIsInstance(state, GameState)
            self.assertEqual(state.stage, Stage.PREFLOP)
            self.assertEqual(len(state.my_player.cards), 2)  # Should have 2 cards
            self.assertEqual(len(state.other_players), len(self.game.players) - 1)
            
            # Check core game reference
            self.assertEqual(state.core_game, self.game)
            
    def test_calculate_min_bet(self):
        """Test calculation of minimum bet to continue"""
        # Set the current stage for testing
        self.game.current_stage = Stage.PREFLOP
        
        # Calculate min bets for all players
        min_bet1 = GameStateHelper._calculate_min_bet(self.game, self.player1)
        min_bet2 = GameStateHelper._calculate_min_bet(self.game, self.player2)
        min_bet3 = GameStateHelper._calculate_min_bet(self.game, self.player3)
        
        # Check that the bets match our expectations
        self.assertEqual(min_bet1, 5)   # Small blind (posted 5) needs 5 more to match big blind
        self.assertEqual(min_bet2, 0)   # Big blind already posted the full amount
        self.assertEqual(min_bet3, 10)  # Third player needs to post full big blind
        
    def test_create_state_player(self):
        """Test creation of state player objects"""
        # Adjust positions for testing - player1 (SB), player2 (BB), player3 (button)
        self.game.dealer_idx = 2  # Player 3 is dealer
        self.game.small_blind_idx = 0  # Player 1 is small blind
        self.game.big_blind_idx = 1  # Player 2 is big blind
        
        state_player1 = GameStateHelper._create_state_player(self.player1, Stage.PREFLOP)
        
        # Check that position is calculated correctly - player1 is 2 spots from BB
        self.assertEqual(state_player1.spots_left_bb, 2)
        
        # Check that cards are copied correctly
        self.assertEqual(len(state_player1.cards), 2)
        
    def test_update_player_actions(self):
        """Test updating game states with player actions"""
        game_states = GameStateHelper.create_game_states(self.game, Stage.PREFLOP)
        
        # Create some player actions
        player_actions = {
            self.player1: (Action.FOLD, None),
            self.player2: (Action.CHECK_CALL, None),
            self.player3: (Action.RAISE, 20)
        }
        
        # Update the game states
        GameStateHelper.update_player_actions(game_states, player_actions)
        
        # Check that actions were updated correctly
        self.assertEqual(game_states[self.player1].my_player_action, (Action.FOLD, None))
        self.assertEqual(game_states[self.player2].my_player_action, (Action.CHECK_CALL, None))
        self.assertEqual(game_states[self.player3].my_player_action, (Action.RAISE, 20))
        
    def test_get_all_game_states_by_stage(self):
        """Test getting game states for all stages"""
        # Set current stage to FLOP to test multiple stages
        self.game.current_stage = Stage.FLOP
        
        all_states = GameStateHelper.get_all_game_states_by_stage(self.game)
        
        # Should have states for PREFLOP and FLOP
        self.assertEqual(len(all_states), 2)
        self.assertIn(Stage.PREFLOP, all_states)
        self.assertIn(Stage.FLOP, all_states)
        
        # Each stage should have states for all players
        self.assertEqual(len(all_states[Stage.PREFLOP]), len(self.game.players))
        self.assertEqual(len(all_states[Stage.FLOP]), len(self.game.players))
        
if __name__ == '__main__':
    unittest.main()
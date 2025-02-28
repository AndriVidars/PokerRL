import os
import sys
from typing import List, Dict, Tuple, Optional, Any

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poker.parsers.pluribus_parser import PluribusParser, Action as ParserAction
from poker.core.deck import Deck
from poker.core.player import Player
from poker.core.game import Game
from poker.core.gamestage import Stage
from poker.agents.game_state import GameState
from poker.parsers.test_pluribus_parser import PluribusParserTester

class GameStateRetriever:
    """
    A class to make it easy to retrieve game states from Pluribus logs
    """
    def __init__(self, log_dir: str = None):
        """
        Initialize with the logs directory
        """
        if log_dir is None:
            # Use default logs directory
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                 "pluribus_converted_logs")
        
        self.log_dir = log_dir
        self.tester = None
        self.is_initialized = False
        
    def initialize(self, verbose: bool = False):
        """
        Parse all logs (can be time-consuming for all files)
        """
        if self.is_initialized:
            return
            
        self.tester = PluribusParserTester(self.log_dir)
        
        # Parse all files
        self.tester.parse_all_files()
        
        # Extract decisions
        if verbose:
            self.tester.analyze_hand_histories()
            self.tester.extract_pluribus_decisions()
            
        self.tester.extract_any_player_decisions()
        self.is_initialized = True
        
    def get_game_state(self, hand_id: str, player_name: str, stage: Stage) -> List[GameState]:
        """
        Get game states for a player in a specific hand and stage
        
        Args:
            hand_id: The hand ID string
            player_name: Player name (e.g., "Pluribus", "MrBlue", etc.)
            stage: Game stage (PREFLOP, FLOP, TURN, RIVER)
            
        Returns:
            List of GameState objects for each decision point
        """
        if not self.is_initialized:
            self.initialize()
            
        return self.tester.get_game_state_for_player(hand_id, player_name, stage)
    
    def get_decisions(self, criteria: Dict[str, Any] = None) -> List[Tuple]:
        """
        Get decisions based on criteria
        
        Args:
            criteria: Dict with any of these keys:
                - player_name: Player name (e.g., "Pluribus")
                - stage: Game stage (e.g., Stage.FLOP)
                - action: Action type (e.g., CoreAction.FOLD)
                
        Returns:
            List of tuples (player_name, stage, game_state, action, amount)
        """
        if not self.is_initialized:
            self.initialize()
            
        return self.tester.retrieve_decisions(criteria)
    
    def get_pluribus_decisions(self) -> List[Tuple]:
        """
        Get all decisions made by Pluribus
        
        Returns:
            List of tuples (player_name, stage, game_state, action, amount)
            where player_name is always "Pluribus" 
        """
        return self.get_decisions({'player_name': 'Pluribus'})
    
    def get_hand_count(self) -> int:
        """
        Get the total number of hands parsed
        """
        if not self.is_initialized:
            self.initialize()
            
        return len(self.tester.hand_histories)

# Example usage
if __name__ == "__main__":
    retriever = GameStateRetriever()
    
    # Initialize with verbose output 
    print("Initializing retriever...")
    retriever.initialize(verbose=True)
    
    # Get Pluribus decisions in FLOP stage
    print("\nGetting Pluribus FLOP decisions...")
    flop_decisions = retriever.get_decisions({
        'player_name': 'Pluribus',
        'stage': Stage.FLOP
    })
    print(f"Found {len(flop_decisions)} Pluribus decisions on the flop")
    
    # Show details about one decision
    if flop_decisions:
        player, stage, state, action, amount = flop_decisions[0]
        print(f"\nSample decision:")
        print(f"Player: {player}")
        print(f"Stage: {stage.name}")
        print(f"Action: {action.name}")
        print(f"Amount: {amount}")
        print(f"Community cards: {[str(card.rank.name) + str(card.suit.name) for card in state.community_cards]}")
        print(f"Hand strength: {state.hand_strength}")
        print(f"Pot size: {state.pot_size}")
        print(f"Min bet to continue: {state.min_bet_to_continue}")
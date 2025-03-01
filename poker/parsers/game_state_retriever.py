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
    
    def get_player_game_states(self, player_name: str) -> List[GameState]:
        """
        Get all game states for a specific player across all hands.
        This function returns GameState objects that represent each decision point
        for the specified player, with their history properly populated.
        
        Args:
            player_name: Name of the player to get game states for
            
        Returns:
            List of GameState objects representing each decision point for the player
        """
        if not self.is_initialized:
            self.initialize()
        
        game_states = []
        
        # Get all decisions made by this player
        player_decisions = self.get_decisions({'player_name': player_name})
        
        # Extract the game states from the decisions
        for _, stage, game_state, action, amount in player_decisions:
            # Set the action that was taken
            game_state.my_player_action = (action, amount)
            game_states.append(game_state)
        
        return game_states
    
    def get_player_game_sequence(self, player_name: str, hand_id: str = None) -> List[List[GameState]]:
        """
        Get all game states for a player organized by hand and in sequence.
        This is useful for tracking a player's decision sequence through a hand.
        
        Args:
            player_name: Name of the player to get game states for
            hand_id: Optional hand ID to filter by
            
        Returns:
            List of lists, where each inner list contains GameState objects 
            for a single hand in stage order (preflop, flop, turn, river)
        """
        if not self.is_initialized:
            self.initialize()
        
        # Dictionary to organize game states by hand ID
        hand_game_states = {}
        
        # Get all relevant decisions
        criteria = {'player_name': player_name}
        if hand_id:
            criteria['hand_id'] = hand_id
            
        player_decisions = self.get_decisions(criteria)
        
        # Group by hand ID
        for p_name, stage, game_state, action, amount in player_decisions:
            # Create a unique key for the hand
            # We need to extract hand_id from the game_state or associated data
            # This depends on how hand_id is stored in your implementation
            hand_key = "unknown"
            if hasattr(game_state, 'core_game') and game_state.core_game:
                # If hand ID is accessible through core_game
                if hasattr(game_state.core_game, 'hand_id'):
                    hand_key = game_state.core_game.hand_id
            
            # Initialize list for this hand if needed
            if hand_key not in hand_game_states:
                hand_game_states[hand_key] = []
                
            # Set the action that was taken
            game_state.my_player_action = (action, amount)
            
            # Add to list for this hand
            hand_game_states[hand_key].append((stage.value, game_state))
        
        # Sort each hand's game states by stage and convert to list of lists
        result = []
        for hand_key, states_with_stage in hand_game_states.items():
            # Sort by stage value (preflop=0, flop=1, etc.)
            sorted_states = [state for _, state in sorted(states_with_stage, key=lambda x: x[0])]
            result.append(sorted_states)
            
        return result
    
    def analyze_player_history(self, player_name: str) -> Dict:
        """
        Analyze a player's action history across all available hands.
        Returns statistics about the player's tendencies at different stages.
        
        Args:
            player_name: Name of the player to analyze
            
        Returns:
            Dictionary containing analysis of player's actions
        """
        if not self.is_initialized:
            self.initialize()
            
        # Get all game states for this player
        game_states = self.get_player_game_states(player_name)
        
        # Initialize counters for each stage
        stage_counts = {
            Stage.PREFLOP: {'FOLD': 0, 'CHECK_CALL': 0, 'RAISE': 0, 'total': 0},
            Stage.FLOP: {'FOLD': 0, 'CHECK_CALL': 0, 'RAISE': 0, 'total': 0},
            Stage.TURN: {'FOLD': 0, 'CHECK_CALL': 0, 'RAISE': 0, 'total': 0},
            Stage.RIVER: {'FOLD': 0, 'CHECK_CALL': 0, 'RAISE': 0, 'total': 0}
        }
        
        # Counters for positional play
        position_counts = {
            'early': {'FOLD': 0, 'CHECK_CALL': 0, 'RAISE': 0, 'total': 0},
            'middle': {'FOLD': 0, 'CHECK_CALL': 0, 'RAISE': 0, 'total': 0},
            'late': {'FOLD': 0, 'CHECK_CALL': 0, 'RAISE': 0, 'total': 0}
        }
        
        # Track raise sizes
        raise_sizes = {
            Stage.PREFLOP: [],
            Stage.FLOP: [],
            Stage.TURN: [],
            Stage.RIVER: []
        }
        
        # Analyze each game state
        for state in game_states:
            if not hasattr(state, 'my_player_action') or state.my_player_action is None:
                continue
                
            action, amount = state.my_player_action
            stage = state.stage
            
            # Count actions by stage
            stage_counts[stage]['total'] += 1
            stage_counts[stage][action.name] += 1
            
            # Determine position
            position = 'middle'  # Default
            if hasattr(state.my_player, 'spots_left_bb'):
                spots_left_bb = state.my_player.spots_left_bb
                if spots_left_bb <= 1:  # Early position
                    position = 'early'
                elif spots_left_bb >= 4:  # Late position
                    position = 'late'
            
            # Count actions by position
            position_counts[position]['total'] += 1
            position_counts[position][action.name] += 1
            
            # Track raise sizes relative to pot
            if action == Action.RAISE and amount is not None and state.pot_size > 0:
                rel_size = amount / state.pot_size
                raise_sizes[stage].append(rel_size)
        
        # Calculate percentages
        stage_percentages = {}
        for stage, counts in stage_counts.items():
            if counts['total'] > 0:
                stage_percentages[stage.name] = {
                    'FOLD': counts['FOLD'] / counts['total'],
                    'CHECK_CALL': counts['CHECK_CALL'] / counts['total'],
                    'RAISE': counts['RAISE'] / counts['total'],
                    'sample_size': counts['total']
                }
        
        position_percentages = {}
        for position, counts in position_counts.items():
            if counts['total'] > 0:
                position_percentages[position] = {
                    'FOLD': counts['FOLD'] / counts['total'],
                    'CHECK_CALL': counts['CHECK_CALL'] / counts['total'],
                    'RAISE': counts['RAISE'] / counts['total'],
                    'sample_size': counts['total']
                }
        
        # Calculate average raise sizes
        avg_raise_sizes = {}
        for stage, sizes in raise_sizes.items():
            if sizes:
                avg_raise_sizes[stage.name] = sum(sizes) / len(sizes)
        
        # Compile the analysis
        analysis = {
            'player_name': player_name,
            'total_decisions': sum(counts['total'] for counts in stage_counts.values()),
            'action_by_stage': stage_percentages,
            'action_by_position': position_percentages,
            'avg_raise_sizes': avg_raise_sizes
        }
        
        return analysis
    
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
    
    # Example of using the new functions
    print("\n*** Demonstrating new functions ***")
    
    # Get a player's game states
    player_name = "Pluribus"  # or any other player in the dataset
    print(f"\nGetting all game states for {player_name}...")
    game_states = retriever.get_player_game_states(player_name)
    print(f"Found {len(game_states)} game states for {player_name}")
    
    # Get a player's game sequences
    print(f"\nGetting game sequences for {player_name}...")
    game_sequences = retriever.get_player_game_sequence(player_name)
    print(f"Found {len(game_sequences)} game hands for {player_name}")
    
    # If we have any sequences, show one as an example
    if game_sequences:
        print(f"\nSample game sequence (first hand):")
        for i, state in enumerate(game_sequences[0]):
            stage_name = state.stage.name
            action_info = ""
            if hasattr(state, 'my_player_action') and state.my_player_action:
                action, amount = state.my_player_action
                action_info = f", Action: {action.name}"
                if amount is not None:
                    action_info += f", Amount: {amount}"
            print(f"  Decision {i+1}: Stage {stage_name}{action_info}")
            # Check player history
            if hasattr(state.my_player, 'history') and state.my_player.history:
                print(f"    Player history: {state.my_player.history}")
    
    # Analyze player history
    print(f"\nAnalyzing {player_name}'s play patterns...")
    analysis = retriever.analyze_player_history(player_name)
    
    print(f"Total decisions: {analysis['total_decisions']}")
    
    # Print action tendencies by stage
    print("\nAction tendencies by stage:")
    for stage, percentages in analysis['action_by_stage'].items():
        print(f"  {stage}: FOLD {percentages['FOLD']:.2f}, CHECK/CALL {percentages['CHECK_CALL']:.2f}, RAISE {percentages['RAISE']:.2f} (sample size: {percentages['sample_size']})")
    
    # Print action tendencies by position
    print("\nAction tendencies by position:")
    for position, percentages in analysis['action_by_position'].items():
        print(f"  {position.capitalize()}: FOLD {percentages['FOLD']:.2f}, CHECK/CALL {percentages['CHECK_CALL']:.2f}, RAISE {percentages['RAISE']:.2f} (sample size: {percentages['sample_size']})")
    
    # Print average raise sizes
    print("\nAverage raise sizes (as multiple of pot):")
    for stage, avg_size in analysis['avg_raise_sizes'].items():
        print(f"  {stage}: {avg_size:.2f}x pot")
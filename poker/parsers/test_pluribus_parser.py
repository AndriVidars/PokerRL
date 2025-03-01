import os
import sys
import time
import glob
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poker.parsers.pluribus_parser import PluribusParser, HandHistory, PluribusDataset, Action as ParserAction
from poker.core.deck import Deck
from poker.core.player import Player
from poker.core.game import Game
from poker.core.card import Card
from poker.core.gamestage import Stage
from poker.agents.game_state import GameState, Player as StatePlayer

# Use the Action enum from parsers for parsing
from poker.core.action import Action as CoreAction

class PluribusParserTester:
    """
    Class to test the PluribusParser by analyzing all logs
    and providing statistics and error reports
    """
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        # Initialize the parser with create_core_games=True to use core components
        self.parser = PluribusParser(log_dir, create_core_games=True)
        self.hand_histories = []
        self.parse_errors = []
        self.file_stats = defaultdict(int)
        self.player_stats = defaultdict(int)
        self.stage_stats = defaultdict(int)
        self.action_stats = defaultdict(int)
        self.pluribus_decisions = []
        self.all_player_decisions = []
        self.core_games = []  # Store reference to core Game objects
        
    def parse_all_files(self):
        """
        Parse all files in the log directory and collect statistics
        """
        start_time = time.time()
        log_files = glob.glob(os.path.join(self.log_dir, "pluribus_*.txt"))
        
        total_files = len(log_files)
        print(f"Found {total_files} log files")
        
        success_count = 0
        total_hands = 0
        total_hands_with_pluribus = 0
        
        for file_idx, file_path in enumerate(log_files):
            file_name = os.path.basename(file_path)
            print(f"Parsing file {file_idx+1}/{total_files}: {file_name}...")
            
            try:
                histories = self.parser.parse_file(file_path)
                
                # Track file-level stats
                self.file_stats[file_name] = len(histories)
                total_hands += len(histories)
                
                # Track hands with Pluribus
                pluribus_hands = sum(1 for h in histories if any(
                    player_name == "Pluribus" for seat, (player_name, _) in h.seats.items()
                ))
                total_hands_with_pluribus += pluribus_hands
                
                # Collect all histories
                self.hand_histories.extend(histories)
                # Store references to core Game objects
                for history in histories:
                    if history.game:
                        self.core_games.append(history.game)
                success_count += 1
                
            except Exception as e:
                error_msg = f"Error parsing {file_name}: {str(e)}"
                print(f"ERROR: {error_msg}")
                self.parse_errors.append(error_msg)
        
        elapsed_time = time.time() - start_time
        print(f"\nParsing completed in {elapsed_time:.2f} seconds")
        print(f"Successfully parsed {success_count}/{total_files} files")
        print(f"Total hands: {total_hands}")
        print(f"Hands with Pluribus: {total_hands_with_pluribus}")
        
        if self.parse_errors:
            print(f"\nEncountered {len(self.parse_errors)} errors:")
            for error in self.parse_errors[:5]:  # Show first 5 errors
                print(f"- {error}")
            if len(self.parse_errors) > 5:
                print(f"... and {len(self.parse_errors) - 5} more errors")
                
        return self.hand_histories
    
    def analyze_hand_histories(self):
        """
        Analyze the parsed hand histories to collect detailed statistics
        """
        print("\nAnalyzing hand histories...")
        
        player_names = set()
        for history in self.hand_histories:
            # Track players
            for seat, (player_name, stack) in history.seats.items():
                player_names.add(player_name)
                
            # Track actions by stage
            stages = [
                ("preflop", history.pre_flop_actions),
                ("flop", history.flop_actions),
                ("turn", history.turn_actions),
                ("river", history.river_actions)
            ]
            
            for stage_name, actions in stages:
                self.stage_stats[stage_name] += len(actions)
                
                # Track actions by player and type
                for player, action, amount in actions:
                    self.player_stats[player] += 1
                    action_key = f"{action.name}"
                    self.action_stats[action_key] += 1
        
        # Print statistics
        print(f"\nFound {len(player_names)} unique players: {', '.join(sorted(player_names))}")
        
        print("\nActions by stage:")
        for stage, count in sorted(self.stage_stats.items()):
            print(f"- {stage}: {count} actions")
            
        print("\nActions by player:")
        for player, count in sorted(self.player_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"- {player}: {count} actions")
            
        print("\nActions by type:")
        for action_type, count in sorted(self.action_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"- {action_type}: {count}")
    
    def extract_pluribus_decisions(self):
        """
        Extract all decision points where Pluribus made a move
        """
        print("\nExtracting Pluribus decisions...")
        
        dataset = PluribusDataset(self.hand_histories)
        self.pluribus_decisions = dataset.extract_pluribus_decisions()
        
        # Summarize the decisions
        actions_count = Counter([action for _, action, _ in self.pluribus_decisions])
        stages_count = Counter([state['stage'] for state, _, _ in self.pluribus_decisions])
        
        print(f"\nExtracted {len(self.pluribus_decisions)} Pluribus decisions")
        print("\nDecisions by action:")
        for action_idx, count in actions_count.items():
            action_name = "FOLD" if action_idx == 0 else "CHECK" if action_idx == 1 else "CALL" if action_idx == 2 else "RAISE"
            print(f"- {action_name}: {count}")
            
        print("\nDecisions by stage:")
        for stage, count in stages_count.items():
            print(f"- {stage.name}: {count}")
            
        return self.pluribus_decisions
    
    def extract_any_player_decisions(self):
        """
        Extract game states for any player's decision points
        """
        print("\nExtracting all player decisions...")
        
        all_decisions = []
        
        for history in self.hand_histories:
            # For each player
            for player_seat, (player_name, _) in history.seats.items():
                
                # Process each stage
                stages = [
                    (Stage.PREFLOP, history.pre_flop_actions, []),
                    (Stage.FLOP, history.flop_actions, history.flop_cards),
                    (Stage.TURN, history.turn_actions, history.flop_cards + ([history.turn_card] if history.turn_card else [])),
                    (Stage.RIVER, history.river_actions, history.flop_cards + ([history.turn_card] if history.turn_card else []) + 
                                ([history.river_card] if history.river_card else []))
                ]
                
                for stage, actions, community_cards in stages:
                    # Find actions by this player
                    for action_idx, (action_player, action, amount) in enumerate(actions):
                        if action_player != player_name:
                            continue
                        
                        # Construct the game state at this decision point
                        state = self._construct_player_game_state(
                            history, player_name, player_seat, stage, community_cards, 
                            actions[:action_idx]  # Actions before this one
                        )
                        
                        # Convert action to our format
                        core_action = self._convert_parser_action_to_core_action(action)
                        
                        all_decisions.append((player_name, stage, state, core_action, amount))
            
        self.all_player_decisions = all_decisions
        
        # Print statistics
        player_counts = Counter([player for player, _, _, _, _ in all_decisions])
        stage_counts = Counter([stage for _, stage, _, _, _ in all_decisions])
        action_counts = Counter([action.value for _, _, _, action, _ in all_decisions])
        
        print(f"\nExtracted {len(all_decisions)} total player decisions")
        print("\nTop players by decision count:")
        for player, count in player_counts.most_common(5):
            print(f"- {player}: {count}")
            
        print("\nDecisions by stage:")
        for stage, count in stage_counts.items():
            print(f"- {stage.name}: {count}")
            
        print("\nDecisions by action type:")
        for action_value, count in action_counts.items():
            action_name = CoreAction(action_value).name
            print(f"- {action_name}: {count}")
            
        return self.all_player_decisions
    
    def _construct_player_game_state(self, history: HandHistory, player_name: str, player_seat: int, 
                                   stage: Stage, community_cards: List[Card],
                                   previous_actions: List[Tuple[str, CoreAction, Optional[int]]]):
        """
        Construct a GameState object at a specific decision point for any player
        This is a more general version of PluribusDataset._construct_game_state
        """
        # Calculate pot size and minimum bet to continue
        pot_size = self._calculate_pot_size(previous_actions)
        min_bet_to_continue = self._get_min_bet_to_continue(previous_actions, player_name)
        
        # Create the player whose decision point we're modeling
        my_player = self._create_state_player(
            history, player_name, player_seat, previous_actions
        )
        
        # Create other players
        other_players = []
        for seat, (other_player_name, _) in history.seats.items():
            if other_player_name != player_name:
                other_players.append(
                    self._create_state_player(
                        history, other_player_name, seat, previous_actions
                    )
                )
        
        # Create GameState
        game_state = GameState(
            stage=stage,
            community_cards=community_cards,
            pot_size=max(1, pot_size),  # Ensure non-zero pot size
            min_bet_to_continue=min_bet_to_continue,
            my_player=my_player,
            other_players=other_players,
            my_player_action=None  # Will be filled later
        )
        
        return game_state
    
    def _create_state_player(self, history: HandHistory, player_name: str, seat: int, 
                           previous_actions: List[Tuple[str, CoreAction, Optional[int]]]):
        """
        Create a Player object for the GameState class
        """
        # Get initial stack and calculate remaining stack
        initial_stack = history.seats[seat][1]
        remaining_stack = initial_stack
        
        for p, _, amount in previous_actions:
            if p == player_name and amount is not None:
                remaining_stack -= amount
        
        # Create state player
        cards_list = history.player_hole_cards.get(player_name, [])
        player = StatePlayer(
            spots_left_bb=seat,  # Using seat as position
            cards=cards_list,  # Pass the cards directly
            stack_size=remaining_stack
        )
        
        # Filter actions by player and stage, then add them to player history using proper methods
        
        # PREFLOP actions - use add_preflop_action
        preflop_actions = [(p, a, amt) for p, a, amt in history.pre_flop_actions if p == player_name]
        if preflop_actions:
            # Get the last action for preflop
            p, a, amt = preflop_actions[-1]
            core_action = self._convert_parser_action_to_core_action(a)
            # Use the specific method to add preflop action
            player.add_preflop_action(core_action, amt)
        
        # FLOP actions - use add_flop_action
        flop_actions = [(p, a, amt) for p, a, amt in history.flop_actions if p == player_name]
        if flop_actions and not player.history:
            # If player has no history yet but has flop actions, 
            # add a default preflop action first to maintain sequence
            player.add_preflop_action(CoreAction.CHECK_CALL, None)
            
        if flop_actions:
            # Get the last action for flop
            p, a, amt = flop_actions[-1]
            core_action = self._convert_parser_action_to_core_action(a)
            # Use the specific method to add flop action
            player.add_flop_action(core_action, amt)
        
        # TURN actions - use add_turn_action
        turn_actions = [(p, a, amt) for p, a, amt in history.turn_actions if p == player_name]
        if turn_actions and len(player.history) < 2:
            # If player doesn't have enough history yet, add placeholders
            while len(player.history) < 1:
                player.add_preflop_action(CoreAction.CHECK_CALL, None)
            if len(player.history) < 2:
                player.add_flop_action(CoreAction.CHECK_CALL, None)
            
        if turn_actions:
            # Get the last action for turn
            p, a, amt = turn_actions[-1]
            core_action = self._convert_parser_action_to_core_action(a)
            # Use the specific method to add turn action
            player.add_turn_action(core_action, amt)
        
        # RIVER actions - use add_river_action
        river_actions = [(p, a, amt) for p, a, amt in history.river_actions if p == player_name]
        if river_actions and len(player.history) < 3:
            # If player doesn't have enough history yet, add placeholders
            while len(player.history) < 1:
                player.add_preflop_action(CoreAction.CHECK_CALL, None)
            if len(player.history) < 2:
                player.add_flop_action(CoreAction.CHECK_CALL, None)
            if len(player.history) < 3:
                player.add_turn_action(CoreAction.CHECK_CALL, None)
            
        if river_actions:
            # Get the last action for river
            p, a, amt = river_actions[-1]
            core_action = self._convert_parser_action_to_core_action(a)
            # Use the specific method to add river action
            player.add_river_action(core_action, amt)
        
        # Handle fold actions specially
        # Check if player has folded
        has_folded = False
        fold_stage = None
        for stage in [Stage.PREFLOP, Stage.FLOP, Stage.TURN, Stage.RIVER]:
            if stage == Stage.PREFLOP:
                stage_data = history.pre_flop_actions
            elif stage == Stage.FLOP:
                stage_data = history.flop_actions
            elif stage == Stage.TURN:
                stage_data = history.turn_actions
            elif stage == Stage.RIVER:
                stage_data = history.river_actions
            
            for p, a, _ in stage_data:
                if p == player_name and (a == CoreAction.FOLD or 
                                     (isinstance(a, ParserAction) and a == ParserAction.FOLD)):
                    has_folded = True
                    fold_stage = stage
                    break
            
            if has_folded:
                break
        
        # If player folded but fold action isn't in history, add it
        if has_folded:
            # Get current history length
            history_len = len(player.history)
            
            # Determine which method to use based on fold stage
            if fold_stage == Stage.PREFLOP and history_len == 0:
                player.add_preflop_action(CoreAction.FOLD, None)
            elif fold_stage == Stage.FLOP and history_len == 1:
                player.add_flop_action(CoreAction.FOLD, None)
            elif fold_stage == Stage.TURN and history_len == 2:
                player.add_turn_action(CoreAction.FOLD, None)
            elif fold_stage == Stage.RIVER and history_len == 3:
                player.add_river_action(CoreAction.FOLD, None)
            # If history already has actions for later stages, it means fold was already recorded
        
        return player
    
    def _calculate_pot_size(self, actions: List[Tuple[str, CoreAction, Optional[int]]]) -> int:
        """
        Calculate the pot size after a sequence of actions
        """
        pot_size = 0
        for _, _, amount in actions:
            if amount is not None:
                pot_size += amount
        return pot_size
    
    def _get_min_bet_to_continue(self, actions: List[Tuple[str, CoreAction, Optional[int]]], player: str) -> int:
        """
        Calculate the minimum bet required for a player to continue in the hand
        """
        if not actions:
            return 0
        
        # Find the highest bet so far
        max_bet = 0
        for _, _, amount in actions:
            if amount is not None:
                max_bet = max(max_bet, amount)
        
        # Find how much the player has already contributed
        player_contribution = 0
        for p, _, amount in actions:
            if p == player and amount is not None:
                player_contribution += amount
        
        return max(0, max_bet - player_contribution)
    
    def _convert_parser_action_to_core_action(self, parser_action) -> CoreAction:
        """
        Convert parser Action enum to core Action enum
        """
        if isinstance(parser_action, ParserAction):
            if parser_action == ParserAction.FOLD:
                return CoreAction.FOLD
            elif parser_action == ParserAction.CHECK_CALL:
                return CoreAction.CHECK_CALL
            elif parser_action == ParserAction.RAISE:
                return CoreAction.RAISE
            else:
                return CoreAction.CHECK_CALL  # Default
        else:
            # Try to handle string or other type
            action_name = parser_action.name if hasattr(parser_action, 'name') else str(parser_action)
            
            if action_name == "FOLD":
                return CoreAction.FOLD
            elif action_name in ["CHECK", "CALL", "CHECK_CALL"]:
                return CoreAction.CHECK_CALL
            elif action_name == "RAISE":
                return CoreAction.RAISE
            else:
                return CoreAction.CHECK_CALL  # Default

    def get_game_state_for_player(self, hand_id: str, player_name: str, stage: Stage) -> List[GameState]:
        """
        Retrieve all game states for a specific player in a specific hand and stage
        """
        states = []
        
        # Find the hand history
        for history in self.hand_histories:
            if history.hand_id != hand_id:
                continue
                
            # Check if player is in this hand
            player_seat = None
            for seat, (name, _) in history.seats.items():
                if name == player_name:
                    player_seat = seat
                    break
                    
            if player_seat is None:
                continue
                
            # Get actions for the requested stage
            stage_actions = []
            community_cards = []
            
            if stage == Stage.PREFLOP:
                stage_actions = history.pre_flop_actions
                community_cards = []
            elif stage == Stage.FLOP:
                stage_actions = history.flop_actions
                community_cards = history.flop_cards
            elif stage == Stage.TURN:
                stage_actions = history.turn_actions
                community_cards = history.flop_cards + ([history.turn_card] if history.turn_card else [])
            elif stage == Stage.RIVER:
                stage_actions = history.river_actions
                community_cards = history.flop_cards + ([history.turn_card] if history.turn_card else []) + ([history.river_card] if history.river_card else [])
            
            # Find actions by this player
            for action_idx, (action_player, action, amount) in enumerate(stage_actions):
                if action_player != player_name:
                    continue
                
                # Construct the game state at this decision point
                state = self._construct_player_game_state(
                    history, player_name, player_seat, stage, community_cards, 
                    stage_actions[:action_idx]  # Actions before this one
                )
                
                states.append(state)
                
        return states
    
    def retrieve_decisions(self, criteria: Dict[str, Any] = None) -> List[Tuple]:
        """
        Retrieve decisions based on specified criteria:
        - player_name: Filter by player name
        - hand_id: Filter by hand ID
        - stage: Filter by game stage
        - action: Filter by action taken
        """
        if criteria is None:
            criteria = {}
        
        filtered_decisions = []
        
        for decision in self.all_player_decisions:
            player_name, stage, state, action, amount = decision
            
            # Apply filters
            if 'player_name' in criteria and player_name != criteria['player_name']:
                continue
                
            if 'stage' in criteria and stage != criteria['stage']:
                continue
                
            if 'action' in criteria and action != criteria['action']:
                continue
            
            # For hand_id, we need to reconstruct which hand this came from
            if 'hand_id' in criteria:
                # This would require storing hand_id with each decision
                # Skipping for simplicity
                pass
                
            filtered_decisions.append(decision)
            
        return filtered_decisions

def main():
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "pluribus_converted_logs")
    tester = PluribusParserTester(log_dir)
    
    # Parse all files
    hand_histories = tester.parse_all_files()
    
    # Analyze the parsed histories
    tester.analyze_hand_histories()
    
    # Extract Pluribus decisions (as in the original code)
    pluribus_decisions = tester.extract_pluribus_decisions()
    
    # Extract decisions for all players (the new functionality)
    all_player_decisions = tester.extract_any_player_decisions()
    
    print("\nTesting completed successfully!")

if __name__ == "__main__":
    main()

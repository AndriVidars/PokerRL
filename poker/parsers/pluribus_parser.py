import re
import os
import glob
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import numpy as np

from poker.core.card import Card, Rank, Suit
from poker.core.action import Action
from poker.core.gamestage import Stage

class HandHistory:
    """
    Represents a single poker hand history from the Pluribus dataset
    """
    def __init__(self):
        self.hand_id: str = ""
        self.blinds: Tuple[int, int] = (0, 0)  # (small_blind, big_blind)
        self.date: str = ""
        self.seats: Dict[int, Tuple[str, int]] = {}  # {seat_num: (player_name, stack)}
        self.button_pos: int = 0
        self.pre_flop_actions: List[Tuple[str, Action, Optional[int]]] = []  # [(player, action, amount)]
        self.flop_cards: List[Card] = []
        self.flop_actions: List[Tuple[str, Action, Optional[int]]] = []
        self.turn_card: Optional[Card] = None
        self.turn_actions: List[Tuple[str, Action, Optional[int]]] = []
        self.river_card: Optional[Card] = None
        self.river_actions: List[Tuple[str, Action, Optional[int]]] = []
        self.showdown: Dict[str, List[Card]] = {}  # {player: [cards]}
        self.winners: List[Tuple[str, int]] = []  # [(player, amount)]
        self.player_hole_cards: Dict[str, List[Card]] = {}  # {player: [cards]}
        
    def __str__(self):
        return f"Hand #{self.hand_id} - {len(self.seats)} players, Button: {self.button_pos}"

class PluribusParser:
    """
    Parser for Pluribus poker hand history files
    """
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.hand_histories: List[HandHistory] = []
        
    def parse_all_files(self) -> List[HandHistory]:
        """
        Parse all Pluribus log files in the specified directory
        """
        log_files = glob.glob(os.path.join(self.log_dir, "pluribus_*.txt"))
        all_histories = []
        
        for file_path in log_files:
            print(f"Parsing {os.path.basename(file_path)}...")
            histories = self.parse_file(file_path)
            all_histories.extend(histories)
            
        self.hand_histories = all_histories
        print(f"Parsed {len(all_histories)} hand histories total")
        return all_histories
    
    def parse_file(self, file_path: str) -> List[HandHistory]:
        """
        Parse a single Pluribus log file
        """
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Split the content into individual hand histories
        hand_blocks = re.split(r'PokerStars Hand #', content)
        if hand_blocks[0].strip() == '':
            hand_blocks = hand_blocks[1:]  # Remove empty first block
        
        hand_histories = []
        for block in hand_blocks:
            block = "PokerStars Hand #" + block  # Add back the split part
            history = self._parse_hand_history(block)
            if history:
                hand_histories.append(history)
        
        return hand_histories
    
    def _parse_hand_history(self, text: str) -> Optional[HandHistory]:
        """
        Parse a single hand history block
        """
        history = HandHistory()
        
        # Parse hand ID, blinds, and date
        header_match = re.search(r'PokerStars Hand #(\d+): Hold\'em No Limit \((\d+)/(\d+)\) - ([\d/]+ [\d:]+)', text)
        if not header_match:
            return None
        
        history.hand_id = header_match.group(1)
        history.blinds = (int(header_match.group(2)), int(header_match.group(3)))
        history.date = header_match.group(4)
        
        # Parse button position
        button_match = re.search(r'Table .* Seat #(\d+) is the button', text)
        if button_match:
            history.button_pos = int(button_match.group(1))
        
        # Parse seats
        seat_matches = re.finditer(r'Seat (\d+): (\w+) \((\d+) in chips\)', text)
        for match in seat_matches:
            seat_num = int(match.group(1))
            player_name = match.group(2)
            stack = int(match.group(3))
            history.seats[seat_num] = (player_name, stack)
        
        # Parse hole cards
        hole_card_matches = re.finditer(r'Dealt to (\w+) \[([\w\s]+)\]', text)
        for match in hole_card_matches:
            player = match.group(1)
            cards_str = match.group(2)
            cards = self._parse_cards(cards_str)
            history.player_hole_cards[player] = cards
        
        # Parse preflop actions
        preflop_section = self._extract_section(text, r'\*\*\* HOLE CARDS \*\*\*', r'\*\*\* FLOP \*\*\*')
        if preflop_section:
            history.pre_flop_actions = self._parse_actions(preflop_section)
        
        # Parse flop
        flop_section = self._extract_section(text, r'\*\*\* FLOP \*\*\* \[([\w\s]+)\]', r'\*\*\* TURN \*\*\*')
        if flop_section:
            flop_match = re.search(r'\*\*\* FLOP \*\*\* \[([\w\s]+)\]', text)
            if flop_match:
                flop_cards_str = flop_match.group(1)
                history.flop_cards = self._parse_cards(flop_cards_str)
            history.flop_actions = self._parse_actions(flop_section)
        
        # Parse turn
        turn_section = self._extract_section(text, r'\*\*\* TURN \*\*\* \[[\w\s]+\] \[([\w\s]+)\]', r'\*\*\* RIVER \*\*\*')
        if turn_section:
            turn_match = re.search(r'\*\*\* TURN \*\*\* \[[\w\s]+\] \[([\w\s]+)\]', text)
            if turn_match:
                turn_card_str = turn_match.group(1)
                cards = self._parse_cards(turn_card_str)
                if cards:
                    history.turn_card = cards[0]
            history.turn_actions = self._parse_actions(turn_section)
        
        # Parse river
        river_section = self._extract_section(text, r'\*\*\* RIVER \*\*\* \[[\w\s]+\] \[([\w\s]+)\]', r'\*\*\* SHOW DOWN \*\*\*')
        if river_section:
            river_match = re.search(r'\*\*\* RIVER \*\*\* \[[\w\s]+\] \[([\w\s]+)\]', text)
            if river_match:
                river_card_str = river_match.group(1)
                cards = self._parse_cards(river_card_str)
                if cards:
                    history.river_card = cards[0]
            history.river_actions = self._parse_actions(river_section)
        
        # Parse showdown
        showdown_section = self._extract_section(text, r'\*\*\* SHOW DOWN \*\*\*', r'\*\*\* SUMMARY \*\*\*')
        if showdown_section:
            shown_matches = re.finditer(r'(\w+): shows \[([\w\s]+)\]', showdown_section)
            for match in shown_matches:
                player = match.group(1)
                cards_str = match.group(2)
                history.showdown[player] = self._parse_cards(cards_str)
        
        # Parse winners
        collected_matches = re.finditer(r'(\w+) collected (\d+) from pot', text)
        for match in collected_matches:
            player = match.group(1)
            amount = int(match.group(2))
            history.winners.append((player, amount))
        
        return history
    
    def _extract_section(self, text: str, start_pattern: str, end_pattern: str) -> Optional[str]:
        """
        Extract a section of text between two patterns
        """
        start_match = re.search(start_pattern, text)
        if not start_match:
            return None
        
        start_idx = start_match.end()
        
        end_match = re.search(end_pattern, text[start_idx:])
        if end_match:
            end_idx = start_idx + end_match.start()
            return text[start_idx:end_idx]
        else:
            return text[start_idx:]
    
    def _parse_cards(self, cards_str: str) -> List[Card]:
        """
        Parse card strings into Card objects
        """
        cards = []
        card_matches = re.finditer(r'(\d+|[AKQJT])([cdhs])', cards_str)
        
        rank_map = {
            'A': Rank.ACE,
            'K': Rank.KING,
            'Q': Rank.QUEEN,
            'J': Rank.JACK,
            'T': Rank.TEN,
            '2': Rank.TWO,
            '3': Rank.THREE,
            '4': Rank.FOUR,
            '5': Rank.FIVE,
            '6': Rank.SIX,
            '7': Rank.SEVEN,
            '8': Rank.EIGHT,
            '9': Rank.NINE
        }
        
        suit_map = {
            'c': Suit.CLUB,
            'd': Suit.DIAMOND,
            'h': Suit.HEART,
            's': Suit.SPADE
        }
        
        for match in card_matches:
            rank_str = match.group(1)
            suit_str = match.group(2)
            
            # Convert rank string to Rank enum
            if rank_str in rank_map:
                rank = rank_map[rank_str]
            else:
                continue  # Skip invalid ranks
            
            # Convert suit string to Suit enum
            if suit_str in suit_map:
                suit = suit_map[suit_str]
            else:
                continue  # Skip invalid suits
            
            cards.append(Card(rank, suit))
        
        return cards
    
    def _parse_actions(self, section: str) -> List[Tuple[str, Action, Optional[int]]]:
        """
        Parse player actions from a section of text
        """
        actions = []
        
        # Find all blind posts
        blind_matches = re.finditer(r'(\w+): posts (small|big) blind (\d+)', section)
        for match in blind_matches:
            player = match.group(1)
            amount = int(match.group(3))
            actions.append((player, Action.CALL, amount))  # Treat blinds as calls
        
        # Find all folds
        fold_matches = re.finditer(r'(\w+): folds', section)
        for match in fold_matches:
            player = match.group(1)
            actions.append((player, Action.FOLD, None))
        
        # Find all checks
        check_matches = re.finditer(r'(\w+): checks', section)
        for match in check_matches:
            player = match.group(1)
            actions.append((player, Action.CHECK, None))
        
        # Find all calls
        call_matches = re.finditer(r'(\w+): calls (\d+)', section)
        for match in call_matches:
            player = match.group(1)
            amount = int(match.group(2))
            actions.append((player, Action.CALL, amount))
        
        # Find all bets
        bet_matches = re.finditer(r'(\w+): bets (\d+)', section)
        for match in bet_matches:
            player = match.group(1)
            amount = int(match.group(2))
            actions.append((player, Action.RAISE, amount))
        
        # Find all raises
        raise_matches = re.finditer(r'(\w+): raises (\d+) to (\d+)', section)
        for match in raise_matches:
            player = match.group(1)
            amount = int(match.group(3))  # Use the final amount
            actions.append((player, Action.RAISE, amount))
        
        return actions

class PluribusDataset:
    """
    Process the parsed Pluribus data for imitation learning
    """
    def __init__(self, hand_histories: List[HandHistory]):
        self.hand_histories = hand_histories
        self.state_action_pairs = []  # [(state, action, raise_amount)]
        
    def extract_pluribus_decisions(self) -> List[Tuple[Dict, int, float]]:
        """
        Extract all decision points where Pluribus made a move
        Returns a list of (state, action_idx, raise_amount) tuples
        """
        pluribus_decisions = []
        
        for history in self.hand_histories:
            # Get Pluribus's seat
            pluribus_seat = None
            for seat, (player_name, _) in history.seats.items():
                if player_name == "Pluribus":
                    pluribus_seat = seat
                    break
            
            if pluribus_seat is None:
                continue  # Skip hands where Pluribus is not present
            
            # Process each stage
            stages = [
                (Stage.PREFLOP, history.pre_flop_actions, []),
                (Stage.FLOP, history.flop_actions, history.flop_cards),
                (Stage.TURN, history.turn_actions, history.flop_cards + ([history.turn_card] if history.turn_card else [])),
                (Stage.RIVER, history.river_actions, history.flop_cards + ([history.turn_card] if history.turn_card else []) + 
                              ([history.river_card] if history.river_card else []))
            ]
            
            for stage, actions, community_cards in stages:
                # Process each action
                for action_idx, (player, action, amount) in enumerate(actions):
                    if player != "Pluribus":
                        continue
                    
                    # Construct the game state at this decision point
                    state = self._construct_game_state(
                        history, pluribus_seat, stage, community_cards, 
                        actions[:action_idx]  # Actions before this one
                    )
                    
                    # Convert action to our format
                    action_idx = self._convert_action_to_idx(action)
                    
                    # Calculate normalized raise amount (0-1)
                    raise_amount = 0.0
                    if action == Action.RAISE and amount is not None:
                        # Normalize raise amount relative to pot and stack
                        pot_size = self._calculate_pot_size(actions[:action_idx])
                        pluribus_stack = history.seats[pluribus_seat][1]  # Initial stack
                        
                        # Subtract previous contributions
                        for p, a, amt in actions[:action_idx]:
                            if p == "Pluribus" and amt is not None:
                                pluribus_stack -= amt
                        
                        if pot_size > 0 and pluribus_stack > 0:
                            # Normalize as percentage of possible raise range
                            min_raise = self._get_min_bet_to_continue(actions[:action_idx], "Pluribus")
                            max_raise = pluribus_stack
                            if max_raise > min_raise:
                                raise_amount = (amount - min_raise) / (max_raise - min_raise)
                                raise_amount = max(0.0, min(1.0, raise_amount))  # Clamp to [0, 1]
                    
                    pluribus_decisions.append((state, action_idx, raise_amount))
        
        print(f"Extracted {len(pluribus_decisions)} Pluribus decisions")
        return pluribus_decisions
    
    def _create_state_player(self, history: HandHistory, player_name: str, seat: int, 
                               previous_actions: List[Tuple[str, Action, Optional[int]]]):
        """
        Create a Player object for the GameState class
        """
        from poker.agents.game_state import Player as StatePlayer
        
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
            cards=cards_list,  # Pass the cards directly, not as a tuple
            stack_size=remaining_stack
        )
        
        # We need to track stages separately to add actions in the correct order
        # First, group actions by player
        player_actions = []
        for p, action, amount in previous_actions:
            if p == player_name:
                player_actions.append((action, amount))
        
        # Add fold action if needed (to indicate player has folded)
        if player_name in [p for p, a, _ in previous_actions if a == Action.FOLD]:
            if not player_actions:
                player_actions.append((Action.FOLD, None))
            elif player_actions[-1][0] != Action.FOLD:
                player_actions.append((Action.FOLD, None))
        
        # Just record the last action for simplicity
        # The GameState class expects a specific history pattern by stage
        if player_actions:
            player.history = [player_actions[-1]]
        
        return player
    
    def _construct_game_state(self, history: HandHistory, pluribus_seat: int, 
                             stage: Stage, community_cards: List[Card],
                             previous_actions: List[Tuple[str, Action, Optional[int]]]):
        """
        Construct a GameState object at a specific decision point
        """
        from poker.agents.game_state import GameState
        
        # Calculate pot size and minimum bet to continue
        pot_size = self._calculate_pot_size(previous_actions)
        min_bet_to_continue = self._get_min_bet_to_continue(previous_actions, "Pluribus")
        
        # Create Pluribus player
        pluribus_player = self._create_state_player(
            history, "Pluribus", pluribus_seat, previous_actions
        )
        
        # Create other players
        other_players = []
        for seat, (player_name, _) in history.seats.items():
            if player_name != "Pluribus":
                other_players.append(
                    self._create_state_player(
                        history, player_name, seat, previous_actions
                    )
                )
        
        # Create GameState
        game_state = GameState(
            stage=stage,
            community_cards=community_cards,
            pot_size=max(1, pot_size),  # Ensure non-zero pot size
            min_bet_to_continue=min_bet_to_continue,
            my_player=pluribus_player,
            other_players=other_players,
            my_player_action=None  # Will be filled later
        )
        
        return game_state
    
    def _calculate_pot_size(self, actions: List[Tuple[str, Action, Optional[int]]]) -> int:
        """
        Calculate the pot size after a sequence of actions
        """
        pot_size = 0
        for _, _, amount in actions:
            if amount is not None:
                pot_size += amount
        return pot_size
    
    def _get_min_bet_to_continue(self, actions: List[Tuple[str, Action, Optional[int]]], player: str) -> int:
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
    
    def _convert_action_to_idx(self, action: Action) -> int:
        """
        Convert Action enum to action index for our DQN
        """
        if action == Action.FOLD:
            return 0
        elif action == Action.CHECK:
            return 1
        elif action == Action.CALL:
            return 2
        elif action == Action.RAISE:
            return 3
        else:
            return 2  # Default to CALL for unknown actions
    
    def _calculate_hand_strength(self, cards: List[Card]) -> float:
        """
        Calculate hand strength from a set of cards (hole + community)
        This is a simplified version and could be improved with actual hand evaluation
        """
        if len(cards) < 2:
            return 0.0
            
        from poker.core.hand_evaluator import evaluate_hand
        
        try:
            if len(cards) <= 5:
                # Just use whatever cards we have, not a complete hand yet
                rank_sum = sum(card.rank.value for card in cards)
                return rank_sum / 100.0  # Simple normalization
            else:
                # Use proper hand evaluation if we have enough cards
                best_rank = 0
                # Try all 5-card combinations
                from itertools import combinations
                for combo in combinations(cards, 5):
                    rank, _ = evaluate_hand(list(combo))
                    best_rank = max(best_rank, rank)
                return best_rank / 10.0  # Normalize
        except:
            return 0.0  # Default if evaluation fails
    
    def _calculate_community_strength(self, community_cards: List[Card]) -> float:
        """
        Calculate the strength of the community cards alone
        """
        if not community_cards:
            return 0.0
            
        return self._calculate_hand_strength(community_cards)

def create_imitation_dataset(log_dir: str, output_file: str):
    """
    Create a dataset for imitation learning from Pluribus logs
    """
    parser = PluribusParser(log_dir)
    histories = parser.parse_all_files()
    
    dataset = PluribusDataset(histories)
    pluribus_decisions = dataset.extract_pluribus_decisions()
    
    # Convert to numpy arrays and save
    states = []
    actions = []
    raise_amounts = []
    
    for game_state, action_idx, raise_amount in pluribus_decisions:
        # Create feature vector for training
        # We need to extract features from the GameState object
        features = [
            game_state.pot_size,
            game_state.min_bet_to_continue,
            game_state.stage.value,
            game_state.my_player.stack_size,
            game_state.hand_strength,  # Use the property
            game_state.community_hand_strength,  # Use the property
            game_state.my_player.spots_left_bb,  # Position relative to big blind
            len([p for p in game_state.other_players if p.in_game])  # Count active players
        ]
        
        states.append(features)
        actions.append(action_idx)
        raise_amounts.append(raise_amount)
    
    if pluribus_decisions:
        np.savez(
            output_file,
            states=np.array(states, dtype=np.float32),
            actions=np.array(actions, dtype=np.int32),
            raise_amounts=np.array(raise_amounts, dtype=np.float32)
        )
        print(f"Saved {len(states)} training examples to {output_file}")
    else:
        print("No training examples found")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python pluribus_parser.py <log_dir> <output_file>")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    output_file = sys.argv[2]
    create_imitation_dataset(log_dir, output_file)
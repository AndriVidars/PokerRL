from poker.core.card import Card, Rank, Suit
from poker.core.gamestage import Stage
from poker.core.action import Action
from poker.core.hand_evaluator import evaluate_hand
from typing import List, Tuple, Dict, Optional
import re
from collections import defaultdict
import os

def remap_numbers(lst):
    # Replace all the cells with numbers so that the smallest number becomes 0,
    # the second smallest becomes 1 and so on. For example,  [None, 3, None 1] becomes
    # [None, 1, None, 0].
    nums = sorted(set(x for x in lst if x is not None))
    mapping = {num: i for i, num in enumerate(nums)}
    return [mapping[x] if x is not None else None for x in lst]

class Player():
    """
    Represents the state of a player. The most important part is the history,
    which holds all the actions the player has made.

    For now, the history assumes that each player only acts once during a stage.
    """
    def __init__(self,
        spots_left_bb: int, # how many seats to the left is player from bb
        cards: Tuple[Card, Card] | None, # None if not visible
        stack_size: int
    ):
        self.spots_left_bb = spots_left_bb
        self.cards = cards
        self.stack_size = stack_size
        self.history: List[Tuple[Action, int]] = []

    def add_preflop_action(self, action: Action, raise_size: int | None):
        #assert len(self.history) == 0
        self.history.append((action, raise_size))
    
    def add_flop_action(self, action: Action, raise_size: int | None):
        #assert len(self.history) == 1
        self.history.append((action, raise_size))

    def add_turn_action(self, action: Action, raise_size: int | None):
        #assert len(self.history) == 2
        self.history.append((action, raise_size))
    
    def add_river_action(self, action: Action, raise_size: int | None):
        #assert len(self.history) == 3
        self.history.append((action, raise_size))

    def turn_to_play(self, stage: Stage, ttl_players:int):
        # returns what turn the player acted/should act on in a given stage
        # assuming no players have folded. 0 is first.
        if stage == Stage.PREFLOP:
            return (self.spots_left_bb - 1) % ttl_players
        return (self.spots_left_bb + 1) % ttl_players
    
    @property
    def in_game(self):
        return not any([action == Action.FOLD for action, _ in self.history])
    
    def __str__(self):
        res = ""
        res += f"in_game: {self.in_game}\n"
        res += f"spots_left_bb: {self.spots_left_bb}\n"
        res += f"cards: {self.cards}\n"
        res += f"history: {self.history}\n"
        return res

class GameState():
    """
    Represents a game state where "my_player" is about to take an action.
    """
    def __init__(self,
            stage : Stage,
            community_cards: List[Card],
            pot_size: int,
            # the minimum bet (0 if check) that my_player should make to continue playing
            min_bet_to_continue: int,
            my_player: Player,
            other_players: List[Player],
            # action the player took in practice
            my_player_action: Tuple[Action, int] | None,
    ):
        self.community_cards = community_cards
        self.pot_size = pot_size
        self.min_bet_to_continue = min_bet_to_continue
        self.stage = stage
        self.my_player = my_player
        self.other_players = other_players
        self.my_player_action = my_player_action

    @staticmethod
    def compute_hand_strength(cards: List[Card]):
        # computes hand strenght leveraging tie breakers
        # there's possibly a smarter/better way to do this
        rank, tbs = evaluate_hand(cards)
        strength = rank*10_000 + tbs[0]*100
        if len(tbs) > 1: strength += tbs[1]
        return strength

    @property
    def hand_strength(self):
        return self.compute_hand_strength(list(self.my_player.cards) + self.community_cards)

    @property
    def community_hand_strenght(self):
        if len(self.community_cards) == 0: return 0
        return self.compute_hand_strength(self.community_cards)

    def get_hand_strength(self): return self.hand_strength
    
    def get_community_hand_strength(self): return self.community_hand_strenght

    def get_effective_turns(self):
        # For every player returns None if the player is not in the game any more.
        # Otherwise, returns their effective turn (i.e. removing all ppl that folded).
        ttl_players = 1 + len(self.other_players)
        all_players = [self.my_player] + self.other_players
        absolute_turns = [player.turn_to_play(self.stage, ttl_players) for player in all_players]
        in_game_turns = [turn if player.in_game else None for turn, player in zip(absolute_turns, all_players)]
        in_game_turns = remap_numbers(in_game_turns)
        
        my_player_in_game_turn = in_game_turns[0]
        other_players_in_game_turn = in_game_turns[1:]
        return my_player_in_game_turn, other_players_in_game_turn
    
    def __str__(self):
        res = ""
        res += f"stage: {self.stage}\n"
        res += f"pot_size: {self.pot_size}\n"
        res += f"min_bet_to_continue: {self.min_bet_to_continue}\n"
        res += f"community cards: {self.community_cards}\n"
        res += f"\n====== my player ======\n"
        res += f"{self.my_player}"
        res += f"my_player_action: {self.my_player_action}\n"
        res += f"=========================\n"
        res += f"\n==== other players =====\n"
        for p in self.other_players: res += f"{p}\n"
        res += f"==========================\n"
        return res
    

class GameStateBuilder:
    def __init__(self):
        self.stages = {
            "PREFLOP": Stage.PREFLOP,
            "FLOP": Stage.FLOP,
            "TURN": Stage.TURN,
            "RIVER": Stage.RIVER
        }
    
    def parse_hand(self, hand_data: str):
        """Parse a poker hand and extract data for GameState creation"""
        # Extract data
        data = self._extract_data(hand_data)
        
        # Parse and organize
        result = {
            "hand_id": data["hand"],
            "variant": data["variant"],
            "players": {},
            "player_ids": {},
            "player_positions": {},
            "board_cards": [],
            "player_hole_cards": {},
            "player_stacks": {},
            "actions_by_stage": defaultdict(list),
            "stage_transitions": [],
            "blinds_or_straddles": data.get("blinds_or_straddles", []),
            "min_bet": data.get("min_bet", None),
        }
        
        # Find blind positions
        self._find_blinds(data, result)
        
        # Set up players
        self._setup_players(data, result)
        
        # Parse actions
        self._parse_actions(data, result)
        
        return result
    
    def _extract_data(self, hand_data: str) -> dict:
        """Extract data from the hand string"""
        data = {}
        for line in hand_data.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                data[key.strip()] = self._parse_value(value.strip())
        
        if not data:
            parts = hand_data.split(' ')
            current_key = None
            current_value = []
            
            for part in parts:
                if '=' in part:
                    if current_key:
                        data[current_key] = self._parse_value(' '.join(current_value))
                    current_key, value_start = part.split('=', 1)
                    current_value = [value_start]
                elif current_key:
                    current_value.append(part)
            
            if current_key:
                data[current_key] = self._parse_value(' '.join(current_value))
        
        return data
    
    def _parse_value(self, value_str: str):
        """Safely parse a value string into the appropriate Python data type"""
        if value_str == 'true':
            return True
        elif value_str == 'false':
            return False
        
        try:
            return eval(value_str)
        except (NameError, SyntaxError):
            return value_str
    
    def _find_blinds(self, data: dict, result: dict):
        """Find blind positions and record them"""
        blinds = data.get("blinds_or_straddles", [])
        result["blinds"] = []
        
        for i, blind in enumerate(blinds):
            if blind > 0:
                result["blinds"].append((i, blind))
        
        # Sort blinds by amount to determine SB and BB
        result["blinds"].sort(key=lambda x: x[1])
        
        if len(result["blinds"]) >= 2:
            result["sb_pos"] = result["blinds"][0][0]
            result["bb_pos"] = result["blinds"][1][0]
        elif len(result["blinds"]) == 1:
            result["bb_pos"] = result["blinds"][0][0]
            result["sb_pos"] = (result["bb_pos"] - 1) % len(data["players"])
        else:
            result["bb_pos"] = 1
            result["sb_pos"] = 0
    
    def _setup_players(self, data: dict, result: dict):
        """Set up player information and calculate spots from BB"""
        # Create a list of player names in order for easier indexing
        result["player_list"] = data["players"]
        
        blinds = data.get("blinds_or_straddles", [0] * len(data["players"]))
        
        for i, player_name in enumerate(data["players"]):
            player_id = f"p{i+1}"
            result["players"][player_id] = player_name
            result["player_ids"][player_name] = player_id
            
            # Calculate spots_left_bb
            spots_left_bb = (i - result["bb_pos"]) % len(data["players"])
            result["player_positions"][player_name] = spots_left_bb
            
            # Record starting stack (with blinds already deducted)
            starting_stack = data["starting_stacks"][i]
            
            # Immediately deduct blinds from stack if this player posted a blind
            if i < len(blinds) and blinds[i] > 0:
                starting_stack -= blinds[i]
                
            result["player_stacks"][player_name] = starting_stack
    
    def _parse_actions(self, data: dict, result: dict):
        """Parse actions and organize by stage"""
        current_stage = "PREFLOP"
        pot_size = sum(data.get("blinds_or_straddles", []))
        
        # Set initial bet to BB amount for preflop
        initial_bet = max(data.get("blinds_or_straddles", []))
        current_bet = initial_bet  
        
        stage_contributions = {
            "PREFLOP": {player: 0 for player in data["players"]},
            "FLOP": {player: 0 for player in data["players"]},
            "TURN": {player: 0 for player in data["players"]},
            "RIVER": {player: 0 for player in data["players"]}
        }
        
        # Add initial blinds to preflop contributions
        blinds = data.get("blinds_or_straddles", [])
        for i, blind in enumerate(blinds):
            if i < len(data["players"]) and blind > 0:
                player_name = data["players"][i]
                stage_contributions["PREFLOP"][player_name] = blind
        
        # Process each action
        for action_idx, action in enumerate(data["actions"]):
            # Deal hole cards
            if action.startswith('d dh'):
                parts = action.split(' ')
                if len(parts) >= 4:
                    player_id = parts[2]
                    player_name = result["players"][player_id]
                    cards_str = parts[3]
                    hole_cards = self._parse_cards(cards_str)
                    result["player_hole_cards"][player_name] = tuple(hole_cards)
                continue
            
            # Deal board cards - update stage
            elif action.startswith('d db'):
                # Record the stage transition
                result["stage_transitions"].append((current_stage, action_idx))
                
                cards_str = action.split(' ')[2]
                board_cards = self._parse_cards(cards_str)
                result["board_cards"].extend(board_cards)
                
                board_len = len(result["board_cards"])
                if board_len == 3:
                    current_stage = "FLOP"
                elif board_len == 4:
                    current_stage = "TURN"
                elif board_len == 5:
                    current_stage = "RIVER"
                
                current_bet = 0
                
                continue
            
            # Show cards (sm) - ignore for betting actions
            elif action.startswith('p') and ' sm ' in action:
                continue
            
            # Player action
            player_match = re.match(r'p(\d+) (\w+)(?: (\d+))?', action)
            if player_match:
                player_num, action_type, amount = player_match.groups()
                player_id = f"p{player_num}"
                player_name = result["players"][player_id]
                
                pot_before = pot_size
                
                player_contribution = stage_contributions[current_stage][player_name]
                
                min_bet_to_continue = max(0, current_bet - player_contribution)
                
                if current_stage == "PREFLOP" and current_bet == initial_bet:
                    # If player hasn't posted a blind, they need to match the BB
                    if player_contribution < initial_bet:
                        min_bet_to_continue = initial_bet
                
                core_action = None
                bet_amount = None
                additional = 0
                
                if action_type == "f":
                    core_action = Action.FOLD
                    bet_amount = None  # No betting with a fold
                    
                elif action_type == "cc":
                    core_action = Action.CHECK_CALL
                    if current_bet > player_contribution:
                        additional = current_bet - player_contribution
                        bet_amount = current_bet
                    else:
                        # This is a check - no additional money
                        bet_amount = 0
                        
                elif action_type == "cbr":
                    core_action = Action.RAISE
                    amount_int = int(amount) if amount else 0
                    additional = amount_int - player_contribution
                    bet_amount = amount_int
                    current_bet = amount_int
                
                pot_size += additional
                
                if bet_amount is not None and bet_amount > 0:
                    stage_contributions[current_stage][player_name] = bet_amount
                
                total_contribution = 0
                for stage in ["PREFLOP", "FLOP", "TURN", "RIVER"]:
                    if stage == current_stage:
                        # For current stage, use contribution before this action
                        total_contribution += player_contribution
                    else:
                        total_contribution += stage_contributions[stage][player_name]
                
                # Record the action if it's a recognized betting action
                if core_action is not None:
                    result["actions_by_stage"][current_stage].append({
                        "player": player_name,
                        "action": core_action,
                        "amount": bet_amount,
                        "stage_contribution": stage_contributions[current_stage][player_name],  
                        "total_contribution": total_contribution + additional, 
                        "pot_before": pot_before,
                        "pot_after": pot_size,
                        "min_bet_to_continue": min_bet_to_continue,  
                        "current_bet": current_bet,  
                        "action_idx": action_idx
                    })
        
        result["stage_contributions"] = stage_contributions
        
        # Record final stage transition
        result["stage_transitions"].append((current_stage, len(data["actions"])))
    
    def _parse_cards(self, cards_str: str) -> List[Card]:
        """Parse a string of cards into Card objects"""
        cards = []
        
        # Parse 2-character cards
        for i in range(0, len(cards_str), 2):
            if i + 1 < len(cards_str):
                rank_char = cards_str[i]
                suit_char = cards_str[i+1]
                
                rank = self._char_to_rank(rank_char)
                suit = self._char_to_suit(suit_char)
                
                if rank and suit:
                    cards.append(Card(rank, suit))
        
        return cards
    
    def _char_to_rank(self, char: str) -> Rank:
        """Convert a character to a Rank enum"""
        mapping = {
            '2': Rank.TWO,
            '3': Rank.THREE,
            '4': Rank.FOUR,
            '5': Rank.FIVE,
            '6': Rank.SIX,
            '7': Rank.SEVEN,
            '8': Rank.EIGHT,
            '9': Rank.NINE,
            'T': Rank.TEN,
            'J': Rank.JACK,
            'Q': Rank.QUEEN,
            'K': Rank.KING,
            'A': Rank.ACE
        }
        return mapping.get(char)
    
    def _char_to_suit(self, char: str) -> Suit:
        """Convert a character to a Suit enum"""
        mapping = {
            'c': Suit.CLUB,
            'd': Suit.DIAMOND,
            'h': Suit.HEART,
            's': Suit.SPADE
        }
        return mapping.get(char)
    
    def build_player_gamestates(self, hand_data: str, target_player: str) -> List[GameState]:
        """Build GameState objects for a specific player's decision points"""
        # Parse the hand data
        parsed_data = self.parse_hand(hand_data)
        
        # Find the target player's decision points
        target_player_actions = []
        for stage_name, actions in parsed_data["actions_by_stage"].items():
            for action in actions:
                if action["player"] == target_player:
                    target_player_actions.append((stage_name, action))
        
        # Limit to maximum of one action per stage (max 4 total)
        stage_seen = set()
        filtered_player_actions = []
        
        for stage_name, action in target_player_actions:
            if stage_name not in stage_seen:
                stage_seen.add(stage_name)
                filtered_player_actions.append((stage_name, action))
                
                # Stop if we have actions for all 4 stages
                if len(stage_seen) == 4:
                    break
        
        # Build a GameState for each decision point
        gamestates = []
        
        for stage_name, player_action in filtered_player_actions:
            stage = self.stages[stage_name]
            action_idx = player_action["action_idx"]
            
            # Get target player's position relative to BB
            target_player_spots_from_bb = parsed_data["player_positions"][target_player]
            
            # Get state of community cards at this point
            community_cards = []
            if stage_name != "PREFLOP":
                board_idx = 0
                if stage_name == "FLOP":
                    board_idx = 3
                elif stage_name == "TURN":
                    board_idx = 4
                elif stage_name == "RIVER":
                    board_idx = 5
                community_cards = parsed_data["board_cards"][:board_idx]
            
            pot_size = player_action["pot_before"]
            
  
            current_stage_actions = parsed_data["actions_by_stage"].get(stage_name, [])
            highest_contribution = 0
            for act in current_stage_actions:
                if act["action_idx"] < action_idx and act["action"] != Action.FOLD and act["amount"] is not None:
                    highest_contribution = max(highest_contribution, act["amount"])
            
            target_player_contribution = 0
            for act in current_stage_actions:
                if act["action_idx"] < action_idx and act["player"] == target_player and act["amount"] is not None:
                    target_player_contribution += act["amount"]
            

            if stage_name == "PREFLOP":
                initial_bet = parsed_data.get("min_bet")
                if initial_bet is None:
                    initial_bet = max(parsed_data.get("blinds_or_straddles", [0]))
                if highest_contribution < initial_bet:
                    highest_contribution = initial_bet
            
            min_bet_to_continue = highest_contribution - target_player_contribution
            if min_bet_to_continue < 0:
                min_bet_to_continue = 0  
            
            current_stack = parsed_data["player_stacks"][target_player]
            
            
            # Then deduct any additional contributions up to current action
            for stage_actions in parsed_data["actions_by_stage"].values():
                for act in stage_actions:
                    if act["action_idx"] < action_idx and act["player"] == target_player:
                        if act["action"] == Action.CHECK_CALL and act["amount"] is not None:
                            blinds = 0
                            if target_player in parsed_data["player_list"]:
                                player_idx = parsed_data["player_list"].index(target_player)
                                if player_idx < len(parsed_data.get("blinds_or_straddles", [])):
                                    blinds = parsed_data.get("blinds_or_straddles", [])[player_idx]
                            
                            if act["amount"] > blinds:
                                current_stack -= (act["amount"] - blinds)
            
            # Create Player object for target player
            my_player = Player(
                spots_left_bb=parsed_data["player_positions"][target_player],
                cards=parsed_data["player_hole_cards"].get(target_player),
                stack_size=current_stack
            )
            
            # Track which stages the player has acted in
            player_has_acted = {
                "PREFLOP": False,
                "FLOP": False,
                "TURN": False,
                "RIVER": False
            }
            
            # Find the player's actions in previous stages (if any)
            for prev_stage_name in ["PREFLOP", "FLOP", "TURN", "RIVER"]:
                # Only process up to the current stage, never future stages
                if self.stages[prev_stage_name].value >= stage.value:
                    break
                    
                # Find the player's action in this previous stage (if any)
                player_action_in_stage = None
                for act in parsed_data["actions_by_stage"][prev_stage_name]:
                    if act["player"] == target_player:
                        player_action_in_stage = act
                        player_has_acted[prev_stage_name] = True
                
                # If player acted in this stage, add it to history
                if player_action_in_stage:
                    if prev_stage_name == "PREFLOP":
                        my_player.add_preflop_action(player_action_in_stage["action"], player_action_in_stage["amount"])
                    elif prev_stage_name == "FLOP":
                        if player_has_acted["PREFLOP"]:
                            my_player.add_flop_action(player_action_in_stage["action"], player_action_in_stage["amount"])
                    elif prev_stage_name == "TURN":
                        if player_has_acted["PREFLOP"] and player_has_acted["FLOP"]:
                            my_player.add_turn_action(player_action_in_stage["action"], player_action_in_stage["amount"])
                    elif prev_stage_name == "RIVER":
                        if player_has_acted["PREFLOP"] and player_has_acted["FLOP"] and player_has_acted["TURN"]:
                            my_player.add_river_action(player_action_in_stage["action"], player_action_in_stage["amount"])
            
            # Create Player objects for other players
            other_players = []
            for player_name in parsed_data["player_stacks"].keys():
                if player_name != target_player:
                    player_stack = parsed_data["player_stacks"][player_name]
                    
                    player_spots_from_bb = parsed_data["player_positions"][player_name]
                    
                    if player_name in parsed_data["player_list"]:
                        player_idx = parsed_data["player_list"].index(player_name)
                        if player_idx < len(parsed_data.get("blinds_or_straddles", [])) and parsed_data.get("blinds_or_straddles", [])[player_idx] > 0:
                            player_stack -= parsed_data.get("blinds_or_straddles", [])[player_idx]
                    
                    # Deduct any bets, calls, or raises made up to this point
                    for stage_actions in parsed_data["actions_by_stage"].values():
                        for act in stage_actions:
                            if act["action_idx"] < action_idx and act["player"] == player_name:
                                if act["action"] == Action.CHECK_CALL and act["amount"] is not None:
                                    player_stack -= act["amount"]
                                elif act["action"] == Action.RAISE and act["amount"] is not None:
                                    if player_name in parsed_data["player_list"]:
                                        player_idx = parsed_data["player_list"].index(player_name)
                                        blinds = 0
                                        if player_idx < len(parsed_data.get("blinds_or_straddles", [])):
                                            blinds = parsed_data.get("blinds_or_straddles", [])[player_idx]
                                        
                                        player_stack -= (act["amount"] - blinds)
                    
                    other_player = Player(
                        spots_left_bb=player_spots_from_bb,
                        # Other players' cards are hidden
                        cards=None,
                        stack_size=player_stack
                    )
                    
                    # Track which stages this player has acted in
                    other_player_has_acted = {
                        "PREFLOP": False,
                        "FLOP": False,
                        "TURN": False,
                        "RIVER": False
                    }
                    
                    # Process previous complete stages - all actions are visible
                    for prev_stage_name in ["PREFLOP", "FLOP", "TURN", "RIVER"]:
                        # Only process up to the current stage
                        if self.stages[prev_stage_name].value >= stage.value:
                            break
                            
                        # Find player's action in this previous stage (if any)
                        player_action_in_stage = None
                        for act in parsed_data["actions_by_stage"][prev_stage_name]:
                            if act["player"] == player_name:
                                player_action_in_stage = act
                                other_player_has_acted[prev_stage_name] = True
                        
                        # Add to history if player acted
                        if player_action_in_stage:
                            if prev_stage_name == "PREFLOP":
                                other_player.add_preflop_action(player_action_in_stage["action"], player_action_in_stage["amount"])
                            elif prev_stage_name == "FLOP":
                                if other_player_has_acted["PREFLOP"]:
                                    other_player.add_flop_action(player_action_in_stage["action"], player_action_in_stage["amount"])
                            elif prev_stage_name == "TURN":
                                if other_player_has_acted["PREFLOP"] and other_player_has_acted["FLOP"]:
                                    other_player.add_turn_action(player_action_in_stage["action"], player_action_in_stage["amount"])
                            elif prev_stage_name == "RIVER":
                                if other_player_has_acted["PREFLOP"] and other_player_has_acted["FLOP"] and other_player_has_acted["TURN"]:
                                    other_player.add_river_action(player_action_in_stage["action"], player_action_in_stage["amount"])
                    
                    # Process current stage - only for players who already acted BEFORE target player's turn
                    if stage_name in parsed_data["actions_by_stage"]:
                        # Get this player's action order in the current stage
                        player_turn_order = self._get_player_turn_order(player_spots_from_bb, stage_name)
                        target_player_turn_order = self._get_player_turn_order(target_player_spots_from_bb, stage_name)
                        
                        if player_turn_order < target_player_turn_order:
                            # Find the player's action in current stage
                            player_action_in_current_stage = None
                            for act in parsed_data["actions_by_stage"][stage_name]:
                                if act["player"] == player_name and act["action_idx"] < action_idx:
                                    player_action_in_current_stage = act
                            
                            # Add action to history if found
                            if player_action_in_current_stage:
                                # Determine if all previous stages have actions
                                previous_stages_complete = True
                                for prev_stage in ["PREFLOP", "FLOP", "TURN"]:
                                    if self.stages[prev_stage].value < stage.value and not other_player_has_acted[prev_stage]:
                                        previous_stages_complete = False
                                        break
                                
                                if previous_stages_complete:
                                    if stage_name == "PREFLOP":
                                        other_player.add_preflop_action(player_action_in_current_stage["action"], player_action_in_current_stage["amount"])
                                    elif stage_name == "FLOP":
                                        other_player.add_flop_action(player_action_in_current_stage["action"], player_action_in_current_stage["amount"])
                                    elif stage_name == "TURN":
                                        other_player.add_turn_action(player_action_in_current_stage["action"], player_action_in_current_stage["amount"])
                                    elif stage_name == "RIVER":
                                        other_player.add_river_action(player_action_in_current_stage["action"], player_action_in_current_stage["amount"])
                    
                    other_players.append(other_player)
            
            # Create the GameState
            gamestate = GameState(
                stage=stage,
                community_cards=community_cards,
                pot_size=pot_size,
                min_bet_to_continue=min_bet_to_continue,
                my_player=my_player,
                other_players=other_players,
                my_player_action=(player_action["action"], player_action["amount"])
            )
            
            gamestates.append(gamestate)
        
        return gamestates
    
    def _get_player_turn_order(self, spots_from_bb: int, stage_name: str) -> int:
        """
        Determine a player's turn order based on their position relative to BB
        """
        if stage_name == "PREFLOP":
            # UTG is first to act
            if spots_from_bb == 1:
                return 0
            # SB acts next-to-last
            elif spots_from_bb == -1:  
                return float('inf') - 1
            # BB acts last
            elif spots_from_bb == 0:
                return float('inf')
            else:
                return spots_from_bb - 1  
        
        # POSTFLOP (FLOP, TURN, RIVER): SB acts first, then BB, then UTG, etc.
        else:
            if spots_from_bb == -1:
                return 0
            elif spots_from_bb == 0:
                return 1
            else:
                return spots_from_bb + 1  
        
        return spots_from_bb
    
    def format_gamestate(self, gamestate: GameState) -> str:
        """Format a GameState object into a readable string"""
        lines = []
        lines.append(f"Stage: {gamestate.stage.name}")
        
        # Community cards
        if gamestate.community_cards:
            lines.append(f"Board: {', '.join(str(card) for card in gamestate.community_cards)}")
        else:
            lines.append("Board: []")
        
        # Pot and bet info
        lines.append(f"Pot size: {gamestate.pot_size}")
        
        if gamestate.stage.name == "PREFLOP":
            lines.append(f"Min bet to continue: {gamestate.min_bet_to_continue} (BB or match current bet)")
        else:
            lines.append(f"Min bet to continue: {gamestate.min_bet_to_continue}")
        
        # My player info
        lines.append(f"\nMy Player:")
        lines.append(f"  Position: {gamestate.my_player.spots_left_bb} spots left of BB")
        lines.append(f"  Cards: {', '.join(str(card) for card in gamestate.my_player.cards) if gamestate.my_player.cards else 'Hidden'}")
        lines.append(f"  Stack: {gamestate.my_player.stack_size}")
        lines.append(f"  History: {gamestate.my_player.history}")
        
        # Action taken
        lines.append(f"  Action: {gamestate.my_player_action[0].name} {gamestate.my_player_action[1] if gamestate.my_player_action[1] is not None else ''}")
        
        # Other players
        lines.append(f"\nOther Players:")
        for i, player in enumerate(gamestate.other_players):
            lines.append(f"  Player {i+1}:")
            lines.append(f"    Position: {player.spots_left_bb} spots left of BB")
            lines.append(f"    Cards: {', '.join(str(card) for card in player.cards) if player.cards else 'Hidden'}")
            lines.append(f"    Stack: {player.stack_size}")
            lines.append(f"    History: {player.history}")
        
        return "\n".join(lines)



class GameStateRetriever:
    def __init__(self, directory_path, verbose=True):
        self.directory_path = directory_path
        self.pph_files_dict = defaultdict(list)
        self.verbose = verbose
        self._load_files()

    def _load_files(self):
        subfolders = [d for d in os.listdir(self.directory_path) if os.path.isdir(os.path.join(self.directory_path, d))]
        for subfolder in subfolders:
            subfolder_path = os.path.join(self.directory_path, subfolder)
            for file in os.listdir(subfolder_path):
                if file.endswith('.phh'):
                    self.pph_files_dict[subfolder].append(file)
        if self.verbose:
            print(f"Loaded files: {len(self.pph_files_dict)}")

    def get_player_game_states(self, player_name):
        builder = GameStateBuilder()
        all_game_states = []
        for subfolder, files in self.pph_files_dict.items():
            for file in files:
                with open(os.path.join(self.directory_path, subfolder, file), 'r') as f:
                    hand_data = f.read()
                    game_states = builder.build_player_gamestates(hand_data, player_name)
                    all_game_states.extend(game_states)
        return all_game_states
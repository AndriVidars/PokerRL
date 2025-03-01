from Poker.core.card import Card
from Poker.core.gamestage import Stage
from Poker.core.action import Action
from Poker.core.hand_evaluator import evaluate_hand
from typing import List, Tuple
from Poker.agents.util import remap_numbers

class Player():
    """
    Represents the state of a player. The most important part is the history,
    which holds all the actions the player has made.

    For now, the history assumes that each player only acts once during a stage.
    """
    def __init__(self,
        spots_left_bb: int, # how many seats to the left is player from bb
        cards: Tuple[Card, Card] | None, # None if not visible
        stack_size: int,
    ):
        self.spots_left_bb = spots_left_bb
        self.cards = cards
        self.stack_size = stack_size
        self.history: List[Tuple[Action, int]] = []

    def add_preflop_action(self, action: Action, raise_size: int | None):
        assert len(self.history) == 0
        self.history.append((action, raise_size))
    
    def add_flop_action(self, action: Action, raise_size: int | None):
        assert len(self.history) == 1
        self.history.append((action, raise_size))

    def add_turn_action(self, action: Action, raise_size: int | None):
        assert len(self.history) == 2
        self.history.append((action, raise_size))
    
    def add_river_action(self, action: Action, raise_size: int | None):
        assert len(self.history) == 3
        self.history.append((action, raise_size))

    def turn_to_play(self, stage: Stage, ttl_players:int):
        # returns what turn the player acted/should act on in a given stage
        # assuming no players have folded. 0 is first.
        if stage == Stage.PREFLOP:
            return (self.spots_left_bb - 1) % ttl_players
        return (self.spots_left_bb + 1) % ttl_players
    
    def get_visible_history(self, current_stage: Stage) -> List[Tuple[Action, int]]:
        """
        Returns visible history based on the current stage.
        Player can only see their own history up to the previous stage.
        """
        # If we're at the preflop, no history is visible
        if current_stage == Stage.PREFLOP:
            return []
        
        # Otherwise, return history up to the current stage (excluding current stage)
        return self.history[:current_stage.value]
    
    def get_visible_history_at_stage(self, stage: Stage, my_turn: int, their_turn: int) -> List[Tuple[Action, int]]:
        """
        Returns visible history at a specific stage based on turns.
        Players can only see other players' actions if those players have already acted in the current stage.
        
        Args:
            stage: The game stage
            my_turn: The current player's turn (0-indexed)
            their_turn: The other player's turn (0-indexed)
            
        Returns:
            Filtered history list
        """
        # Previous stages are always visible
        if stage.value > len(self.history):
            return self.history
            
        # For current stage, only show if they've already acted (their turn < my turn)
        if their_turn < my_turn:
            return self.history
        
        # Otherwise, only show history up to the previous stage
        return self.history[:stage.value - 1] if stage.value > 0 else []
    
    @property
    def in_game(self):
        return not any([action == Action.FOLD for action, _ in self.history])

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
            core_game = None,
            apply_visibility_rules: bool = True
    ):
        self.community_cards = community_cards
        self.pot_size = pot_size
        self.min_bet_to_continue = min_bet_to_continue
        self.stage = stage
        
        # Store the original player objects
        self._original_my_player = my_player
        self._original_other_players = other_players
        
        # Apply visibility rules by default
        if apply_visibility_rules:
            # My player only sees their history up to previous stage (not current)
            visible_my_player = self._apply_my_player_visibility(my_player)
            # Other players' history is visible based on turn order
            visible_other_players = self._apply_other_players_visibility(my_player, other_players)
            
            self.my_player = visible_my_player
            self.other_players = visible_other_players
        else:
            # Use original players without filtering history
            self.my_player = my_player
            self.other_players = other_players
            
        self.my_player_action = my_player_action
        self.core_game = core_game
        
    def _apply_my_player_visibility(self, my_player: Player) -> Player:
        """Filters my_player's history based on visibility rules"""
        # Create a copy of the player object
        filtered_player = Player(
            spots_left_bb=my_player.spots_left_bb,
            cards=my_player.cards,
            stack_size=my_player.stack_size
        )
        
        # Add only the visible part of history (up to previous stage)
        visible_history = my_player.get_visible_history(self.stage)
        
        # Rebuild history in the new player object
        for i, (action, amount) in enumerate(visible_history):
            if i == 0:
                filtered_player.add_preflop_action(action, amount)
            elif i == 1:
                filtered_player.add_flop_action(action, amount)
            elif i == 2:
                filtered_player.add_turn_action(action, amount)
            elif i == 3:
                filtered_player.add_river_action(action, amount)
                
        return filtered_player
        
    def _apply_other_players_visibility(self, my_player: Player, other_players: List[Player]) -> List[Player]:
        """Filters other players' history based on turn order and visibility rules"""
        filtered_players = []
        
        # Get effective turns
        total_players = 1 + len(other_players)
        my_turn = my_player.turn_to_play(self.stage, total_players)
        
        for other_player in other_players:
            other_turn = other_player.turn_to_play(self.stage, total_players)
            
            # Create a copy of the player object
            filtered_player = Player(
                spots_left_bb=other_player.spots_left_bb,
                cards=other_player.cards,
                stack_size=other_player.stack_size
            )
            
            # Get visible history based on turns
            visible_history = other_player.get_visible_history_at_stage(self.stage, my_turn, other_turn)
            
            # Rebuild history in the new player object
            for i, (action, amount) in enumerate(visible_history):
                if i == 0:
                    filtered_player.add_preflop_action(action, amount)
                elif i == 1:
                    filtered_player.add_flop_action(action, amount)
                elif i == 2:
                    filtered_player.add_turn_action(action, amount)
                elif i == 3:
                    filtered_player.add_river_action(action, amount)
                    
            filtered_players.append(filtered_player)
            
        return filtered_players

    @staticmethod
    def compute_hand_strength(cards: List[Card]):
        # computes hand strenght leveraging tie breakers
        # there's possibly a smarter/better way to do this
        rank, tbs = evaluate_hand(cards)
        strength = rank*10_000 + tbs[0]*100
        if len(tbs) > 1: strength += tbs[1]
        return strength

    def get_hand_strength(self):
        """ Returns the hand strength of the hand with respect to all possible hands.
        """
        if self.my_player.cards is None:
            return 0  # Return 0 strength if we don't have cards
        return self.compute_hand_strength(list(self.my_player.cards) + self.community_cards)

    def get_community_hand_strength(self):
        """ Returns the hand strength of the community hand with respect to all possible hands.
        """
        if len(self.community_cards) == 0: return 0
        return self.compute_hand_strength(self.community_cards)

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
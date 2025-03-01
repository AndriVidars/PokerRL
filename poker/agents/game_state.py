from poker.core.card import Card
from poker.core.gamestage import Stage
from poker.core.action import Action
from poker.core.hand_evaluator import evaluate_hand
from typing import List, Tuple
from poker.agents.util import remap_numbers

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
            core_game = None
    ):
        self.community_cards = community_cards
        self.pot_size = pot_size
        self.min_bet_to_continue = min_bet_to_continue
        self.stage = stage
        self.my_player = my_player
        self.other_players = other_players
        self.my_player_action = my_player_action
        self.core_game = core_game

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
        """ Returns the hand strength of the hand with respect to all possible hands.
        """
        if self.my_player.cards is None:
            return 0  # Return 0 strength if we don't have cards
        return self.compute_hand_strength([self.my_player.cards] + self.community_cards)

    @property
    def community_hand_strength(self):
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
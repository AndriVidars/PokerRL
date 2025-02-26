from core.action import Card
from core.gamestage import Stage
from core.action import Action
from typing import List, Tuple

class Player():
    """
    Represents the state of a player. The most important part is the history,
    which holds all the actions the player has made.

    For now, the history assumes that each player only acts once during a stage.
    """
    def __init__(self,
        in_game: bool, # whether the player is still in the game
        spots_left_bb: int, # how many seats to the left is player from bb
        cards: Tuple[Card, Card] | None, # None if not visible
        stack_size: int,
    ):
        self.in_game = in_game
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

    @property
    def hand_strength(self):
        """
        TODO: implement this method so that it returns the ranking of the hand with respect
        to all possible hands.
        """
        assert False

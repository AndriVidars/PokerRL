from typing import List
from poker.core.deck import Deck
from poker.core.card import Card
from poker.core.player import Player
from poker.core.pot import Pot
from poker.core.gamestage import Stage
from poker.core.action import Action
import poker.core.hand_evaluator as hand_eval
from typing import Optional, List, Set
from itertools import combinations


class Game:
    def __init__(self, players: List[Player], big_amount: int, small_amount: int):
        self.players = players
        self.big_amount = big_amount
        self.small_amount = small_amount
        self.dealer_idx = 0  # Dealer position
        self.small_blind_idx = (self.dealer_idx + 1) % len(self.players)
        self.big_blind_idx = (self.dealer_idx + 2) % len(self.players)
        self.current_stage = Stage.PREFLOP
        self.community_cards:List[Card] = []
        self.min_bet = self.big_amount # TODO track min re-raise?
        self.pots:List[Pot] = []
        self.deck: Optional[Deck] = None 
        self.active_players:Optional[Set[Player]] = set(self.players) # active players in each round - not all in or folded

        self.game_completed = False
        self.raise_in_round = False
        self.rounds_played = 0
        self.verbose = True

        for player in self.players:
            player.game = self

        self.players_eliminated = []

    def next_stage(self):
        self.current_stage = Stage((self.current_stage.value + 1) % len(Stage))

    def move_blinds(self):
        self.dealer_idx = (self.dealer_idx + 1) % len(self.players)
        self.small_blind_idx = (self.dealer_idx + 1) % len(self.players)
        self.big_blind_idx = (self.dealer_idx + 2) % len(self.players)
    
    # check if every player except one has folded -> autmatic win
    def auto_pot(self):
        non_folded = [p for p in self.players if not p.folded]
        return len(non_folded) == 1
    
    def gameplay_loop(self):
        while True:
            self.raise_in_round = False

            if self.auto_pot():
                self.decide_pot()
                self.move_blinds()
                self.current_stage = Stage.PREFLOP
            else:
                match self.current_stage:
                    case Stage.PREFLOP:
                        self.preflop()
                    case Stage.FLOP:
                        self.flop()
                    case Stage.TURN:
                        self.turn()
                    case Stage.RIVER:
                        self.river()
                        self.move_blinds()
                self.next_stage()
            
            if len(self.players) == 1:
                self.game_completed = True
                if self.verbose:
                    print(f"Game won by player: {self.players[0]} after {self.rounds_played} rounds")
                return

    def next_player(self, curr_player_idx: int):
        next_idx = (curr_player_idx + 1) % len(self.players)
        while self.players[next_idx] not in self.active_players:
            next_idx = (next_idx + 1) % len(self.players)
        
        return next_idx

    def betting_loop(self):
        if len(self.active_players) == 1:
            return
        
        match self.current_stage:
            case Stage.PREFLOP:
                curr_player_idx = (self.big_blind_idx + 1) % len(self.players)  # UTG starts preflop
            case _:
                curr_player_idx = (self.dealer_idx + 1) % len(self.players)  # First to act after dealer

        while self.players[curr_player_idx] not in self.active_players:
            curr_player_idx = self.next_player(curr_player_idx)
        
        init_player_idx = -1
        while init_player_idx != curr_player_idx: # full round around table form first action(non-fold)d
            curr_player = self.players[curr_player_idx]
            action = curr_player.act()
            if self.verbose:
                print(f"Player {curr_player}, Action: {action}")
                if curr_player.all_in:
                    print(f"Player: {curr_player} is now ALL IN")

        
            if init_player_idx == -1 and action != Action.FOLD:
                init_player_idx = curr_player_idx
            
            if action == Action.RAISE:
                self.raise_in_round = True
                init_player_idx = curr_player_idx
            
            if curr_player.all_in or curr_player.folded:
                self.active_players.remove(curr_player)
            
            if len(self.active_players) == 1:
                break

            curr_player_idx = self.next_player(curr_player_idx) # next player

    
    def handle_blinds(self):
        small_player, big_player = self.players[self.small_blind_idx], self.players[self.big_blind_idx]

        big_player_amount_in = min(big_player.stack, self.big_amount)
        small_player_amount_in = min(small_player.stack, self.small_amount)
        big_player.stack -= big_player_amount_in
        small_player.stack -= small_player_amount_in

        small_pot = Pot()
        small_pot_amt = min(big_player_amount_in, small_player_amount_in)
        small_pot.add_contribution(small_player, small_pot_amt)
        small_pot.add_contribution(big_player, small_pot_amt)
        big_player_amount_in -= small_pot_amt
        small_player_amount_in -= small_pot_amt

        self.pots.append(small_pot)

        if small_player_amount_in > 0:
            small_pot_ = Pot()
            small_pot_.add_contribution(small_player, small_player_amount_in)
            self.pots.append(small_pot_)
        
        if big_player_amount_in > 0:
            big_pot = Pot()
            big_pot.add_contribution(big_player, big_player_amount_in)
            self.pots.append(big_pot)
        
        small_player.all_in == small_player.stack == 0
        big_player.all_in == big_player.stack == 0

        if self.verbose:
            if small_player.all_in:
                print(f"Player: {small_player} is now ALL IN")
            if big_player.all_in:
                print(f"Player: {big_player} is now ALL IN")
     
    def preflop(self):
        if self.verbose:
            print(f"\n{25*'-'}\nPreflop\n{25*'-'}\n")

        self.active_players = set(self.players)
        self.handle_blinds()

        self.deck = Deck()
        for _ in range(2):
            for p in self.players:
                p.hand.append(self.deck.deck.pop())

        self.betting_loop()
    
    def flop(self):
        if self.verbose:
            print(f"\n{25*'-'}\nFlop\n{25*'-'}\n")

        burn = self.deck.deck.pop()
        for _ in range(3):
            self.community_cards.append(self.deck.deck.pop())
        
        if self.verbose:
            print(f"\n\nCommunity Cards After Flop:\n{self.community_cards}\n")
        
        self.betting_loop()
    
    def turn(self):
        if self.verbose:
            print(f"\n{25*'-'}\nTurn\n{25*'-'}\n")
        
        burn = self.deck.deck.pop()
        self.community_cards.append(self.deck.deck.pop())
        
        if self.verbose:
            print(f"\n\nCommunity Cards After Turn:\n{self.community_cards}\n")
        
        self.betting_loop()
    
    def river(self):
        if self.verbose:
            print(f"\n{25*'-'}\nRiver\n{25*'-'}\n")

        burn = self.deck.deck.pop()
        self.community_cards.append(self.deck.deck.pop())
        
        if self.verbose:
            print(f"\n\nCommunity Cards After River:\n{self.community_cards}\n")
        
        self.betting_loop()
        self.decide_pot()

    
    def decide_pot(self):
        self.rounds_played += 1
        # this concludes every round
        player_hands_rankings = {}
        for p in self.players:
            if not p.folded:
                best_rank, best_tiebreakers = hand_eval._find_best_hand(self.community_cards + p.hand)
                player_hands_rankings[p] = (best_rank, best_tiebreakers)

        for pot in self.pots:
            if not pot.eligible_players:
                continue
            pot_players = sorted(list(pot.eligible_players), key=lambda x: player_hands_rankings[x], reverse=True)
            tied_players = [pot_players[0]]
            for i in range(1, len(pot_players)):
                if player_hands_rankings[pot_players[i]] == player_hands_rankings[pot_players[0]]:
                    tied_players.append(pot_players[i])
                else:
                    break

            amount_each = pot.total_amount // len(tied_players)
            remainder = pot.total_amount % len(tied_players)

            for p in tied_players:
                p.stack += amount_each
            
            # hacky since chips are int
            if remainder > 0:
                tied_players[0].stack += remainder


        # reset for next round
        players = list(self.players)
        for p in players:
            if p.stack == 0:
                self.players_eliminated.append((p, self.rounds_played))
                self.players.remove(p)
                if self.verbose:
                    print(f"Player {p} has been eliminated")
            else:
                if self.verbose:
                    print(f"Player: {p} stack after {self.rounds_played} rounds: {p.stack}")
                p.folded = False
                p.all_in = False
                p.hand = []

        self.active_players = set(self.players)
        self.pots = [] # reset
        self.community_cards = []
        
              
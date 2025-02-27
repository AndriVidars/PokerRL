from typing import List
from poker.core.deck import Deck
from poker.core.card import Card
from poker.core.player import Player
from poker.core.pot import Pot
from poker.core.pokerhand import PokerHand
from poker.core.gamestage import Stage
from poker.core.action import Action
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
        self.min_bet = self.big_amount
        self.pots:List[Pot] = []
        self.deck: Optional[Deck] = None 
        self.active_players:Optional[Set[Player]] = None # active players in each round - not all in or folded
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
    
    def gameplay_loop(self):
        # Play one complete hand
        try:
            # Play each stage in order
            self.preflop()
            
            # Only continue if more than one player is active
            if len(self.active_players) > 1:
                self.next_stage()  # Move to FLOP
                self.flop()
                
                if len(self.active_players) > 1:
                    self.next_stage()  # Move to TURN
                    self.turn()
                    
                    if len(self.active_players) > 1:
                        self.next_stage()  # Move to RIVER
                        self.river()
            
            # Reset to PREFLOP for next hand
            self.current_stage = Stage.PREFLOP
            self.move_blinds()
            
            # Clear hands and reset player states
            for player in self.players:
                player.hand = []
                player.folded = False
                player.all_in = False
                
            # Reset community cards and pots
            self.community_cards = []
            self.pots = []
        except Exception as e:
            print(f"Error in gameplay loop: {e}")
            raise

    def next_player(self, curr_player_idx: int):
        next_idx = (curr_player_idx + 1) % len(self.players)
        while self.players[next_idx] not in self.active_players:
            next_idx = (next_idx + 1) % len(self.players)
        
        return next_idx

    def betting_loop(self):
        match self.current_stage:
            case Stage.PREFLOP:
                init_player_idx = (self.big_blind_idx + 1) % len(self.players)  # UTG starts preflop
            case _:
                init_player_idx = (self.dealer_idx + 1) % len(self.players)  # First to act after dealer

        while self.players[init_player_idx] not in self.active_players:
            init_player_idx = self.next_player(init_player_idx)
        
        curr_player_idx = -1
        while init_player_idx != curr_player_idx: # full round around table
            if curr_player_idx == -1:
                curr_player_idx = init_player_idx

            curr_player = self.players[curr_player_idx]
            action = curr_player.act()
            
            if action == Action.RAISE:
                init_player_idx = curr_player_idx
            
            if curr_player.all_in or curr_player.folded:
                self.active_players.remove(curr_player)
            
            curr_player_idx = self.next_player(curr_player_idx)
    
    def handle_blind(self, blind_idx, blind_amt):
        player = self.players[blind_idx]
        amount_in = min(player.stack, blind_amt)
        player.stack -= amount_in
        pot = Pot()
        pot.add_contribution(player, amount_in)
        self.pots.append(pot)
        if player.stack == 0:
            player.all_in = True
            
    def preflop(self):
        if self.verbose:
            print(f"\n{25*'-'}\nPreflop\n{25*'-'}\n")

        self.active_players = set(self.players)
        self.handle_blind(self.small_blind_idx, self.small_amount)
        self.handle_blind(self.big_blind_idx, self.big_amount)

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
        player_hands = []
        for p in self.players:
            if not p.folded:
                best_hand = sorted([PokerHand(list(c)) for c in combinations(self.community_cards + p.hand, 5)], reverse=True)[0]
                player_hands.append((best_hand, p))

        player_hands.sort(key=lambda x: x[0], reverse=True)
        player_hands_dict = {x[1]: x[0] for x in player_hands}
        player_hand_rankings = {x[1]: i for i, x in enumerate(player_hands)}  # player -> ranking

        for pot in self.pots:
            # Filter out players not eligible for this pot
            eligible_pot_players = [p for p in pot.eligible_players if not p.folded]
            if not eligible_pot_players:
                continue
                
            # Sort eligible players by hand ranking
            pot_players = sorted(list(eligible_pot_players), key=lambda x: player_hand_rankings.get(x, 999))
            if not pot_players:
                continue
                
            tied_players = [pot_players[0]]
            for i in range(1, len(pot_players)):
                if player_hands_dict[pot_players[i]] == player_hands_dict[pot_players[0]]:
                    tied_players.append(pot_players[i])
                else:
                    break

            amount_each = pot.total_amount // len(tied_players)
            remainder = pot.total_amount % len(tied_players)

            # Store the winners for this pot
            pot.winners = tied_players.copy()
            
            for p in tied_players:
                p.stack += amount_each
            
            # hacky since chips are int
            if remainder > 0:
                tied_players[0].stack += remainder

        players = list(self.players)
        for p in players:
            if p.stack == 0:
                self.players.remove(p)
                print(f"Player {p} has been eliminated")
            else:
                print(f"Player: {p} stack after round: {p.stack}")
                p.folded = False
                p.all_in = False


            
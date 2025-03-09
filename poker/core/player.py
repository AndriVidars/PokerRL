from poker.core.card import Card
from poker.core.action import Action
from poker.core.gamestage import Stage
from poker.core.pot import Pot
from typing import List
from abc import ABC, abstractmethod

class Player(ABC):
    def __init__(self, name:str, stack:int=0):
        self.name = name
        self.stack=stack
        self.hand: List[Card] = []
        self.all_in = False
        self.folded = False
        self.game = None # type Game
    
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        if isinstance(other, Player):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)
    
    @abstractmethod
    def _act(self) -> Action:
        pass
    
    def act(self) -> Action:
        action = self._act()
        self.post_act_hook(action) 
        return action
    
    def post_act_hook(self, action):
        self.game.game_state_players[self].stack_size = self.stack

        if action == Action.FOLD:
            self.folded = True
            self.game.game_state_players[self].history.append((Action.FOLD, 0))
        elif action == Action.CHECK_CALL:
            self.game.game_state_players[self].history.append((Action.CHECK_CALL, 0))
        # NOTE for the raise action history is added inside the handle_raise

        # this is also controlled in other places(where it matters in loop)
        if self.stack == 0:
            self.all_in == True
            if self.game.verbose:
                print(f"Player: {self} is All In")

        if self.folded or self.all_in:
            self.game.active_players.remove(self)
            del self.game.game_state_players[self]
            for p in self.game.active_players:
                self.game.game_state_players[p].spots_left_bb = self.game.pos_from_big_blind(p)
        
    
    def handle_fold(self):
        self.folded = True
        for pot in self.game.pots:
            if self in pot.eligible_players:
                pot.eligible_players.remove(self)
        
          
    def handle_check_call(self):
        for i, pot in enumerate(self.game.pots):            
            pot_max = max(pot.contributions.values())
            due = pot_max if self not in pot.eligible_players else pot_max - pot.contributions[self]
            # need to handle this, maybe the due is always pot max? since
            if due >= self.stack:
                self.all_in = True
                amount_in = self.stack
                self.stack = 0
                pot.add_contribution(self, amount_in)
                side_pot = pot.split_pot(self)
                if side_pot:
                    self.game.pots.insert(i+1, side_pot)  
            
            elif due > 0:
                pot.add_contribution(self, due)
                self.stack -= due
                self.all_in = self.stack == 0

            if self.all_in:
                return
        
        if not self.all_in and self.game.current_stage == Stage.PREFLOP:
            sum_max_pots = sum(max(pot.contributions.values()) for pot in self.game.pots if pot.contributions)
            if sum_max_pots < self.game.big_amount:
                diff = self.game.big_amount - sum_max_pots
                amount_in = min(self.stack, diff)
                self.stack -= amount_in
                pot = Pot()
                pot.add_contribution(self, amount_in)
                self.game.pots.append(pot)
                if self.stack == 0:
                    self.all_in = True
        

    def handle_raise(self, raise_amt):
        self.handle_check_call() 
        # validate funds
        assert raise_amt <= self.stack, f"Raise amount ({raise_amt}) exceeds stack ({self.stack})"
        assert raise_amt >= self.game.min_bet
        
        self.stack -= raise_amt
        if self.stack == 0:
            self.all_in = True

        pot = Pot()
        pot.add_contribution(self, raise_amt)
        self.game.pots.append(pot)
        self.game.game_state_players[self].history.append((Action.RAISE, raise_amt))


    def get_call_amt_due(self):
        pots = self.game.pots
        call_amt_due = 0
        pot_maxs = []
        for pot in pots:
            pot_max = max(pot.contributions.values())
            pot_maxs.append(pot_max)
            call_amt_due += (pot_max if self not in pot.eligible_players else pot_max - pot.contributions[self])

        # preflop hack
        if self.game.current_stage == Stage.PREFLOP and (self.stack-call_amt_due) > 0:
            sum_max_pots = sum(pot_maxs)
            if sum_max_pots < self.game.big_amount:
                call_amt_due += (self.game.big_amount - sum_max_pots)

        return call_amt_due
            
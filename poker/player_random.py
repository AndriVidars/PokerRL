from poker.core.player import Player
from poker.core.action import Action
from poker.core.gamestage import Stage
import numpy as np

import random

class PlayerRandom(Player):
    def act(self):
        max_raise_amt = self.max_raise_amount()
        actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE]
        if max_raise_amt < self.game.min_bet:
            actions.pop() # cannot raise
        
        if self.game.current_stage != Stage.PREFLOP and not self.game.raise_in_round:
            actions.pop(0) # never fold in this scenario
        
        action = random.choice(actions)
        match action:
            case Action.FOLD:
                self.handle_fold()
            case Action.CHECK_CALL:
                self.handle_check_call()
            case Action.RAISE:
                self.handle_raise(self.get_raise_amt(max_raise_amt))
        
        return action
  
    def max_raise_amount(self):
        # for safety, game api should prevent this though
        if self.all_in:
            return 0
        
        pots = self.game.pots
        call_amt_due = 0
        for pot in pots:
            pot_max = max(pot.contributions.values())
            call_amt_due += pot_max if self not in pot.eligible_players else pot_max - pot.contributions[self]
        
        return self.stack - call_amt_due
    
    
    def get_raise_amt(self, max_raise_amount):
        if max_raise_amount < self.game.min_bet:
            return 0

        min_raise = self.game.min_bet
        choices = np.arange(min_raise, max_raise_amount + 1)
        probabilities = np.linspace(1, 0.1, len(choices))
        probabilities /= probabilities.sum()

        return np.random.choice(choices, p=probabilities)

        

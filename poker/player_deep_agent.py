from poker.core.player import Player
from poker.core.action import Action
from poker.core.card import Rank
from poker.core.gamestage import Stage
from poker.agents.deep_learning_agent import PokerPlayerNetV1
from poker.agents.game_state import Player as GameStatePlayer
from poker.agents.game_state import GameState
import numpy as np
import torch
import copy

class PlayerDeepAgent(Player):
    def __init__(self, name, stack = 0, state_dict_dir="poker/a9e8c8.14060308.st"):
        super().__init__(name, stack)
        self.agent = PokerPlayerNetV1(use_batchnorm=False)
        self.agent.load_state_dict(state_dict=torch.load(state_dict_dir))
    
    def _act(self):
        stack_pre_action = self.stack
        game_state = GameState(
            stage = self.game.current_stage,
            community_cards = self.game.community_cards,
            pot_size = self.game.pot_size(), # amount needed to call, see how this is computed
            min_bet_to_continue = self.get_call_amt_due(), # see how this is implemented, are we on the same page
            my_player = self.game.game_state_players[self],
            other_players = [copy.deepcopy(v) for k, v in self.game.game_state_players.items() if k != self],
            my_player_action=None
        )

        # hide cards
        for op in game_state.other_players:
            op.cards = None
        
        game_state = copy.deepcopy(game_state) # HACK so that actions dont leak into the history before action is taken
        action_probs, raise_ratio = self.agent.eval_game_state(game_state)
        # moving out of torch - do we need to do the following inside torch? for gradient updates?

        action_probs = action_probs.detach().numpy() # to cpu? for cuda?
        raise_ratio = raise_ratio.item()
        
        # TODO(roberto), this happens sometimes(not very frequent)
        if np.isnan(action_probs).any():
            self.handle_check_call()
            return Action.CHECK_CALL

        actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE]
        action = np.random.choice(actions, p=action_probs)
        match action:
            case Action.FOLD:
                self.handle_fold()
            case Action.CHECK_CALL:
                self.handle_check_call()
            case Action.RAISE:
                # model outputs raise proportional to current pot(entire pot, not just within round bets?)
                raise_amt = int(game_state.pot_size * raise_ratio)
                assert raise_amt <= self.stack, "insufficient funds"
                
                # the game environment treats raise as betting the call amount + an additional amount(raise amount)
                # this makes some adjustment so that the agent doesnt violate the environment
                raise_amt -= game_state.min_bet_to_continue
                
                # TODO(roberto): handle this in model and change this to an assert
                if raise_amt < self.game.min_bet: # this is the big blind amount
                    # cannot raise, revert to check/call
                    self.handle_check_call()
                    return Action.CHECK_CALL
            
                self.handle_raise(raise_amt)
        
        return action

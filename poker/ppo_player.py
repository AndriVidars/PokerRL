from poker.core.player import Player
from poker.core.action import Action
from poker.core.card import Rank, RANK_ORDER
from poker.core.gamestage import Stage
from poker.agents.game_state import Player as GameStatePlayer
from poker.agents.game_state import GameState
from poker.agents.ppo_agent import PPOAgent
import copy
import torch
import numpy as np

class PlayerPPO(Player):
    """
    Player that uses PPO to make decisions in the poker game.
    """
    def __init__(self, name, ppo_agent=None, stack=0, primary=False):
        super().__init__(name, stack)
        self.ppo_agent = ppo_agent
        self.accumulated_reward = 0
        self.round_start_stack = stack
        self.is_training = True
        self.last_stack_before_action = None
        self.recent_actions = []
        self.primary = primary
    
    def _act(self):
        """
        Main method to determine the player's action using PPO
        """
        # Record stack before action
        self.last_stack_before_action = self.stack
        
        # Create game state for the agent
        game_state = GameState(
            stage=self.game.current_stage,
            community_cards=self.game.community_cards,
            pot_size=self.game.pot_size(),
            min_bet_to_continue=self.get_call_amt_due(),
            my_player=self.game.game_state_players[self],
            other_players=[copy.deepcopy(v) for k, v in self.game.game_state_players.items() if k != self],
            my_player_action=None,
            min_allowed_bet=self.game.min_bet
        )
        
        # Hide other players' cards
        for op in game_state.other_players:
            op.cards = None
        
        # Make a deep copy to prevent action history leakage
        game_state = copy.deepcopy(game_state)
        
        # Use the PPO agent to select an action
        action, raise_amount, action_probs, raise_size_dist = self.ppo_agent.select_action(game_state)
        

        # Store the action for reward calculation later
        self.recent_actions.append((action, raise_amount))
        if action == Action.FOLD and self.get_call_amt_due() == 0: 
             action = Action.CHECK_CALL
        
        # Execute the action
        match action:
            case Action.FOLD:
                self.handle_fold()
            case Action.CHECK_CALL:
                self.handle_check_call()
            case Action.RAISE:
                # Handle special case for re-raise when only one active player left
                if len(self.game.active_players) == 1:
                    self.handle_check_call()
                    action = Action.CHECK_CALL
                else:
                    raise_amount = max(self.game.min_bet, raise_amount)
           
                    raise_amount = min(int(raise_amount), int(self.stack))
                    
                    self.handle_raise(raise_amount)
        
        stack_post_action = self.stack
        action_amt = 0
        if action == Action.CHECK_CALL:
            action_amt = self.last_stack_before_action - stack_post_action
        elif action == Action.RAISE:
            action_amt = raise_amount
        
        self.game.current_round_game_states[self][1].append(
            (game_state, action_probs, raise_size_dist, (action, action_amt))
        )
        
        # Calculate immediate reward for this action
        self._calculate_immediate_reward(action, stack_post_action)
            
        return action
    
    def _calculate_immediate_reward(self, action, current_stack):
        """
        Calculate immediate reward after taking an action
        """
        # Calculate stack change as immediate reward
        stack_diff = current_stack - self.last_stack_before_action
        
        # Base reward is the change in stack size relative to pot
        pot_size = max(1, self.game.pot_size())  # Avoid division by zero
        immediate_reward = stack_diff / pot_size
        
        # If folding, give small negative reward
        if action == Action.FOLD:
            # Calculate relative strength to determine if folding was reasonable
            if hasattr(self, 'game') and self.game.community_cards:
                hand_strength = self._estimate_hand_strength()
                # If hand is strong, penalize folding more
                if hand_strength > 0.7:  # Strong hand threshold
                    immediate_reward -= 0.2
            else:
                immediate_reward -= 0.05  # Small penalty for folding early
        
        
        self.ppo_agent.store_reward(immediate_reward, is_terminal=False)
        
        return immediate_reward
    
    def _estimate_hand_strength(self):
        """
        Estimate hand strength based on current cards
        Returns a value between 0 and 1
        """
        if not hasattr(self, 'game') or not self.game.community_cards:
            # Preflop hand strength estimation
            if not self.hand or len(self.hand) < 2:
                return 0.5  # Default to moderate strength
            
            # Simple preflop hand strength heuristic
            ranks = [card.rank for card in self.hand]
            is_pair = ranks[0] == ranks[1]
            high_card = max(RANK_ORDER[rank] for rank in ranks)
            
            if is_pair:
                return 0.5 + (RANK_ORDER[ranks[0]] / 28.0)  # Range 0.5 - 1.0 for pairs
            else:
                return 0.3 + (high_card / 42.0)  # Range 0.3 - 0.63 for non-pairs
        else:
            # Use game state for hand strength
            from poker.core import hand_evaluator
            hand_strength = 0.0
            
            # Calculate hand strength based on evaluator
            if self.hand and self.game.community_cards:
                rank, _ = hand_evaluator.evaluate_hand(self.hand + self.game.community_cards)
                # Normalize rank (1-9) to 0-1 scale
                hand_strength = min(1.0, rank / 9.0)
            
            return hand_strength
    
    def post_act_hook(self, action, stack_diff):
        """
        Override post_act_hook to calculate rewards after the action is executed
        """
        # Call parent method first
        super().post_act_hook(action, stack_diff)
        
        # Additional reward logic for PPO can be added here if needed
    
    def reset_for_new_round(self):
        """
        Reset player state for a new round
        """
        self.round_start_stack = self.stack
        self.recent_actions = []
        
    def store_terminal_reward(self, end_stack):
        """
        Store terminal reward at the end of a round/game
        """
        # Calculate final reward as relative stack change over the round
        relative_stack_change = (end_stack - self.round_start_stack) / max(1, self.round_start_stack)
        
        # Boost or penalize based on the outcome
        terminal_reward = relative_stack_change * 3.0  # Amplify the terminal reward
        
        self.ppo_agent.store_reward(terminal_reward, is_terminal=True)
        
        # Reset for next round
        self.accumulated_reward = 0

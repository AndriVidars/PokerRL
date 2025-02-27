import torch
import numpy as np
import os
import time
from typing import List
import matplotlib.pyplot as plt
from collections import deque

from poker.core.player import Player
from poker.core.action import Action
from poker.core.game import Game
from poker.core.gamestage import Stage
from poker.agents.dqn_agent import DQNAgent
from poker.agents.game_state import GameState

class DQNPlayer(Player):
    def __init__(self, stack_size: int, position: int, agent: DQNAgent, is_training: bool = True):
        super().__init__(f"DQN_Player_{position}", stack_size)
        self.position = position
        self.agent = agent
        self.is_training = is_training
        self.current_state = None
        self.last_action = None
        self.last_action_idx = None
        self.last_raise_amount = None
        self.initial_stack = stack_size
        
    def act(self) -> Action:
        # Get game state from player's perspective
        game_state = self._create_game_state()
        self.current_state = self.agent.preprocess_state(game_state)
        
        # Select action using agent
        action_idx, raise_amount = self.agent.select_action(self.current_state)
        
        # Convert action to poker action
        action, raise_size = self.agent.convert_action_to_poker_action(
            action_idx, raise_amount, game_state
        )
        
        # Store action for later use in training
        self.last_action = action
        self.last_action_idx = action_idx
        self.last_raise_amount = raise_amount
        
        # Execute action in the game
        return self._execute_action(action, raise_size)
    
    def _create_game_state(self) -> GameState:
        """
        Create a GameState object from the current game state
        """
        other_players = [p for p in self.game.players if p != self]
        
        # Calculate min bet to continue
        min_bet = 0
        for pot in self.game.pots:
            if self in pot.contributions:
                min_bet = max(min_bet, max(pot.contributions.values()) - pot.contributions.get(self, 0))
            else:
                min_bet = max(min_bet, max(pot.contributions.values())) if pot.contributions else 0
        
        # Calculate pot size
        pot_size = sum(pot.total_amount for pot in self.game.pots)
        if pot_size == 0:
            pot_size = 1  # Avoid division by zero
        
        # Create a simplified game state
        return {
            'stage': self.game.current_stage,
            'community_cards': self.game.community_cards,
            'pot_size': pot_size,
            'min_bet_to_continue': min_bet,
            'my_player': {
                'stack_size': self.stack,
                'hand': self.hand,
                'folded': self.folded,
                'all_in': self.all_in,
                'position': self.position
            },
            'other_players': [{
                'stack_size': p.stack,
                'position': p.position if hasattr(p, 'position') else 0,
                'folded': p.folded,
                'all_in': p.all_in
            } for p in other_players],
            'hand_strength': self._calculate_hand_strength(),
            'community_strength': self._calculate_community_strength()
        }
        
    def _calculate_hand_strength(self):
        """Calculate the strength of the player's hand"""
        if not self.hand or len(self.hand) < 2:
            return 0
            
        # Simple hand strength calculation - could be improved
        from poker.core.hand_evaluator import evaluate_hand
        
        try:
            if not self.game.community_cards:
                # Preflop hand strength based on card ranks
                card1, card2 = self.hand
                rank1, rank2 = card1.rank.value, card2.rank.value
                suited = 1 if card1.suit == card2.suit else 0
                # Simple preflop ranking based on high card, pair, and suited
                return (max(rank1, rank2) * 10 + min(rank1, rank2) + suited * 50) / 1000
            else:
                # Postflop hand strength using hand evaluator
                cards = self.hand + self.game.community_cards
                rank, _ = evaluate_hand(cards)
                return rank / 10  # Normalize
        except:
            return 0  # Default if evaluation fails
            
    def _calculate_community_strength(self):
        """Calculate the strength of the community cards alone"""
        if not self.game.community_cards:
            return 0
            
        from poker.core.hand_evaluator import evaluate_hand
        
        try:
            if len(self.game.community_cards) >= 5:
                rank, _ = evaluate_hand(self.game.community_cards[:5])
                return rank / 10
            else:
                # Not enough community cards for a full hand
                return sum(card.rank.value for card in self.game.community_cards) / 100
        except:
            return 0  # Default if evaluation fails
    
    def _execute_action(self, action: Action, raise_size: int) -> Action:
        """
        Execute action in the game
        """
        if action == Action.FOLD:
            self.folded = True
            print(f"Player {self.position} folds")
            return Action.FOLD
            
        elif action == Action.CHECK:
            # Check if we can check
            can_check = True
            for pot in self.game.pots:
                if self in pot.contributions:
                    if pot.contributions[self] < max(pot.contributions.values()):
                        can_check = False
                else:
                    can_check = False
            
            if can_check:
                print(f"Player {self.position} checks")
                return Action.CHECK
            else:
                # If we can't check, we fold
                self.folded = True
                print(f"Player {self.position} folds (invalid check)")
                return Action.FOLD
                
        elif action == Action.CALL:
            # Calculate amount to call
            call_amount = 0
            for pot in self.game.pots:
                if self in pot.contributions:
                    call_amount = max(call_amount, max(pot.contributions.values()) - pot.contributions[self])
                else:
                    call_amount = max(call_amount, max(pot.contributions.values()))
            
            # If we can't afford to call, go all-in
            if call_amount >= self.stack:
                call_amount = self.stack
                self.all_in = True
            
            # Execute call
            self.stack -= call_amount
            for pot in self.game.pots:
                if self in pot.eligible_players:
                    pot.add_contribution(self, call_amount)
                    break
            
            print(f"Player {self.position} calls {call_amount}")
            return Action.CALL
            
        elif action == Action.RAISE:
            # Ensure raise is valid
            if raise_size > self.stack:
                raise_size = self.stack
                self.all_in = True
            
            self.stack -= raise_size
            for pot in self.game.pots:
                if self in pot.eligible_players:
                    pot.add_contribution(self, raise_size)
                    break
            
            print(f"Player {self.position} raises {raise_size}")
            return Action.RAISE
            
        return Action.FOLD  # Default
    
    def receive_reward(self, reward: float, terminal: bool = False):
        """
        Receive reward for last action
        """
        if not self.is_training or self.last_action is None:
            return
        
        # Get current state
        try:
            next_state = self._create_game_state()
            next_state_features = self.agent.preprocess_state(next_state)
        except Exception:
            # If we can't create a valid state (e.g., game ended), use a default state
            next_state_features = np.zeros(self.agent.state_dim)
        
        # Store transition
        self.agent.store_transition(
            self.current_state,
            self.last_action_idx,
            reward,
            next_state_features,
            terminal
        )
        
        # Update agent
        self.agent.update()
        
        # Reset last action
        self.last_action = None
        self.last_action_idx = None
        self.last_raise_amount = None

def calculate_reward(player: DQNPlayer, game: Game) -> float:
    """
    Calculate reward for a player based on the game result
    """
    # Negative reward for folding
    if player.folded:
        return -0.5
    
    # Base reward on money won/lost
    reward = 0
    
    # Check if player won pots
    pots_won = 0
    for pot in game.pots:
        if player in pot.winners:
            pots_won += pot.total_amount / len(pot.winners)
    
    # Calculate money invested
    money_invested = player.initial_stack - player.stack
    
    # Reward is the profit
    reward = pots_won - money_invested
    
    # Normalize reward
    if reward > 0:
        reward = min(1.0, reward / player.initial_stack)
    else:
        reward = max(-1.0, reward / player.initial_stack)
    
    return reward

def train_dqn_agent(num_episodes: int = 10000, 
                   eval_frequency: int = 100, 
                   num_players: int = 6, 
                   initial_stack: int = 1000,
                   big_blind: int = 20,
                   small_blind: int = 10,
                   save_dir: str = "models"):
    """
    Train a DQN agent to play poker
    """
    # Create agent
    state_dim = 8  # Number of features in state
    action_dim = 4  # Fold, Check, Call, Raise
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training metrics
    episode_rewards = []
    avg_rewards = deque(maxlen=100)
    
    for episode in range(num_episodes):
        # Create players
        dqn_player = DQNPlayer(initial_stack, 0, agent, is_training=True)
        other_players = [
            DQNPlayer(initial_stack, i+1, agent, is_training=False) 
            for i in range(num_players-1)
        ]
        all_players = [dqn_player] + other_players
        
        # Create game
        game = Game(all_players, big_blind, small_blind)
        
        # Save initial stack
        for player in all_players:
            player.initial_stack = player.stack
        
        # Play game
        try:
            game.gameplay_loop()
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            continue
        
        # Calculate reward
        reward = calculate_reward(dqn_player, game)
        dqn_player.receive_reward(reward, terminal=True)
        
        # Update metrics
        episode_rewards.append(reward)
        avg_rewards.append(reward)
        
        # Logging
        if episode % 10 == 0:
            print(f"Episode: {episode}, Reward: {reward:.4f}, Avg Reward: {sum(avg_rewards)/len(avg_rewards):.4f}, Epsilon: {agent.epsilon:.4f}")
        
        # Evaluation
        if episode % eval_frequency == 0:
            # Save model
            agent.save(os.path.join(save_dir, f"dqn_agent_episode_{episode}.pt"))
            
            # Plot reward
            plt.figure(figsize=(10, 5))
            plt.plot(episode_rewards)
            plt.plot(np.convolve(episode_rewards, np.ones(100)/100, mode='valid'))
            plt.title("Episode Reward")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.savefig(os.path.join(save_dir, f"reward_plot_episode_{episode}.png"))
            plt.close()
    
    # Save final model
    agent.save(os.path.join(save_dir, "dqn_agent_final.pt"))
    
    return agent

if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Train agent
    train_dqn_agent(num_episodes=10000, 
                   eval_frequency=100, 
                   num_players=6,
                   initial_stack=1000,
                   big_blind=20,
                   small_blind=10,
                   save_dir="models")
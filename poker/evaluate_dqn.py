import torch
import numpy as np
import os
import argparse
from typing import List

from poker.core.player import Player
from poker.core.action import Action
from poker.core.game import Game
from poker.agents.dqn_agent import DQNAgent
from poker.train_dqn import DQNPlayer

def evaluate_agent(model_path: str, num_games: int = 100, num_players: int = 6, 
                  initial_stack: int = 1000, big_blind: int = 20, small_blind: int = 10):
    """
    Evaluate a trained DQN agent against random agents
    """
    # Create agent
    state_dim = 8  # Number of features in state
    action_dim = 4  # Fold, Check, Call, Raise
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Load model
    agent.load(model_path)
    agent.epsilon = 0.05  # Small epsilon for some exploration during evaluation
    
    # Metrics
    total_rewards = []
    win_count = 0
    
    for game_idx in range(num_games):
        print(f"Game {game_idx+1}/{num_games}")
        
        # Create players
        dqn_player = DQNPlayer(initial_stack, 0, agent, is_training=False)
        
        # Create random players
        random_agent = DQNAgent(state_dim, action_dim)
        random_agent.epsilon = 1.0  # Always random
        other_players = [
            DQNPlayer(initial_stack, i+1, random_agent, is_training=False) 
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
            print(f"Error in game {game_idx}: {e}")
            continue
        
        # Calculate results
        profit = dqn_player.stack - dqn_player.initial_stack
        total_rewards.append(profit)
        
        # Check if DQN player won
        if not dqn_player.folded and dqn_player.stack > 0:
            has_won = True
            for player in other_players:
                if player.stack > dqn_player.stack:
                    has_won = False
                    break
            if has_won:
                win_count += 1
        
        print(f"Game {game_idx+1} result: Profit = {profit}, Stack = {dqn_player.stack}")
    
    # Print overall results
    avg_reward = sum(total_rewards) / len(total_rewards)
    win_rate = win_count / num_games
    
    print("\n=== Evaluation Results ===")
    print(f"Model: {model_path}")
    print(f"Number of games: {num_games}")
    print(f"Average profit: {avg_reward:.2f}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Total profit: {sum(total_rewards)}")
    
    # More detailed statistics
    print(f"Max profit: {max(total_rewards)}")
    print(f"Min profit: {min(total_rewards)}")
    
    # Calculate percentiles
    percentiles = np.percentile(total_rewards, [25, 50, 75])
    print(f"25th percentile: {percentiles[0]:.2f}")
    print(f"Median: {percentiles[1]:.2f}")
    print(f"75th percentile: {percentiles[2]:.2f}")
    
    return avg_reward, win_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN poker agent")
    parser.add_argument("--model", type=str, default="models/dqn_agent_final.pt", help="Path to the model file")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--players", type=int, default=6, help="Number of players in each game")
    parser.add_argument("--stack", type=int, default=1000, help="Initial stack size")
    parser.add_argument("--big-blind", type=int, default=20, help="Big blind amount")
    parser.add_argument("--small-blind", type=int, default=10, help="Small blind amount")
    
    args = parser.parse_args()
    
    evaluate_agent(args.model, args.games, args.players, args.stack, args.big_blind, args.small_blind)
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from poker.core.game import Game
from poker.ppo_player import PlayerPPO
from poker.player_heuristic import PlayerHeuristic
from poker.player_random import PlayerRandom
from poker.player_deep_agent import PlayerDeepAgent
from poker.agents.ppo_agent import PPOAgent
from poker.agents.deep_learning_agent import PokerPlayerNetV1
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained PPO agent for poker')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained PPO model')
    parser.add_argument('--num_games', type=int, default=1000, help='Number of games to evaluate')
    parser.add_argument('--start_stack', type=int, default=400, help='Starting chip stack')
    parser.add_argument('--big_blind', type=int, default=10, help='Big blind amount')
    parser.add_argument('--small_blind', type=int, default=5, help='Small blind amount')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--opponent_type', type=str, default='heuristic', 
                        choices=['heuristic', 'random', 'imitation'],
                        help='Type of opponents to evaluate against')
    parser.add_argument('--imitation_model_path', type=str, 
                        default='poker/e55f94.12150310.st',
                        help='Path to the imitation learning model (if using imitation opponents)')
    return parser.parse_args()

def evaluate_ppo_agent(args):
    """
    Evaluate the PPO agent against different opponents
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create and load PPO agent
    ppo_agent = PPOAgent(device=device)
    ppo_agent.load(args.model_path)
    print(f"Loaded PPO agent from {args.model_path}")
    
    # Track metrics
    wins = 0
    stack_changes = []
    round_counts = []
    
    # Create imitation learning model if needed
    imitation_model = None
    if args.opponent_type == 'imitation':
        imitation_model = PokerPlayerNetV1()
        imitation_model.load_state_dict(torch.load(args.imitation_model_path))
        print(f"Loaded imitation learning model from {args.imitation_model_path}")
    
    # Run evaluation games
    for game_num in tqdm(range(args.num_games), desc="Evaluating"):
        # Create PPO player
        ppo_player = PlayerPPO("PPO_Player", ppo_agent, args.start_stack)
        ppo_player.is_training = False  # Disable training during evaluation
        
        # Create opponents based on selected type
        players = [ppo_player]
        
        if args.opponent_type == 'heuristic':
            for i in range(3):  # 3 opponents (total 4 players)
                players.append(PlayerHeuristic(f"Heuristic_{i}", args.start_stack))
        elif args.opponent_type == 'random':
            for i in range(3):
                players.append(PlayerRandom(f"Random_{i}", args.start_stack))
        elif args.opponent_type == 'imitation':
            for i in range(3):
                players.append(PlayerDeepAgent(f"Imitation_{i}", imitation_model, args.start_stack))
        
        # Randomize player order
        random.shuffle(players)
        
        # Create and run game
        game = Game(players, args.big_blind, args.small_blind, verbose=False)
        winner, rounds_played, eliminated, _ = game.gameplay_loop()
        
        # Record results
        if winner == ppo_player:
            wins += 1
        
        # Calculate stack change
        initial_stack = args.start_stack
        final_stack = ppo_player.stack if ppo_player.stack > 0 else 0
        stack_change = (final_stack - initial_stack) / initial_stack
        stack_changes.append(stack_change)
        round_counts.append(rounds_played)
    
    # Calculate and display results
    win_rate = wins / args.num_games
    avg_stack_change = np.mean(stack_changes)
    avg_rounds = np.mean(round_counts)
    
    print(f"\nEvaluation Results against {args.opponent_type} opponents:")
    print(f"Win Rate: {win_rate:.4f}")
    print(f"Average Stack Change: {avg_stack_change:.4f}")
    print(f"Average Game Rounds: {avg_rounds:.2f}")
    
    # Plot stack change distribution
    plt.figure(figsize=(10, 6))
    plt.hist(stack_changes, bins=20)
    plt.title(f'PPO Agent Stack Change Distribution vs {args.opponent_type.capitalize()} Opponents')
    plt.xlabel('Relative Stack Change')
    plt.ylabel('Frequency')
    
    # Save the plot
    results_dir = os.path.join('poker', 'policy_training', 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, f'ppo_vs_{args.opponent_type}_results.png')
    plt.savefig(plot_path)
    print(f"Results plot saved to {plot_path}")
    
    return win_rate, avg_stack_change, avg_rounds

if __name__ == '__main__':
    args = parse_args()
    evaluate_ppo_agent(args)

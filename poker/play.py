from poker.core.game import Game
from poker.player_io import PlayerIO
from poker.player_random import PlayerRandom
from poker.player_heuristic import PlayerHeuristic
from poker.player_deep_agent import PlayerDeepAgent
from poker.agents.deep_learning_agent import PokerPlayerNetV1
from poker.agents.ppo_agent import PPOAgent
from poker.ppo_player import PlayerPPO
from poker.utils import init_players
import time
import pickle
from tqdm import tqdm
import random
import torch
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Run poker games with different agent types')
    
    # Game parameters
    parser.add_argument('--num_games', type=int, default=1000, help='Number of games to run')
    parser.add_argument('--start_stack', type=int, default=400, help='Starting chip stack')
    parser.add_argument('--big_blind', type=int, default=10, help='Big blind amount')
    parser.add_argument('--small_blind', type=int, default=5, help='Small blind amount')
    
    # Player counts
    parser.add_argument('--ppo_agents', type=int, default=0, help='Number of PPO agents')
    parser.add_argument('--deep_agents', type=int, default=2, help='Number of deep learning agents')
    parser.add_argument('--heuristic_agents', type=int, default=2, help='Number of heuristic agents')
    parser.add_argument('--random_agents', type=int, default=0, help='Number of random agents')
    parser.add_argument('--io_agents', type=int, default=0, help='Number of IO (human) agents')
    
    # Model paths
    parser.add_argument('--deep_model', type=str, default='poker/193c5c.05050310.st', 
                        help='Path to deep learning model state dict')
    parser.add_argument('--ppo_model', type=str, default=None, 
                        help='Path to PPO model state dict')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='pkl', 
                        help='Directory to save statistics and game state batches')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Track statistics
    st = time.time()
    winner_stats = []
    eliminated_stats = []
    game_state_batches = []

    # Setup player types
    player_type_dict = {}
    if args.heuristic_agents > 0:
        player_type_dict[PlayerHeuristic] = args.heuristic_agents
    if args.deep_agents > 0:
        player_type_dict[PlayerDeepAgent] = args.deep_agents
    if args.random_agents > 0:
        player_type_dict[PlayerRandom] = args.random_agents
    if args.io_agents > 0:
        player_type_dict[PlayerIO] = args.io_agents
    if args.ppo_agents > 0:
        player_type_dict[PlayerPPO] = args.ppo_agents

    # Load deep learning model if needed
    agent_model = None
    if args.deep_agents > 0:
        print(f"Loading deep learning model from {args.deep_model}")
        agent_model = PokerPlayerNetV1(use_batchnorm=False)
        agent_model.load_state_dict(state_dict=torch.load(args.deep_model))
        agent_model.eval()  # Set to evaluation mode

    # Load PPO model if needed
    ppo_agent = None
    if args.ppo_agents > 0:
        if args.ppo_model is None:
            raise ValueError("PPO model path must be provided when using PPO agents")
        
        print(f"Loading PPO model from {args.ppo_model}")
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create and load PPO agent
        ppo_agent = PPOAgent(device=device)
        ppo_agent.load(args.ppo_model)
        
        # Set to evaluation mode (disable training)
        for param in ppo_agent.policy.parameters():
            param.requires_grad = False
        ppo_agent.policy.eval()
        ppo_agent.old_policy.eval()

    # Run games
    for _ in tqdm(range(args.num_games)):
        # Initialize players with both models
        players, players_str = init_players(
            player_type_dict, 
            agent_model=agent_model, 
            ppo_agent=ppo_agent, 
            start_stack=args.start_stack
        )
        
        # Randomize player order
        random.shuffle(players)

        # Run the game
        game = Game(players, args.big_blind, args.small_blind, False)
        winner, rounds_total, eliminated, game_state_batch = game.gameplay_loop()
        
        # Record statistics
        winner_stats.append((winner.__class__.__name__, rounds_total))
        for e in eliminated:
            eliminated_stats.append(
                (
                    e[0].__class__.__name__,
                    e[1],
                    e[1] / rounds_total
                )                    
            )

        # Record game state batches
        game_state_batches.append(game_state_batch)
    
    # Calculate and report timing
    et = time.time()
    elapsed_time = et - st
    print(f"Total time: {elapsed_time:.4f}, time per game: {elapsed_time/args.num_games:.4f}")

    # Summarize results
    win_counts = {}
    for winner, _ in winner_stats:
        win_counts[winner] = win_counts.get(winner, 0) + 1
    
    print("\nWin statistics:")
    for player_type, count in win_counts.items():
        print(f"{player_type}: {count} wins ({count/args.num_games*100:.2f}%)")

    # Save results
    fname_stats = os.path.join(args.output_dir, f'stats_{players_str}_{args.num_games}.pkl')
    fname_batches = os.path.join(args.output_dir, f'game_state_batches_{players_str}_{args.num_games}.pkl')

    with open(fname_stats, 'wb') as f:
        pickle.dump((winner_stats, eliminated_stats), f)
    
    if game_state_batches:
        # Convert keys to strings for pickling
        gsb = [{f"{k.__class__.__name__}_{k.name}": v for k, v in gb.items()} for gb in game_state_batches]
        with open(fname_batches, 'wb') as f:
            pickle.dump(gsb, f)

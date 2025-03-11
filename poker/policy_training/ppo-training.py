from poker.core.game import Game
from poker.ppo_player import PlayerPPO
from poker.player_heuristic import PlayerHeuristic
from poker.player_random import PlayerRandom
from poker.player_deep_agent import PlayerDeepAgent
from poker.agents.ppo_agent import PPOAgent
from poker.agents.deep_learning_agent import PokerPlayerNetV1
import torch
import numpy as np
from poker.utils import init_logging, init_players
import argparse
import os
import pickle
from datetime import datetime
from tqdm import tqdm
import logging
import random
from itertools import chain

def extract_game_states_and_actions(game_state_batch):
    """Extract game states, actions, and rewards from the game state batch"""
    episodes = []  # list of tuples ((game states, raise ratios, actions), stack_delta)
    
    # Iterate over game states for all players
    for gsb in game_state_batch.values():
        
        for round_data in gsb:
            pre_stack, post_stack, stack_delta, game_states = round_data
            
            game_states_list = []
            raise_ratios_list = []
            actions_list = []
            
            for game_state, action_probs, raise_ratio, action_data in game_states:
                game_states_list.append(game_state)
                raise_ratios_list.append(raise_ratio)
                actions_list.append(action_data)  # (Action, amount)

            # Only include rounds with actions
            if game_states_list:
                episodes.append(((game_states_list, raise_ratios_list, actions_list), stack_delta))

    return episodes

def process_batch_for_ppo(episodes, ppo_agent, device):
    """Process a batch of episodes for PPO training"""
    # Extract game states and convert to tensors
    game_states = [game_state for episode in episodes for game_state in episode[0][0]]
    
    # Preprocess game states for the PPO network
    x_player_games = []
    x_acted_histories = []
    x_to_act_histories = []
    
    for game_state in game_states:
        x_player_game, x_acted_history, x_to_act_history, _ = PokerPlayerNetV1.game_state_to_batch(game_state)
        x_player_games.append(x_player_game)
        x_acted_histories.append(x_acted_history)
        x_to_act_histories.append(x_to_act_history)
    
    # Stack tensors
    x_player_game_batch = torch.stack(x_player_games).to(device)
    x_acted_history_batch = torch.stack(x_acted_histories).to(device)
    x_to_act_history_batch = torch.stack(x_to_act_histories).to(device)
    
    # Extract actions, raise ratios, and rewards
    actions = [action[0].value for episode in episodes for action in episode[0][2]]
    raise_ratios = [ratio if ratio is not None else 0.0 for episode in episodes for ratio in episode[0][1]]
    
    # Create rewards that match each action (repeat stack_delta for each action in an episode)
    rewards = list(chain(*[[episode[1]] * len(episode[0][0]) for episode in episodes]))
    
    # Convert to tensors
    actions_tensor = torch.tensor(actions, dtype=torch.int64).to(device)
    raise_ratios_tensor = torch.tensor(raise_ratios, dtype=torch.float32).to(device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
    
    # Normalize rewards
    if len(rewards_tensor) > 1:
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-7)
    
    return x_player_game_batch, x_acted_history_batch, x_to_act_history_batch, actions_tensor, raise_ratios_tensor, rewards_tensor

def train_ppo_from_batch(ppo_agent, episode_batch, device, k_epochs=4, eps_clip=0.2):
    """Train PPO from a batch of episodes"""
    # Process batch data
    x_player_game, x_acted_history, x_to_act_history, actions, raise_sizes, rewards = process_batch_for_ppo(
        episode_batch, ppo_agent, device
    )
    
    # Get old action probabilities and state values
    with torch.no_grad():
        action_logits, raise_size_dist, old_state_values = ppo_agent.policy(
            x_player_game, x_acted_history, x_to_act_history
        )
        
        # Fix NaN values in action logits
        action_logits = torch.nan_to_num(action_logits, nan=0.0)
        
        # Get action probabilities
        action_probs = torch.nn.functional.softmax(action_logits, dim=-1)
        action_probs = torch.nan_to_num(action_probs, nan=1.0/3.0)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        
        action_dist = torch.distributions.Categorical(action_probs)
        old_action_log_probs = action_dist.log_prob(actions).detach()
        
        # Get raise size log probabilities
        old_raise_log_probs = torch.nan_to_num(raise_size_dist.log_prob(raise_sizes), nan=0.0).detach()
        
        # Get state values
        old_state_values = old_state_values.squeeze(-1).detach()
    
    # Calculate advantages
    advantages = rewards - old_state_values
    
    # Optimize policy for K epochs
    total_loss = 0
    
    for _ in range(k_epochs):
        # Evaluate current policy
        logprobs_actions, logprobs_raises, state_values, dist_entropy = ppo_agent.policy.evaluate(
            x_player_game, x_acted_history, x_to_act_history, actions, raise_sizes
        )
        
        # Calculate ratios
        ratios_actions = torch.exp(logprobs_actions - old_action_log_probs)
        ratios_raises = torch.exp(logprobs_raises - old_raise_log_probs)
        
        # Combined ratio (geometric mean)
        combined_ratios = torch.sqrt(ratios_actions * ratios_raises)
        
        # Calculate surrogate losses
        surr1 = combined_ratios * advantages
        surr2 = torch.clamp(combined_ratios, 1-eps_clip, 1+eps_clip) * advantages
        
        # Final loss
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * ppo_agent.MseLoss(state_values.squeeze(-1), rewards)
        entropy_bonus = -0.01 * dist_entropy.mean()  # Encourage exploration
        
        loss = policy_loss + value_loss + entropy_bonus
        total_loss += loss.item()
        
        # Perform gradient step
        ppo_agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ppo_agent.policy.parameters(), max_norm=0.5)
        ppo_agent.optimizer.step()
    
    # Update old policy to current policy
    ppo_agent.old_policy.load_state_dict(ppo_agent.policy.state_dict())
    
    return total_loss / k_epochs

def parse_args():
    parser = argparse.ArgumentParser(description='Train a PPO agent for poker')
    
    # Training parameters
    parser.add_argument('--num_games', type=int, default=10000, help='Number of games to train for')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for PPO updates')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--k_epochs', type=int, default=4, help='Number of epochs for PPO updates')
    parser.add_argument('--replay_buffer_cap', type=int, default=10000, help='Maximum size of replay buffer')
    
    # Game parameters
    parser.add_argument('--ppo_agents', type=int, default=2, help='Number of PPO agents')
    parser.add_argument('--heuristic_agents', type=int, default=2, help='Number of heuristic agents')
    parser.add_argument('--random_agents', type=int, default=0, help='Number of random agents')
    parser.add_argument('--deep_agents', type=int, default=0, help='Number of deep agents')
    parser.add_argument('--start_stack', type=int, default=400, help='Starting chip stack')
    parser.add_argument('--big_blind', type=int, default=10, help='Big blind amount')
    parser.add_argument('--small_blind', type=int, default=5, help='Small blind amount')
    
    # Model parameters
    parser.add_argument('--warm_start', type=str, default=None, help='Path to model for warm start')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='Save checkpoints every N games')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluate every N games')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--save_state_dict', action='store_true', help='Save model as state_dict instead of full checkpoint')
    parser.add_argument('--use_replay', action='store_true', help='Use experience replay buffer for training')
    
    return parser.parse_args()

def setup_training(args):
    """
    Setup training environment
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create PPO agent
    ppo_agent = PPOAgent(
        lr=args.lr,
        gamma=args.gamma,
        eps_clip=args.eps_clip,
        K_epochs=args.k_epochs,
        device=device
    )
    
    # Warm start from pre-trained model if specified
    if args.warm_start:
        print(f"Warm starting from: {args.warm_start}")
        ppo_agent.warm_start(args.warm_start)
    
    # Add deep agent indicator if using deep agents
    deep_str = f"_DEEP_{args.deep_agents}" if args.deep_agents > 0 else ""
    replay_str = "_REPLAY" if args.use_replay else ""
    
    # Setup logging
    timestamp = datetime.now().strftime('%m%d_%H%M')
    setup_str = f"PPO_{args.ppo_agents}_HEUR_{args.heuristic_agents}_RAND_{args.random_agents}{deep_str}{replay_str}"
    log_dir = os.path.join('poker', 'policy_training', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{setup_str}_{timestamp}.log")
    init_logging(log_file_path)
    
    # Setup checkpoint directory
    checkpoint_dir = os.path.join('poker', 'policy_training', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup metrics directory
    metrics_dir = os.path.join('poker', 'policy_training', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    return ppo_agent, setup_str, checkpoint_dir, metrics_dir

def create_players(args, ppo_agent):
    """
    Create players for the game
    """
    players = []
    
    # Create PPO players
    for i in range(args.ppo_agents):
        player = PlayerPPO(f"PPO_{i}", ppo_agent, args.start_stack)
        players.append(player)
    
    # Create heuristic players
    for i in range(args.heuristic_agents):
        player = PlayerHeuristic(f"Heuristic_{i}", args.start_stack)
        players.append(player)
    
    # Create random players
    for i in range(args.random_agents):
        player = PlayerRandom(f"Random_{i}", args.start_stack)
        players.append(player)
    
    return players

def save_checkpoint(ppo_agent, metrics, setup_str, game_number, checkpoint_dir, metrics_dir, save_state_dict=False):
    """
    Save agent checkpoint and training metrics
    
    Args:
        ppo_agent: The PPO agent to save
        metrics: Training metrics to save
        setup_str: String describing the training setup
        game_number: Current game number
        checkpoint_dir: Directory to save checkpoints
        metrics_dir: Directory to save metrics
        save_state_dict: If True, save only the model state_dict (not the full checkpoint)
    """
    # Determine file extension based on save type
    file_ext = ".st" if save_state_dict else ".pt"
    
    # Save model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"{setup_str}_{game_number}{file_ext}")
    
    if save_state_dict:
        ppo_agent.save_state_dict(checkpoint_path)
        logging.info(f"Saved model state_dict to {checkpoint_path}")
    else:
        ppo_agent.save(checkpoint_path)
        logging.info(f"Saved full checkpoint to {checkpoint_path}")
    
    # Save metrics
    metrics_path = os.path.join(metrics_dir, f"{setup_str}_{game_number}_metrics.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    logging.info(f"Saved metrics to {metrics_path}")

def evaluate_agent(ppo_agent, args, num_eval_games=50):
    """
    Evaluate the agent against heuristic players
    """
    win_count = 0
    avg_stack_change = 0
    
    for _ in range(num_eval_games):
        # Create players for evaluation
        players = []
        
        # Create one PPO player for evaluation (not training)
        ppo_player = PlayerPPO("PPO_Eval", ppo_agent, args.start_stack)
        ppo_player.is_training = False  # Disable training during evaluation
        players.append(ppo_player)
        
        # Create heuristic opponents
        for i in range(args.ppo_agents + args.heuristic_agents + args.random_agents - 1):
            player = PlayerHeuristic(f"Heuristic_Eval_{i}", args.start_stack)
            players.append(player)
        
        # Randomize player order
        random.shuffle(players)
        
        # Create and run game
        game = Game(players, args.big_blind, args.small_blind, verbose=False)
        winner, _, _, _ = game.gameplay_loop()
        
        # Record results
        if type(winner) == PlayerPPO:
            win_count += 1
        
        # Calculate final stack for PPO player
        for player in players:
            if isinstance(player, PlayerPPO):
                avg_stack_change += (player.stack - args.start_stack) / args.start_stack
                break
    
    win_rate = win_count / num_eval_games
    avg_stack_change = avg_stack_change / num_eval_games
    
    return win_rate, avg_stack_change

def train_ppo_agent():
    """
    Main training loop for PPO agent
    """
    args = parse_args()
    ppo_agent, setup_str, checkpoint_dir, metrics_dir = setup_training(args)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize player types based on arguments
    player_types = {}
    if args.heuristic_agents > 0:
        player_types[PlayerHeuristic] = args.heuristic_agents
    if args.random_agents > 0:
        player_types[PlayerRandom] = args.random_agents
    if args.deep_agents > 0:
        # Create imitation model for deep agents if needed
        imitation_model = PokerPlayerNetV1(use_batchnorm=False)
        if args.warm_start:
            # Use warm_start model for deep agents too if specified
            imitation_model.load_state_dict(torch.load(args.warm_start))
        else:
            # Default to standard model
            imitation_model.load_state_dict(torch.load('poker/193c5c.05050310.st'))
        player_types[PlayerDeepAgent] = args.deep_agents
    else:
        imitation_model = None
    
    # Initialize metrics tracking
    metrics = {
        'game_numbers': [],
        'win_rates': [],
        'avg_stack_changes': [],
        'ppo_losses': [],
        'eval_win_rates': [],
        'eval_stack_changes': []
    }
    
    games_played = 0
    total_games_won = 0
    update_counter = 0
    ppo_losses = []
    
    # Initialize replay buffer if using experience replay
    replay_buffer = []
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Main training loop
    for game_number in tqdm(range(1, args.num_games + 1), desc="Training PPO agent"):
        # Create regular players
        players, _ = init_players(player_types, agent_model=imitation_model, start_stack=args.start_stack)
        
        # Add PPO players
        ppo_players = []
        for i in range(args.ppo_agents):
            ppo_player = PlayerPPO(f"PPO_{i}", ppo_agent, args.start_stack)
            players.append(ppo_player)
            ppo_players.append(ppo_player)
            
        random.shuffle(players)  # Randomize player order
        
        # Setup game
        game = Game(players, args.big_blind, args.small_blind, verbose=args.verbose)
        
        # Initialize game state tracking for PPO players
        for player in ppo_players:
            player.reset_for_new_round()
        
        # Run the game
        winner, rounds_played, eliminated, game_state_batch = game.gameplay_loop()
        games_played += 1
        
        # Process game results
        if any(isinstance(winner, PlayerPPO) for player in ppo_players):
            total_games_won += 1
        
        # Record terminal rewards for all PPO players
        for player in ppo_players:
            player.store_terminal_reward(player.stack)
        
        if args.use_replay:
            # Extract game states for experience replay
            episodes = extract_game_states_and_actions(game_state_batch)
            
            # Add to replay buffer and maintain buffer size
            replay_buffer = episodes + replay_buffer
            if len(replay_buffer) > args.replay_buffer_cap:
                replay_buffer = replay_buffer[:args.replay_buffer_cap]
            
            # Train from batch if we have enough episodes
            if len(replay_buffer) >= args.batch_size:
                training_batch = random.sample(replay_buffer, args.batch_size)
                loss = train_ppo_from_batch(
                    ppo_agent, 
                    training_batch, 
                    device, 
                    k_epochs=args.k_epochs, 
                    eps_clip=args.eps_clip
                )
                ppo_losses.append(loss)
        else:
            # Update PPO agent using standard method if we have enough experience
            update_counter += 1
            if update_counter >= args.batch_size:
                loss = ppo_agent.update()
                ppo_losses.append(loss)
                update_counter = 0
        
        # Calculate current win rate
        current_win_rate = total_games_won / games_played
        
        # Periodically evaluate the agent
        if game_number % args.eval_interval == 0:
            eval_win_rate, eval_stack_change = evaluate_agent(ppo_agent, args)
            
            # Log evaluation results
            logging.info(f"Game {game_number}: Training win rate: {current_win_rate:.4f}, " +
                         f"Eval win rate: {eval_win_rate:.4f}, " +
                         f"Avg stack change: {eval_stack_change:.4f}")
            
            # Store metrics
            metrics['game_numbers'].append(game_number)
            metrics['win_rates'].append(current_win_rate)
            metrics['ppo_losses'].append(np.mean(ppo_losses) if ppo_losses else 0)
            metrics['eval_win_rates'].append(eval_win_rate)
            metrics['eval_stack_changes'].append(eval_stack_change)
            
            # Reset metrics for next interval
            ppo_losses = []
        
        # Save checkpoint periodically
        if game_number % args.checkpoint_interval == 0:
            save_checkpoint(ppo_agent, metrics, setup_str, game_number, checkpoint_dir, metrics_dir, args.save_state_dict)
    
    # Final evaluation and checkpoint
    eval_win_rate, eval_stack_change = evaluate_agent(ppo_agent, args, num_eval_games=100)
    logging.info(f"Final evaluation - Win rate: {eval_win_rate:.4f}, Avg stack change: {eval_stack_change:.4f}")
    
    # Save final checkpoint
    save_checkpoint(ppo_agent, metrics, setup_str, args.num_games, checkpoint_dir, metrics_dir, args.save_state_dict)
    
    # Always save a state_dict version at the end for easy loading in other models
    if not args.save_state_dict:
        state_dict_path = os.path.join(checkpoint_dir, f"{setup_str}_{args.num_games}.st")
        ppo_agent.save_state_dict(state_dict_path)
        logging.info(f"Saved additional model state_dict to {state_dict_path}")
    
    return ppo_agent, metrics

if __name__ == '__main__':
    train_ppo_agent()
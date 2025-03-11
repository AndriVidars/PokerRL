from poker.core.game import Game
from poker.player_io import PlayerIO
from poker.player_random import PlayerRandom
from poker.player_heuristic import PlayerHeuristic
from poker.player_deep_agent import PlayerDeepAgent
from poker.agents.deep_learning_agent import PokerPlayerNetV1, clamp_ste
import torch
from poker.utils import init_players, init_logging
import torch.optim as optim
from tqdm import tqdm
import random
from itertools import chain
import pickle
import logging
from datetime import datetime
import argparse


def extract_game_states_and_actions(game_state_batch):
    episodes = [] # list of tuples ((game states and actions), reward)
    # iterate over game states for all deep agent players
    for gsb in game_state_batch.values():
        for round in gsb:
            _, _, stack_delta, game_states = round
            game_states_ = []
            raise_ratios_ = []
            actions_ = []
            for game_state, _, raise_ratio, action in game_states:
                game_states_.append(game_state)
                raise_ratios_.append(raise_ratio)
                actions_.append(action)

            episodes.append(((game_states_, raise_ratios_, actions_), stack_delta))

    return episodes


def training_loop(player_type_dict, agent_model_primary:PokerPlayerNetV1, agent_model_secondary=None,
                  lr=1e-4, batch_size=16, max_grad_norm=0.25, num_games=10_000, replay_buffer_cap=10_000,
                   metric_interval=1000, checkpoint_interval=10_000, stack_size=400, setup_str='', verbose=True):
    
    # NOTE batch_size is in terms of number of trajectories, not number of actions, so the emperical batch size that is passed to get_log_probs is larger(#traj # action per traj)
    optimizer = optim.Adam(agent_model_primary.parameters(), lr) # TODO tune lr, 
    replay_buffer = []
    
    
    # NOTE that each game adds multiple trajectories(multiple rounds per game, per (potentially) multiple deep agent players) to the buffer
    games_won = [0] # games won by PlayerDeepAgent players
    win_rates = []

    i = 0
    progress_bar = tqdm(total=num_games)
    while i < num_games:
        players = init_players(player_type_dict, agent_model_primary, agent_model_secondary, stack_size)
        random.shuffle(players)
        
        game = Game(players, 10, 5, verbose=False)
        try:
            winner, _, _, game_state_batch = game.gameplay_loop()
        except:
            logging.exception("Error in gameplay loop")
            # TODO if the Normal error occurs in forward, re-initialize the model from latest state dir and reset the optimizer accordingly
            raise
        
        if type(winner) == PlayerDeepAgent and winner.primary:
            games_won[-1] += 1

        game_states_episodes = extract_game_states_and_actions(game_state_batch)
        replay_buffer = game_states_episodes + replay_buffer 
        if len(replay_buffer) > replay_buffer_cap:
            replay_buffer = replay_buffer[:replay_buffer_cap] 
        
        
        if len(replay_buffer) < batch_size:
            # this should never happen, just being safe
            continue

        # prob best to still use "full" trajectories, so batch size is 
        training_batch = random.sample(replay_buffer, batch_size)

        rewards_extended = list(chain(*[[x[1]] * len(x[0][1]) for x in training_batch])) # repeat reward per action in episode
        rewards_tensor = torch.tensor(rewards_extended)
        actions_tensor =  torch.tensor([y[0].value for x in training_batch for y in x[0][2]])
        raise_sizes_tensor =  torch.tensor([y for x in training_batch for y in x[0][1]])
        game_states = [y for x in training_batch for y in x[0][0]]

        action_log_probs, raise_log_probs = agent_model_primary.get_log_probs(actions_tensor, raise_sizes_tensor, game_states)
        should_raise_mask = actions_tensor == 2
        raise_log_probs = torch.where(should_raise_mask, raise_log_probs, torch.zeros_like(raise_log_probs))

        # TODO(remove this when we do PPO, really small probabilities can only happen because of distribution shift?)
        action_log_probs = torch.clamp_min(action_log_probs, torch.log(torch.tensor(0.01)))
        raise_log_probs = torch.clamp_min(raise_log_probs, torch.log(torch.tensor(0.01)))
        loss = (-rewards_tensor * (action_log_probs + raise_log_probs)).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent_model_primary.parameters(), max_norm=max_grad_norm) # TODO tune max_norm
        optimizer.step()

        if (i+1) % metric_interval == 0 or ((i+1) <= 100 and (i+1) % 10 == 0):
            win_rate = games_won[-1] / (metric_interval if i+1 > 100 else 10)
            win_rates.append(win_rate)
            games_won.append(0)
            if verbose:
                logging.info(f"Win Rate after {i+1} games: {win_rate:.4f}")
            
        if (i+1) % checkpoint_interval == 0:
            state_dict = agent_model_primary.state_dict()
            dump_checkpoint(state_dict, f'{setup_str}_{i+1}_{int(round(win_rates[-1]*100, 2))}')
            dump_eval_stats(win_rates, f'{setup_str}_{i+1}_interval_{metric_interval}')
        
        i += 1
        progress_bar.update(1)
        
    return win_rates # maybe use this when doing hyperparam tuning

def dump_eval_stats(eval_stats, f_name):
    base_dir = 'poker/policy_training/metric_results'
    file = f'{base_dir}/{f_name}.pkl'
    with open(file, 'wb') as f:
        pickle.dump(eval_stats, f)

def dump_checkpoint(state_dict, f_name):
    base_dir = 'poker/policy_training/checkpoints'
    file = f'{base_dir}/{f_name}.st'
    torch.save(state_dict, file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument("--max_grad_norm", type=float, default=0.25)
    parser.add_argument("--num_games", type=int, default=10_000)
    parser.add_argument("--replay_buffer_cap", type=int, default=5000)
    parser.add_argument("--metric_interval", type=int, default=100)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--policy_state_dict", type=str, default='poker/193c5c.05050310.st')
    parser.add_argument("--frozen_state_dict", type=str, default='poker/193c5c.05050310.st') # if playing against some imitation agents that dont learn
    parser.add_argument("--num_H_players", type=int, default=2) # number of heuristic players in training setup
    parser.add_argument("--num_R_players", type=int, default=0)
    parser.add_argument("--num_D_players", type=int, default=2) # number of deep agent policy players
    parser.add_argument("--num_DF_players", type=int, default=0) # number of frozen pretrained deep agent players

    args = parser.parse_args()
    agent_model_primary = PokerPlayerNetV1(use_batchnorm=False)
    agent_model_primary.load_state_dict(state_dict=torch.load(args.policy_state_dict))
    
    agent_model_secondary = None
    if args.num_DF_players != 0:
        agent_model_secondary = PokerPlayerNetV1(use_batchnorm=False)
        agent_model_secondary.load_state_dict(state_dict=torch.load(args.frozen_state_dict))

    setup_str = ''
    p_types = ['H', 'R', 'D', 'DF']
    player_args = [args.num_H_players, args.num_R_players, args.num_D_players, args.num_DF_players]
    player_type_dict = {}

    for i, n in enumerate(player_args):
        if n == 0:
            continue
        if p_types[i] == 'H':
            player_type_dict[(PlayerHeuristic, False)] = n
        elif p_types[i] == 'R':
            player_type_dict[(PlayerRandom, False)] = n
        else:
            player_type_dict[(PlayerDeepAgent, p_types[i] == 'D')] = n

        setup_str += f'{p_types[i]}{n}_'
    
    
    setup_str = setup_str[:-1]
    timestamp = datetime.now().strftime('%d%H_%M')
    log_file_path = f'poker/policy_training/logs/{setup_str}_{timestamp}.log'
    init_logging(log_file_path)

    training_loop(player_type_dict, agent_model_primary, agent_model_secondary, args.lr, args.batch_size,
                  args.max_grad_norm, args.num_games, args.replay_buffer_cap, args.metric_interval,
                  args.checkpoint_interval, setup_str=setup_str)
                  
            
if __name__ == '__main__':
    main()
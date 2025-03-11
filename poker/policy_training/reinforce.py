from poker.core.game import Game
from poker.player_io import PlayerIO
from poker.player_random import PlayerRandom
from poker.player_heuristic import PlayerHeuristic
from poker.player_deep_agent import PlayerDeepAgent
from poker.agents.deep_learning_agent import PokerPlayerNetV1
import torch
from poker.utils import init_players
import torch.optim as optim
from tqdm import tqdm
import random
from itertools import chain
import pickle

# NOTE, need to tune and change this to run with different configs
player_type_dict = {
        PlayerHeuristic: 2,
        PlayerDeepAgent: 2,
        #PlayerRandom: 2,
    }

start_stack_size = 400

setup_str = '_'.join(f"{x.__name__}_{v}" for x, v in player_type_dict.items())
log_file_path = f'poker/policy_training/logs/{setup_str}.log' # TODO log to file, not just print

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


def training_loop(player_type_dict, state_dict_dir='poker/193c5c.05050310.st', num_games=10_000, replay_buffer_cap=10_000,
                   batch_size=16, metric_interval=1000, checkpoint_interval=10_000, verbose=True):
    # NOTE batch_size is in terms of number of trajectories, not number of actions, so the emperical batch size that is passed to get_log_probs is larger(#traj # action per traj)
    agent_model = PokerPlayerNetV1(use_batchnorm=False)
    agent_model.load_state_dict(state_dict=torch.load(state_dict_dir))
    optimizer = optim.Adam(agent_model.parameters(), 5e-5) # TODO tune lr, 
    replay_buffer = []
    
    
    # NOTE that each game adds multiple trajectories(multiple rounds per game, per (potentially) multiple deep agent players) to the buffer
    games_won = [0] # games won by PlayerDeepAgent players
    win_rates = []

    # TODO, add eval in increments of some number of games to track how win rate evolves?
    i = 0
    for _ in tqdm(range(num_games), disable=not verbose):
        players, _ = init_players(player_type_dict, agent_model, start_stack_size)

        game = Game(players, 10, 5, verbose=False)
        winner, _, _, game_state_batch = game.gameplay_loop()
        if type(winner) == PlayerDeepAgent:
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

        action_log_probs, raise_log_progs = agent_model.get_log_probs(actions_tensor, raise_sizes_tensor, game_states)

        loss = (-rewards_tensor * (action_log_probs + raise_log_progs)).mean() # TODO verify
        optimizer.zero_grad()
        loss.backward()
        
        #max_grad = max(p.grad.abs().max().item() for p in agent_model.parameters() if p.grad is not None)
        #print(f"Max Gradient Value: {max_grad:.6f}")

        torch.nn.utils.clip_grad_norm_(agent_model.parameters(), max_norm=0.5) # TODO tune max_norm
        optimizer.step()

        if (i+1) % metric_interval == 0 or ((i+1) <= 100 and (i+1) % 10 == 0):
            win_rate = games_won[-1] / (metric_interval if i+1 > 100 else 10)
            win_rates.append(win_rate)
            games_won.append(0)
            if verbose:
                print(f"Win Rate after {i+1} games: {win_rate:.4f}")
            
        if (i+1) % checkpoint_interval == 0:
            state_dict = agent_model.state_dict()
            dump_checkpoint(state_dict, f'{setup_str}_{i+1}')
            dump_eval_stats(win_rates, f'{setup_str}_{i+1}_interval_{metric_interval}')
        
        i += 1 # because of the if len(replay_buffer) < batch_size

def dump_eval_stats(eval_stats, f_name):
    base_dir = 'poker/policy_training/metric_results'
    file = f'{base_dir}/{f_name}.pkl'
    with open(file, 'wb') as f:
        pickle.dump(eval_stats, f)

def dump_checkpoint(state_dict, f_name):
    base_dir = 'poker/policy_training/checkpoints'
    file = f'{base_dir}/{f_name}.st'
    torch.save(state_dict, file)
    

if __name__ == '__main__':
    training_loop(player_type_dict, num_games=10_000, metric_interval=100)


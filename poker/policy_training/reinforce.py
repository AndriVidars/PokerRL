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


# NOTE, need to tune and change this to run with different configs
player_type_dict = {
        PlayerHeuristic: 2,
        PlayerDeepAgent: 2,
        #PlayerRandom: 2,
    }

start_stack_size = 400
lr = 5e-4 # TODO tune this?

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


def training_loop(player_type_dict, state_dict_dir='poker/193c5c.05050310.st', num_games=10_000, replay_buffer_cap=10_000, batch_size=16):
    # NOTE batch_size is in terms of number of trajectories, not number of actions, so the emperical batch size that is passed to get_log_probs is larger(#traj # action per traj)
    agent_model = PokerPlayerNetV1(use_batchnorm=False)
    agent_model.load_state_dict(state_dict=torch.load(state_dict_dir))
    optimizer = optim.Adam(agent_model.parameters(), lr)
    replay_buffer = []
    
    
    # NOTE that each game adds multiple trajectories(multiple rounds per game, per (potentially) multiple deep agent players) to the buffer
    games_won = 0 # games won by PlayerDeepAgent players
    cumm_rewards = [] # maybe worth counting the cummilative rewards for each game, that is rewards
    # in lead up to either win or elimination

    rounds_total_ls = [] # how many rounds does each game last, interesting to track this?
    # or track this only retio for deep players? prob not

    # TODO, add eval in increments of some number of games to track how win rate evolves?
    for i in tqdm(range(num_games)):
        players, _ = init_players(player_type_dict, agent_model, start_stack_size)


        game = Game(players, 10, 5, verbose=False)
        winner, _, _, game_state_batch = game.gameplay_loop()
        if type(winner) == PlayerDeepAgent:
            games_won += 1

        game_states_episodes = extract_game_states_and_actions(game_state_batch)
        if len(game_states_episodes) + len(replay_buffer) > replay_buffer_cap:
            replay_buffer_cap = replay_buffer_cap[:replay_buffer_cap-len(game_states_episodes)]
        
        replay_buffer = game_states_episodes + replay_buffer
        
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
        loss.backward() # TODO maybe add gradient clipping?
        optimizer.step()
        

    return games_won # TODO change return type


if __name__ == '__main__':
    games_w = training_loop(player_type_dict, num_games=10)
    print(games_w)

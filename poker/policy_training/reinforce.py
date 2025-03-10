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


# NOTE, need to tune and change this to run with different configs
player_type_dict = {
        PlayerHeuristic: 2,
        PlayerDeepAgent: 2,
        #PlayerRandom: 2,
    }

start_stack_size = 400
lr = 5e-4 # TODO tune this?


def training_loop(player_type_dict, state_dict_dir='poker/6afb9.02010310.st', num_games=10_000, exp_replay_buffer_cap = 10_000):
    agent_model = PokerPlayerNetV1(use_batchnorm=False)
    agent_model.load_state_dict(state_dict=torch.load(state_dict_dir))
    optimizer = optim.Adam(agent_model.parameters(), lr)
    
    
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

        # loop over the game state batch to collect the trajectories and the relevant data for backprop
    
    return games_won # TODO change return type


if __name__ == '__main__':
    games_w = training_loop(player_type_dict, num_games=10)
    print(games_w)

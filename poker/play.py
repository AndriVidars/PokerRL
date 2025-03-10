from poker.core.game import Game
from poker.player_io import PlayerIO
from poker.player_random import PlayerRandom
from poker.player_heuristic import PlayerHeuristic
from poker.player_deep_agent import PlayerDeepAgent
from poker.agents.deep_learning_agent import PokerPlayerNetV1
from poker.utils import init_players
import time
import pickle
from tqdm import tqdm
import random
import torch


if __name__ == '__main__':
    st = time.time()
    n_games = 100
    winner_stats = []
    eliminated_stats = []
    game_state_batches = []

    # setup of each game, number of players of each type
    player_type_dict = {
        #PlayerHeuristic: 2,
        PlayerDeepAgent: 2,
        PlayerRandom: 2,
    }


    state_dict_dir = 'poker/6afb9.02010310.st'
    agent_model = PokerPlayerNetV1(use_batchnorm=False)
    agent_model.load_state_dict(state_dict=torch.load(state_dict_dir))


    for _ in tqdm(range(n_games)):
        players, playrs_str = init_players(player_type_dict, agent_model=agent_model)
        random.shuffle(players)

        game = Game(players, 10, 5, verbose=False) # NOTE set verbose true for detailed print logging of actions and results
        winner, rounds_total, eliminated, game_state_batch = game.gameplay_loop()
        winner_stats.append((winner.__class__.__name__, rounds_total))
        for e in eliminated:
            eliminated_stats.append(
                (
                    e[0].__class__.__name__,
                    e[1],
                    e[1] / rounds_total
                )                    
            )

        game_state_batches.append(game_state_batch)
    
    et = time.time()
    elapased_time = et-st
    print(f"Total time: {elapased_time:.4f}, time per game: {elapased_time/n_games:.4f}")

    fname_stats = f'pkl/stats_{playrs_str}_{n_games}.pkl'
    fname_batches = f'pkl/game_state_batches_{playrs_str}_{n_games}.pkl'

    with open(fname_stats, 'wb') as f:
        pickle.dump((winner_stats, eliminated_stats), f)
    

    if game_state_batches:
        # HACK, can't unpickle if I use Player as key in dict for some reason
        gsb = [{f"{k.__class__.__name__}_{k.name}": v for k, v in gb.items()} for gb in game_state_batches]

        with open(fname_batches, 'wb') as f:
            pickle.dump(gsb, f)

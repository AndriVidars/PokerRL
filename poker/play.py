from poker.core.game import Game
from poker.player_io import PlayerIO
from poker.player_random import PlayerRandom
from poker.player_heuristic import PlayerHeuristic
from poker.player_deep_agent import PlayerDeepAgent
import time
import pickle
from tqdm import tqdm
import random

def init_players(player_type_dict, start_stack=400):
    players = []
    player_str_list = []
    
    for i, (player_class, count) in enumerate(player_type_dict.items()):
        player_str_list.append(f"{player_class.__name__}_{count}")
        for j in range(count):
            player_name = f"{player_class.__name__} {i+1}_{j+1}"
            players.append(player_class(player_name, start_stack))
    
    return players, "_".join(player_str_list)


if __name__ == '__main__':
    st = time.time()
    n_games = 5000
    winner_stats = []
    eliminated_stats = []

    # setup of each game, number of players of each type
    player_type_dict = {
        PlayerHeuristic: 2,
        PlayerDeepAgent: 2
    }


    for _ in tqdm(range(n_games)):
        players, playrs_str = init_players(player_type_dict)
        random.shuffle(players)

        game = Game(players, 10, 5, verbose=False) # NOTE set verbose true for detailed print logging of actions and results
        winner, rounds_total, eliminated = game.gameplay_loop()
        winner_stats.append((winner.__class__.__name__, rounds_total))
        for e in eliminated:
            eliminated_stats.append(
                (
                    e[0].__class__.__name__,
                    e[1],
                    e[1] / rounds_total
                )                    
            )
    
    et = time.time()
    elapased_time = et-st
    print(f"Total time: {elapased_time:.4f}, time per game: {elapased_time/n_games:.4f}")

    fname = f'stats_{playrs_str}_{n_games}_.pkl'

    with open(fname, 'wb') as f:
        pickle.dump((winner_stats, eliminated_stats), f)

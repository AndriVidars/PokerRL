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
    n_games = 1000
    winner_stats = []
    eliminated_stats = []
    game_state_batches = []

    # setup of each game, number of players of each type
    player_type_dict = {
        PlayerRandom: 2,
        PlayerDeepAgent: 2
    }


    for _ in tqdm(range(n_games)):
        players, playrs_str = init_players(player_type_dict)
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

    fname_stats = f'stats_{playrs_str}_{n_games}.pkl'
    fname_batches = f'game_state_batches_{playrs_str}_{n_games}.pkl'

    with open(fname_stats, 'wb') as f:
        pickle.dump((winner_stats, eliminated_stats), f)
    

    if game_state_batches:
        # HACK, can't unpickle if I use Player as key in dict for some reason
        gsb = [{f"{k.__class__.__name__}_{k.name}": v for k, v in gb.items()} for gb in game_state_batches]

        with open(fname_batches, 'wb') as f:
            pickle.dump(gsb, f)

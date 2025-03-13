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
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_games", type=int, default=1000)
    parser.add_argument("--num_H_players", type=int, default=0) # number of heuristic players in training setup
    parser.add_argument("--num_R_players", type=int, default=0)
    parser.add_argument("--deep_players", nargs="+")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    setup_str = ''
    p_types = ['H', 'R']
    player_args = [args.num_H_players, args.num_R_players]
    player_type_dict = {}

    for i, n in enumerate(player_args):
        if n == 0:
            continue
        if p_types[i] == 'H':
            player_type_dict[(PlayerHeuristic, False, 'H')] = (n, None)
        elif p_types[i] == 'R':
            player_type_dict[(PlayerRandom, False), 'R'] = (n, None)
            
        setup_str += f'{p_types[i]}{n}_'

    if args.deep_players:
        val_state_dicts = args.deep_players[0::2]
        val_num_players = list(map(int, args.deep_players[1::2]))
        
        for i, state_dict in enumerate(val_state_dicts):
            model = PokerPlayerNetV1(use_batchnorm=False)
            model.load_state_dict(state_dict)
            name_prefix = state_dict.split('/')[-1].split('.')[0]
            player_type_dict[(PlayerDeepAgent, False, name_prefix)] = (val_num_players[i], model)
            setup_str += f'{name_prefix}_{val_num_players[i]}_'

    setup_str = setup_str[:-1]

    st = time.time()
    winner_stats = []
    eliminated_stats = []
    game_state_batches = []

    for _ in tqdm(range(args.num_games)):
        players = init_players(player_type_dict) # using default stack
        random.shuffle(players)
        game = Game(players, 10, 5, verbose=args.verbose)
        winner, rounds_total, eliminated, _, game_state_batch = game.gameplay_loop()
        winner_stats.append(('_'.join(winner.name.split("_")[:-1]), rounds_total))
        for e in eliminated:
            eliminated_stats.append(
                (
                    '_'.join(e[0].name.split("_")[:-1]),
                    e[1],
                    e[1] / rounds_total
                )                    
            )

        game_state_batches.append(game_state_batch)
    
    et = time.time()
    elapased_time = et-st
    print(f"Total time: {elapased_time:.4f}, time per game: {elapased_time/args.num_games:.4f}")

    fname_stats = f'pkl/stats_{setup_str}_{args.num_games}.pkl'
    fname_batches = f'pkl/game_state_batches_{setup_str}_{args.num_games}.pkl'

    with open(fname_stats, 'wb') as f:
        pickle.dump((winner_stats, eliminated_stats), f)
    

    if game_state_batches:
        # HACK, can't unpickle if I use Player as key in dict for some reason
        gsb = [{f"{k.__class__.__name__}_{k.name}": v for k, v in gb.items()} for gb in game_state_batches]

        with open(fname_batches, 'wb') as f:
            pickle.dump(gsb, f)

    
if __name__ == '__main__':
    main()
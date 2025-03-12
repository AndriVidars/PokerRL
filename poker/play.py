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
    parser.add_argument("--num_games", type=int, default=10_000)
    parser.add_argument("--primary_state_dict", type=str, default='poker/193c5c.05050310.st') # TODO, call the imitation state dicts sometihng more descriptive
    parser.add_argument("--validation_state_dict", type=str, default='poker/a9e8c8.14060308.st') # if playing against some imitation agents that dont learn
    parser.add_argument("--num_H_players", type=int, default=2) # number of heuristic players in training setup
    parser.add_argument("--num_R_players", type=int, default=0)
    parser.add_argument("--num_D_primary_players", type=int, default=2) # deep players that we are evaluating
    parser.add_argument("--num_D_validation_players", type=int, default=0) # other deep players(using another state dict)

    args = parser.parse_args()
    agent_model_primary = PokerPlayerNetV1(use_batchnorm=False)
    agent_model_primary.load_state_dict(args.primary_state_dict)

    agent_model_validation = None
    if args.num_D_validation_players != 0:
        assert args.primary_state_dict != args.validation_state_dict
        agent_model_validation = PokerPlayerNetV1(use_batchnorm=False)
        agent_model_validation.load_state_dict(args.frozen_state_dict)
    
    setup_str = ''
    p_types = ['H', 'R', 'D', 'DV']
    player_args = [args.num_H_players, args.num_R_players, args.num_D_primary_players, args.num_D_validation_players]
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
    st = time.time()
    winner_stats = []
    eliminated_stats = []
    game_state_batches = []

    for _ in tqdm(range(args.num_games)):
        players = init_players(player_type_dict, agent_model_primary, agent_model_validation) # using default stack
        random.shuffle(players)
        game = Game(players, 10, 5, verbose=False)
        winner, rounds_total, eliminated, game_state_batch = game.gameplay_loop()
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
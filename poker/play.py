from poker.core.game import Game
from poker.player_io import PlayerIO
from poker.player_random import PlayerRandom
from poker.player_heuristic import PlayerHeuristic
from poker.player_deep_agent import PlayerDeepAgent
import time
import pickle
from tqdm import tqdm

if __name__ == '__main__':
    st = time.time()
    n_games = 1000
    winner_stats = []
    eliminated_stats = []

    for _ in tqdm(range(n_games)):
        player1 = PlayerRandom("Player 1", 400)
        player2 = PlayerDeepAgent("Player 2 Deep", 400)
        player3 = PlayerRandom("Player 3", 400)
        player4 = PlayerDeepAgent("Player 4 Deep", 400)

        game = Game([player1, player2, player3, player4], 10, 5, verbose=False) # NOTE set verbose true for detailed print logging of actions and results
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

    fname = f'stats_2_random_2_deep_{n_games}_.pkl'

    with open(fname, 'wb') as f:
        pickle.dump((winner_stats, eliminated_stats), f)

from poker.core.game import Game
from poker.player_io import PlayerIO
from poker.player_random import PlayerRandom
from poker.player_heuristic import PlayerHeuristic
import time
import pickle
from tqdm import tqdm

if __name__ == '__main__':
    st = time.time()
    n_games = 100_000
    winner_stats = []
    eliminated_stats = []

    for _ in tqdm(range(n_games)):
        player1 = PlayerRandom("Player 1", 400)
        player2 = PlayerHeuristic("Player 2H", 400)
        player3 = PlayerRandom("Player 3", 400)
        player4 = PlayerHeuristic("Player 4H", 400)

        game = Game([player1, player2, player3, player4], 10, 5, verbose=False)
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

    with open(f'stats_2_random_2_heur_{n_games}_.pkl', 'wb') as f:
        pickle.dump((winner_stats, eliminated_stats), f)

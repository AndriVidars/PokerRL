from poker.core.game import Game
from poker.player_io import PlayerIO
from poker.player_random import PlayerRandom
import time

if __name__ == '__main__':
    """
    player1 = PlayerIO("Andri", 100)
    player2 = PlayerIO("Abi", 120)
    player3 = PlayerIO("Roberto", 120)
    """
    st = time.time()
    n_games = 1000
    for _ in range(n_games):
        player1 = PlayerRandom("Player 1", 200)
        player2 = PlayerRandom("Player 2", 200)
        player3 = PlayerRandom("Player 3", 200)
        player4 = PlayerRandom("Player 4", 200)

        game = Game([player1, player2, player3, player4], 10, 5)
        game.gameplay_loop()
    
    et = time.time()
    elapased_time = et-st
    print(f"Total time: {elapased_time:.4f}, time per game: {elapased_time/n_games:.4f}")


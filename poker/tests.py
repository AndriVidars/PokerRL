from poker.core.deck import Deck
from poker.core.game import Game
from poker.player_io import PlayerIO

if __name__ == '__main__':
    player1 = PlayerIO("Andri", 100)
    player2 = PlayerIO("Abi", 120)
    player3 = PlayerIO("Roberto", 120)

    game = Game([player1, player2, player3], 10, 5)
    game.gameplay_loop()

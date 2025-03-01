import random
from Poker.core.card import Card, Suit, Rank

class Deck:
    def __init__(self):
        self.deck = [Card(rank, suit) for suit in Suit for rank in Rank]
        random.shuffle(self.deck)

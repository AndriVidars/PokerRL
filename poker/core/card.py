from enum import Enum

class Suit(Enum):
    DIAMOND = "♦"
    HEART = "♥"
    SPADE = "♠"
    CLUB = "♣"

class Rank(Enum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"

RANK_ORDER = {Rank.TWO: 2, Rank.THREE: 3, Rank.FOUR: 4, Rank.FIVE: 5, Rank.SIX: 6, Rank.SEVEN: 7,
              Rank.EIGHT: 8, Rank.NINE: 9, Rank.TEN: 10, Rank.JACK: 11, Rank.QUEEN: 12, Rank.KING: 13, Rank.ACE: 14}

class Card:
    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        return f"{self.rank.value}{self.suit.value}"
    
    def __repr__(self):
        return self.__str__()

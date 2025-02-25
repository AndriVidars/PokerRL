from collections import Counter
from poker.core.card import Card, Suit, Rank, RANK_ORDER

class PokerHand:
    def __init__(self, cards):
        self.cards = cards

    def rank(self):
        ranks = [card.rank for card in self.cards]
        suits = [card.suit for card in self.cards]
        rank_counts = Counter(ranks)

        is_flush = len(set(suits)) == 1
        sorted_ranks = sorted([RANK_ORDER[rank] for rank in ranks], reverse=True)
        is_straight = all(sorted_ranks[i] - 1 == sorted_ranks[i + 1] for i in range(4)) or sorted_ranks == [14, 5, 4, 3, 2]

        if is_straight and is_flush:
            if sorted_ranks[0] == 14 and sorted_ranks[1] == 13:
                return (10, sorted_ranks)  # Royal Flush
            return (9, sorted_ranks)  # Straight Flush

        if 4 in rank_counts.values():
            four_rank = [rank for rank, count in rank_counts.items() if count == 4][0]
            kicker = [rank for rank in sorted_ranks if rank != RANK_ORDER[four_rank]]
            return (8, [RANK_ORDER[four_rank]] + kicker)  # Four of a Kind

        if 3 in rank_counts.values() and 2 in rank_counts.values():
            three_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
            return (7, [RANK_ORDER[three_rank], RANK_ORDER[pair_rank]])  # Full House

        if is_flush:
            return (6, sorted_ranks)  # Flush

        if is_straight:
            return (5, sorted_ranks)  # Straight

        if 3 in rank_counts.values():
            three_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            kickers = [rank for rank in sorted_ranks if rank != RANK_ORDER[three_rank]]
            return (4, [RANK_ORDER[three_rank]] + kickers)  # Three of a Kind

        if list(rank_counts.values()).count(2) == 2:
            pairs = [rank for rank, count in rank_counts.items() if count == 2]
            kickers = [rank for rank in sorted_ranks if rank not in [RANK_ORDER[p] for p in pairs]]
            return (3, sorted([RANK_ORDER[p] for p in pairs], reverse=True) + kickers)  # Two Pair

        if 2 in rank_counts.values():
            pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
            kickers = [rank for rank in sorted_ranks if rank != RANK_ORDER[pair_rank]]
            return (2, [RANK_ORDER[pair_rank]] + kickers)  # One Pair

        return (1, sorted_ranks)  # High Card

    def __repr__(self):
        return f"{' '.join(str(card) for card in self.cards)}"

    def __lt__(self, other):
        return self.rank() < other.rank()
    
    def __eq__(self, other):
        return self.rank() == other.rank()

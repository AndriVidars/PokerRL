from typing import List, Tuple
from collections import Counter
from poker.core.card import RANK_ORDER, Card
import itertools

def evaluate_hand(cards: List[Card]) -> Tuple[int, List]:
    """ Evaluates a hand for anywhere from 2 to 7 cards.

    The ranks are:
    1->high card
    2->one pair
    3->two pair
    4->three of a kind
    5->straigth
    6->flush
    7->full house
    8->four of a kind
    9->straight flush
    """
    if not cards:
        return (0, [])
    
    if len(cards) > 5:
        return _find_best_hand(cards)
    
    card_values = [(RANK_ORDER[card.rank], card.suit) for card in cards]
    ranks = [rank for rank, _ in card_values]
    suits = [suit for _, suit in card_values]
    
    if is_flush(suits) and is_straight(ranks):
        return (9, [max(ranks)])
    
    rank_counts = Counter(ranks)
    sorted_ranks = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    
    # four of a kind
    if sorted_ranks[0][1] == 4:
        four_rank = sorted_ranks[0][0]
        kicker = sorted_ranks[1][0] if len(sorted_ranks) > 1 else 0
        return (8, [four_rank, kicker])
    
    # full
    if len(sorted_ranks) >= 2 and sorted_ranks[0][1] == 3 and sorted_ranks[1][1] >= 2:
        three_rank = sorted_ranks[0][0]
        two_rank = sorted_ranks[1][0]
        return (7, [three_rank, two_rank])
    
    # flush
    if is_flush(suits):
        flush_ranks = sorted(ranks, reverse=True)
        return (6, flush_ranks)
    
    # straight
    if is_straight(ranks):
        return (5, [max(ranks)])
    
    # three of a kind
    if sorted_ranks[0][1] == 3:
        three_rank = sorted_ranks[0][0]
        kickers = [r[0] for r in sorted_ranks[1:]]
        return (4, [three_rank] + sorted(kickers, reverse=True))
    
    # two pair
    if len(sorted_ranks) >= 2 and sorted_ranks[0][1] == 2 and sorted_ranks[1][1] == 2:
        high_pair = sorted_ranks[0][0]
        low_pair = sorted_ranks[1][0]
        kicker = sorted_ranks[2][0] if len(sorted_ranks) > 2 else 0
        return (3, [high_pair, low_pair, kicker])
    
    # one pair
    if sorted_ranks[0][1] == 2:
        pair_rank = sorted_ranks[0][0]
        kickers = [r[0] for r in sorted_ranks[1:]]
        return (2, [pair_rank] + sorted(kickers, reverse=True))
    
    # high card
    return (1, sorted(ranks, reverse=True))

def is_flush(suits):
    if len(suits) < 5: return False
    return len(set(suits)) == 1 if suits else False

def is_straight(ranks):
    if len(ranks) < 5:
        return False
    
    unique_ranks = sorted(set(ranks))
    for i in range(len(unique_ranks) - 4):
        if unique_ranks[i+4] - unique_ranks[i] == 4:
            return True
    if set([2, 3, 4, 5, 14]).issubset(set(ranks)):
        return True
    if len(unique_ranks) == len(ranks) and unique_ranks[-1] - unique_ranks[0] == len(ranks) - 1:
        return True
    
    return False

def _find_best_hand(cards):
    # Gets the best 5 card hand when more than 5 cards are given.
    best_rank = 0
    best_tiebreakers = []
    for five_cards in itertools.combinations(cards, min(5, len(cards))):
        rank, tiebreakers = evaluate_hand(list(five_cards))
        if rank > best_rank or (rank == best_rank and tiebreakers > best_tiebreakers):
            best_rank = rank
            best_tiebreakers = tiebreakers
    
    return (best_rank, best_tiebreakers)

def suit_counter(cards:List[Card]):
    possible_hands = itertools.combinations(cards, min(5, len(cards)))
    suit_combs = [[card.suit for card in hand] for hand in possible_hands]
    max_suit_count = max(max(Counter(hand).values()) for hand in suit_combs)
    return max_suit_count

def high_card(cards:List[Card]):
    return max(RANK_ORDER[c.rank] for c in cards)

def high_pair(cards:List[Card]):
    c = Counter([RANK_ORDER[c.rank] for c in cards])
    pairs = [k for k,v in c.items() if v == 2]
    if pairs:
        return max(pairs)
    return 0

import os
import sys
import re
from typing import List, Dict, Tuple, Optional

# Add the parent directory to sys.path so we can import Poker module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Poker.core.card import Card, Rank, Suit
from Poker.core.action import Action
from Poker.core.gamestage import Stage

def convert_card_string(card_str: str) -> Card:
    """Convert a card string (e.g., 'Tc') to a Card object."""
    rank_map = {
        '2': Rank.TWO, '3': Rank.THREE, '4': Rank.FOUR, '5': Rank.FIVE,
        '6': Rank.SIX, '7': Rank.SEVEN, '8': Rank.EIGHT, '9': Rank.NINE,
        'T': Rank.TEN, 'J': Rank.JACK, 'Q': Rank.QUEEN, 'K': Rank.KING, 'A': Rank.ACE
    }
    suit_map = {
        'c': Suit.CLUB, 'd': Suit.DIAMOND, 'h': Suit.HEART, 's': Suit.SPADE
    }
    
    rank_char = card_str[0]
    suit_char = card_str[1]
    
    return Card(rank_map[rank_char], suit_map[suit_char])

class PokerHand:
    def __init__(self, hand_id: str, players: List[str], blinds: List[int], starting_stacks: List[int]):
        self.hand_id = hand_id
        self.players = players
        self.blinds = blinds
        self.starting_stacks = starting_stacks
        self.player_cards: Dict[str, List[Card]] = {}
        self.community_cards: List[Card] = []
        self.actions: List[Tuple[str, Action, Optional[int]]] = []  # (player, action, amount)
        self.finishing_stacks: List[int] = []
        
    def add_player_cards(self, player: str, cards: List[Card]):
        self.player_cards[player] = cards
        
    def add_community_card(self, card: Card):
        self.community_cards.append(card)
        
    def add_action(self, player: str, action: Action, amount: Optional[int] = None):
        self.actions.append((player, action, amount))
        
    def set_finishing_stacks(self, stacks: List[int]):
        self.finishing_stacks = stacks
        
    def __str__(self):
        return f"Hand {self.hand_id}: {len(self.players)} players, {len(self.actions)} actions"

def parse_pluribus_hand(file_path: str) -> PokerHand:
    """Parse a single Pluribus hand file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data = {}
    for line in lines:
        line = line.strip()
        if '=' in line:
            key, value = line.split('=', 1)
            # Handle 'true' and 'false' which are JavaScript/JSON literals
            value = value.strip()
            if value == 'true':
                data[key.strip()] = True
            elif value == 'false':
                data[key.strip()] = False
            else:
                try:
                    data[key.strip()] = eval(value)
                except:
                    print(f"Error evaluating value in {file_path}: {value}")
                    continue
    
    # Extract basic info
    hand_id = data.get('hand', os.path.basename(file_path))
    players = data['players']
    blinds = data['blinds_or_straddles']
    starting_stacks = data['starting_stacks']
    
    # Create the hand object
    hand = PokerHand(str(hand_id), players, blinds, starting_stacks)
    
    # Process actions
    actions = data['actions']
    for action_str in actions:
        parts = action_str.split()
        
        # Deal hole cards
        if parts[0] == 'd' and parts[1] == 'dh':
            player_idx = int(parts[2][1:]) - 1  # p1 -> index 0
            player = players[player_idx]
            cards_str = parts[3]
            cards = [convert_card_string(cards_str[0:2]), convert_card_string(cards_str[2:4])]
            hand.add_player_cards(player, cards)
        
        # Deal board cards
        elif parts[0] == 'd' and parts[1] == 'db':
            for i in range(0, len(parts[2]), 2):
                card_str = parts[2][i:i+2]
                hand.add_community_card(convert_card_string(card_str))
        
        # Player actions
        else:
            player_idx = int(parts[0][1:]) - 1  # p1 -> index 0
            player = players[player_idx]
            
            if parts[1] == 'f':
                hand.add_action(player, Action.FOLD)
            elif parts[1] == 'cc':
                hand.add_action(player, Action.CHECK_CALL)
            elif parts[1] == 'cbr':
                amount = int(parts[2])
                hand.add_action(player, Action.RAISE, amount)
    
    # Set finishing stacks
    hand.set_finishing_stacks(data['finishing_stacks'])
    
    return hand

def parse_all_pluribus_hands(directory: str) -> List[PokerHand]:
    """Parse all Pluribus hand files in a directory."""
    hands = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.phh'):
                file_path = os.path.join(root, file)
                try:
                    hand = parse_pluribus_hand(file_path)
                    hands.append(hand)
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")
    
    # Sort hands by hand_id if possible
    try:
        hands.sort(key=lambda h: int(h.hand_id))
    except:
        hands.sort(key=lambda h: h.hand_id)
    
    return hands

if __name__ == "__main__":
    # Example usage - limit to just the 100 folder for testing
    pluribus_dir = "/Users/huram-abi/Desktop/PokerRL/pluribus/100"
    hands = parse_all_pluribus_hands(pluribus_dir)
    print(f"Parsed {len(hands)} hands")
    
    # Print sample info from first few hands
    for i, hand in enumerate(hands[:5]):
        print(f"Hand {hand.hand_id}:")
        print(f"  Players: {hand.players}")
        print(f"  Community cards: {hand.community_cards}")
        print(f"  Number of actions: {len(hand.actions)}")
        print("  Sample actions:")
        for j, (player, action, amount) in enumerate(hand.actions[:3]):
            action_str = f"{player} {action.name}"
            if amount is not None:
                action_str += f" {amount}"
            print(f"    {action_str}")
        print()
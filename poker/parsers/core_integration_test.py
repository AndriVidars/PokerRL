import os
import sys
import random
from collections import Counter
import numpy as np

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import core components
from poker.core.card import Card, Rank, Suit
from poker.core.action import Action
from poker.core.gamestage import Stage
from poker.core.deck import Deck
from poker.core.player import Player
from poker.core.game import Game

# Import parser components 
from poker.parsers.pluribus_parser import PluribusParser, HandHistory

def test_core_integration():
    """
    Test the integration between the parser and core game components
    """
    # Parse a specific file
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                         "pluribus_converted_logs")
    parser = PluribusParser(log_dir, create_core_games=True)
    
    # Parse a single file
    sample_file = os.path.join(log_dir, 'pluribus_30.txt')
    hand_histories = parser.parse_file(sample_file)
    
    print(f"Parsed {len(hand_histories)} hand histories")
    
    # Check if core game objects were created
    core_games_count = sum(1 for h in hand_histories if h.game is not None)
    print(f"Core Game objects created: {core_games_count}")
    
    if core_games_count > 0:
        # Show details for a sample game
        sample_history = random.choice([h for h in hand_histories if h.game is not None])
        game = sample_history.game
        
        print("\nSample Game Details:")
        print(f"Hand ID: {sample_history.hand_id}")
        print(f"Game players: {len(game.players)}")
        print(f"Player names: {[player.name for player in game.players]}")
        print(f"Big blind: {game.big_amount}")
        print(f"Small blind: {game.small_amount}")
        print(f"Dealer position: {game.dealer_idx}")
        print(f"Community cards: {game.community_cards}")
        
        # Check player attributes
        pluribus_player = next((p for p in game.players if p.name == "Pluribus"), None)
        if pluribus_player:
            print("\nPluribus Player:")
            print(f"Stack: {pluribus_player.stack}")
            print(f"Hand: {pluribus_player.hand}")
    
    # Test creating a new game from scratch
    print("\nCreating a new Game from scratch:")
    players = [
        Player("Player1", 1000),
        Player("Player2", 1000),
        Player("Player3", 1000),
    ]
    
    new_game = Game(players, 100, 50)  # big blind 100, small blind 50
    print(f"New game created with {len(new_game.players)} players")
    
    # Create a deck and deal cards
    deck = Deck()
    for player in new_game.players:
        player.hand = [deck.deck.pop() for _ in range(2)]
    
    print("Dealt cards to players:")
    for player in new_game.players:
        print(f"{player.name}: {player.hand}")
    
    # Deal community cards
    new_game.community_cards = [deck.deck.pop() for _ in range(5)]
    print(f"Community cards: {new_game.community_cards}")

if __name__ == "__main__":
    test_core_integration()
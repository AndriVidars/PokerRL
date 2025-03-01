import os
import sys
from typing import List, Dict, Tuple, Optional, Any

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from poker.parsers.game_state_retriever import GameStateRetriever
from poker.core.gamestage import Stage
from poker.core.action import Action

def main():
    # Initialize the game state retriever with the logs directory
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pluribus_converted_logs")
    retriever = GameStateRetriever(log_dir)
    
    # Initialize with no verbose output (faster)
    print("Initializing retriever and parsing logs...")
    retriever.initialize(verbose=True)
    
    # Get the number of hands parsed
    hand_count = retriever.get_hand_count()
    print(f"Parsed {hand_count} hands from the logs")
    
    # Specify the player to analyze
    player_name = "Pluribus"  # You can change this to any player in the dataset
    
    # Get all game states for this player
    print(f"\nGetting all game states for {player_name}...")
    game_states = retriever.get_player_game_states(player_name)
    print(f"Found {len(game_states)} game states for {player_name}")
    

if __name__ == "__main__":
    main()
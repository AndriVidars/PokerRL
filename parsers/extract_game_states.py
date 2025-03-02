import os
import sys
from typing import List, Dict, Tuple, Optional

# Add the parent directory to sys.path so we can import Poker module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Poker.core.card import Card
from Poker.core.action import Action
from Poker.core.gamestage import Stage
from Poker.agents.game_state import Player, GameState
from parse_pluribus_hands import parse_all_pluribus_hands, PokerHand

class GameStateExtractor:
    def __init__(self, hand: PokerHand):
        self.hand = hand
        self.player_positions = self._determine_player_positions()
        self.game_states = []
    
    def _determine_player_positions(self) -> Dict[str, int]:
        """
        Determine player positions relative to the big blind.
        Returns a dictionary mapping player names to positions (0=BB, 1=UTG, etc.)
        
        In blinds_or_straddles = [50, 100, 0, 0, 0, 0],
        the first position (index 0) is small blind (SB),
        the second position (index 1) is big blind (BB),
        and remaining positions follow clockwise.
        """
        positions = {}
        
        # The blinds array directly corresponds to player positions in order
        # The SB is at index 0, BB at index 1, etc.
        # So we know that player at index 1 is the BB (position 0)
        bb_position = 1  # Big blind is always at position 1 in the blinds array
        
        for i, player in enumerate(self.hand.players):
            # Position relative to BB (how many spots to the left of BB)
            # In poker terminology, BB is position 0, then UTG is 1, etc.
            rel_pos = (i - bb_position) % len(self.hand.players)
            positions[player] = rel_pos
        
        return positions
    
    def extract_player_objects(self) -> Dict[str, Player]:
        """
        Create Player objects for all players in the hand.
        """
        players = {}
        
        for player_name in self.hand.players:
            cards = self.hand.player_cards.get(player_name)
            player_idx = self.hand.players.index(player_name)
            stack = self.hand.starting_stacks[player_idx]
            position = self.player_positions[player_name]
            
            # Create a Player object
            player_obj = Player(
                spots_left_bb=position,
                cards=tuple(cards) if cards else None,
                stack_size=stack
            )
            
            players[player_name] = player_obj
        
        return players
        
    def update_player_stacks_for_stage(self, player_objects: Dict[str, Player], stage: Stage):
        """
        Update player stack sizes based on game stage.
        If we're at the RIVER stage, use finishing_stacks instead of starting_stacks.
        """
        if stage == Stage.RIVER:
            for player_name, player_obj in player_objects.items():
                player_idx = self.hand.players.index(player_name)
                player_obj.stack_size = self.hand.finishing_stacks[player_idx]
    
    def determine_game_stage(self, action_index: int) -> Stage:
        """
        Determine the game stage at a given action index.
        """
        # Count community cards
        community_cards = self.hand.community_cards
        
        # Simplified implementation based on the number of community cards available
        if len(community_cards) == 0:
            return Stage.PREFLOP
        elif len(community_cards) >= 3 and len(community_cards) < 4:
            return Stage.FLOP
        elif len(community_cards) >= 4 and len(community_cards) < 5:
            return Stage.TURN
        elif len(community_cards) >= 5:
            return Stage.RIVER
        
        # Should not reach here
        return Stage.PREFLOP
    
    def calculate_pot_size(self, action_index: int) -> int:
        """
        Calculate the pot size at a given action index.
        """
        pot_size = sum(self.hand.blinds)  # Start with blinds
        
        for i in range(action_index):
            _, action, amount = self.hand.actions[i]
            if action == Action.RAISE and amount is not None:
                pot_size += amount
            elif action == Action.CHECK_CALL:
                # For simplicity, assume call adds the minimum bet amount
                # This is a simplification - in reality would need to track betting rounds
                pot_size += self.hand.blinds[1]  # Add big blind amount
        
        return pot_size
    
    def extract_community_cards(self, action_index: int) -> List[Card]:
        """
        Extract community cards at a given action index.
        """
        # Simplified implementation based on current game stage
        current_stage = self.determine_game_stage(action_index)
        
        if current_stage == Stage.PREFLOP:
            return []
        elif current_stage == Stage.FLOP and len(self.hand.community_cards) >= 3:
            return self.hand.community_cards[:3]
        elif current_stage == Stage.TURN and len(self.hand.community_cards) >= 4:
            return self.hand.community_cards[:4]
        elif current_stage == Stage.RIVER and len(self.hand.community_cards) >= 5:
            return self.hand.community_cards[:5]
        
        return []
    
    def calculate_min_bet(self, action_index: int) -> int:
        """
        Calculate minimum bet to continue at a given action index.
        """
        # For simplicity, use the min_bet value from the hand data
        # In real implementation, would track betting for each street
        return self.hand.blinds[1]  # Big blind amount
    
    def extract_all_game_states(self) -> List[Tuple[str, GameState]]:
        """
        Extract game states for all decision points in the hand.
        Returns a list of (player_name, GameState) tuples.
        """
        all_states = []
        player_objects = self.extract_player_objects()
        
        # Track player action histories by stage
        player_action_stages = {player: Stage.PREFLOP for player in player_objects}
        
        # For each action, create a game state for the player about to act
        for i, (player, action, amount) in enumerate(self.hand.actions):
            # Skip dealer actions
            if player == 'd':
                continue
            
            # Get the current stage
            stage = self.determine_game_stage(i)
            
            # Update player stacks if we're at river stage
            self.update_player_stacks_for_stage(player_objects, stage)
            
            # Get player objects
            my_player = player_objects[player]
            other_players = [p for name, p in player_objects.items() if name != player]
            
            # Get community cards
            community_cards = self.extract_community_cards(i)
            
            # Calculate pot size and min bet
            pot_size = self.calculate_pot_size(i)
            min_bet = self.calculate_min_bet(i)
            
            # Create the game state
            game_state = GameState(
                stage=stage,
                community_cards=community_cards,
                pot_size=pot_size,
                min_bet_to_continue=min_bet,
                my_player=my_player,
                other_players=other_players,
                my_player_action=(action, amount),
                apply_visibility_rules=False  # Disable visibility rules for simplicity
            )
            
            # Add the action to player's history based on tracked stage
            current_player_stage = player_action_stages[player]
            
            # Only add action if the player hasn't acted in this stage yet
            # We track which stage the player has acted in
            if current_player_stage == Stage.PREFLOP and len(my_player.history) == 0:
                my_player.add_preflop_action(action, amount)
                player_action_stages[player] = Stage.FLOP
            elif current_player_stage == Stage.FLOP and len(my_player.history) == 1:
                my_player.add_flop_action(action, amount)
                player_action_stages[player] = Stage.TURN
            elif current_player_stage == Stage.TURN and len(my_player.history) == 2:
                my_player.add_turn_action(action, amount)
                player_action_stages[player] = Stage.RIVER
            elif current_player_stage == Stage.RIVER and len(my_player.history) == 3:
                # Skip if we've already added a river action
                # This can happen if players act multiple times in a stage
                pass
            else:
                # Player has already acted in this stage, we'll skip adding this action
                # This is a simplification - in real implementation you'd track multiple actions per stage
                pass
            
            all_states.append((player, game_state))
        
        return all_states

def extract_all_hands_game_states(hands: List[PokerHand]) -> Dict[str, List[Tuple[str, GameState]]]:
    """
    Extract game states for all hands.
    Returns a dictionary mapping hand IDs to lists of (player, GameState) tuples.
    """
    all_game_states = {}
    
    for hand in hands:
        extractor = GameStateExtractor(hand)
        hand_states = extractor.extract_all_game_states()
        all_game_states[hand.hand_id] = hand_states
    
    return all_game_states

def extract_player_game_states(hands: List[PokerHand], player_name: str) -> List[GameState]:
    """
    Extract all game states for a specific player across all hands.
    
    Args:
        hands: List of PokerHand objects
        player_name: Name of the player to extract game states for
    
    Returns:
        List of GameState objects where the specified player was making a decision
    """
    player_states = []
    
    for hand in hands:
        # Skip hands where the player isn't present
        if player_name not in hand.players:
            continue
            
        extractor = GameStateExtractor(hand)
        hand_states = extractor.extract_all_game_states()
        
        # Filter for states where this player was acting
        for acting_player, state in hand_states:
            if acting_player == player_name:
                # Add the hand ID as metadata to the state for reference
                state.hand_id = hand.hand_id
                player_states.append(state)
    
    return player_states

if __name__ == "__main__":
    # Example usage - limit to just the 100 folder for testing
    pluribus_dir = "/Users/huram-abi/Desktop/PokerRL/pluribus/100"
    hands = parse_all_pluribus_hands(pluribus_dir)
    print(f"Parsed {len(hands)} hands")
    
    # First, show general game states from first few hands
    sample_hands = hands[:5]
    game_states_dict = extract_all_hands_game_states(sample_hands)
    
    print("\n===== Sample of All Game States =====")
    for hand_id, states in game_states_dict.items():
        print(f"Hand {hand_id}: {len(states)} game states")
        for i, (player, state) in enumerate(states[:3]):  # Print first 3 states
            print(f"  State {i+1}: Player {player}, Stage: {state.stage.name}")
            print(f"    Community cards: {state.community_cards}")
            print(f"    Pot size: {state.pot_size}")
            print(f"    Action taken: {state.my_player_action}")
            print()
    
    # Now extract and show Pluribus game states
    pluribus_states = extract_player_game_states(hands, "Pluribus")
    print(f"\n===== Pluribus Game States =====")
    print(f"Found {len(pluribus_states)} game states for Pluribus")
    
    # Show sample of Pluribus states
    for i, state in enumerate(pluribus_states[:10]):  # Show first 10
        print(f"  Pluribus State {i+1} (Hand {state.hand_id}):")
        print(f"    Stage: {state.stage.name}")
        print(f"    Community cards: {state.community_cards}")
        print(f"    My cards: {state.my_player.cards}")
        print(f"    Pot size: {state.pot_size}")
        print(f"    Action taken: {state.my_player_action}")
        
        # Show other players' visible actions at this point
        print(f"    Other players ({len(state.other_players)}):")
        for j, other_player in enumerate(state.other_players[:2]):  # Show first 2 other players
            print(f"      Player {j+1}: Position: {other_player.spots_left_bb}, History: {other_player.history}")
        print()
import os
import sys
import torch
import random
from typing import List, Dict, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Poker.core.game import Game
from Poker.core.player import Player
from Poker.core.action import Action
from Poker.core.card import Card, Rank, Suit
from Poker.core.gamestage import Stage
from Poker.core.deck import Deck
from Poker.player_random import PlayerRandom
from Poker.agents.imitation_agent import ImitationAgent, DeepLearningPlayer


def colored_card(card: Card) -> str:
    """Return a colored string representation of a card"""
    suits = {
        Suit.HEART: '♥',
        Suit.DIAMOND: '♦',
        Suit.CLUB: '♣',
        Suit.SPADE: '♠'
    }
    
    colors = {
        Suit.HEART: '\033[91m',  # Red
        Suit.DIAMOND: '\033[91m',  # Red
        Suit.CLUB: '\033[0m',  # Default
        Suit.SPADE: '\033[0m'  # Default
    }
    
    end_color = '\033[0m'
    
    rank_str = str(card.rank).split('.')[-1]
    if rank_str == 'TEN':
        rank_str = 'T'
    elif rank_str == 'JACK':
        rank_str = 'J'
    elif rank_str == 'QUEEN':
        rank_str = 'Q'
    elif rank_str == 'KING':
        rank_str = 'K'
    elif rank_str == 'ACE':
        rank_str = 'A'
    else:
        rank_str = rank_str[0]
    
    return f"{colors[card.suit]}{rank_str}{suits[card.suit]}{end_color}"


def display_cards(cards: List[Card]) -> str:
    """Display a list of cards"""
    return " ".join(colored_card(card) for card in cards)


def display_game_state(game: Game) -> None:
    """Display the current game state"""
    print("\n" + "="*60)
    print(f"Stage: {game.current_stage.name}")
    print(f"Pot: ${sum(pot.total_amount for pot in game.pots)}")
    print(f"Community cards: {display_cards(game.community_cards)}")
    print("-"*60)
    
    # Print player information
    for player in game.players:
        status = ""
        if player.folded:
            status = "(folded)"
        elif player.all_in:
            status = "(all-in)"
            
        print(f"{player.name}: ${player.stack} {status}")
        print(f"  Hand: {display_cards(player.hand)}")
    
    print("="*60)


def determine_winners(game: Game) -> List[Tuple[Player, int]]:
    """
    Determine the winners of the hand
    Returns a list of (player, pot_idx) tuples
    """
    from Poker.core.hand_evaluator import evaluate_hand
    
    # Skip if there's only one active player
    active_players = [p for p in game.players if not p.folded]
    if len(active_players) <= 1:
        return [(active_players[0], i) for i in range(len(game.pots))] if active_players else []
    
    # Evaluate all player hands
    player_hand_rankings = {}
    for player in active_players:
        best_rank, best_tiebreakers = evaluate_hand(game.community_cards + player.hand)
        player_hand_rankings[player] = (best_rank, best_tiebreakers)
    
    winners = []
    
    # Determine winners for each pot
    for pot_idx, pot in enumerate(game.pots):
        # Skip empty pots
        if not pot.eligible_players:
            continue
            
        # Filter eligible players from the pot
        pot_players = [p for p in active_players if p in pot.eligible_players]
        
        # Skip if no eligible players
        if not pot_players:
            continue
            
        # Sort players by hand strength
        pot_players = sorted(pot_players, key=lambda p: player_hand_rankings[p], reverse=True)
        best_hand = player_hand_rankings[pot_players[0]]
        
        # Find players with the same hand strength
        tied_players = [pot_players[0]]
        for i in range(1, len(pot_players)):
            if player_hand_rankings[pot_players[i]] == best_hand:
                tied_players.append(pot_players[i])
            else:
                break
        
        # Add winners to the list
        for player in tied_players:
            winners.append((player, pot_idx))
    
    return winners


def award_pot(game: Game, winners: List[Tuple[Player, int]]) -> None:
    """
    Award the pot to the winners
    winners: list of (player, pot_idx) tuples
    """
    # Group winners by pot
    pot_winners = {}
    for player, pot_idx in winners:
        if pot_idx not in pot_winners:
            pot_winners[pot_idx] = []
        pot_winners[pot_idx].append(player)
    
    # Award money from each pot
    for pot_idx, players in pot_winners.items():
        if pot_idx >= len(game.pots):
            continue
            
        pot = game.pots[pot_idx]
        amount_each = pot.total_amount // len(players)
        remainder = pot.total_amount % len(players)
        
        print(f"\nPot #{pot_idx+1} (${pot.total_amount}) winners:")
        for i, player in enumerate(players):
            # Last player gets the remainder
            extra = remainder if i == len(players) - 1 and remainder > 0 else 0
            player.stack += amount_each + extra
            
            # Show hand strength if we have multiple winners
            if len(players) > 1 and hasattr(player, 'hand') and player.hand:
                from Poker.core.hand_evaluator import evaluate_hand
                hand_rank, _ = evaluate_hand(game.community_cards + player.hand)
                hand_names = {
                    9: "Straight Flush",
                    8: "Four of a Kind", 
                    7: "Full House",
                    6: "Flush",
                    5: "Straight",
                    4: "Three of a Kind",
                    3: "Two Pair",
                    2: "One Pair", 
                    1: "High Card"
                }
                hand_name = hand_names.get(hand_rank, "Unknown")
                print(f"  {player.name} wins ${amount_each + extra} with {hand_name}: {display_cards(player.hand)}")
            else:
                print(f"  {player.name} wins ${amount_each + extra}")
        
    # Clear the pots
    game.pots = []


def log_action(player: Player, action: Action, amount: Optional[int] = None) -> None:
    """Log a player action"""
    action_str = action.name
    if action == Action.RAISE and amount is not None:
        action_str += f" {amount}"
    print(f"{player.name} -> {action_str}")


class GameLogger:
    """Class to log game actions"""
    
    def __init__(self, game: Game):
        self.game = game
        self.original_act_methods = {}
        self.wrap_act_methods()
        
    def wrap_act_methods(self):
        """Wrap act methods to log actions"""
        for player in self.game.players:
            self.original_act_methods[player] = player.act
            player.act = self.create_wrapper(player, player.act)
    
    def create_wrapper(self, player, original_method):
        """Create a wrapper for the act method"""
        def wrapped_act(*args, **kwargs):
            # Print current state
            if self.game.current_stage == Stage.PREFLOP:
                display_game_state(self.game)
            
            # Call original method
            action = original_method(*args, **kwargs)
            
            # Log action
            if action == Action.RAISE:
                # Find the raise amount from the contributions
                amount = None
                if isinstance(player, DeepLearningPlayer) and hasattr(player, 'game_states') and player.game_states:
                    # Try to extract the last raise amount from agent's action history 
                    for pot in self.game.pots:
                        if player in pot.contributions:
                            amount = pot.contributions[player]
                
                # If we couldn't get amount from pots, use a default
                if amount is None:
                    amount = 20
                    
                log_action(player, action, amount)
            else:
                log_action(player, action)
            
            # If it's the last action in a stage, show the updated state
            active_players = [p for p in self.game.players if not p.folded]
            if len(active_players) == 1 or all(p.all_in for p in active_players):
                print("\nStage completed - showing cards and pot")
                display_game_state(self.game)
            
            return action
        return wrapped_act
    
    def restore_act_methods(self):
        """Restore original act methods"""
        for player, method in self.original_act_methods.items():
            player.act = method


def play_test_game(num_hands: int = 1):
    """Play test games between AI and random players"""
    # Load the AI agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = ImitationAgent(device=device)
    
    try:
        agent.load('./models')
        print(f"Loaded AI model from ./models")
    except FileNotFoundError:
        print(f"Could not find model in ./models - using a random player instead")
        return
    
    # Create players
    ai_player = DeepLearningPlayer("AI (Pluribus)", 1000, agent)
    random_player1 = PlayerRandom("RandomPlayer1", 1000)
    random_player2 = PlayerRandom("RandomPlayer2", 1000)
    
    all_players = [ai_player, random_player1, random_player2]
    
    # Our custom game loop to ensure cards are properly reset between hands
    hand_num = 0
    
    # Increase big blind over time to speed up elimination
    big_blind = 20
    small_blind = 10
    
    while len(all_players) > 1 and hand_num < num_hands:
        # Increase blinds every 5 hands
        if hand_num > 0 and hand_num % 5 == 0:
            big_blind *= 2
            small_blind = big_blind // 2
            print(f"\n*** BLINDS INCREASED TO ${small_blind}/${big_blind} ***\n")
        hand_num += 1
        print(f"\n{'#'*80}")
        print(f"Hand {hand_num}/{num_hands}")
        print(f"{'#'*80}")
        
        # Create new game with fresh players
        game = Game(all_players.copy(), big_blind, small_blind)
        
        # Ensure all players have empty hands before starting
        for player in game.players:
            player.hand = []
            player.folded = False
            player.all_in = False
        
        # Set up logging
        logger = GameLogger(game)
        
        try:
            # Generate a new deck
            game.deck = Deck()
            
            # Run preflop
            print("\n" + "-"*25 + "\nPreflop\n" + "-"*25 + "\n")
            game.active_players = set(game.players)
            game.handle_blinds()
            
            # Deal cards - only 2 per player
            for player in game.players:
                player.hand = []  # Clear any existing cards
                for _ in range(2):  # Deal exactly 2 cards
                    player.hand.append(game.deck.deck.pop())
            
            # Run betting rounds
            game.betting_loop()
            
            # Always proceed through all stages, even if there are folds
            # First, handle the preflop case where everyone folded except one player
            if len(game.active_players) <= 1 or all(p.all_in for p in game.active_players if not p.folded):
                # Skip betting but show cards
                # Flop
                print("\n" + "-"*25 + "\nFlop\n" + "-"*25 + "\n")
                game.current_stage = Stage.FLOP
                game.community_cards = []
                
                # Burn a card and deal 3 community cards
                game.deck.deck.pop()  # Burn
                for _ in range(3):
                    game.community_cards.append(game.deck.deck.pop())
                
                print("\nCommunity Cards After Flop:")
                print(display_cards(game.community_cards))
                
                # Turn
                print("\n" + "-"*25 + "\nTurn\n" + "-"*25 + "\n")
                game.current_stage = Stage.TURN
                
                # Burn a card and deal 1 community card
                game.deck.deck.pop()  # Burn
                game.community_cards.append(game.deck.deck.pop())
                
                print("\nCommunity Cards After Turn:")
                print(display_cards(game.community_cards))
                
                # River
                print("\n" + "-"*25 + "\nRiver\n" + "-"*25 + "\n")
                game.current_stage = Stage.RIVER
                
                # Burn a card and deal 1 community card
                game.deck.deck.pop()  # Burn
                game.community_cards.append(game.deck.deck.pop())
                
                print("\nCommunity Cards After River:")
                print(display_cards(game.community_cards))
                
                # Determine winner and award pot
                winners = determine_winners(game)
                award_pot(game, winners)
            else:
                # Normal gameplay with betting rounds
                # Flop
                print("\n" + "-"*25 + "\nFlop\n" + "-"*25 + "\n")
                game.current_stage = Stage.FLOP
                game.community_cards = []
                
                # Burn a card and deal 3 community cards
                game.deck.deck.pop()  # Burn
                for _ in range(3):
                    game.community_cards.append(game.deck.deck.pop())
                
                print("\nCommunity Cards After Flop:")
                print(display_cards(game.community_cards))
                game.betting_loop()
                
                # Check for all-in or 1 player remaining
                if len(game.active_players) <= 1 or all(p.all_in for p in game.active_players if not p.folded):
                    # Turn without betting
                    print("\n" + "-"*25 + "\nTurn\n" + "-"*25 + "\n")
                    game.current_stage = Stage.TURN
                    
                    # Burn a card and deal 1 community card
                    game.deck.deck.pop()  # Burn
                    game.community_cards.append(game.deck.deck.pop())
                    
                    print("\nCommunity Cards After Turn:")
                    print(display_cards(game.community_cards))
                    
                    # River without betting
                    print("\n" + "-"*25 + "\nRiver\n" + "-"*25 + "\n")
                    game.current_stage = Stage.RIVER
                    
                    # Burn a card and deal 1 community card
                    game.deck.deck.pop()  # Burn
                    game.community_cards.append(game.deck.deck.pop())
                    
                    print("\nCommunity Cards After River:")
                    print(display_cards(game.community_cards))
                    
                    # Determine winner
                    game.decide_pot()
                else:
                    # Continue with Turn
                    print("\n" + "-"*25 + "\nTurn\n" + "-"*25 + "\n")
                    game.current_stage = Stage.TURN
                    
                    # Burn a card and deal 1 community card
                    game.deck.deck.pop()  # Burn
                    game.community_cards.append(game.deck.deck.pop())
                    
                    print("\nCommunity Cards After Turn:")
                    print(display_cards(game.community_cards))
                    game.betting_loop()
                    
                    # Check for all-in or 1 player remaining
                    if len(game.active_players) <= 1 or all(p.all_in for p in game.active_players if not p.folded):
                        # River without betting
                        print("\n" + "-"*25 + "\nRiver\n" + "-"*25 + "\n")
                        game.current_stage = Stage.RIVER
                        
                        # Burn a card and deal 1 community card
                        game.deck.deck.pop()  # Burn
                        game.community_cards.append(game.deck.deck.pop())
                        
                        print("\nCommunity Cards After River:")
                        print(display_cards(game.community_cards))
                        
                        # Determine winner and award pot
                        winners = determine_winners(game)
                        award_pot(game, winners)
                    else:
                        # Continue with River
                        print("\n" + "-"*25 + "\nRiver\n" + "-"*25 + "\n")
                        game.current_stage = Stage.RIVER
                        
                        # Burn a card and deal 1 community card
                        game.deck.deck.pop()  # Burn
                        game.community_cards.append(game.deck.deck.pop())
                        
                        print("\nCommunity Cards After River:")
                        print(display_cards(game.community_cards))
                        game.betting_loop()
                        
                        # Determine winner and award pot
                        winners = determine_winners(game)
                        award_pot(game, winners)
            
            # Print end of hand summary
            print("\nHand completed!")
            print("Final community cards:", display_cards(game.community_cards))
            print("Player stacks:")
            players_to_remove = []
            for player in all_players:
                print(f"Player: {player.name} stack after hand {hand_num}: ${player.stack}")
                
                # Check if player is eliminated
                if player.stack <= 0:
                    print(f"Player {player.name} has been eliminated!")
                    players_to_remove.append(player)
            
            # Remove eliminated players after the loop
            for player in players_to_remove:
                if player in all_players:
                    all_players.remove(player)
            
            # Move dealer button
            game.move_blinds()
            
        except Exception as e:
            print(f"Error during gameplay: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore original act methods
            logger.restore_act_methods()


if __name__ == "__main__":
    # Play until only one player remains
    play_test_game(num_hands=20)
import torch
from torch import nn
import os
import sys
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poker.agents.game_state import GameState, Stage
from poker.core.player import Player
from poker.core.action import Action
from typing import Tuple, List, Optional

class FFN(nn.Module):
    def __init__(self, idim, hdim, odim, n_hidden):
        super().__init__()
        
        layers = []
        # First layer with BatchNorm
        layers.append(nn.BatchNorm1d(idim))
        layers.append(nn.Linear(idim, hdim))
        
        # Hidden layers
        for _ in range(n_hidden):
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hdim, hdim))
        
        # Final output layer
        layers.append(nn.BatchNorm1d(hdim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hdim, odim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        assert len(x.shape) == 2
        return self.net(x)

class PokerPlayerNetV1(nn.Module):
    def __init__(self):
        super().__init__()

        self.player_game_net = FFN(
            idim=5, # stack_size, turn_to_act, hand_rank, c_hand_rank, min_bet
            hdim=10,
            odim=10,
            n_hidden=2,
        )
        # consider only players that are in the game
        self.acted_player_history_net = FFN(
            idim=3, # stack_size, turn_to_act, bet
            hdim=20,
            odim=20,
            n_hidden=3,
        )
        self.to_act_player_history_net = FFN(
            idim=3, # stack_size, turn_to_act, bet
            hdim=20,
            odim=20,
            n_hidden=3,
        )

        self.gather_net = FFN(
            idim=30, # stack_size, turn_to_act, bet
            hdim=30,
            odim=2, # (fold / no_fold), raise size
            n_hidden=3,
        )

    def forward(self, x_player_game, x_acted_history, x_to_act_history):
        # x_player_game : (B, 5)
        # x_acted_history : (B, 9, 3)
        # x_to_act_history : (B, 9, 3)
        player_game_state = self.player_game_net(x_player_game)
        
        # Process each player in the batch for acted history
        batch_size = x_acted_history.shape[0]
        acted_outputs = []
        for i in range(batch_size):
            player_hist = x_acted_history[i]
            if player_hist.sum() != 0:  # Only process non-zero padded data
                acted_output = self.acted_player_history_net(player_hist)
                acted_outputs.append(acted_output.sum(dim=0))
            else:
                acted_outputs.append(torch.zeros(20))  # Match the output dimension
        acted_player_history_state = torch.stack(acted_outputs, dim=0)
        
        # Process each player in the batch for to-act history
        to_act_outputs = []
        for i in range(batch_size):
            player_hist = x_to_act_history[i]
            if player_hist.sum() != 0:  # Only process non-zero padded data
                to_act_output = self.to_act_player_history_net(player_hist)
                to_act_outputs.append(to_act_output.sum(dim=0))
            else:
                to_act_outputs.append(torch.zeros(20))  # Match the output dimension
        to_act_player_history_state = torch.stack(to_act_outputs, dim=0)

        all_game_state = torch.cat([
            player_game_state,
            acted_player_history_state,
            to_act_player_history_state,
        ], dim=-1)

        out = self.gather_net(all_game_state)
        fold_logits = out[:, 0]
        raise_size = torch.sigmoid(out[:, 1])

        # if raise size <= 10%, just call
        raise_size = raise_size.clamp_min(0.1) * (raise_size >= 0.1).float()

        return fold_logits, raise_size
    
    @staticmethod
    def game_state_to_batch(state: GameState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        player_turn, other_player_turns = state.get_effective_turns()
        
        # Ensure pot_size is not zero to avoid division by zero
        pot_size = max(1, state.pot_size)
        
        # Construct player / game state
        rel_stack_size = state.my_player.stack_size / pot_size
        turn_to_act = player_turn if player_turn is not None else 0
        hand_rank = state.hand_strength
        c_hand_rank = state.community_hand_strength  # Fixed typo
        rel_min_bet = state.min_bet_to_continue / pot_size
        
        # Create tensor and add batch dimension
        x_player_game = torch.tensor([rel_stack_size, turn_to_act, hand_rank, c_hand_rank, rel_min_bet], 
                                   dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # separate acted and non-acted players
        acted_players = [(player, turn) for player, turn in
                    zip(state.other_players, other_player_turns)
                    if turn is not None and turn < player_turn]
        to_act_players = [(player, turn) for player, turn in
                    zip(state.other_players, other_player_turns)
                    if turn is not None and turn > player_turn]

        # Construct acted players state
        # note that order does not matter right now 
        acted_players_data = []
        for p, t in acted_players:
            # Handle case where history might be empty
            if p.history:
                raise_size = p.history[-1][1] if p.history[-1][1] is not None else 0
                raise_size = raise_size / pot_size
            else:
                raise_size = 0
                
            acted_players_data.append([
                p.stack_size / pot_size,  # rel stack size
                t,  # turn to act
                raise_size,  # normalized raise size
            ])
        
        # Create tensor of correct shape with padding
        if acted_players_data:
            x_acted_players = torch.tensor(acted_players_data, dtype=torch.float32)
            # Add padding to reach size 9
            if x_acted_players.shape[0] < 9:
                padding_size = 9 - x_acted_players.shape[0]
                padding = torch.zeros((padding_size, 3), dtype=torch.float32)
                x_acted_players = torch.cat([x_acted_players, padding], dim=0)
        else:
            x_acted_players = torch.zeros((9, 3), dtype=torch.float32)
        
        # Add batch dimension
        x_acted_players = x_acted_players.unsqueeze(0)

        # Construct to-act players state, only if not in preflop
        to_act_players_data = []
        if state.stage != Stage.PREFLOP:
            for p, t in to_act_players:
                # Handle case where history might be empty
                if p.history:
                    raise_size = p.history[-1][1] if p.history[-1][1] is not None else 0
                    raise_size = raise_size / pot_size
                else:
                    raise_size = 0
                    
                to_act_players_data.append([
                    p.stack_size / pot_size,  # rel stack size
                    t,  # turn to act
                    raise_size,  # normalized raise size
                ])
        
        # Create tensor of correct shape with padding
        if to_act_players_data:
            x_to_act_players = torch.tensor(to_act_players_data, dtype=torch.float32)
            # Add padding to reach size 9
            if x_to_act_players.shape[0] < 9:
                padding_size = 9 - x_to_act_players.shape[0]
                padding = torch.zeros((padding_size, 3), dtype=torch.float32)
                x_to_act_players = torch.cat([x_to_act_players, padding], dim=0)
        else:
            x_to_act_players = torch.zeros((9, 3), dtype=torch.float32)
        
        # Add batch dimension
        x_to_act_players = x_to_act_players.unsqueeze(0)

        return x_player_game, x_acted_players, x_to_act_players


class DeepLearningAgent(Player):
    """
    A poker player that uses a trained neural network to make decisions.
    This is the implementation of the PokerPlayerNetV1 architecture.
    """
    def __init__(self, name: str, stack: int, model_path: Optional[str] = None):
        """
        Initialize the deep learning agent.
        
        Args:
            name: Player name
            stack: Starting stack size
            model_path: Path to the saved model (if None, creates a new model)
        """
        super().__init__(name, stack)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = PokerPlayerNetV1().to(self.device)
        
        # Load the model if a path is provided
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        self.model.eval()  # Set to evaluation mode
        self.game_states = []  # Track game states for analysis
    
    def act(self) -> Action:
        """
        Take an action based on the current game state.
        
        Returns:
            Action to take (FOLD, CHECK_CALL, or RAISE)
        """
        # Add randomness to prevent infinite loops in self-play
        if random.random() < 0.4:  # 40% chance to take a random action
            # Make a random decision
            actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE]
            valid_actions = []
            
            # Check which actions are valid
            if self.stack > 0 and not self.folded:
                # Check if there's any bet to call
                current_bet = 0
                for pot in self.game.pots:
                    if pot.eligible_players and self in pot.eligible_players:
                        current_bet = max(current_bet, pot.contributions.get(self, 0))
                
                highest_bet = 0
                for player in self.game.players:
                    for pot in self.game.pots:
                        if player in pot.contributions:
                            highest_bet = max(highest_bet, pot.contributions.get(player, 0))
                
                call_amount = highest_bet - current_bet
                
                if call_amount > 0:
                    valid_actions.append(Action.FOLD)
                valid_actions.append(Action.CHECK_CALL)
                
                # Can only raise if we have enough chips
                min_raise = self.game.min_bet
                if self.stack >= min_raise:
                    valid_actions.append(Action.RAISE)
            
            if not valid_actions:
                valid_actions = [Action.CHECK_CALL]
                
            # Choose random action from valid actions
            return random.choice(valid_actions)
        
        # Create a game state for the current situation
        from poker.game_state_helper import GameStateHelper
        
        # Use GameStateHelper to create game states
        current_game_states = GameStateHelper.create_game_states(self.game, self.game.current_stage)
        
        if self not in current_game_states:
            # Fallback: make a random decision
            actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE]
            valid_actions = []
            
            # Check which actions are valid
            if self.stack > 0 and not self.folded:
                # Check if there's any bet to call
                current_bet = 0
                for pot in self.game.pots:
                    if pot.eligible_players and self in pot.eligible_players:
                        current_bet = max(current_bet, pot.contributions.get(self, 0))
                
                highest_bet = 0
                for player in self.game.players:
                    for pot in self.game.pots:
                        if player in pot.contributions:
                            highest_bet = max(highest_bet, pot.contributions.get(player, 0))
                
                call_amount = highest_bet - current_bet
                
                if call_amount > 0:
                    valid_actions.append(Action.FOLD)
                valid_actions.append(Action.CHECK_CALL)
                
                # Can only raise if we have enough chips
                min_raise = self.game.min_bet
                if self.stack >= min_raise:
                    valid_actions.append(Action.RAISE)
            
            if not valid_actions:
                valid_actions = [Action.CHECK_CALL]
                
            # Choose random action from valid actions
            return random.choice(valid_actions)
        
        # Get the game state for this player
        game_state = current_game_states[self]
        
        # Save game state for analysis
        self.game_states.append(game_state)
        
        # Use the neural network to predict an action
        with torch.no_grad():
            # Convert game state to batch format
            x_player_game, x_acted_history, x_to_act_history = PokerPlayerNetV1.game_state_to_batch(game_state)
            
            # Move to device
            x_player_game = x_player_game.to(self.device)
            x_acted_history = x_acted_history.to(self.device)
            x_to_act_history = x_to_act_history.to(self.device)
            
            # Forward pass
            fold_logits, raise_size = self.model(x_player_game, x_acted_history, x_to_act_history)
            
            # Determine action based on fold logits
            fold_prob = torch.sigmoid(fold_logits[0]).item()
            
            # Determine current game situation
            pot_size = game_state.pot_size
            min_bet = game_state.min_bet_to_continue
            current_stack = game_state.my_player.stack_size
            
            # Decision logic
            if fold_prob > 0.7 and min_bet > 0:  # High probability of fold and there's a bet to call
                return Action.FOLD
            elif raise_size.item() > 0.3 and current_stack > min_bet * 2:  # Significant raise and we have chips
                return Action.RAISE
            else:
                return Action.CHECK_CALL


def train_model(game_states_dataset, epochs=20, batch_size=32, learning_rate=0.001, save_path='./models/poker_model.pt'):
    """
    Train the poker player model on a dataset of game states.
    
    Args:
        game_states_dataset: List of (game_state, action, raise_amount) tuples
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        save_path: Path to save the trained model
        
    Returns:
        Trained model
    """
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PokerPlayerNetV1().to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss functions
    fold_criterion = nn.BCEWithLogitsLoss()
    raise_criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        fold_accuracy = 0
        raise_error = 0
        
        # Shuffle data
        random.shuffle(game_states_dataset)
        
        # Process in batches
        for i in range(0, len(game_states_dataset), batch_size):
            batch = game_states_dataset[i:i+batch_size]
            
            # Convert batch to tensors
            player_game_batch = []
            acted_history_batch = []
            to_act_history_batch = []
            fold_targets = []
            raise_targets = []
            
            for game_state, action, raise_amount in batch:
                # Convert game state to batch format
                player_game, acted_history, to_act_history = PokerPlayerNetV1.game_state_to_batch(game_state)
                
                player_game_batch.append(player_game.squeeze(0))
                acted_history_batch.append(acted_history.squeeze(0))
                to_act_history_batch.append(to_act_history.squeeze(0))
                
                # Convert action to fold target (1 for fold, 0 for not fold)
                fold_target = 1.0 if action == Action.FOLD else 0.0
                fold_targets.append(fold_target)
                
                # Convert raise amount to normalized target (0-1)
                if action == Action.RAISE and raise_amount is not None:
                    # Normalize by pot size
                    pot_size = max(1, game_state.pot_size)
                    raise_target = min(raise_amount / pot_size, 1.0)
                else:
                    raise_target = 0.0
                raise_targets.append(raise_target)
            
            # Stack tensors
            player_game_batch = torch.stack(player_game_batch).to(device)
            acted_history_batch = torch.stack(acted_history_batch).to(device)
            to_act_history_batch = torch.stack(to_act_history_batch).to(device)
            fold_targets = torch.tensor(fold_targets, dtype=torch.float32).to(device)
            raise_targets = torch.tensor(raise_targets, dtype=torch.float32).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            fold_logits, raise_preds = model(player_game_batch, acted_history_batch, to_act_history_batch)
            
            # Calculate loss
            fold_loss = fold_criterion(fold_logits, fold_targets)
            raise_loss = raise_criterion(raise_preds, raise_targets)
            
            # Combine losses
            loss = fold_loss + raise_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            fold_preds = (torch.sigmoid(fold_logits) > 0.5).float()
            fold_accuracy += (fold_preds == fold_targets).float().mean().item()
            raise_error += torch.abs(raise_preds - raise_targets).mean().item()
        
        # Calculate epoch metrics
        avg_loss = total_loss / (len(game_states_dataset) / batch_size)
        avg_fold_accuracy = fold_accuracy / (len(game_states_dataset) / batch_size)
        avg_raise_error = raise_error / (len(game_states_dataset) / batch_size)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Fold Acc: {avg_fold_accuracy:.4f} | Raise Err: {avg_raise_error:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deep Learning Poker Agent")
    parser.add_argument("--mode", type=str, choices=["train", "play"], default="play",
                        help="Mode: train a model or play with a trained model")
    parser.add_argument("--model", type=str, default="./models/poker_model.pt",
                        help="Path to model file (for loading or saving)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("Training mode not fully implemented yet. Need game state dataset.")
        print("To create a dataset, play many games and record (game_state, action, raise_amount) tuples.")
    else:
        from poker.play_vs_ai import play_game
        
        print("Playing with deep learning agent...")
        play_game(model_path=args.model, use_deep_learning=True, num_hands=3)

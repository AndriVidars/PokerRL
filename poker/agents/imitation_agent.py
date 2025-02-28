import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import pickle

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poker.core.action import Action
from poker.core.gamestage import Stage
from poker.core.player import Player
from poker.agents.game_state import GameState
from poker.parsers.game_state_retriever import GameStateRetriever


class PokerDataset(Dataset):
    """
    Dataset for poker decision points.
    Each sample contains state features and the target action taken by Pluribus.
    """
    def __init__(self, features: np.ndarray, action_labels: np.ndarray, raise_amounts: np.ndarray = None):
        """
        Initialize the poker dataset.
        
        Args:
            features: Array of state features [n_samples, n_features]
            action_labels: Array of action labels [n_samples]
            raise_amounts: Optional array of raise amounts for RAISE actions [n_samples]
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.action_labels = torch.tensor(action_labels, dtype=torch.long)
        
        if raise_amounts is not None:
            self.raise_amounts = torch.tensor(raise_amounts, dtype=torch.float32)
        else:
            self.raise_amounts = None
            
        self.has_raise_amounts = raise_amounts is not None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.has_raise_amounts:
            return self.features[idx], self.action_labels[idx], self.raise_amounts[idx]
        else:
            return self.features[idx], self.action_labels[idx]


class ActionPredictionModel(nn.Module):
    """
    Neural network model for predicting poker actions.
    """
    def __init__(self, input_size: int, hidden_size: int = 128, dropout_rate: float = 0.3):
        """
        Initialize the action prediction model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            dropout_rate: Dropout probability
        """
        super(ActionPredictionModel, self).__init__()
        
        # Action prediction network (3 actions: FOLD, CHECK_CALL, RAISE)
        self.action_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 3)  # 3 action types
        )
        
    def forward(self, x):
        """Forward pass through the network"""
        action_logits = self.action_network(x)
        return action_logits


class RaiseAmountModel(nn.Module):
    """
    Neural network model for predicting raise amounts when RAISE action is chosen.
    """
    def __init__(self, input_size: int, hidden_size: int = 128, dropout_rate: float = 0.3):
        """
        Initialize the raise amount prediction model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            dropout_rate: Dropout probability
        """
        super(RaiseAmountModel, self).__init__()
        
        # Raise amount prediction network (predicts raise amount as fraction of pot)
        self.raise_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Range (0, 1) - will be scaled by pot size
        )
        
    def forward(self, x):
        """Forward pass through the network"""
        raise_amount = self.raise_network(x)
        return raise_amount


class ImitationAgent:
    """
    Poker agent trained via imitation learning to mimic Pluribus's playing style.
    """
    def __init__(self, 
                 action_model: ActionPredictionModel = None,
                 raise_model: RaiseAmountModel = None,
                 device: str = 'cpu'):
        """
        Initialize the imitation agent.
        
        Args:
            action_model: Pre-trained action prediction model (or None to create new)
            raise_model: Pre-trained raise amount model (or None to create new)
            device: Device to run models on ('cpu' or 'cuda')
        """
        self.device = device
        self.action_model = action_model
        self.raise_model = raise_model
        
        # Feature standardization parameters
        self.feature_means = None
        self.feature_stds = None
        
        # Track training history
        self.training_history = {
            'action_loss': [],
            'action_accuracy': [],
            'raise_loss': [],
            'val_action_loss': [],
            'val_action_accuracy': [],
            'val_raise_loss': []
        }
    
    def create_models(self, input_size: int, hidden_size: int = 128, dropout_rate: float = 0.3):
        """
        Create new models for action prediction and raise amount prediction.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            dropout_rate: Dropout probability
        """
        self.action_model = ActionPredictionModel(input_size, hidden_size, dropout_rate).to(self.device)
        self.raise_model = RaiseAmountModel(input_size, hidden_size, dropout_rate).to(self.device)
    
    def standardize_features(self, features: np.ndarray, is_training: bool = False) -> np.ndarray:
        """
        Standardize features by subtracting mean and dividing by standard deviation.
        
        Args:
            features: Array of features [n_samples, n_features]
            is_training: Whether this is being called during training
            
        Returns:
            Standardized features
        """
        if is_training:
            # Calculate mean and std from training data
            self.feature_means = np.mean(features, axis=0)
            self.feature_stds = np.std(features, axis=0)
            self.feature_stds[self.feature_stds == 0] = 1  # Avoid division by zero
            
        if self.feature_means is not None and self.feature_stds is not None:
            # Standardize using saved mean and std
            return (features - self.feature_means) / self.feature_stds
        else:
            return features
    
    def prepare_features(self, game_states: List[GameState]) -> np.ndarray:
        """
        Extract and prepare features from game states for prediction.
        
        Args:
            game_states: List of GameState objects
            
        Returns:
            Array of standardized features [n_samples, n_features]
        """
        # Extract features from game states
        features = []
        
        for state in game_states:
            # Basic features
            feature_vec = [
                state.stage.value,  # Game stage
                state.pot_size,
                state.min_bet_to_continue,
                state.my_player.stack_size,
                state.my_player.spots_left_bb,  # Position
                len(state.community_cards),
                state.hand_strength,
                state.community_hand_strength,
                len([p for p in state.other_players if p.in_game])  # Active opponents
            ]
            
            # Add derived features
            pot_to_stack = state.pot_size / max(1, state.my_player.stack_size)
            feature_vec.append(pot_to_stack)
            
            # Add pot odds if applicable
            if state.pot_size > 0 and state.min_bet_to_continue > 0:
                pot_odds = state.min_bet_to_continue / (state.pot_size + state.min_bet_to_continue)
            else:
                pot_odds = 0
            feature_vec.append(pot_odds)
            
            # One-hot encoding for stage
            stage_onehot = [0, 0, 0, 0]  # PREFLOP, FLOP, TURN, RIVER
            stage_onehot[state.stage.value] = 1
            feature_vec.extend(stage_onehot)
            
            # Position categories
            position = state.my_player.spots_left_bb
            is_early = 1 if position <= 1 else 0
            is_middle = 1 if 1 < position <= 3 else 0
            is_late = 1 if position > 3 else 0
            feature_vec.extend([is_early, is_middle, is_late])
            
            features.append(feature_vec)
        
        # Convert to numpy array and standardize
        features_array = np.array(features, dtype=np.float32)
        return self.standardize_features(features_array)
    
    def train(self,
              features: np.ndarray,
              action_labels: np.ndarray,
              raise_amounts: np.ndarray,
              val_split: float = 0.1,
              batch_size: int = 64,
              epochs: int = 20,
              lr: float = 1e-3,
              input_size: int = None,
              hidden_size: int = 128,
              dropout_rate: float = 0.3):
        """
        Train the imitation agent on Pluribus data.
        
        Args:
            features: Array of state features [n_samples, n_features]
            action_labels: Array of action labels [n_samples]
            raise_amounts: Array of raise amounts (pot multipliers) [n_samples]
            val_split: Fraction of data to use for validation
            batch_size: Batch size for training
            epochs: Number of training epochs
            lr: Learning rate
            input_size: Input feature size (if None, inferred from features)
            hidden_size: Hidden layer size
            dropout_rate: Dropout probability
        """
        # Standardize features
        features = self.standardize_features(features, is_training=True)
        
        # Split into train and validation sets
        indices = np.random.permutation(len(features))
        val_size = int(len(features) * val_split)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        X_train, y_train, r_train = features[train_indices], action_labels[train_indices], raise_amounts[train_indices]
        X_val, y_val, r_val = features[val_indices], action_labels[val_indices], raise_amounts[val_indices]
        
        # Create datasets and dataloaders
        train_dataset = PokerDataset(X_train, y_train, r_train)
        val_dataset = PokerDataset(X_val, y_val, r_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create models if not already created
        if input_size is None:
            input_size = features.shape[1]
            
        if self.action_model is None or self.raise_model is None:
            self.create_models(input_size, hidden_size, dropout_rate)
        
        # Define optimizers and loss functions
        action_optimizer = optim.Adam(self.action_model.parameters(), lr=lr)
        raise_optimizer = optim.Adam(self.raise_model.parameters(), lr=lr)
        
        action_criterion = nn.CrossEntropyLoss()
        raise_criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.action_model.train()
            self.raise_model.train()
            
            total_action_loss = 0
            total_raise_loss = 0
            correct_actions = 0
            total_actions = 0
            
            for batch in train_loader:
                features, action_targets, raise_targets = batch
                features = features.to(self.device)
                action_targets = action_targets.to(self.device)
                raise_targets = raise_targets.to(self.device)
                
                # Train action model
                action_optimizer.zero_grad()
                action_preds = self.action_model(features)
                action_loss = action_criterion(action_preds, action_targets)
                action_loss.backward()
                action_optimizer.step()
                
                # Train raise model (only on RAISE examples)
                raise_mask = (action_targets == 2)  # Action.RAISE has index 2
                if raise_mask.sum() > 0:
                    raise_features = features[raise_mask]
                    raise_targets_filtered = raise_targets[raise_mask].view(-1, 1)
                    
                    raise_optimizer.zero_grad()
                    raise_preds = self.raise_model(raise_features)
                    raise_loss = raise_criterion(raise_preds, raise_targets_filtered)
                    raise_loss.backward()
                    raise_optimizer.step()
                    
                    total_raise_loss += raise_loss.item() * len(raise_features)
                
                # Calculate metrics
                total_action_loss += action_loss.item() * len(features)
                _, predicted_actions = torch.max(action_preds, 1)
                correct_actions += (predicted_actions == action_targets).sum().item()
                total_actions += len(action_targets)
            
            avg_action_loss = total_action_loss / total_actions
            avg_raise_loss = total_raise_loss / max(1, sum(action_targets == 2).item())
            action_accuracy = correct_actions / total_actions
            
            # Validation phase
            self.action_model.eval()
            self.raise_model.eval()
            
            val_action_loss = 0
            val_raise_loss = 0
            val_correct_actions = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    features, action_targets, raise_targets = batch
                    features = features.to(self.device)
                    action_targets = action_targets.to(self.device)
                    raise_targets = raise_targets.to(self.device)
                    
                    # Action predictions
                    action_preds = self.action_model(features)
                    val_action_loss += action_criterion(action_preds, action_targets).item() * len(features)
                    
                    _, predicted_actions = torch.max(action_preds, 1)
                    val_correct_actions += (predicted_actions == action_targets).sum().item()
                    
                    # Raise predictions (only on RAISE examples)
                    raise_mask = (action_targets == 2)
                    if raise_mask.sum() > 0:
                        raise_features = features[raise_mask]
                        raise_targets_filtered = raise_targets[raise_mask].view(-1, 1)
                        
                        raise_preds = self.raise_model(raise_features)
                        val_raise_loss += raise_criterion(raise_preds, raise_targets_filtered).item() * len(raise_features)
            
            val_avg_action_loss = val_action_loss / len(val_indices)
            val_avg_raise_loss = val_raise_loss / max(1, sum(y_val == 2))
            val_action_accuracy = val_correct_actions / len(val_indices)
            
            # Save training metrics
            self.training_history['action_loss'].append(avg_action_loss)
            self.training_history['action_accuracy'].append(action_accuracy)
            self.training_history['raise_loss'].append(avg_raise_loss)
            self.training_history['val_action_loss'].append(val_avg_action_loss)
            self.training_history['val_action_accuracy'].append(val_action_accuracy)
            self.training_history['val_raise_loss'].append(val_avg_raise_loss)
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Action Loss: {avg_action_loss:.4f} | "
                  f"Action Acc: {action_accuracy:.4f} | "
                  f"Raise Loss: {avg_raise_loss:.4f} | "
                  f"Val Action Loss: {val_avg_action_loss:.4f} | "
                  f"Val Action Acc: {val_action_accuracy:.4f} | "
                  f"Val Raise Loss: {val_avg_raise_loss:.4f}")
    
    def predict_action(self, game_state: GameState) -> Tuple[Action, Optional[float]]:
        """
        Predict the best action for a given game state.
        
        Args:
            game_state: The current game state
            
        Returns:
            Tuple of (predicted action, raise amount if applicable)
        """
        self.action_model.eval()
        self.raise_model.eval()
        
        # Extract and prepare features
        features = self.prepare_features([game_state])
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # Predict action
        with torch.no_grad():
            action_logits = self.action_model(features_tensor)
            action_idx = torch.argmax(action_logits, dim=1).item()
            
            # Convert to Action enum
            if action_idx == 0:
                action = Action.FOLD
                raise_amount = None
            elif action_idx == 1:
                action = Action.CHECK_CALL
                raise_amount = None
            else:  # action_idx == 2
                action = Action.RAISE
                
                # Predict raise amount
                raise_multiplier = self.raise_model(features_tensor).item()
                
                # Scale by pot size (minimum 1x big blind)
                # Handle case where core_game is None (during evaluation)
                big_blind = 20  # Default big blind if core_game is None
                if game_state.core_game is not None:
                    big_blind = game_state.core_game.big_amount
                
                min_bet = max(game_state.min_bet_to_continue, big_blind)
                pot_size = max(game_state.pot_size, big_blind * 2)
                
                # Calculate raise amount - scale from 0.5x to 3x pot
                base_amount = min_bet
                additional_amount = pot_size * (0.5 + raise_multiplier * 2.5)
                
                # Ensure raise is at least min_bet and not zero
                raise_amount = min(max(base_amount + additional_amount, min_bet), game_state.my_player.stack_size)
                if raise_amount < min_bet:
                    raise_amount = min_bet
                
                # Must be at least min_bet
                if raise_amount < big_blind:
                    raise_amount = big_blind
                    
                # Round to nearest multiple of big blind
                raise_amount = max(min_bet, round(raise_amount / big_blind) * big_blind)
        
        return action, raise_amount
    
    def act(self, game_state: GameState) -> Action:
        """
        Take an action based on the current game state.
        This method interfaces with the poker framework.
        
        Args:
            game_state: The current game state
            
        Returns:
            Action to take
        """
        action, raise_amount = self.predict_action(game_state)
        
        # Handle the action based on the core game
        player = game_state.my_player
        game = game_state.core_game
        
        # Return the action instead of handling it - game.py will handle the action
        return action
    
    def save(self, save_dir: str = './models'):
        """
        Save the trained models and parameters.
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        torch.save(self.action_model.state_dict(), os.path.join(save_dir, 'action_model.pt'))
        torch.save(self.raise_model.state_dict(), os.path.join(save_dir, 'raise_model.pt'))
        
        # Save feature normalization parameters
        normalization_dict = {
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'history': self.training_history
        }
        
        with open(os.path.join(save_dir, 'normalization_params.pkl'), 'wb') as f:
            pickle.dump(normalization_dict, f)
    
    def load(self, load_dir: str = './models', input_size: int = None, hidden_size: int = 128, dropout_rate: float = 0.3):
        """
        Load trained models and parameters.
        
        Args:
            load_dir: Directory to load models from
            input_size: Number of input features (must match saved model)
            hidden_size: Size of hidden layers
            dropout_rate: Dropout probability
        """
        # Load normalization parameters
        with open(os.path.join(load_dir, 'normalization_params.pkl'), 'rb') as f:
            normalization_dict = pickle.load(f)
            
        self.feature_means = normalization_dict['feature_means']
        self.feature_stds = normalization_dict['feature_stds']
        self.training_history = normalization_dict.get('history', {})
        
        # Create models
        if input_size is None:
            input_size = len(self.feature_means)
            
        self.create_models(input_size, hidden_size, dropout_rate)
        
        # Load model weights
        self.action_model.load_state_dict(torch.load(os.path.join(load_dir, 'action_model.pt'), map_location=self.device))
        self.raise_model.load_state_dict(torch.load(os.path.join(load_dir, 'raise_model.pt'), map_location=self.device))
        
        # Set models to evaluation mode
        self.action_model.eval()
        self.raise_model.eval()
    
    def plot_training_history(self):
        """Plot the training history."""
        plt.figure(figsize=(15, 10))
        
        # Plot action loss
        plt.subplot(2, 2, 1)
        plt.plot(self.training_history['action_loss'], label='Train')
        plt.plot(self.training_history['val_action_loss'], label='Validation')
        plt.title('Action Prediction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot action accuracy
        plt.subplot(2, 2, 2)
        plt.plot(self.training_history['action_accuracy'], label='Train')
        plt.plot(self.training_history['val_action_accuracy'], label='Validation')
        plt.title('Action Prediction Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot raise loss
        plt.subplot(2, 2, 3)
        plt.plot(self.training_history['raise_loss'], label='Train')
        plt.plot(self.training_history['val_raise_loss'], label='Validation')
        plt.title('Raise Amount Prediction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


class DeepLearningPlayer(Player):
    """
    A player that uses a trained imitation learning model to make decisions.
    This class serves as an adapter between the poker game and the ImitationAgent.
    """
    def __init__(self, name: str, stack: int, agent: ImitationAgent):
        """
        Initialize the deep learning player.
        
        Args:
            name: Player name
            stack: Starting stack
            agent: Trained ImitationAgent model
        """
        super().__init__(name, stack)
        self.agent = agent
        self.game_states = []  # Track game states for analysis
        
    def act(self) -> Action:
        """
        Take an action based on the current game state.
        
        Returns:
            Action to take (FOLD, CHECK_CALL, or RAISE)
        """
        # Create a game state for the current situation
        from poker.agents.game_state import GameState, Player as StatePlayer
        from poker.game_state_helper import GameStateHelper
        
        # Use GameStateHelper to create a game state
        current_game_states = GameStateHelper.create_game_states(self.game, self.game.current_stage)
        
        if self not in current_game_states:
            # Fallback: make a random decision
            actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE]
            valid_actions = []
            
            # Check which actions are valid
            if self.stack > 0 and not self.folded:
                valid_actions.append(Action.FOLD)
                valid_actions.append(Action.CHECK_CALL)
                
                # Can only raise if we have enough chips
                max_raise = self.game.min_bet
                if self.stack >= max_raise:
                    valid_actions.append(Action.RAISE)
            
            if not valid_actions:
                valid_actions = [Action.CHECK_CALL]
                
            # Choose random action from valid actions
            action = random.choice(valid_actions)
            
            # Handle the action
            if action == Action.FOLD:
                self.handle_fold()
            elif action == Action.CHECK_CALL:
                self.handle_check_call()
            elif action == Action.RAISE:
                min_raise = self.game.min_bet
                self.handle_raise(min_raise)
                
            return action
        
        # Get the game state for this player
        game_state = current_game_states[self]
        
        # Save game state for analysis
        self.game_states.append(game_state)
        
        # Use the agent to choose an action
        return self.agent.act(game_state)


def train_imitation_agent(log_dir: str = None, save_dir: str = './models', device: str = 'cpu'):
    """
    Train an imitation agent on Pluribus data.
    
    Args:
        log_dir: Directory containing Pluribus logs
        save_dir: Directory to save models
        device: Device to train on ('cpu' or 'cuda')
    
    Returns:
        Trained ImitationAgent
    """
    # Initialize game state retriever
    retriever = GameStateRetriever(log_dir)
    print("Initializing retriever (parsing logs)...")
    retriever.initialize(verbose=True)
    print(f"Successfully parsed {retriever.get_hand_count()} hands")
    
    # Get all Pluribus decisions
    pluribus_decisions = retriever.get_pluribus_decisions()
    print(f"Found {len(pluribus_decisions)} Pluribus decisions")
    
    # Extract features and labels
    features = []
    action_labels = []
    raise_amounts = []
    
    for player, stage, state, action, amount in pluribus_decisions:
        # Basic features
        feature_vec = [
            stage.value,  # Game stage
            state.pot_size,
            state.min_bet_to_continue,
            state.my_player.stack_size,
            state.my_player.spots_left_bb,  # Position
            len(state.community_cards),
            state.hand_strength,
            state.community_hand_strength,
            len([p for p in state.other_players if p.in_game])  # Active opponents
        ]
        
        # Add derived features
        pot_to_stack = state.pot_size / max(1, state.my_player.stack_size)
        feature_vec.append(pot_to_stack)
        
        # Add pot odds if applicable
        if state.pot_size > 0 and state.min_bet_to_continue > 0:
            pot_odds = state.min_bet_to_continue / (state.pot_size + state.min_bet_to_continue)
        else:
            pot_odds = 0
        feature_vec.append(pot_odds)
        
        # One-hot encoding for stage
        stage_onehot = [0, 0, 0, 0]  # PREFLOP, FLOP, TURN, RIVER
        stage_onehot[stage.value] = 1
        feature_vec.extend(stage_onehot)
        
        # Position categories
        position = state.my_player.spots_left_bb
        is_early = 1 if position <= 1 else 0
        is_middle = 1 if 1 < position <= 3 else 0
        is_late = 1 if position > 3 else 0
        feature_vec.extend([is_early, is_middle, is_late])
        
        features.append(feature_vec)
        
        # Convert action to index: FOLD=0, CHECK_CALL=1, RAISE=2
        action_idx = action.value
        action_labels.append(action_idx)
        
        # Normalize raise amount as fraction of pot size
        if action == Action.RAISE and amount is not None and state.pot_size > 0:
            norm_amount = min(amount / state.pot_size, 5.0)  # Cap at 5x pot for stability
            raise_amounts.append(norm_amount)
        else:
            raise_amounts.append(0.0)
    
    # Convert to numpy arrays
    features = np.array(features, dtype=np.float32)
    action_labels = np.array(action_labels, dtype=np.int64)
    raise_amounts = np.array(raise_amounts, dtype=np.float32)
    
    print(f"Feature shape: {features.shape}")
    print(f"Action label shape: {action_labels.shape}")
    print(f"Raise amount shape: {raise_amounts.shape}")
    
    # Initialize the agent
    agent = ImitationAgent(device=device)
    
    # Train the agent
    agent.train(
        features=features,
        action_labels=action_labels,
        raise_amounts=raise_amounts,
        val_split=0.1,
        batch_size=64,
        epochs=20,
        lr=1e-3,
        input_size=features.shape[1],
        hidden_size=128,
        dropout_rate=0.3
    )
    
    # Save the trained agent
    agent.save(save_dir)
    
    # Plot training history
    agent.plot_training_history()
    
    return agent


def evaluate_agent(agent: ImitationAgent, log_dir: str = None, num_samples: int = 1000):
    """
    Evaluate an imitation agent on held-out Pluribus data.
    
    Args:
        agent: Trained ImitationAgent
        log_dir: Directory containing Pluribus logs
        num_samples: Number of samples to evaluate on
        
    Returns:
        Evaluation metrics
    """
    # Initialize game state retriever
    retriever = GameStateRetriever(log_dir)
    print("Initializing retriever for evaluation...")
    retriever.initialize(verbose=False)
    
    # Get all Pluribus decisions
    all_decisions = retriever.get_pluribus_decisions()
    print(f"Found {len(all_decisions)} Pluribus decisions")
    
    # Sample decisions for evaluation
    if num_samples < len(all_decisions):
        eval_decisions = random.sample(all_decisions, num_samples)
    else:
        eval_decisions = all_decisions
    
    # Track metrics
    correct_actions = 0
    action_type_correct = defaultdict(int)
    action_type_total = defaultdict(int)
    raise_errors = []
    
    # Evaluate on each decision
    for player, stage, state, true_action, true_amount in eval_decisions:
        # Predict action
        pred_action, pred_amount = agent.predict_action(state)
        
        # Check if action prediction is correct
        if pred_action == true_action:
            correct_actions += 1
            action_type_correct[true_action.name] += 1
        
        action_type_total[true_action.name] += 1
        
        # Calculate raise amount error
        if true_action == Action.RAISE and pred_action == Action.RAISE and true_amount is not None and pred_amount is not None:
            # Calculate error as percentage of pot
            pot_size = max(1, state.pot_size)
            error = abs(true_amount - pred_amount) / pot_size
            raise_errors.append(error)
    
    # Calculate metrics
    accuracy = correct_actions / len(eval_decisions)
    action_accuracies = {action: action_type_correct[action] / max(1, action_type_total[action]) 
                        for action in action_type_total.keys()}
    
    avg_raise_error = np.mean(raise_errors) if raise_errors else 0
    
    # Print metrics
    print(f"Overall accuracy: {accuracy:.4f}")
    for action, acc in action_accuracies.items():
        print(f"{action} accuracy: {acc:.4f} ({action_type_correct[action]}/{action_type_total[action]})")
    print(f"Average raise amount error: {avg_raise_error:.4f} x pot")
    
    # Plot confusion matrix
    action_names = ["FOLD", "CHECK_CALL", "RAISE"]
    confusion_matrix = np.zeros((3, 3))
    
    for player, stage, state, true_action, _ in eval_decisions:
        pred_action, _ = agent.predict_action(state)
        confusion_matrix[pred_action.value, true_action.value] += 1
    
    # Normalize by true labels
    for i in range(3):
        if np.sum(confusion_matrix[:, i]) > 0:
            confusion_matrix[:, i] = confusion_matrix[:, i] / np.sum(confusion_matrix[:, i])
    
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, cmap='Blues')
    
    # Add labels
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{confusion_matrix[i, j]:.2f}", ha='center', va='center', color='black')
    
    plt.colorbar()
    plt.xlabel('True Action')
    plt.ylabel('Predicted Action')
    plt.xticks(range(3), action_names)
    plt.yticks(range(3), action_names)
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'action_accuracies': action_accuracies,
        'avg_raise_error': avg_raise_error,
        'confusion_matrix': confusion_matrix
    }


if __name__ == "__main__":
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Define log directory
    log_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'pluribus_converted_logs'))
    
    # Train an imitation agent
    print("Training imitation agent...")
    agent = train_imitation_agent(log_dir=log_dir, device=device)
    
    # Evaluate the agent
    print("\nEvaluating agent...")
    evaluate_agent(agent, log_dir=log_dir, num_samples=1000)
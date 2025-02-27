import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, List, Optional

from poker.core.action import Action
from poker.core.gamestage import Stage
from poker.agents.dqn_agent import PokerDQN

class ImitationLearningAgent:

    def __init__(self, 
                 state_dim: int = 8, 
                 action_dim: int = 4,
                 hidden_dim: int = 128,
                 learning_rate: float = 1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy network for action selection
        self.policy_net = PokerDQN(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Raise amount prediction network
        self.raise_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        ).to(self.device)
        
        # Optimizer
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.raise_optimizer = optim.Adam(self.raise_net.parameters(), lr=learning_rate)
        
        # For evaluation mode
        self.epsilon = 0.05  # Small exploration during evaluation
        
    def select_action(self, state) -> Tuple[int, float]:
        """
        Select an action using the policy network
        Returns action index and raise amount (0-1 for no-raise actions)
        """
        # Exploration during evaluation 
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
            raise_amount = 0.0 if action < 3 else np.random.uniform(0.1, 1.0)
            return action, raise_amount
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action probabilities
            q_values = self.policy_net(state_tensor)
            action = q_values.max(1)[1].item()
            
            # Get raise amount if needed
            raise_amount = 0.0
            if action == 3:  # RAISE action
                raise_amount = self.raise_net(state_tensor).item()
                
            return action, raise_amount
    
    def convert_action_to_poker_action(self, action_idx: int, raise_amount: float, game_state) -> Tuple[Action, int]:
        """
        Convert action index and raise amount to poker game action
        Works with either GameState objects or dictionary representation
        """
        if action_idx == 0:
            return Action.FOLD, None
        elif action_idx == 1:
            return Action.CHECK, None
        elif action_idx == 2:
            return Action.CALL, None
        elif action_idx == 3:
            # Calculate raise size based on pot and raise_amount
            # Support both GameState objects and dictionary representation
            if hasattr(game_state, 'min_bet_to_continue'):
                # GameState object
                min_raise = game_state.min_bet_to_continue
                max_raise = game_state.my_player.stack_size
            else:
                # Dictionary representation
                min_raise = game_state['min_bet_to_continue']
                max_raise = game_state['my_player']['stack_size']
            
            # Scale raise between min and max based on raise_amount
            raise_size = int(min_raise + (max_raise - min_raise) * raise_amount)
            return Action.RAISE, max(min_raise, raise_size)
        else:
            return Action.CALL, None  # Default to CALL
    
    def preprocess_state(self, game_state) -> np.ndarray:
        """
        Convert game state to feature vector
        Works with both GameState objects and dictionary representation
        """
        # Support both GameState objects and dictionary representation
        if hasattr(game_state, 'pot_size'):
            # GameState object
            features = [
                game_state.pot_size,
                game_state.min_bet_to_continue,
                game_state.stage.value,
                game_state.my_player.stack_size,
                game_state.hand_strength,  # Property from GameState
                game_state.community_hand_strenght,  # Property from GameState
                game_state.my_player.spots_left_bb,  # Position
                len([p for p in game_state.other_players if p.in_game])  # Active players
            ]
        else:
            # Dictionary representation
            features = [
                game_state['pot_size'],
                game_state['min_bet_to_continue'],
                game_state['stage'].value,
                game_state['my_player']['stack_size'],
                game_state['hand_strength'],
                game_state['community_strength'],
                game_state['my_player']['position'],
                len([p for p in game_state['other_players'] if not p['folded']])
            ]
        
        return np.array(features, dtype=np.float32)
    
    def train(self, dataset_file: str, epochs: int = 100, batch_size: int = 64):
        """
        Train the agent on the imitation learning dataset
        """
        print(f"Loading dataset from {dataset_file}")
        data = np.load(dataset_file)
        states = data['states']
        actions = data['actions']
        raise_amounts = data['raise_amounts']
        
        dataset_size = len(states)
        print(f"Dataset size: {dataset_size} examples")
        
        # Create PyTorch dataset
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        raise_amounts_tensor = torch.FloatTensor(raise_amounts).to(self.device)
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data for each epoch
            indices = torch.randperm(dataset_size)
            states_shuffled = states_tensor[indices]
            actions_shuffled = actions_tensor[indices]
            raise_amounts_shuffled = raise_amounts_tensor[indices]
            
            total_policy_loss = 0
            total_raise_loss = 0
            num_batches = 0
            
            for i in range(0, dataset_size, batch_size):
                # Get batch
                if i + batch_size > dataset_size:
                    break
                    
                states_batch = states_shuffled[i:i+batch_size]
                actions_batch = actions_shuffled[i:i+batch_size]
                raise_amounts_batch = raise_amounts_shuffled[i:i+batch_size]
                
                # Only use raise examples for training the raise network
                raise_mask = (actions_batch == 3)
                
                # Policy network training
                self.policy_optimizer.zero_grad()
                q_values = self.policy_net(states_batch)
                
                # Cross-entropy loss for policy
                policy_loss = nn.CrossEntropyLoss()(q_values, actions_batch)
                policy_loss.backward()
                self.policy_optimizer.step()
                
                # Raise amount network training (only if we have raise examples)
                if raise_mask.sum() > 0:
                    self.raise_optimizer.zero_grad()
                    raise_pred = self.raise_net(states_batch[raise_mask]).squeeze()
                    raise_target = raise_amounts_batch[raise_mask]
                    
                    # MSE loss for raise amounts
                    raise_loss = nn.MSELoss()(raise_pred, raise_target)
                    raise_loss.backward()
                    self.raise_optimizer.step()
                    
                    total_raise_loss += raise_loss.item()
                else:
                    total_raise_loss += 0  # No raise examples in this batch
                
                total_policy_loss += policy_loss.item()
                num_batches += 1
            
            # Print epoch results
            if num_batches > 0:
                avg_policy_loss = total_policy_loss / num_batches
                avg_raise_loss = total_raise_loss / num_batches
                print(f"Epoch {epoch+1}/{epochs}, Policy Loss: {avg_policy_loss:.4f}, Raise Loss: {avg_raise_loss:.4f}")
        
        print("Training complete!")
    
    def save(self, path: str):
        """
        Save the model
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'raise_net': self.raise_net.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load the model
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.raise_net.load_state_dict(checkpoint['raise_net'])
        print(f"Model loaded from {path}")


def train_imitation_agent(dataset_path: str, model_path: str, epochs: int = 100):
    """
    Train an imitation learning agent and save it
    """
    agent = ImitationLearningAgent()
    agent.train(dataset_path, epochs=epochs)
    agent.save(model_path)
    return agent

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train an imitation learning agent on Pluribus data")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the Pluribus dataset (.npz file)")
    parser.add_argument("--model", type=str, default="models/imitation_agent.pt", help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    
    args = parser.parse_args()
    
    train_imitation_agent(args.dataset, args.model, args.epochs)
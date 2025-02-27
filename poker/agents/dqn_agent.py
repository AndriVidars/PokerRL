import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict
from poker.core.action import Action
from poker.core.card import Card
from poker.core.gamestage import Stage
from .game_state import GameState

class PokerDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PokerDQN, self).__init__()
        
        # Using a simpler architecture to avoid batch norm with single samples
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple:
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        sample = random.sample(self.buffer, batch_size)
        for state, action, reward, next_state, done in sample:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        return (torch.FloatTensor(np.array(states)), 
                torch.LongTensor(np.array(actions)), 
                torch.FloatTensor(np.array(rewards)),
                torch.FloatTensor(np.array(next_states)), 
                torch.FloatTensor(np.array(dones)))
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dim: int = 128,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_final: float = 0.1,
                 epsilon_decay: float = 10000,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 1000):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q Networks
        self.policy_net = PokerDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = PokerDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training info
        self.steps = 0
        self.updates = 0
        
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Select an action using epsilon-greedy policy
        Returns action index and raise amount (0-1 for no-raise actions)
        """
        if np.random.random() < self.epsilon:
            # Random action
            action = np.random.randint(self.action_dim)
            # For fold, check, call, raise_amount=0
            # For raise, generate random raise amount between 0.1 and 1.0
            raise_amount = 0.0 if action < 3 else np.random.uniform(0.1, 1.0)
            return action, raise_amount
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.max(1)[1].item()
            # For raise, we need to decide raise amount
            raise_amount = 0.0 if action < 3 else 0.5  # Default 0.5 for deterministic policy
            return action, raise_amount
    
    def convert_action_to_poker_action(self, action_idx: int, raise_amount: float, game_state) -> Tuple[Action, int]:
        """
        Convert DQN action to poker game action
        """
        if action_idx == 0:
            return Action.FOLD, None
        elif action_idx == 1:
            return Action.CHECK, None
        elif action_idx == 2:
            return Action.CALL, None
        elif action_idx == 3:
            # Calculate raise size based on pot and raise_amount
            min_raise = game_state['min_bet_to_continue']
            max_raise = game_state['my_player']['stack_size']
            # Scale raise between min and max based on raise_amount
            raise_size = int(min_raise + (max_raise - min_raise) * raise_amount)
            return Action.RAISE, max(min_raise, raise_size)
    
    def preprocess_state(self, game_state) -> np.ndarray:
        """
        Convert state dictionary to numpy array for DQN input
        """
        # Featurize game state
        features = []
        
        # Basic game info
        features.append(game_state['pot_size'])
        features.append(game_state['min_bet_to_continue'])
        features.append(game_state['stage'].value)  # Game stage as integer
        
        # Player info
        features.append(game_state['my_player']['stack_size'])
        
        # Hand strength features
        features.append(game_state['hand_strength'])
        features.append(game_state['community_strength'])
        
        # Position info
        features.append(game_state['my_player']['position'])
            
        # Add feature for number of active players
        active_players = sum(1 for p in game_state['other_players'] if not p['folded'])
        features.append(active_players)
        
        # Normalize features
        return np.array(features, dtype=np.float32)
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay buffer
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update(self):
        """
        Update network parameters using replay buffer
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q values
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.updates += 1
        if self.updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # Update epsilon
        self.epsilon = max(self.epsilon_final, 
                           self.epsilon - (1.0 - self.epsilon_final) / self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path: str):
        """
        Save model parameters
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'updates': self.updates,
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """
        Load model parameters
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.updates = checkpoint['updates']
        self.epsilon = checkpoint['epsilon']
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from poker.agents.deep_learning_agent import PokerPlayerNetV1, FFN, TruncatedNormal
from poker.agents.game_state import Stage
from poker.core.action import Action
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.states_player_game = []
        self.states_acted_history = []
        self.states_to_act_history = []
        self.actions = []
        self.raise_sizes = []
        self.logprobs_actions = []
        self.logprobs_raise_sizes = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.states_player_game[:]
        del self.states_acted_history[:]
        del self.states_to_act_history[:]
        del self.actions[:]
        del self.raise_sizes[:]
        del self.logprobs_actions[:]
        del self.logprobs_raise_sizes[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class PPOPokerNet(PokerPlayerNetV1):
    """
    PPO Network for poker that extends the PokerPlayerNetV1.
    Adds a value function while keeping the same action policy structure.
    """
    def __init__(self, use_batchnorm=False):
        super(PPOPokerNet, self).__init__(use_batchnorm=use_batchnorm)
        
        # Add critic network for value function estimation
        self.critic_net = FFN(
            idim=60,  # Same input dimension as the gather_net
            hdim=30,
            odim=1,   # Scalar value output
            n_hidden=3,
            use_batchnorm=use_batchnorm,
        )
    
    def forward(self, x_player_game, x_acted_history, x_to_act_history):
        # Get action logits and raise size from the base network
        action_logits, raise_size = super().forward(x_player_game, x_acted_history, x_to_act_history)
        
        # Calculate state value using the critic network
        player_game_state = self.player_game_net(x_player_game)
        acted_player_history_state = self.acted_player_history_net(x_acted_history)
        to_act_player_history_state = self.to_act_player_history_net(x_to_act_history)
        stage = x_player_game[:, 0].to(torch.int64)

        stage_embeds = self.stage_embed(stage)
        acted_player_history_state = acted_player_history_state.sum(dim=1)
        to_act_player_history_state = to_act_player_history_state.sum(dim=1)

        all_game_state = torch.concat([
            player_game_state,
            stage_embeds,
            acted_player_history_state,
            to_act_player_history_state,
        ], dim=-1)
        
        state_value = self.critic_net(all_game_state)
        
        return action_logits, raise_size, state_value
    
    def act(self, x_player_game, x_acted_history, x_to_act_history):
        """
        Select an action given the current state
        Returns the selected action, raise size, log probabilities, and state value
        """
        with torch.no_grad():
            action_logits, raise_size_dist, state_value = self(x_player_game, x_acted_history, x_to_act_history)
            
            action_logits = torch.nan_to_num(action_logits, nan=0.0)

            action_logits[:, 0] *= 1.5
            action_logits[:, 1] *= 1
            action_logits[:, 2] *= 1
            
            # Sample from action distribution
            action_probs = F.softmax(action_logits, dim=-1)
            # action_probs = torch.nan_to_num(action_probs, nan=1.0/3.0)
            actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE]


            
            
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            action_logprob = action_dist.log_prob(action)
            
            # Sample from raise size distribution if applicable
            raise_size = raise_size_dist.sample()
            raise_size_logprob = torch.nan_to_num(raise_size_dist.log_prob(raise_size), nan=0.0)

            # Get the index of the chosen action to return as a tensor
            action_idx = action_dist.sample()
            
            # Convert index to Action enum for the game logic
            paction = actions[action_idx.item()]
            
            # Log probability based on the sampled index
            action_logprob = action_dist.log_prob(action_idx)
            
        return (paction, action_probs), raise_size, action_logprob, raise_size_logprob, state_value
    
    def forward_and_evaluate(self, x_player_game, x_acted_history, x_to_act_history, actions, raise_sizes):
        """
        Run a forward pass through the model and compute log probabilities, entropy, and state values for PPO updates
        """
        # Forward pass through the network
        action_logits, raise_size_dist, state_values = self(x_player_game, x_acted_history, x_to_act_history)
        
        # Fix NaN values in action logits
        action_logits = torch.nan_to_num(action_logits, nan=0.0)
        
        # Action distribution
        action_probs = F.softmax(action_logits, dim=-1)
        # Ensure valid probabilities by handling potential NaN or inf values
        action_probs = torch.nan_to_num(action_probs, nan=1.0/3.0)
        # Normalize to ensure probabilities sum to 1
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        
        # Create categorical distribution and compute log probabilities and entropy
        action_dist = Categorical(action_probs)
        action_logprobs = action_dist.log_prob(actions)
        action_entropy = action_dist.entropy()
        
        # Compute raise size log probabilities with NaN handling
        raise_logprobs = torch.nan_to_num(raise_size_dist.log_prob(raise_sizes), nan=0.0)
        
        return action_logprobs, raise_logprobs, state_values, action_entropy
        
    # For backward compatibility
    def evaluate(self, x_player_game, x_acted_history, x_to_act_history, actions, raise_sizes):
        """
        Alias for forward_and_evaluate for backward compatibility
        """
        return self.forward_and_evaluate(x_player_game, x_acted_history, x_to_act_history, actions, raise_sizes)
    
    def warm_start_from_model(self, model_state_dict):
        """
        Initialize the policy network from a pre-trained PokerPlayerNetV1 model
        """
        if isinstance(model_state_dict, str):
            model_state_dict = torch.load(model_state_dict)
        
        # Special handling for the gather_net layer that might have different dimensions
        if "gather_net.net.8.weight" in model_state_dict:
            orig_gather_net_weight = model_state_dict.pop("gather_net.net.8.weight")
            orig_gather_net_bias = model_state_dict.pop("gather_net.net.8.bias")
            
            # Filter out parameters that don't exist in the target model
            filtered_state_dict = {k: v for k, v in model_state_dict.items() 
                                if k in self.state_dict() and 'critic_net' not in k}
            
            # Load the filtered state dict
            torch.nn.Module.load_state_dict(self, filtered_state_dict, strict=False)
            
            # Handle the last layer separately
            if hasattr(self, 'gather_net') and hasattr(self.gather_net, 'net'):
                last_layer = self.gather_net.net[-1]
                output_size = last_layer.weight.size(0)
                input_size = last_layer.weight.size(1)
                
                # Copy weights for the dimensions that match
                min_output = min(output_size, orig_gather_net_weight.size(0))
                last_layer.weight.data[:min_output, :input_size] = orig_gather_net_weight.data[:min_output, :]
                last_layer.bias.data[:min_output] = orig_gather_net_bias.data[:min_output]
                
                # Initialize any additional dimensions
                if output_size > min_output:
                    last_layer.weight.data[min_output:, :] = 0
                    last_layer.bias.data[min_output:] = -2.5  # Bias initialization for new dimensions
        else:
            # For other models without specific gather_net layer
            filtered_state_dict = {k: v for k, v in model_state_dict.items() 
                                if k in self.state_dict() and 'critic_net' not in k}
            torch.nn.Module.load_state_dict(self, filtered_state_dict, strict=False)
        
        print("Model warm-started from pre-trained PokerPlayerNetV1")


class PPOAgent:
    def __init__(self, 
                 lr=1e-4, 
                 gamma=0.99, 
                 eps_clip=0.2, 
                 K_epochs=100, 
                 use_batchnorm=False,
                 device='cpu'):
        
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        
        # Initialize policy network
        self.policy = PPOPokerNet(use_batchnorm=use_batchnorm).to(device)
        self.old_policy = PPOPokerNet(use_batchnorm=use_batchnorm).to(device)
        
        # Copy parameters from policy to old_policy
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Initialize buffer
        self.buffer = RolloutBuffer()
        
        # Loss function for value network
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, game_state):
        """
        Select an action for the given game state
        Also stores the state, action, and log probability in the buffer
        """
        # Convert game state to tensors
        x_player_game, x_acted_history, x_to_act_history, _ = PokerPlayerNetV1.game_state_to_batch(game_state)
        
        # Convert to appropriate tensor shapes and move to device
        x_player_game = x_player_game.unsqueeze(0).to(self.device)
        x_acted_history = x_acted_history.unsqueeze(0).to(self.device)
        x_to_act_history = x_to_act_history.unsqueeze(0).to(self.device)
        
        # Get action using old policy
        action_info, raise_size, action_logprob, raise_size_logprob, state_value = self.old_policy.act(
            x_player_game, x_acted_history, x_to_act_history
        )
        
        action, action_probs = action_info
        
        # Store in buffer
        self.buffer.states_player_game.append(x_player_game)
        self.buffer.states_acted_history.append(x_acted_history)
        self.buffer.states_to_act_history.append(x_to_act_history)
        # Store the index of the action (enum.value) as a tensor
        action_idx = torch.tensor([action.value], dtype=torch.int64, device=self.device)
        self.buffer.actions.append(action_idx)
        self.buffer.raise_sizes.append(raise_size)
        self.buffer.logprobs_actions.append(action_logprob)
        self.buffer.logprobs_raise_sizes.append(raise_size_logprob)
        self.buffer.state_values.append(state_value)
        
        # Convert to appropriate PokerAction and raise amount
        poker_action = action
        raise_amount = raise_size.item() * game_state.pot_size

        if poker_action == Action.RAISE:
            raise_amount = max(game_state.min_allowed_bet, raise_amount - game_state.min_bet_to_continue)
        
        return poker_action, raise_amount, action_probs.detach().numpy()[0], raise_size.item()
    
    def store_reward(self, reward, is_terminal=False):
        """
        Store a reward in the buffer
        """
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(is_terminal)
    
    def update(self):
        """
        Update the policy using PPO
        """
        # Return early if buffer is empty
        if len(self.buffer.rewards) == 0:
            return 0.0
        
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Convert rewards to tensor and normalize
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        if len(rewards) > 1:  # Only normalize if we have more than one reward
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Check for buffer size mismatch
        if len(self.buffer.rewards) != len(self.buffer.state_values):
            # If the buffer sizes don't match, get the minimum length and trim all arrays
            min_length = min(len(self.buffer.rewards), len(self.buffer.state_values),
                            len(self.buffer.states_player_game), len(self.buffer.actions))
            
            # Trim all buffer data to the same length
            self.buffer.states_player_game = self.buffer.states_player_game[:min_length]
            self.buffer.states_acted_history = self.buffer.states_acted_history[:min_length]
            self.buffer.states_to_act_history = self.buffer.states_to_act_history[:min_length]
            self.buffer.actions = self.buffer.actions[:min_length]
            self.buffer.raise_sizes = self.buffer.raise_sizes[:min_length]
            self.buffer.logprobs_actions = self.buffer.logprobs_actions[:min_length]
            self.buffer.logprobs_raise_sizes = self.buffer.logprobs_raise_sizes[:min_length]
            self.buffer.state_values = self.buffer.state_values[:min_length]
            rewards = rewards[:min_length]
        
        # Convert buffer data to tensors
        old_states_player_game = torch.cat(self.buffer.states_player_game).to(self.device)
        old_states_acted_history = torch.cat(self.buffer.states_acted_history).to(self.device)
        old_states_to_act_history = torch.cat(self.buffer.states_to_act_history).to(self.device)
        old_actions = torch.cat(self.buffer.actions).to(self.device)
        old_raise_sizes = torch.cat(self.buffer.raise_sizes).to(self.device)
        old_logprobs_actions = torch.cat(self.buffer.logprobs_actions).to(self.device)
        old_logprobs_raise_sizes = torch.cat(self.buffer.logprobs_raise_sizes).to(self.device)
        old_state_values = torch.cat(self.buffer.state_values).squeeze(-1).to(self.device)
        
        # Ensure rewards and state_values have the same length
        if len(rewards) > len(old_state_values):
            rewards = rewards[:len(old_state_values)]
        elif len(old_state_values) > len(rewards):
            old_state_values = old_state_values[:len(rewards)]
            
        # Calculate advantages
        advantages = rewards - old_state_values.detach()
        
        # Optimize policy for K epochs
        total_loss = 0.0
        policy_losses = 0.0
        value_losses = 0.0
        entropy_losses = 0.0
        
        for _ in range(self.K_epochs):
            # Run forward pass through the policy network and get new action probabilities and values
            logprobs_actions, logprobs_raises, state_values, dist_entropy = self.policy.forward_and_evaluate(
                old_states_player_game,
                old_states_acted_history,
                old_states_to_act_history,
                old_actions,
                old_raise_sizes
            )
            
            # Check for NaN values and handle them
            if torch.isnan(logprobs_actions).any() or torch.isnan(logprobs_raises).any():
                print("Warning: NaN detected in log probabilities. Using fallback values.")
                # Replace NaN with safe values for log probabilities (very small negative value)
                logprobs_actions = torch.nan_to_num(logprobs_actions, nan=-10.0)
                logprobs_raises = torch.nan_to_num(logprobs_raises, nan=-10.0)
            
            # Check state values for NaN
            if torch.isnan(state_values).any():
                print("Warning: NaN detected in state values. Using fallback values.")
                state_values = torch.nan_to_num(state_values, nan=0.0)
            
            # Calculate policy ratio for actions in a numerically stable way
            # Add small epsilon to prevent division by zero or log of zero
            epsilon = 1e-8
            # Clamp the old log probs to prevent extreme values
            old_logprobs_actions_safe = old_logprobs_actions.detach().clamp(min=-20.0, max=20.0)
            old_logprobs_raises_safe = old_logprobs_raise_sizes.detach().clamp(min=-20.0, max=20.0)
            
            ratios_actions = torch.exp(logprobs_actions - old_logprobs_actions_safe)
            
            # Calculate policy ratio for raise sizes
            ratios_raises = torch.exp(logprobs_raises - old_logprobs_raises_safe)
            
            # Combined ratio (geometric mean) with safeguards
            # Ensure ratios are positive before taking square root
            ratios_actions = torch.clamp(ratios_actions, min=epsilon)
            ratios_raises = torch.clamp(ratios_raises, min=epsilon)
            
            combined_ratios = torch.sqrt(ratios_actions * ratios_raises)
            
            # Clip the ratio to prevent extreme changes
            clipped_ratios = torch.clamp(combined_ratios, 1-self.eps_clip, 1+self.eps_clip)
            
            # Calculate surrogate losses with advantage clipping for stability
            advantages_clipped = torch.clamp(advantages, min=-10.0, max=10.0)
            surr1 = combined_ratios * advantages_clipped
            surr2 = clipped_ratios * advantages_clipped
            
            # Compute policy loss (negative because we're minimizing)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value function loss
            value_loss = 0.5 * self.MseLoss(state_values.squeeze(-1), rewards)
            
            # Compute entropy bonus to encourage exploration
            entropy_loss = -0.01 * dist_entropy.mean()
            
            # Total loss (PPO objective function)
            loss = policy_loss + value_loss + entropy_loss
            
            # Check if loss is NaN
            if torch.isnan(loss).any():
                print("Warning: NaN detected in loss. Using component losses for update.")
                # If total loss is NaN, use the component that isn't NaN
                if not torch.isnan(policy_loss).any():
                    loss = policy_loss
                elif not torch.isnan(value_loss).any():
                    loss = value_loss
                else:
                    print("All loss components are NaN. Skipping update.")
                    continue
            
            # Track losses for each component
            policy_losses += policy_loss.item()
            value_losses += value_loss.item()
            entropy_losses += entropy_loss.item() 
            total_loss += loss.item()
            
            # Perform gradient step
            self.optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN gradients
            has_nan_grads = False
            for name, param in self.policy.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grads = True
                    print(f"Warning: NaN gradient detected in {name}")
            
            # Skip update if gradients contain NaN
            if has_nan_grads:
                print("Skipping parameter update due to NaN gradients")
                continue
                
            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
        
        # Copy new weights to old policy
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Clear buffer
        self.buffer.clear()
        
        # Return comprehensive loss information as a dictionary
        if self.K_epochs > 0:
            avg_loss = total_loss / self.K_epochs
            avg_policy_loss = policy_losses / self.K_epochs
            avg_value_loss = value_losses / self.K_epochs
            avg_entropy_loss = entropy_losses / self.K_epochs
        else:
            avg_loss = 0.0
            avg_policy_loss = 0.0
            avg_value_loss = 0.0
            avg_entropy_loss = 0.0
            
        return {
            'total': avg_loss,
            'policy': avg_policy_loss,
            'value': avg_value_loss,
            'entropy': avg_entropy_loss
        }
    
    def save(self, filepath, state_dict_only=False):
        """
        Save the model
        
        Args:
            filepath: Path to save the model
            state_dict_only: If True, only save the policy state_dict (not the optimizer)
                            This is useful for deployment or when using the model with other frameworks
        """
        if state_dict_only:
            # Save only the model state_dict
            torch.save(self.policy.state_dict(), filepath)
        else:
            # Save the complete checkpoint (policy and optimizer)
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, filepath)
    
    def save_state_dict(self, filepath):
        """
        Convenience method to save only the model state_dict
        """
        self.save(filepath, state_dict_only=True)
    
    def load(self, filepath):
        """
        Load the model - handles both checkpoint format and state_dict format
        """
        try:
            # Try loading as a dictionary (checkpoint format)
            checkpoint = torch.load(filepath)
            
            if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
                # It's a checkpoint with policy and optimizer
                self.policy.load_state_dict(checkpoint['policy_state_dict'])
                self.old_policy.load_state_dict(checkpoint['policy_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                # Assume it's a direct state dict
                try:
                    self.policy.load_state_dict(checkpoint)
                    self.old_policy.load_state_dict(checkpoint)
                    print("Loaded state_dict only (no optimizer state)")
                except:
                    self.warm_start(filepath)
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def warm_start(self, model_path):
        """
        Warm start from a pre-trained PokerPlayerNetV1 model
        """
        state_dict = torch.load(model_path)
        self.policy.warm_start_from_model(state_dict)
        torch.nn.Module.load_state_dict(self.old_policy, self.policy.state_dict())

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch import nn
from poker.agents.game_state import Stage
from poker.core.action import Action
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
import pandas as pd
import numpy as np

class GameStateTensorDataset(Dataset):
    def __init__(self, data_list): self.data_list = data_list
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

def collate_fn(batch):
    return tuple(torch.stack(tensors) for tensors in zip(*batch))
class FFN(nn.Module):
    def __init__(self, idim, hdim, odim, n_hidden, use_batchnorm=False):
        super().__init__()
        self.net = []

        if use_batchnorm: self.net.append(nn.BatchNorm1d(idim))
        self.net.append(nn.Linear(idim, hdim))
        for _ in range(n_hidden):
            if use_batchnorm: self.net.append(nn.BatchNorm1d(hdim))
            self.net.append(nn.ReLU())
            self.net.append(nn.Linear(hdim, hdim))
        if use_batchnorm: self.net.append(nn.BatchNorm1d(hdim))
        self.net.append(nn.ReLU())
        self.net.append(nn.Linear(hdim, odim))

        self.net = nn.ModuleList(self.net)

    def forward(self, x):
        assert len(x.shape) == 2 or len(x.shape) == 3
        if len(x.shape) == 2:
            for layer in self.net: x = layer(x)
            return x

        for layer in self.net:
            if type(layer) == nn.BatchNorm1d:
                x = layer(x.permute(0, 2, 1)).permute(0, 2, 1)
            else: x = layer(x)
        return x

class PokerPlayerNetV1(nn.Module):
    def __init__(self, use_batchnorm=False):
        super().__init__()

        self.stage_embed = nn.Embedding(4, 10)
        self.player_game_net = FFN(
            idim=6, # stage, stack_size, turn_to_act, hand_rank, c_hand_rank, min_bet
            hdim=10,
            odim=10,
            n_hidden=2,
            use_batchnorm=use_batchnorm,
        )
        # consider only players that are in the game
        self.acted_player_history_net = FFN(
            idim=3, # stack_size, turn_to_act, bet
            hdim=20,
            odim=20,
            n_hidden=3,
            use_batchnorm=use_batchnorm,
        )
        self.to_act_player_history_net = FFN(
            idim=3, # stack_size, turn_to_act, bet
            hdim=20,
            odim=20,
            n_hidden=3,
            use_batchnorm=use_batchnorm,
        )

        self.gather_net = FFN(
            idim=60,
            hdim=30,
            odim=4, # (fold,check/call,raise), raise size
            n_hidden=3,
            use_batchnorm=use_batchnorm,
        )
        self.aggressiveness_call = 1.0 # agressiveness just controls probabilities
        self.aggressiveness_raise = 1.0 # agressiveness just controls probabilities

    def forward(self, x_player_game, x_acted_history, x_to_act_history):
        # x_player_game : (B, 5)
        # x_acted_history : (B, 9, 3)
        # x_to_act_history : (B, 9, 3)
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

        out = self.gather_net(all_game_state)
        action_logits = out[:, :-1]
        raise_size = out[:, -1]

        return action_logits, raise_size

    def run_eval(self, dl, device=None):
        self.eval()
        val_loss = 0
        val_action_acc = 0
        val_raise_size_mse = 0
        with torch.no_grad():
            for batch in dl:
                batch = tuple(t.to(device) for t in batch)
                loss = self.compute_loss(batch)
                action_acc, raise_loss = self.eval_acc_batch(batch)
                val_loss += loss.item()
                val_action_acc += action_acc.item()
                val_raise_size_mse += raise_loss.item()
        avg_val_loss = val_loss / len(dl)
        avg_val_action_acc = val_action_acc / len(dl)
        avg_val_raise_mse = val_raise_size_mse / len(dl)
        return avg_val_loss, avg_val_action_acc, avg_val_raise_mse

    def get_actions(self, *args):
        action, raise_size = self(*args)
        return torch.argmax(action, dim=-1), raise_size

    @torch.no_grad()
    def eval_acc_batch(self, batch):
        x_player_game, x_acted_players, x_to_act_players, player_action = batch
        action, raise_size = self.get_actions(x_player_game, x_acted_players, x_to_act_players)
        
        real_player_action, real_raise_size = player_action[:, 0], player_action[:, 1]
        should_raise = real_player_action == 2
        
        acc = (action == real_player_action).to(float).mean()
        raise_size_mse = torch.tensor(0)
        if should_raise.any():
            raise_size_mse = nn.functional.mse_loss(raise_size[should_raise], real_raise_size[should_raise])

        return acc, raise_size_mse

    def compute_loss(self, batch):
        x_player_game, x_acted_players, x_to_act_players, player_action = batch
        action, raise_size = self(x_player_game, x_acted_players, x_to_act_players)
        real_player_action, real_raise_size = player_action[:, 0], player_action[:, 1]

        action_loss = nn.functional.cross_entropy(action, real_player_action.to(torch.int64)) 
        raise_loss = 0
        not_folded = real_player_action >= 2
        if torch.any(not_folded):
            raise_loss = nn.functional.mse_loss(raise_size[not_folded], real_raise_size[not_folded])
        loss = raise_loss + action_loss
        return loss

    def train_model(self, train_loader, val_loader=None, num_epochs=10, lr=1e-3, device=None, eval_steps=100):
        optimizer = optim.AdamW(self.parameters(), lr=lr)

        train_losses = []
        valid_lossess = []
        valid_metrics = {"action_acc":[], "raise_size_mse":[]}
        steps = []
        _step = 0
        train_loss = 0
        for epoch in tqdm(range(num_epochs)):
            for batch in train_loader:
                self.train()
                batch = tuple(t.to(device) for t in batch)
                loss = self.compute_loss(batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                if (_step + 1) % eval_steps == 0:
                    steps.append(_step)
                    avg_train_loss = train_loss / eval_steps
                    train_losses.append(avg_train_loss)
                    train_loss = 0
                    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
                    
                    if val_loader:
                        avg_val_loss, avg_val_action_acc, avg_val_raise_mse = self.run_eval(val_loader, device) 
                        print(f"Validation Loss: {avg_val_loss:.4f}, avg_val_action_acc: {avg_val_action_acc:.4f}, avg_val_raise_size_mse: {avg_val_raise_mse}")
                        valid_lossess.append(avg_val_loss)
                        valid_metrics["action_acc"].append(avg_val_action_acc)
                        valid_metrics["raise_size_mse"].append(avg_val_raise_mse)
                    
                _step += 1
        return pd.DataFrame(dict(train_loss=train_losses, valid_loss=valid_lossess, step=steps, **valid_metrics)).set_index("step")

    def eval_game_state(self, game_state):
        self.eval()
        batch = self.game_state_to_batch(game_state)
        with torch.no_grad():
            action_logits, raise_size = self(batch[0].unsqueeze(0), batch[1].unsqueeze(0), batch[2].unsqueeze(0))
        action_logits[:, 1] *= self.aggressiveness_call # call
        action_logits[:, 2] *= self.aggressiveness_raise # raise
        
        raise_size = raise_size[0]
        action_logits = action_logits[0]
        
        # making actions 
        raise_size = torch.minimum(raise_size, torch.tensor(game_state.my_player.stack_size / game_state.pot_size))
        if game_state.my_player.stack_size == 0: # can not raise
            action_logits[2] = -1e10
        return torch.softmax(action_logits, dim=-1), raise_size

    @staticmethod
    def game_state_to_batch(state) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        player_turn, other_player_turns = state.get_effective_turns()

        # Construct player / game state
        rel_stack_size = state.my_player.stack_size / state.pot_size
        turn_to_act = player_turn
        hand_rank = state.get_hand_strength()
        c_hand_rank = state.get_community_hand_strength()
        rel_min_bet = state.min_bet_to_continue / state.pot_size
        x_player_game = torch.Tensor([state.stage.value, rel_stack_size, turn_to_act, hand_rank, c_hand_rank, rel_min_bet])

        # separate acted and non-acted players
        acted_players = [(player, turn) for player, turn in
                    zip(state.other_players, other_player_turns)
                    if turn is not None and turn < player_turn]
        to_act_players = [(player, turn) for player, turn in
                    zip(state.other_players, other_player_turns)
                    if turn is not None and turn > player_turn]

        # Construct acted players state
        # note that order does not matter right now 
        acted_players_data = [(
                p.stack_size / state.pot_size, # rel stack size
                t, #t, # turn to act
                0 if not p.history else p.history[-1][1] / state.pot_size, # raise_size TODO(roberto): consider using something better NOTE andri change
            ) for p, t in acted_players
        ]
        # Construct to-act players state, only if not in preflop
        to_act_players_data = [(
                p.stack_size / state.pot_size, # rel stack size
                t, #, # turn to act
                p.history[-1][1] / state.pot_size if p.history else 0, # raise_size TODO(roberto): consider using something better
            ) for p, t in to_act_players
        ] if state.stage != Stage.PREFLOP else []

        x_acted_players = torch.Tensor(acted_players_data)
        x_to_act_players = torch.Tensor(to_act_players_data)

        # desired_shape = (9, (x_acted_players.shape[1] if len(x_acted_players.shape) >= 2 else x_to_act_players.shape[1]))
        desired_shape = (9, 3)

        # pad with 0s
        pad = torch.zeros(desired_shape[0]-x_acted_players.shape[0], desired_shape[1])
        x_acted_players = torch.concat((x_acted_players, pad) , dim=0)

        # pad with 0s
        pad = torch.zeros(desired_shape[0]-x_to_act_players.shape[0], desired_shape[1])
        x_to_act_players = torch.concat((x_to_act_players, pad) , dim=0)

        if state.my_player_action:
            # only for supervised training
            raise_size = state.my_player_action[1]
            if state.my_player_action[0].value == 1: raise_size = state.min_bet_to_continue # if call/check
            raise_size = 0 if raise_size is None else raise_size / state.pot_size
            player_action = torch.Tensor([state.my_player_action[0].value, raise_size])
        else:
            player_action = torch.Tensor([1, 0]) # not used ?

        return x_player_game, x_acted_players, x_to_act_players, player_action

    @staticmethod
    def get_game_state_data_loader(states: List, batch_size=16, shuffle=True) -> DataLoader:
        tensor_states = []
        for game_state in tqdm(states):
            tensor_states.append(PokerPlayerNetV1.game_state_to_batch(game_state))
        dataset = GameStateTensorDataset(tensor_states)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return data_loader
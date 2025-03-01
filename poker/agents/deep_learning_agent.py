import torch
from torch import nn
from agents.game_state import GameState, Stage
from core.action import Action
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim

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

        self.player_game_net = FFN(
            idim=5, # stack_size, turn_to_act, hand_rank, c_hand_rank, min_bet
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
            idim=50,
            hdim=30,
            odim=2, # (fold / no_fold), raise size
            n_hidden=3,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x_player_game, x_acted_history, x_to_act_history):
        # x_player_game : (B, 5)
        # x_acted_history : (B, 9, 3)
        # x_to_act_history : (B, 9, 3)
        player_game_state = self.player_game_net(x_player_game)
        acted_player_history_state = self.acted_player_history_net(x_acted_history)
        to_act_player_history_state = self.to_act_player_history_net(x_to_act_history)

        acted_player_history_state = acted_player_history_state.sum(dim=1)
        to_act_player_history_state = to_act_player_history_state.sum(dim=1)

        all_game_state = torch.concat([
            player_game_state,
            acted_player_history_state,
            to_act_player_history_state,
        ], dim=-1)

        out = self.gather_net(all_game_state)
        fold_logits = out[:, 0]
        raise_size = out[:, 1]

        return torch.sigmoid(fold_logits), raise_size

    def compute_loss(self, batch):
        x_player_game, x_acted_players, x_to_act_players, player_action = batch
        fold_logits, raise_size = self(x_player_game, x_acted_players, x_to_act_players)
        real_fold_p, real_raise_size = player_action[:, 0], player_action[:, 1]
        
        fold_loss = nn.functional.binary_cross_entropy(fold_logits, real_fold_p)
        
        not_fold = ~real_fold_p.to(torch.bool)
        raise_loss = nn.functional.mse_loss(raise_size[not_fold], real_raise_size[not_fold])

        loss = fold_loss + raise_loss
        return loss

    def train_model(self, train_loader, val_loader=None, num_epochs=10, lr=1e-3, device=None, eval_steps=100):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        optimizer = optim.AdamW(self.parameters(), lr=lr)

        valid_lossess = []
        for epoch in tqdm(range(num_epochs)):
            total_loss = 0
            step = 0
            for batch in train_loader:
                self.train()
                batch = tuple(t.to(device) for t in batch)
                loss = self.compute_loss(batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
                if step % eval_steps == 0:
                    avg_train_loss = total_loss / len(train_loader)
                    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
                    
                    if val_loader:
                        self.eval()
                        val_loss = 0
                        with torch.no_grad():
                            for batch in val_loader:
                                batch = tuple(t.to(device) for t in batch)
                                loss = self.compute_loss(loss)
                                val_loss += loss.item()
                        avg_val_loss = val_loss / len(val_loader)
                        print(f"Validation Loss: {avg_val_loss:.4f}")
                        valid_lossess.append(avg_val_loss)
            step += 1


    @staticmethod
    def game_state_to_batch(state: GameState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        player_turn, other_player_turns = state.get_effective_turns()
        
        # Construct player / game state
        rel_stack_size = state.my_player.stack_size / state.pot_size
        turn_to_act = player_turn
        hand_rank = state.hand_strength
        c_hand_rank = state.community_hand_strenght
        rel_min_bet = state.min_bet_to_continue / state.pot_size
        x_player_game = torch.Tensor([rel_stack_size, turn_to_act, hand_rank, c_hand_rank, rel_min_bet])

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
                t, # turn to act
                p.history[-1][1] / state.pot_size, # raise_size TODO(roberto): consider using something better
            ) for p, t in acted_players
        ]
        # Construct to-act players state, only if not in preflop
        to_act_players_data = [(
                p.stack_size / state.pot_size, # rel stack size
                t, # turn to act
                p.history[-1][1] / state.pot_size, # raise_size TODO(roberto): consider using something better
            ) for p, t in to_act_players
        ] if state.stage != Stage.PREFLOP else []

        x_acted_players = torch.Tensor(acted_players_data)
        x_to_act_players = torch.Tensor(to_act_players_data)

        desired_shape = (9, (x_acted_players.shape[1] if len(x_acted_players.shape) >= 2 else x_to_act_players.shape[1]))
        
        # pad with 0s
        pad = torch.zeros(desired_shape[0]-x_acted_players.shape[0], desired_shape[1])
        x_acted_players = torch.concat((x_acted_players, pad) , dim=0)

        # pad with 0s
        pad = torch.zeros(desired_shape[0]-x_to_act_players.shape[0], desired_shape[1])
        x_to_act_players = torch.concat((x_to_act_players, pad) , dim=0)

        raise_size = state.my_player_action[1]
        raise_size = 0 if raise_size is None else raise_size / state.pot_size
        player_action = torch.Tensor([float(state.my_player_action[0] == Action.FOLD), raise_size])
        return x_player_game, x_acted_players, x_to_act_players, player_action

    @staticmethod
    def get_game_state_data_loader(states: List[GameState], batch_size=16, shuffle=True) -> DataLoader:
        tensor_states = []
        for game_state in tqdm(states):
            tensor_states.append(PokerPlayerNetV1.game_state_to_batch(game_state))
        dataset = GameStateTensorDataset(tensor_states)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return data_loader

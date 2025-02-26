import torch
from torch import nn
from game_state import GameState, Stage
from typing import Tuple

class FFN(nn.Module):
    def __init__(self, idim, hdim, odim, n_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(),
            nn.Linear(idim, hdim),
            *[
                nn.BatchNorm1d(),
                nn.ReLU(),
                nn.Linear(hdim, hdim)
            ],
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Linear(hdim, odim),
        )

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
        raise_size = torch.sigmoid(out[:, 1])

        # if raise size <= 10%, just call
        raise_size = raise_size.clamp_min(0.1) * (raise_size >= 0.1).float()

        return fold_logits, raise_size
    
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
        x_acted_players = torch.Tensor(acted_players_data)
        # pad with 0s
        pad = torch.zeros(9-x_acted_players.shape[0], x_acted_players.shape[1])
        x_acted_players = torch.concat(x_acted_players, pad , dim=0)

        # Construct to-act players state, only if not in preflop
        to_act_players_data = [(
                p.stack_size / state.pot_size, # rel stack size
                t, # turn to act
                p.history[-1][1] / state.pot_size, # raise_size TODO(roberto): consider using something better
            ) for p, t in to_act_players
        ] if state.stage != Stage.PREFLOP else []
        x_to_act_players = torch.Tensor(to_act_players_data)
        # pad with 0s
        pad = torch.zeros(9-x_to_act_players.shape[0], x_to_act_players.shape[1])
        x_to_act_players = torch.concat(x_to_act_players, pad , dim=0)

        return x_player_game, x_acted_players, x_to_act_players

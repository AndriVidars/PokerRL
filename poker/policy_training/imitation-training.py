import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))
from poker.agents.game_state import *
from poker.core.card import *
from poker.agents.deep_learning_agent import PokerPlayerNetV1
from poker.agents.game_state import GameStateRetriever

import random
import numpy as np
import torch
import itertools
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    retriever = GameStateRetriever('./pluribus', verbose=False)
    all_game_states = []
    for player in ['MrBlue', 'MrBlonde', 'MrWhite', 'MrPink', 'MrBrown', 'Pluribus']:
        game_states = retriever.get_player_game_states(player)
        all_game_states.append(game_states)
        print(f"Found {len(game_states)} GameStates for {player}")

    game_states = list(itertools.chain.from_iterable(all_game_states))
    random.shuffle(game_states)
    train_game_states = game_states[:-1000]
    valid_game_states = game_states[-1000:]

    train_dl = PokerPlayerNetV1.get_game_state_data_loader(train_game_states)
    valid_dl = PokerPlayerNetV1.get_game_state_data_loader(valid_game_states)

    my_agent = PokerPlayerNetV1(use_batchnorm=False, use_mse_loss_for_raise=True)
    train_df = my_agent.train_model(train_dl, valid_dl, num_epochs=5, lr=1e-4, device=None, eval_steps=100)
    pd.to_pickle(train_df, "imitation_training_train_df.pkl")
    torch.save(my_agent.state_dict(), "imitation.st")


if __name__ == "__main__":
    main()
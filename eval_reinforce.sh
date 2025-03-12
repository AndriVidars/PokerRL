#!/bin/bash

# eval against imitation(frozen) set --primary_state_dict by some checkpoint for reinforced deep player

#python -m poker.play  --num_H_players 2 --num_D_validation_players 2 --primary_state_dict poker/policy_training/checkpoints/lr_1e-05_batch_size_16_max_grad_norm_0.25_num_games_4000_replay_buffer_cap_1000_D2_DF2_3000_69.st

# play against H, trained agaisnt I
python -m poker.play  --num_H_players 2 --primary_state_dict poker/policy_training/checkpoints/lr_1e-05_batch_size_16_max_grad_norm_0.25_num_games_4000_replay_buffer_cap_1000_D2_DF2_3000_69.st

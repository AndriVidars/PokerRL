#!/bin/bash

python -m poker.policy_training.reinforce --num_H_players 2 --num_D_players 2 --num_games 2000 --lr 1e-5 --max_grad_norm 0.2
python -m poker.policy_training.reinforce --num_H_players 2 --num_D_players 2 --num_games 2000 --lr 5e-5 --max_grad_norm 0.5 --replay_buffer_cap 1000
python -m poker.policy_training.reinforce --num_H_players 2 --num_D_players 2 --num_games 2000 --lr 5e-5 --max_grad_norm 0.25 --replay_buffer_cap 1000
python -m poker.policy_training.reinforce --num_H_players 2 --num_D_players 2 --num_games 4000 --lr 5e-5 --max_grad_norm 0.5

python -m poker.policy_training.reinforce --num_R_players 2 --num_D_players 2 --num_games 2000 --lr 1e-5 --max_grad_norm 0.25
python -m poker.policy_training.reinforce --num_R_players 2 --num_D_players 2 --num_games 2000 --lr 1e-4 --max_grad_norm 0.5 --replay_buffer_cap 5000
python -m poker.policy_training.reinforce --num_R_players 2 --num_D_players 2 --num_games 2000 --lr 5e-5 --max_grad_norm 0.25 --replay_buffer_cap 1000
python -m poker.policy_training.reinforce --num_R_players 2 --num_D_players 2 --num_games 4000 --lr 5e-5 --max_grad_norm 0.5

python -m poker.policy_training.reinforce --num_DF_players 2 --num_D_players 2 --num_games 2000 --lr 5e-5 --max_grad_norm 0.2
python -m poker.policy_training.reinforce --num_DF_players 2 --num_D_players 2 --num_games 4000

python -m poker.policy_training.reinforce --num_H_players 1 --num_DF_players 1 --num_D_players 2 --num_games 4000
python -m poker.policy_training.reinforce --num_R_players 1 --num_DF_players 1 --num_D_players 2 --num_games 4000

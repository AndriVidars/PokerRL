#!/bin/bash

python -m poker.policy_training.reinforce --num_H_players 2 --num_D_players 2 --num_games 3000
python -m poker.policy_training.reinforce --num_H_players 2 --num_D_players 2 --num_games 3000 --lr 1e-5 --max_grad_norm 0.1
python -m poker.policy_training.reinforce --num_H_players 2 --num_D_players 2 --num_games 3000 --lr 1e-5 --max_grad_norm 1
python -m poker.policy_training.reinforce --num_H_players 2 --num_D_players 2 --num_games 3000 --lr 5e-5 --max_grad_norm 0.25
python -m poker.policy_training.reinforce --num_H_players 2 --num_D_players 2 --num_games 3000 --lr 5e-6 --max_grad_norm 0.1
python -m poker.policy_training.reinforce --num_H_players 2 --num_D_players 2 --num_games 3000 --lr 5e-6 --max_grad_norm 0.5


python -m poker.policy_training.reinforce --num_H_players 2 --num_D_players 2 --num_games 3000 --lr 1e-5 --max_grad_norm 0.1 --replay_buffer_cap 500
python -m poker.policy_training.reinforce --num_H_players 2 --num_D_players 2 --num_games 3000 --lr 1e-5 --max_grad_norm 0.25 --replay_buffer_cap 500
python -m poker.policy_training.reinforce --num_H_players 2 --num_D_players 2 --num_games 3000 --lr 1e-5 --max_grad_norm 0.05 --replay_buffer_cap 500


python -m poker.policy_training.reinforce --num_H_players 0 --num_R_players 2 --num_D_players 2 --num_games 3000
python -m poker.policy_training.reinforce --num_H_players 0 --num_R_players 2 --num_D_players 2 --num_games 3000 --lr 1e-5 --max_grad_norm 0.1
python -m poker.policy_training.reinforce --num_H_players 0 --num_R_players 2 --num_D_players 2 --num_games 3000 --lr 1e-5 --max_grad_norm 0.25
python -m poker.policy_training.reinforce --num_H_players 0 --num_R_players 2 --num_D_players 2 --num_games 3000 --lr 1e-5 --max_grad_norm 0.05
python -m poker.policy_training.reinforce --num_H_players 0 --num_R_players 2 --num_D_players 2 --num_games 3000 --lr 5e-6 --max_grad_norm 0.1


python -m poker.policy_training.reinforce --num_H_players 0 --num_R_players 2 --num_D_players 2 --num_games 3000 --lr 1e-5 --max_grad_norm 0.1 --replay_buffer_cap 500
python -m poker.policy_training.reinforce --num_H_players 0 --num_R_players 2 --num_D_players 2 --num_games 3000 --lr 1e-5 --max_grad_norm 0.25 --replay_buffer_cap 500
python -m poker.policy_training.reinforce --num_H_players 0 --num_R_players 2 --num_D_players 2 --num_games 3000 --lr 1e-5 --max_grad_norm 0.05 --replay_buffer_cap 500


python -m poker.policy_training.reinforce --num_DF_players 2 --num_H_players 0 --num_D_players 2 --num_games 4000 --lr 1e-5 --max_grad_norm 0.1 
python -m poker.policy_training.reinforce --num_DF_players 2 --num_H_players 0 --num_D_players 2 --num_games 4000 --replay_buffer_cap 500

python -m poker.policy_training.reinforce --num_H_players 1 --num_DF_players 1 --num_D_players 2 --num_games 4000
python -m poker.policy_training.reinforce --num_H_players 0 --num_R_players 1 --num_DF_players 1 --num_D_players 2 --num_games 4000

python -m poker.policy_training.reinforce --num_H_players 1 --num_DF_players 1 --num_D_players 2 --num_games 4000 --replay_buffer_cap 500
python -m poker.policy_training.reinforce --num_H_players 0 --num_R_players 1 --num_DF_players 1 --num_D_players 2 --num_games 4000 --lr 1e-5 --replay_buffer_cap 500

# todo add a single or two training runs that run for 10k or 20k games-much longer that is
#!/bin/bash


# train against itself
# R2 training
python -m poker.policy_training.reinforce --num_DF_players 2 --num_H_players 0 --num_D_players 2 --num_games 4000 --policy_state_dict R1.st --frozen_state_dict R1.st
python -m poker.policy_training.reinforce --num_DF_players 3 --num_H_players 0 --num_D_players 1 --num_games 4000 --policy_state_dict R1.st --frozen_state_dict R1.st



# R1 training
#python -m poker.policy_training.reinforce --num_DF_players 2 --num_H_players 0 --num_D_players 2 --num_games 4000
#python -m poker.policy_training.reinforce --num_DF_players 2 --num_H_players 0 --num_D_players 2 --num_games 4000 --lr 5e-6 --max_grad_norm 0.1 
#python -m poker.policy_training.reinforce --num_DF_players 2 --num_H_players 0 --num_D_players 2 --num_games 4000 --lr 5e-5 --max_grad_norm 1


python -m poker.policy_training.reinforce --num_DF_players 3 --num_H_players 0 --num_D_players 1 --num_games 4000
python -m poker.policy_training.reinforce --num_DF_players 3 --num_H_players 0 --num_D_players 1 --num_games 4000 --lr 5e-6 --max_grad_norm 0.1 
python -m poker.policy_training.reinforce --num_DF_players 3 --num_H_players 0 --num_D_players 1 --num_games 4000 --lr 5e-5 --max_grad_norm 1


# other R1 setups
python -m poker.policy_training.reinforce --num_H_players 1 --num_DF_players 1 --num_D_players 2 --num_games 4000
python -m poker.policy_training.reinforce --num_H_players 0 --num_R_players 1 --num_DF_players 1 --num_D_players 2 --num_games 4000

python -m poker.policy_training.reinforce --num_H_players 2 --num_D_players 2 --num_games 3000
python -m poker.policy_training.reinforce --num_H_players 2 --num_D_players 2 --num_games 3000 --lr 5e-6 --max_grad_norm 0.1

python -m poker.policy_training.reinforce --num_H_players 0 --num_R_players 2 --num_D_players 2 --num_games 3000 
python -m poker.policy_training.reinforce --num_H_players 0 --num_R_players 2 --num_D_players 2 --num_games 3000 --lr 5e-6 --max_grad_norm 0.1 

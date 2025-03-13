#!/bin/bash


# train against itself
# R2 training
python -m poker.policy_training.reinforce --validation_players R2.st 3 --num_D_players 1 --num_games 4000
python -m poker.policy_training.reinforce --validation_players R2.st 2 --num_games 4000


python -m poker.policy_training.reinforce --validation_players R2.st 3 --num_D_players 1 --num_games 4000 --lr 5e-6 --max_grad_norm 0.1 
python -m poker.policy_training.reinforce --validation_players R2.st 2 --num_games 4000  --lr 5e-5 --max_grad_norm 1

# with heuristic and random
python -m poker.policy_training.reinforce --validation_players R2.st 1 --num_D_players 1 --num_H_players 2 --num_games 4000
python -m poker.policy_training.reinforce --validation_players R2.st 1 --num_D_players 1 --num_H_players 1 --num_R_players 1 --num_games 4000



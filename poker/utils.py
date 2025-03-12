from poker.player_deep_agent import PlayerDeepAgent
from poker.ppo_player import PlayerPPO
import os
import logging

def init_players(player_type_dict, agent_model=None, ppo_agent=None, start_stack=400, primary_ppo_idx=0, primary_deep_idx=0):
    players = []
    player_str_list = []
    
    for i, (player_class, count) in enumerate(player_type_dict.items()):
        player_str_list.append(f"{player_class.__name__}_{count}")
        for j in range(count):
            player_name = f"{player_class.__name__} {i+1}_{j+1}"
            
            if player_class == PlayerDeepAgent:
                # Set primary flag for the specified deep agent index
                is_primary = (j == primary_deep_idx)
                player = player_class(player_name, agent_model, start_stack, primary=is_primary)
            elif player_class == PlayerPPO:
                if ppo_agent is None:
                    raise ValueError("PPO agent must be provided to create PlayerPPO instances")
                
                # Set primary flag for the specified PPO player index
                is_primary = (j == primary_ppo_idx)
                player = player_class(player_name, ppo_agent, start_stack, primary=is_primary)
            else:
                player = player_class(player_name, start_stack)
                
            players.append(player)
    
    return players, "_".join(player_str_list)

def init_logging(filename):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

from poker.player_deep_agent import PlayerDeepAgent
import os
import logging

def init_players(player_type_dict, agent_model=None, start_stack=400):
    players = []
    player_str_list = []
    
    for i, (player_class, count) in enumerate(player_type_dict.items()):
        player_str_list.append(f"{player_class.__name__}_{count}")
        for j in range(count):
            player_name = f"{player_class.__name__} {i+1}_{j+1}"
            player = player_class(player_name, agent_model, start_stack) if player_class == PlayerDeepAgent else player_class(player_name, start_stack)
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

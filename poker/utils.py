from poker.player_deep_agent import PlayerDeepAgent
from poker.ppo_player import PlayerPPO
import logging

def init_players(player_type_dict, start_stack=400):
    players = []
    n = len(player_type_dict.keys())
    for i, ((player_class, primary, name_prefix), (count, model)) in enumerate(player_type_dict.items()):
        for j in range(count):
            player_name = f"{name_prefix}_{player_class.__name__}_{i*n+j}"
            if player_class == PlayerDeepAgent:
                if primary:
                    player = PlayerDeepAgent(player_name, model, start_stack, primary=True)
                else:
                    player = PlayerDeepAgent(player_name, model, start_stack, primary=False)
            elif player_class == PlayerPPO:
                if primary:
                    player = PlayerPPO(player_name, model, start_stack, primary=True)
                else:
                    player = PlayerPPO(player_name, model, start_stack, primary=False)
            else:
                player = player_class(player_name, start_stack)

            players.append(player)
            
    return players

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

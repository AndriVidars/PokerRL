from poker.player_deep_agent import PlayerDeepAgent
import logging

def init_players(player_type_dict, agent_model_primary=None, agent_model_secondary=None, start_stack=400):
    players = []
    n = len(player_type_dict.keys())
    for i, (player_class, (count, primary)) in enumerate(player_type_dict.items()):
        for j in range(count):
            player_name = f"{player_class.__name__}{'_Primary_' if primary else '_'}{i*n+j}"
            if player_class == PlayerDeepAgent:
                if primary:
                    player = PlayerDeepAgent(player_name, agent_model_primary, start_stack, primary=True)
                else:
                    player = PlayerDeepAgent(player_name, agent_model_secondary, start_stack, primary=False)
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

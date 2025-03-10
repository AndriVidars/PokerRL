from poker.player_deep_agent import PlayerDeepAgent

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
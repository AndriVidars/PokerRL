

class Pot:
    def __init__(self):
        self.total_amount = 0
        self.contributions = {}  # {Player: Amount Contributed}
        self.eligible_players = set()  # Players who can win this pot

    def add_contribution(self, player, amount):
        if player in self.contributions:
            self.contributions[player] += amount
        else:
            self.contributions[player] = amount
        self.total_amount += amount
        self.eligible_players.add(player)

    def split_pot(self, all_in_player):
        all_in_amount = self.contributions[all_in_player]
        
        side_pot = Pot()
        for player, contributed in list(self.contributions.items()):
            if contributed > all_in_amount:
                excess = contributed - all_in_amount
                self.contributions[player] -= excess  
                side_pot.add_contribution(player, excess) 

        return side_pot if side_pot.total_amount > 0 else None

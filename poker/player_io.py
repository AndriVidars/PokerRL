from poker.core.player import Player
from poker.core.action import Action

class PlayerIO(Player):
    def act(self):
        pots = self.game.pots
        call_amt = 0
        print("Current Pot State:")
        for p in pots:
            print(f"Contributions: {p.contributions}, Eligible: {p.eligible_players}")
            max_p = max(p.contributions.values())
            if self not in p.contributions.keys():
                call_amt += max_p
            else:
                call_amt += max_p - p.contributions[self]
        
        print(f"Call Amount: {call_amt}")
        print(f"\nIt is {self}'s  turn - Hand: {self.hand} - Current Stack: {self.stack}")
        action = Action(int(input("Enter action (0=Fold, 1=Check/Call, 2=Raise):")))
        print("\n")
        
        match action:
            case Action.FOLD:
                self.handle_fold()
            case Action.CHECK_CALL:
                self.handle_check_call()
            case Action.RAISE:
                self.raise_select()

        return action
    
    def raise_select(self):
        amt = int(input(f"Enter raise amount: "))
        assert amt <= self.stack
        self.handle_raise(amt)

from poker.core.player import Player
from poker.core.action import Action

class PlayerIO(Player):
    def _act(self):
        pots = self.game.pots
        call_amt = 0
        print("Current Pot State:")
        for p in pots:
            print(f"Contributions: {p.contributions}, Eligible: {p.eligible_players}")
            if p.contributions:  # Check if contributions is not empty
                max_p = max(p.contributions.values())
                if self not in p.contributions.keys():
                    call_amt += max_p
                else:
                    call_amt += max_p - p.contributions[self]
        
        print(f"Call Amount: {call_amt}")
        print(f"\nIt is {self}'s turn - Hand: {self.hand} - Current Stack: {self.stack}")
        
        try:
            action = Action(int(input("Enter action (0=Fold, 1=Check/Call, 2=Raise): ")))
            print("\n")
            
            match action:
                case Action.FOLD:
                    self.handle_fold()
                case Action.CHECK_CALL:
                    self.handle_check_call()
                case Action.RAISE:
                    self.raise_select()
            
            return action
            
        except (EOFError, KeyboardInterrupt, ValueError):
            print("\nGame terminated or invalid input. Folding automatically.")
            self.handle_fold()
            return Action.FOLD
    
    def raise_select(self):
        min_raise = max(self.game.min_bet, 1)
        while True:
            try:
                amt = int(input(f"Enter raise amount (min {min_raise}, max {self.stack}): "))
                if min_raise <= amt <= self.stack:
                    self.handle_raise(amt)
                    return
                else:
                    print(f"Invalid amount. Must be between {min_raise} and {self.stack}.")
            except (ValueError, EOFError, KeyboardInterrupt):
                print("Invalid input. Using minimum raise.")
                self.handle_raise(min_raise)
                return

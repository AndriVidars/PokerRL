from poker.core.player import Player
from poker.core.action import Action
from poker.core.card import Rank
from poker.core.gamestage import Stage
import poker.core.hand_evaluator as hand_eval
import random

class PlayerHeuristic(Player):
    def _act(self):
        action, raise_amt = None, 0
        match self.game.current_stage:
            case Stage.PREFLOP:
                action, raise_amt = self.act_preflop()
            case Stage.FLOP:
                action, raise_amt = self.act_flop()
            case Stage.TURN:
                action, raise_amt = self.act_turn()
            case Stage.RIVER:
                action, raise_amt = self.act_river()
        
        match action:
            case Action.FOLD:
                self.handle_fold()
            case Action.CHECK_CALL:
                self.handle_check_call()
            case Action.RAISE:
                if len(self.game.active_players) == 1:
                    self.handle_check_call() # revert to check/call, raise is not an option
                    action = Action.CHECK_CALL
                else:
                    self.handle_raise(raise_amt)

        return action
    
    def act_preflop(self):
        n_active_players = len(self.game.active_players)
        hand_rank, _ = hand_eval.evaluate_hand(self.hand)
        max_suit_count = hand_eval.suit_counter(self.hand) # position wrt flush possibility - only taken into account preflop
        high_card = hand_eval.high_card(self.hand)
        call_amt_due = self.get_call_amt_due()
        max_raise = self.stack - call_amt_due

        # always check if stack is less than big blind
        if self.stack <= self.game.big_amount:
            return Action.CHECK_CALL, 0

        # pair
        if hand_rank == 2:
            pair_rank = self.hand[0].rank
            if call_amt_due >= self.stack or max_raise < self.game.min_bet:
                # will all-in if check
                if pair_rank in [Rank.ACE, Rank.KING, Rank.QUEEN]:
                    return Action.CHECK_CALL, 0
                else:
                    return Action.FOLD, 0

            # raise 50% of the time when pair preflop - to keep some level of unpredictability
            elif random.random() < 0.7:
                max_bet_size = min(self.game.min_bet*5, max_raise)
                steps = max_bet_size // self.game.min_bet
                if steps  == 1:
                    return Action.RAISE, self.game.min_bet

                bet_sizes = [self.game.min_bet*i for i in range(1, steps+1)]
                lo = [Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SEVEN, Rank.EIGHT]
                med = [Rank.NINE, Rank.TEN, Rank.JACK]
                hi = [Rank.QUEEN, Rank.KING, Rank.ACE]

                if pair_rank in lo:
                    probabilities = [0.6 if i == self.game.min_bet else 0.4/(len(bet_sizes)-1) for i in bet_sizes]
                elif pair_rank in med:
                    probabilities = [0.4 if i == self.game.min_bet else 0.6/(len(bet_sizes)-1) for i in bet_sizes]
                else:
                    if n_active_players == 2 and random.random() < 0.05:
                        return Action.RAISE, max_raise             
                    probabilities = [1/len(bet_sizes) for i in bet_sizes]
        
                chosen_bet = random.choices(bet_sizes, probabilities)[0]
                return Action.RAISE, chosen_bet
            
            return Action.CHECK_CALL, 0

        elif max_suit_count == 2 or high_card >= 13:
            if call_amt_due >= self.stack or max_raise < self.game.min_bet:
                return Action.CHECK_CALL, 0
            
            if call_amt_due == 0 and random.random() < 0.2:
                return Action.RAISE, self.game.min_bet
            
        if call_amt_due == 0:
            return Action.CHECK_CALL, 0
        
        return Action.FOLD, 0
    
    def act_flop(self):
        hand_ = self.hand + self.game.community_cards
        hand_rank, _ = hand_eval.evaluate_hand(hand_)
        call_amt_due = self.get_call_amt_due()
        max_raise = self.stack - call_amt_due
        max_raise_ratio = max_raise // self.game.min_bet

        # straight or better
        if hand_rank >= 5:
            if call_amt_due >= self.stack or max_raise < self.game.min_bet:
                return Action.CHECK_CALL, 0
            
            elif random.random() < 0.75:  
                bet_sizes = [self.game.min_bet * i for i in range(1, max_raise_ratio)] + [max_raise]
                prob_all_in = 0.2  
                bet_probs = [(1-prob_all_in)/(len(bet_sizes)-1) for _ in range(len(bet_sizes)-1)] + [prob_all_in]
                chosen_bet = random.choices(bet_sizes, bet_probs)[0]
                return Action.RAISE, chosen_bet
            
            return Action.CHECK_CALL, 0
        
        # two pair or better
        if hand_rank >= 3:
            if call_amt_due >= self.stack or max_raise < self.game.min_bet:
                return Action.CHECK_CALL, 0
            
            if random.random() < 0.6:
                max_bet_size = min(max_raise, 8*self.game.min_bet)
                max_bet_size_ratio = max_bet_size // self.game.min_bet
                bet_sizes = [self.game.min_bet * i for i in range(1, max_bet_size_ratio)]
                if bet_sizes:
                    chosen_bet = random.choice(bet_sizes)
                    return Action.RAISE, chosen_bet
            
            return Action.CHECK_CALL, 0

        # pair
        if hand_rank == 2:
            if call_amt_due > 0.3 * self.stack:
                return Action.FOLD, 0
            if hand_eval.high_pair(hand_) >= 11 and random.random() < 0.25:
                min_raise = self.game.min_bet
                max_raise_size = min(1.5 * self.game.min_bet, max_raise)
                if max_raise_size >= min_raise:
                    return Action.RAISE, random.randint(min_raise, max_raise_size)
                return Action.CHECK_CALL, 0

        if call_amt_due == 0:
            return Action.CHECK_CALL, 0
        
        return Action.FOLD, 0

    def act_turn(self):
        hand_ = self.hand + self.game.community_cards
        hand_rank, _ = hand_eval.evaluate_hand(hand_)
        call_amt_due = self.get_call_amt_due()
        max_raise = self.stack - call_amt_due

        if hand_rank >= 5:
            if call_amt_due >= self.stack or max_raise < self.game.min_bet:
                return Action.CHECK_CALL, 0
            return (Action.RAISE, min(max_raise, 5 * self.game.min_bet)) if random.random() < 0.8 else (Action.CHECK_CALL, 0)

        if hand_rank >= 3:
            if call_amt_due >= self.stack or max_raise < self.game.min_bet:
                return Action.CHECK_CALL, 0
            return (Action.RAISE, min(max_raise, 3 * self.game.min_bet)) if random.random() < 0.65 else (Action.CHECK_CALL, 0)

        if hand_rank == 2:
            if hand_eval.high_pair(hand_) >= 12 and call_amt_due <= 0.35 * self.stack:
                if max_raise > self.game.min_bet and random.random() < 0.25:
                    return Action.RAISE, random.randint(self.game.min_bet, max(int(max_raise*0.25), self.game.min_bet))
                return Action.CHECK_CALL, 0
        
        if call_amt_due == 0:
            return Action.CHECK_CALL, 0
        
        return Action.FOLD, 0

    def act_river(self):
        hand_ = self.hand + self.game.community_cards
        hand_rank, _ = hand_eval.evaluate_hand(hand_)
        call_amt_due = self.get_call_amt_due()
        max_raise = self.stack - call_amt_due

        if hand_rank >= 5:
            if call_amt_due >= self.stack or max_raise < self.game.min_bet:
                return Action.CHECK_CALL, 0
            return (Action.RAISE, min(max_raise, 6 * self.game.min_bet)) if random.random() < 0.9 else (Action.CHECK_CALL, 0)

        if hand_rank >= 3:
            if call_amt_due >= self.stack or max_raise < self.game.min_bet:
                return Action.CHECK_CALL, 0
            return (Action.RAISE, min(max_raise, 4 * self.game.min_bet)) if random.random() < 0.75 else (Action.CHECK_CALL, 0)
        
        if hand_rank == 2:
            if hand_eval.high_pair(hand_) >= 12 and call_amt_due <= 0.25 * self.stack:
                if max_raise > self.game.min_bet and random.random() < 0.25:
                    return Action.RAISE, random.randint(self.game.min_bet, max(int(max_raise*0.25), self.game.min_bet))
                return Action.CHECK_CALL, 0
        
        if call_amt_due == 0:
            return Action.CHECK_CALL, 0
        
        
        return Action.FOLD, 0


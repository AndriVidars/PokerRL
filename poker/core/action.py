from enum import Enum

class Action(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE = 3
    
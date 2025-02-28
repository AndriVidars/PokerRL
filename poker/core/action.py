from enum import Enum

class Action(Enum):
    FOLD = 0
    CHECK_CALL = 1 # treated the same
    RAISE = 2
    
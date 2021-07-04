# DO NOT use numpy's random number generator, use this one instead
# Generates a random number from [0, n)
import random
import time
from stubs import TetrisBot

class Model(TetrisBot):
    def __init__ (self):
        # Have your own "Global" Variables here
        random.seed(438)
        self.timing = 0


    def next_move (self):
        """
        At every tick, the internal representation of the tetris board state is
        updated before this `next_move` method is called. You can use all the
        get methods below

        Return an array of numbers denoting moves in a given frame. The tetris program will interpret the moves to the
        best of its ability.
        Alternatively, return a number for a single command.
        Simultaneous key presses are allowed but not multiple repeated key presses
        Invalid codes are ignored
        The array is evaluated in the order it is returned

        If next_move() takes more milliseconds than provided by game_state["allotted_time"] the game will
        compensate by skipping frames

        Controls:
        0: No input
        1: Rotates Counter-Clockwise
        2: Drop by one row immediately and gain 1 point
        3: Moves block to the left
        4: Moves block to the right
        5: Instant Drop

        Usage:
        return [2, 3, 1]
        return 2
        return []
        """

        # Replace with your AI
        self.timing += 1

        time.sleep(.2)
        return [1]

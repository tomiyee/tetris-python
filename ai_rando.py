# DO NOT use numpy's random number generator, use this one instead
# Generates a random number from [0, n)
from random import randrange as rand

class Model:
    def __init__ (self):
        pass

    def next_move (self):
        """
        Returns a String, the keyboard name

        Either UP, DOWN, LEFT, or RIGHT
        """

        x = rand(4)

        if x == 0:
            return 'UP'
        elif x == 1:
            return 'DOWN'
        elif x == 2:
            return 'LEFT'
        elif x == 3:
            return 'RIGHT'

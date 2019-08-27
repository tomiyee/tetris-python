# DO NOT use numpy's random number generator, use this one instead
# Generates a random number from [0, n)
from random import randrange as rand

# Feel free to change the seed as you see fit
rand.seed(420)

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


    def get_current_piece ():
        """
        Returns the matrix of the current piece
        """
        pass

    def get_next_piece ():
        """
        Returns the matrix of the next piece
        """
        pass

    def get_current_piece_map ():
        """
        Returns a 2d map of the entire grid.
        A space is 1 if the space has a portion of the current piece,
        0 otherwise
        """
        pass

    def get_current_piece_coord ():
        """
        Returns a tuple with (row, col) for the top left of the current piece's matrix
        """
        pass

    def get_score ():
        """
        Returns an integer value of the current score
        """
        pass

    def get_board_state ():
        """
        Will Retrun 2D grid of 0s, 1s,

        0 indicates empty space
        1 indicates a locked in piece.

        current piece is not included
        """
        pass

    def updateState ():
        """
        Updates the model's internal representation of the board.
        The model does not have access to this method, and honestly doesn't need to
        """
        pass

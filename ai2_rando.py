# DO NOT use numpy's random number generator, use this one instead
# Generates a random number from [0, n)
from random import randrange as rand
from random import seed
# Feel free to change the seed as you see fit
seed(420)

class Model:
    def __init__ (self):
        # Have your own "Global" Variables here
        pass

    def next_move (self):
        """
        Returns a int, or None if no button is going to be pressed

        ret
        Controls:
        UP: Rotates Counter-Clockwise(?)
        DOWN: Drop by one line immediately and gain 1 point
        LEFT: Self-explanatory
        RIGHT: Self-explanatory
        RETURN: Instant Drop
        """

        # Replace this method with your AI
        return_value = rand(4)

    def get_board_width (self):
        """
        Returns the number of columns
        """
        return 10

    def get_board_height (self):
        """
        Returns the number of rows in the board
        """
        return 22

    def get_current_piece (self):
        """
        Returns the matrix of the current piece,
        includes rotation, not the position in space
        """
        return self.current_piece

    def get_next_piece (self):
        """
        Returns the matrix of the next piece
        Includes rotation,, not the position in space
        """
        return self.next_piece

    def get_current_piece_id (self):
        """
        The integer id of the piece
        0: T Piece
        1: S Piece
        2: Z Piece
        3: J Piece
        4: L Piece
        5: I Piece (Line)
        6: O Piece (Block)
        """
        return self.current_piece_id

    def get_next_piece_id (self):
        """
        The integer id of the piece
        0: T Piece
        1: S Piece
        2: Z Piece
        3: J Piece
        4: L Piece
        5: I Piece (Line)
        6: O Piece (Block)
        """
        return self.next_piece_id

    def get_current_piece_map (self):
        """
        Returns a 2d map of the entire grid.
        A space is 1 if the space has a portion of the current piece,
        0 otherwise
        """
        return self.current_piece_map

    def get_current_piece_coord (self):
        """
        Returns a tuple with (row, col) for the top left of the current piece's matrix
        """
        return self.coord

    def get_score (self):
        """
        Returns an integer value of the current score
        """
        return self.score;

    def get_board_state (self):
        """
        Will Retrun 2D grid of 0s, 1s,

        0 indicates empty space
        1 indicates a locked in piece.

        current piece is not included
        """
        self.board

    def get_allotted_time (self):
        """
        Returns an integer with the number of milliseconds until your code's termination
        """
        return self.time

    def update_state (self, game_state):
        """
        Updates the model's internal representation of the board.
        The model does not have access to this method, and honestly doesn't need to
        """
        self.current_piece = game_state["current_piece"]
        self.current_piece_id = game_state["current_piece_id"]
        self.next_piece = game_state["next_piece"]
        self.next_piece_id = game_state["next_piece_id"]
        self.score = game_state["score"]
        self.time = game_state["allotted_time"]
        self.board = game_state["current_board"]
        self.coord = game_state["position"]
        self.currrent_piece_map = game_state["current_piece_map"]
        pass

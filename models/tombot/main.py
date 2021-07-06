# DO NOT use numpy's random number generator, use this one instead
# Generates a random number from [0, n)
import random
import time
from stubs import TetrisBot
from copy import deepcopy
random = random.Random() # Ensures Arena (running in multiple threads) have same seed

"""
Idea: Leave the far right column for I blocks, and try to closely pack the
remaining columns as best I can.
"""

def has_holes(board):
    """Returns true if there are no gaps in a column"""
    # dimensions of the board
    rows = len(board)
    cols = len(board[0])

    for c in range(cols):
        col = [board[r][c] for r in range(rows)][::-1]
        s = 0
        for i in range(rows):
            if col[i] == 0:
                s = 1
            if col[i] == 1 and s == 1:
                return True
    return False

def get_max_height(board):
    """ Returns the maximum height"""
    # dimensions of the board
    rows = len(board)
    cols = len(board[0])
    heights = []
    for c in range(cols):
        col = [board[r][c] for r in range(rows)]
        if 1 not in col:
            heights.append(0)
            continue
        heights.append(rows - col.index(1))
    return max(heights)

def get_min_height(board):
    """ Returns the minimum height """
    # dimensions of the board
    rows = len(board)
    cols = len(board[0])
    heights = []
    for c in range(cols):
        col = [board[r][c] for r in range(rows)]
        if 1 not in col:
            heights.append(0)
            continue
        heights.append(rows - col.index(1))
    return min(heights)

def drop_piece(current_piece, board, c):
    """
    Places the board in the given orientation onto the board, None if cannot

    Parameters:
        curent_piece (int[][]): A 2D array of the current piece
        board (int[][]): A 2D array of the board state, without current piece


    """

    # dimensions of the board
    rows = len(board)
    cols = len(board[0])
    # dimensions of the current piece
    piece_width  = len(current_piece[0])
    piece_height = len(current_piece)

    r = rows - piece_height

    while r > 0:
        collision_found = False
        for piece_r in range(len(current_piece)):
            for piece_c in range(len(current_piece[piece_r])):
                if current_piece[piece_r][piece_c] == 0:
                    continue
                if board[r+piece_r][c+piece_c] == 1:
                    r -= 1
                    collision_found = True
                    break
            if collision_found: break
        if collision_found: continue
        # No collisions
        new_board = deepcopy(board)

        for piece_r in range(len(current_piece)):
            for piece_c in range(len(current_piece[piece_r])):
                if current_piece[piece_r][piece_c] == 0:
                    continue
                new_board[r+piece_r][c+piece_c] = 1
        return new_board

    return None

def rotate_counter_clockwise(shape, times=1):
    """Given a shape, rotates it counter clockwise"""
    new_shape = shape
    for i in range(times % 4):
        new_shape = [
            [new_shape[y][x] for y in range(len(new_shape))]
            for x in range(len(new_shape[0]) - 1, -1, -1)
        ]
    return new_shape

def find_good_space(current_piece, board):
    """
    Tries to find the optimal space to place this piece, without rotation

    Parameters:
        curent_piece (int[][]): A 2D array of the current piece
        board (int[][]): A 2D array of the board state, without current piece

    Returns:
        col (int): The column of the left most corner of the current piece for
                   the optimal placement
        max_height
        perfect (bool): True if no gaps were found by doing this method

    """
    # dimensions of the board
    rows = len(board)
    cols = len(board[0])
    # dimensions of the current piece
    piece_width  = len(current_piece[0])
    piece_height = len(current_piece)

    candidates = []
    for c in range(cols - piece_width + 1):
        result_board = drop_piece(current_piece, board, c)

        if result_board == None:
            continue
        if not has_holes(result_board):
            max_height = get_max_height(result_board)
            candidates.append((c, max_height))
    if candidates:
        optimal_col, max_height = min(candidates, key=lambda x: x[1])
        perfect = True
    else:
        optimal_col = 0
        max_height = rows + 1
        perfect = False

    return (optimal_col, max_height, perfect)

def find_optimal_space(current_piece, board):

    good_places = [(find_good_space(rotate_counter_clockwise(current_piece, times=t), board), t) for t in range(4)]
    perfect_places = [x for x in good_places if x[0][2]]

    candidates = good_places
    if len(perfect_places) > 0:
        candidates = perfect_places

    (opt_col, _, perfect), t = min(candidates, key=lambda x: x[0][1])
    return (opt_col, perfect, t)

def pretty_print(b):
    print('')
    for r in b:
        print(r)

b = [
[0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0],
[1,1,0,0,0,0,0,1,0],
[1,1,0,0,0,0,0,1,0],
[1,1,0,0,1,1,1,1,0],
[1,1,1,1,1,1,1,1,0]
]

p = [
[1,1,1],
[0,1,0]
]

find_optimal_space(p, b)

class Model(TetrisBot):
    def __init__ (self):
        # Have your own "Global" Variables here
        random.seed(438)

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

        # Get Board and Piece state
        current_piece = self.get_current_piece()
        board = self.get_board_state()
        if board == None or current_piece == None:
            return [1]
        # remove the right most col
        # board = [r[:-1] for r in board]

        optimal_col, perfect, ccw_rotations = find_optimal_space(current_piece, board)

        # Always move down, and rotate only if necessary
        outputs = [2]
        if ccw_rotations != 0:
            return outputs + [1]

        r, c = self.get_current_piece_coord()

        if c > optimal_col:
            return outputs + [3]
        if c < optimal_col:
            return outputs + [4]
        if c == optimal_col:
            return outputs + [5]

        # Replace with your AI
        return [random.randint(0, 4)]

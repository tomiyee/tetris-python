# -*- coding: utf-8 -*-

# NOTE FOR WINDOWS USERS:
# You can download a "exefied" version of this game at:
# http://hi-im.laria.me/progs/tetris_py_exefied.zipp
# If a DLL is missing or something like this, write an E-Mail (me@laria.me)
# or leave a comment on this gist.

# Very simple tetris implementation
#
# Control keys:
#       Down - Drop stone faster
# Left/Right - Move stone
#         Up - Rotate Stone clockwise
#     Escape - Quit game
#          P - Pause game
#     Return - Instant drop
#
# Have fun!

# Copyright (c) 2010 "Laria Carolin Chabowski"<me@laria.me>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import sys
import time
import importlib
import contextlib
import random
import numpy as np
from numpy.random import randint
from enum import Enum
from math import ceil
from argparse import ArgumentParser
with contextlib.redirect_stdout(None):
    import pygame  # silences pygame's message

from stubs import import_player
# Import Constants
from constants import tetris_shapes, colors, rotation_offsets

# The configuration
CELL_SIZE = 32
COLS = 10
ROWS = 22
MAX_FPS = 120
DROP_INCENTIVE = 1

code_map = [0, "UP", "DOWN", "LEFT", "RIGHT", "RETURN"]


class TimeoutException(Exception):
    """A custom exception for when a model takes too long to make a decision"""

    pass

class Direction(Enum):
    CW = CLOCKWISE = 1
    CCW = COUNTER_CLOCKWISE = 2

# The default orientation of all pieces is UP
Orientation = {
    'UP': 0,
    'RIGHT': 1,
    'DOWN': 2,
    'LEFT': 3
}

def rotate_counter_clockwise(shape):
    """Given a shape, rotates it counter clockwise"""
    return [
        [shape[y][x] for y in range(len(shape))]
        for x in range(len(shape[0]) - 1, -1, -1)
    ]


def rotate_clockwise(shape):
    """ Given a shape, rotates it clockwise """
    shape = rotate_counter_clockwise(shape)
    shape = rotate_counter_clockwise(shape)
    shape = rotate_counter_clockwise(shape)
    return shape


def check_collision(board, shape, offset):
    off_x, off_y = offset
    for cy, row in enumerate(shape):
        for cx, cell in enumerate(row):
            try:
                if cell and board[cy + off_y][cx + off_x]:
                    return True
            except IndexError:
                return True
    return False


def remove_row(board, row):
    """Deletes the row in the board matrix and adds an empty row to the top"""
    del board[row]
    return [[0 for i in range(COLS)]] + board


def join_matrixes(mat1, mat2, mat2_off):
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            mat1[cy + off_y - 1][cx + off_x] += val
    return mat1


def new_board():
    """
    Returns a 2D array ( ROWS+1 x COLS ). The top ROWS x COLS are empty (0), and
    the bottom row is marked full (1) for purposes of collision detection
    """
    board = [[0 for x in range(COLS)] for y in range(ROWS)]
    board += [[1 for x in range(COLS)]]
    return board


class TetrisApp(object):
    def __init__(self, model, debug=False, seed=483):
        if type(model) == str: model = import_player(model)

        # Initialize the seed for the blocks
        np.random.seed(seed)

        pygame.init()
        self.debug = debug
        self.model = model

        self.allotted_time = 100 # time in ms
        self.overtime = 0
        self.queued_commands = []
        self.total_game_ticks = 0

        pygame.key.set_repeat(250, 25)
        self.width = CELL_SIZE * (COLS + 6)
        self.height = CELL_SIZE * ROWS
        self.rlim = CELL_SIZE * COLS
        self.bground_grid = [
            [8 if x % 2 == y % 2 else 0 for x in range(COLS)] for y in range(ROWS)
        ]
        self.default_font = pygame.font.Font(pygame.font.get_default_font(), 12)
        self.screen = pygame.display.set_mode((self.width, self.height))
        # We do not need mouse movement events, so we block them.
        pygame.event.set_blocked(pygame.MOUSEMOTION)

        self.next_stone_id = randint(len(tetris_shapes))
        self.next_stone = tetris_shapes[self.next_stone_id]
        # The number of lines cleared in the previous game tick
        self.cleared_lines = 0
        self.init_game()

    def restart_game(self):
        """Initializes a new game only if the game is not currently running"""
        if self.gameover:
            self.init_game()
            self.gameover = False

    def init_game(self):
        self.board = new_board()
        self.new_stone()
        self.level = 1
        self.score = 0
        self.lines = 0
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000)

    def new_stone(self):
        self.rotation_state = Orientation['UP']
        self.stone_id = self.next_stone_id
        self.stone = self.next_stone[:]
        self.next_stone_id = randint(len(tetris_shapes))
        self.next_stone = tetris_shapes[self.next_stone_id]
        self.stone_x = int(COLS / 2 - len(self.stone[0]) / 2)
        self.stone_y = 0

        if check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
            self.gameover = True

    def disp_msg(self, msg, topleft):
        x, y = topleft
        for line in msg.splitlines():
            self.screen.blit(
                self.default_font.render(line, False, (255, 255, 255), (0, 0, 0)),
                (x, y),
            )
            y += 14

    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image = self.default_font.render(
                line, False, (255, 255, 255), (0, 0, 0)
            )

            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2

            self.screen.blit(
                msg_image,
                (
                    self.width // 2 - msgim_center_x,
                    self.height // 2 - msgim_center_y + i * COLS,
                ),
            )

    def draw_matrix(self, matrix, offset):
        off_x, off_y = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(
                        self.screen,
                        colors[val],
                        pygame.Rect(
                            (off_x + x) * CELL_SIZE,
                            (off_y + y) * CELL_SIZE,
                            CELL_SIZE,
                            CELL_SIZE,
                        ),
                        0,
                    )

    def add_cl_lines(self, n):
        linescores = [0, 40, 100, 300, 1200]
        self.lines += n
        self.score += linescores[n] * self.level
        # if self.lines >= self.level * 6:
            # self.level += 1
            # newdelay = 1000 - 50 * (self.level - 1)
            # newdelay = 100 if newdelay < 100 else newdelay
            # pygame.time.set_timer(pygame.USEREVENT + 1, newdelay)

    def move(self, delta_x):
        """ Moves the current stone horizontally (no vertical movement) """
        # Don't continue if the game is over or paused
        if self.gameover or self.paused:
            return False

        new_x = self.stone_x + delta_x

        # Prevents the stone from colliding with the walls when moving
        if new_x < 0:
            new_x = 0
        if new_x > COLS - len(self.stone[0]):
            new_x = COLS - len(self.stone[0])

        # Commits the movement
        if not check_collision(self.board, self.stone, (new_x, self.stone_y)):
            self.stone_x = new_x

    def drop(self, manual=False):
        """
        The current stone moves down by one row. If manual, the player moved
        the stone down, and thus gains a point.
        """

        # Don't drop if the game is over or paused
        if self.gameover or self.paused:
            return False

        self.score += DROP_INCENTIVE if manual else 0
        self.stone_y += 1
        if check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
            self.board = join_matrixes(
                self.board, self.stone, (self.stone_x, self.stone_y)
            )
            self.new_stone()
            cleared_ROWS = 0
            while True:
                for i, row in enumerate(self.board[:-1]):
                    if 0 not in row:
                        self.board = remove_row(self.board, i)
                        cleared_ROWS += 1
                        break
                else:
                    break

            self.cleared_lines = cleared_ROWS

            self.add_cl_lines(cleared_ROWS)
            return True

    def insta_drop(self):
        """ Manually drops current piece until collision """
        # Don't continue if the game is over or paused
        if self.gameover or self.paused:
            return False

        # Force drops until it fails
        while not self.drop(True):
            pass

    def rotate_stone(self, dir=Direction.COUNTER_CLOCKWISE):
        """ We are assuming / praying this is SRS (Super Rotation System) """

        # Don't continue if the game is over or paused
        if self.gameover or self.paused:
            return False

        # TODO - set the rotation offsets to a more reasonable permutation

        # generate a matrix with the rotated the stone
        if dir == Direction.COUNTER_CLOCKWISE:
            new_stone = rotate_counter_clockwise(self.stone)
            new_rotation_state = (self.rotation_state + 1) % 4
            dx, dy = rotation_offsets[self.stone_id][self.rotation_state]
        if dir == Direction.CLOCKWISE:
            new_stone = rotate_clockwise(self.stone)
            new_rotation_state = (self.rotation_state - 1) % 4
            dx, dy = rotation_offsets[self.stone_id][(self.rotation_state-2)%4]

        self.stone_x += dx
        self.stone_y += dy

        collided = check_collision(self.board, new_stone, (self.stone_x, self.stone_y))

        if not collided and COLS > self.stone_x > 0 and ROWS > self.stone_y > 0:
            self.stone = new_stone
            self.rotation_state = new_rotation_state
        # If new rotation leads to collision, move it back, no commits
        else:
            self.stone_x -= dx
            self.stone_y -= dy

    def toggle_pause(self):
        self.paused = not self.paused

    def quit(self):
        """ Closes the window """
        print("[+] Done!")
        self.center_msg("Exiting...")
        pygame.display.update()
        pygame.quit()

    def render(self):

        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (self.rlim + 1, 0),
            (self.rlim + 1, self.height - 1),
        )
        self.disp_msg("Next:", (self.rlim + CELL_SIZE, 2))
        self.disp_msg(
            f'Score: {self.score}\n\nLevel: {self.level}\nLines: {self.lines}',
            (self.rlim + CELL_SIZE, CELL_SIZE * 5),
        )
        self.draw_matrix(self.bground_grid, (0, 0))
        self.draw_matrix(self.board, (0, 0))
        self.draw_matrix(self.stone, (self.stone_x, self.stone_y))
        self.draw_matrix(self.next_stone, (COLS + 1, 2))

    def run(self, exit_on_end=False):

        self.gameover = False
        self.paused = False
        exited = False

        dont_burn_my_cpu = pygame.time.Clock()
        while not exited:
            self.screen.fill((0, 0, 0))

            if self.gameover:
                self.center_msg(f'Game Over!\nYour score: {self.score} Press space to continue')
            elif self.paused:
                self.center_msg("Paused")
            else:
                self.render()

            pygame.display.update()

            if not self.gameover:
                self.total_game_ticks += 1

                
                # if self.total_game_ticks % 100 == 0:
                    # self.allotted_time -= 5

                if self.overtime > 0:  # if the code was timed out
                    self.overtime = max(0, self.overtime - self.allotted_time)
                    if self.total_game_ticks % 5 == 0:
                        self.drop(False)
                    if self.overtime == 0:
                        self.interpret(self.queued_commands)
                        self.queued_commands = []
                else:
                    # Send the model the current state of the tetris board
                    state_representation = self._build_state_representation()
                    self.model._update_state(state_representation)

                    # Determine the duration of time that the model takes
                    start = time.time()
                    return_value = self.model.next_move()
                    elapsed = time.time() - start
                    elapsed *= 1000

                    # Reset the number of cleared lines
                    self.cleared_lines = 0

                    # If debug is on, do not count the time passed
                    if self.debug:
                        elapsed = 0

                    if elapsed > self.allotted_time:
                        self.overtime += elapsed - self.allotted_time
                        print(
                            "next_move() function took ",
                            ceil(elapsed - state_representation["allotted_time"]),
                            "ms over allotted time to complete. "
                            "game simulated ahead to compensate",
                        )
                        self.queued_commands = return_value
                    else:
                        self.interpret(return_value)

            # Return to normal game execution
            for event in pygame.event.get():
                # if event.type == pygame.USEREVENT+1:
                #    self.drop(False)
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    self.quit()
                    return self.score

            if self.total_game_ticks % 5 == 0:
                self.drop(False)

            if self.gameover and self.debug:
                self.quit()
                return self.score

            dont_burn_my_cpu.tick(MAX_FPS)

    def interpret(self, bot_inputs):
        # takes a list of commands and interprets them as game movements
        key_actions = {
            0: lambda: 0,
            "ESCAPE": self.quit,
            "LEFT": lambda: self.move(-1),
            "RIGHT": lambda: self.move(+1),
            "DOWN": lambda: self.drop(True),
            "UP": self.rotate_stone,
            "p": self.toggle_pause,
            "SPACE": self.restart_game,
            "RETURN": self.insta_drop,
        }

        if type(bot_inputs) == int:
            bot_inputs = [bot_inputs]

        if type(bot_inputs) != list:
            return

        prev_inputs = set()
        for i in bot_inputs:
            if i in prev_inputs:
                print("Ignored repetitive keystroke: ", i)
                continue
            prev_inputs.add(i)

            key_action = code_map[i] if i in range(len(code_map)) else None
            if key_action is None:
                print(f'The code: {i} is not recognized by tetris, command ignored.')
                continue
            key_actions[key_action]()

    def _build_state_representation(self):
        """
        Constructs a dict containing properties of the current game state to send
        to the model.
        """

        # I do some way of internal representation here
        current_piece_map = [[False for c in r] for r in self.board[:-1]]
        for r in range(len(self.stone)):
            for c in range(len(self.stone[r])):
                current_piece_map[r + self.stone_y][c + self.stone_x] = bool(self.stone[r][c])
        # Builds the Internal Representation
        state_representation = {
            'rows': ROWS,
            'cols': COLS,
            "current_piece": [list(map(int,map(bool, row))) for row in self.stone],
            "current_piece_id": self.stone_id,
            'current_piece_orientation': self.rotation_state,
            "next_piece": [list(map(int,map(bool, row))) for row in self.next_stone],
            "next_piece_id": self.next_stone_id,
            'next_piece_orientation': Orientation['UP'],
            'cleared_lines': self.cleared_lines,
            "score": self.score,
            "allotted_time": self.allotted_time,
            "current_board": [
                list(map(int,map(bool, row))) for row in self.board[:-1]
            ],
            "position": (self.stone_y, self.stone_x),
            "current_piece_map": current_piece_map,
        }

        return state_representation

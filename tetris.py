#!/usr/bin/env python2
#-*- coding: utf-8 -*-

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

import random
from numpy.random import randint as rand
from numpy.random import seed
import contextlib
with contextlib.redirect_stdout(None):
    import pygame # silences pygame's message
import sys
import time
from math import ceil
from argparse import ArgumentParser
import importlib

class TimeoutException(Exception):
    """ A custom exception for when a model takes too long to make a decision """
    pass

# The configuration
cell_size = 18
cols =      10
rows =      22
maxfps =    30

colors = [
(0,   0,   0  ),
(255, 85,  85),
(100, 200, 115),
(120, 108, 245),
(255, 140, 50 ),
(50,  120, 52 ),
(146, 202, 73 ),
(150, 161, 218 ),
(35,  35,  35) # Helper color for background grid
]

# Define the shapes of the single parts
tetris_shapes = [
    [[1, 1, 1],
     [0, 1, 0]],

    [[0, 2, 2],
     [2, 2, 0]],

    [[3, 3, 0],
     [0, 3, 3]],

    [[4, 0, 0],
     [4, 4, 4]],

    [[0, 0, 5],
     [5, 5, 5]],

    [[6, 6, 6, 6]],

    [[7, 7],
     [7, 7]]
]

rotation_offsets = [ [(1,-1), (-1,0), (0,0), (0,1)],
                     [(0,0), (0, 0), (0,0), (0,0)],
                     [(1,0), (-1, 0), (1,0), (-1,0)],
                     [(1,-1), (-1,0), (0,0), (0,1)],
                     [(1,-1), (-1,0), (0,0), (0,1)],
                     [(1,-1), (-1,1), (1,-1), (-1,1)],
                     [(0,0), (0, 0), (0,0), (0,0)]
]

code_map = [0, "UP", "DOWN", "LEFT", "RIGHT", "RETURN"]
def rotate_counter_clockwise(shape):
    return [ [ shape[y][x]
            for y in range(len(shape)) ]
        for x in range(len(shape[0]) - 1, -1, -1) ]

def rotate_clockwise(shape):
    """ Given a shape, rotates it counter clockwise """
    raise NotImplementedError()

def check_collision(board, shape, offset):
    off_x, off_y = offset
    for cy, row in enumerate(shape):
        for cx, cell in enumerate(row):
            try:
                if cell and board[ cy + off_y ][ cx + off_x ]:
                    return True
            except IndexError:
                return True
    return False

def remove_row(board, row):
    del board[row]
    return [[0 for i in range(cols)]] + board

def join_matrixes(mat1, mat2, mat2_off):
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            mat1[cy+off_y-1][cx+off_x] += val
    return mat1

def new_board():
    board = [ [ 0 for x in range(cols) ]
            for y in range(rows) ]
    board += [[ 1 for x in range(cols)]]
    return board

class TetrisApp(object):
    def __init__(self):
        pygame.init()
        self.allotted_time = 200
        self.overtime = 0
        self.queued_commands = []
        self.total_game_ticks = 0
        pygame.key.set_repeat(250,25)
        self.width = cell_size*(cols+6)
        self.height = cell_size*rows
        self.rlim = cell_size*cols
        self.bground_grid = [[ 8 if x%2==y%2 else 0 for x in range(cols)] for y in range(rows)]

        self.default_font =  pygame.font.Font(
            pygame.font.get_default_font(), 12)

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.event.set_blocked(pygame.MOUSEMOTION) # We do not need
                                                     # mouse movement
                                                     # events, so we
                                                     # block them.

        self.next_stone_id = rand(len(tetris_shapes))

        self.next_stone = tetris_shapes[self.next_stone_id]
        self.init_game()

    def new_stone(self):
        self.rotatation_state = 0
        self.stone_id = self.next_stone_id
        self.stone = self.next_stone[:]
        self.next_stone_id = rand(len(tetris_shapes))
        self.next_stone = tetris_shapes[self.next_stone_id]
        self.stone_x = int(cols / 2 - len(self.stone[0])/2)
        self.stone_y = 0

        if check_collision(self.board,
                           self.stone,
                           (self.stone_x, self.stone_y)):
            self.gameover = True

    def init_game(self):
        self.board = new_board()
        self.new_stone()
        self.level = 1
        self.score = 0
        self.lines = 0
        pygame.time.set_timer(pygame.USEREVENT+1, 1000)

    def disp_msg(self, msg, topleft):
        x,y = topleft
        for line in msg.splitlines():
            self.screen.blit(
                self.default_font.render(
                    line,
                    False,
                    (255,255,255),
                    (0,0,0)),
                (x,y))
            y+=14

    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image =  self.default_font.render(line, False,
                (255,255,255), (0,0,0))

            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2

            self.screen.blit(msg_image, (
              self.width // 2-msgim_center_x,
              self.height // 2-msgim_center_y+i*22))

    def draw_matrix(self, matrix, offset):
        off_x, off_y  = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(
                        self.screen,
                        colors[val],
                        pygame.Rect(
                            (off_x+x) *
                              cell_size,
                            (off_y+y) *
                              cell_size,
                            cell_size,
                            cell_size),0)

    def add_cl_lines(self, n):
        linescores = [0, 40, 100, 300, 1200]
        self.lines += n
        self.score += linescores[n] * self.level
        if self.lines >= self.level*6:
            self.level += 1
            newdelay = 1000-50*(self.level-1)
            newdelay = 100 if newdelay < 100 else newdelay
            pygame.time.set_timer(pygame.USEREVENT+1, newdelay)

    def move(self, delta_x):
        if not self.gameover and not self.paused:
            new_x = self.stone_x + delta_x
            if new_x < 0:
                new_x = 0
            if new_x > cols - len(self.stone[0]):
                new_x = cols - len(self.stone[0])
            if not check_collision(self.board,
                                   self.stone,
                                   (new_x, self.stone_y)):
                self.stone_x = new_x
    def quit(self):
        self.center_msg("Exiting...")
        pygame.display.update()
        sys.exit()

    def drop(self, manual):
        if not self.gameover and not self.paused:
            self.score += 1 if manual else 0
            self.stone_y += 1
            if check_collision(self.board,
                               self.stone,
                               (self.stone_x, self.stone_y)):
                self.board = join_matrixes(
                  self.board,
                  self.stone,
                  (self.stone_x, self.stone_y))
                self.new_stone()
                cleared_rows = 0
                while True:
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            self.board = remove_row(
                              self.board, i)
                            cleared_rows += 1
                            break
                    else:
                        break
                self.add_cl_lines(cleared_rows)
                return True
        return False

    def insta_drop(self):
        if not self.gameover and not self.paused:
            while(not self.drop(True)):
                pass

    def rotate_stone(self):

        if not self.gameover and not self.paused:
            new_stone = rotate_counter_clockwise(self.stone)
            dx, dy = rotation_offsets[self.stone_id][self.rotatation_state]

            self.stone_x += dx
            self.stone_y += dy
            if not check_collision(self.board, new_stone, (self.stone_x, self.stone_y)) and \
                    10 > self.stone_x > 0 and 22 > self.stone_y > 0:
                self.stone = new_stone
                self.rotatation_state += 1
                self.rotatation_state %= 4
            else:
                self.stone_x -= dx
                self.stone_y -= dy

    def toggle_pause(self):
        self.paused = not self.paused

    def start_game(self):
        if self.gameover:
            self.init_game()
            self.gameover = False

    def run(self):


        self.gameover = False
        self.paused = False

        dont_burn_my_cpu = pygame.time.Clock()
        while 1:
            self.screen.fill((0,0,0))
            if self.gameover:
                self.center_msg("""Game Over!\nYour score: %d
Press space to continue""" % self.score)
            else:
                if self.paused:
                    self.center_msg("Paused")
                else:
                    pygame.draw.line(self.screen,
                        (255,255,255),
                        (self.rlim+1, 0),
                        (self.rlim+1, self.height-1))
                    self.disp_msg("Next:", (
                        self.rlim+cell_size,
                        2))
                    self.disp_msg("Score: %d\n\nLevel: %d\
\nLines: %d" % (self.score, self.level, self.lines),
                        (self.rlim+cell_size, cell_size*5))
                    self.draw_matrix(self.bground_grid, (0,0))
                    self.draw_matrix(self.board, (0,0))
                    self.draw_matrix(self.stone,
                        (self.stone_x, self.stone_y))
                    self.draw_matrix(self.next_stone,
                        (cols+1,2))
            pygame.display.update()
            if not self.gameover:
                self.total_game_ticks += 1

                if self.total_game_ticks % 100 == 0:
                    self.allotted_time -= 5

                if self.overtime > 0: # if the code was timed out
                    self.overtime = max(0, self.overtime - self.allotted_time)
                    if self.total_game_ticks % 5 == 0:
                        self.drop(False)
                    if self.overtime == 0:
                        self.interpret(self.queued_commands)
                        self.queued_commands = []
                else:

                    # I do some way of internal representation here
                    current_piece_map = [[False for c in r] for r in self.board[:-1]]
                    for r in range(len(self.stone)):
                        for c in range(len(self.stone[r])):
                            current_piece_map[r+self.stone_y][c+self.stone_x] = bool(self.stone[r][c])
                    # Builds the Internal Representation
                    internal_state_representation = {
                        "current_piece": [list(map(bool, row)) for row in self.stone],
                        "current_piece_id": self.stone_id,
                        "next_piece": [list(map(bool, row)) for row in self.next_stone],
                        "next_piece_id": self.next_stone_id,
                        "score": self.score,
                        "allotted_time": self.allotted_time,
                        "current_board": [list(map(bool, row)) for row in self.board[:-1]],
                        "position": (self.stone_y, self.stone_x),
                        "current_piece_map": current_piece_map
                    }

                    model.update_state(internal_state_representation)

                    start = time.time()
                    return_value = model.next_move()
                    elapsed = time.time() - start
                    elapsed *= 1000

                    if elapsed  > self.allotted_time:
                        self.overtime += elapsed - self.allotted_time
                        print("next_move() function took ", ceil(elapsed - ir["allotted_time"]), "ms over allotted time to complete. "
                                                                     "game simulated ahead to compensate")
                        self.queued_commands = return_value
                    else:
                        self.interpret(return_value)


            # Return to normal game execution
            for event in pygame.event.get():
                #if event.type == pygame.USEREVENT+1:
                #    self.drop(False)
                if event.type == pygame.QUIT:
                    self.quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.quit()
            if self.total_game_ticks % 5 == 0:
                self.drop(False)

            dont_burn_my_cpu.tick(maxfps)

    def interpret(self, return_value):  # takes a list of commands and interprets them as game movements
        key_actions = {
            0: lambda: print("No input detected"),
            'ESCAPE': self.quit,
            'LEFT': lambda: self.move(-1),
            'RIGHT': lambda: self.move(+1),
            'DOWN': lambda: self.drop(True),
            'UP': self.rotate_stone,
            'p': self.toggle_pause,
            'SPACE': self.start_game,
            'RETURN': self.insta_drop
        }

        if type(return_value) == int:
            return_value = [return_value]
        if type(return_value) == list:
            j = []
            for i in return_value:
                if j.count(i) == 1:
                    print("Ignored repetitive keystroke: ", i)
                    continue
                j.append(i)
                if i in range(6):
                    key_actions[code_map[i]]()
                else:
                    print("the  code: ", i, " is not recognized by tetris; command ignored")

if __name__ == '__main__':

    # Initialize the seed for the blocks
    seed(439)

    # Parse the Arguments
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='file_name')
    args = parser.parse_args()

    # Imports the specified model for the Tetris Game
    module = importlib.import_module(f'models.{args.file_name}.main')
    # Initialize the model
    model = module.Model()

    # Let the games begin
    App = TetrisApp()
    App.run()

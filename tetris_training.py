#!/usr/bin/env python2
#-*- coding: utf-8 -*-

# NOTE FOR WINDOWS USERS:
# You can download a "exefied" version of this game at:
# http://hi-im.laria.me/progs/tetris_py_exefied.zip
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
import numpy as np
from numpy.random import randint as rand
from numpy.random import seed
import contextlib
with contextlib.redirect_stdout(None):
    import pygame # silences pygame's message
import sys
import ai_rando
import time
from math import ceil

import pickle
# Hyperparameters
hidden = 200
batch_size = 10
learning_rate = 1e-4
gamma = 0.99 # This is the discount factor
decay_rate = 0.99
resume = False
# Init the Model
cols = 10
rows = 22
input_size = cols * rows
output_size = 6



class TimeoutException(Exception):
    pass


# The configuration
cell_size =    18
cols =        10
rows =        22
maxfps =     30

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


num_episodes = 0


def rotate_counter_clockwise(shape):
    return [ [ shape[y][x]
            for y in range(len(shape)) ]
        for x in range(len(shape[0]) - 1, -1, -1) ]

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
        pygame.display.quit()
        pygame.quit()

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

                # FOR TRAINING PURPOSES
                self.quit()
                return
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
                    ir = {
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

                    model.update_state(ir)
                    start = time.time()
                    # Generate the model's response
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
                    model.stop = True
                    # FOR TRAINING PURPOSES
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.quit()
                        # FOR TRAINING PURPOSES
                        return
            if self.total_game_ticks % 5 == 0:
                self.drop(False)

            dont_burn_my_cpu.tick(maxfps)

    def interpret(self, return_value):  # takes a list of commands and interprets them as game movements
        key_actions = {
            0: lambda: 0,
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

class Model:
    def __init__ (self, resume=False, load_file_name="save.p"):
        self.stop = False
        # Have your own "Global" Variables here
        self.score = 0
        self.num_ticks = 0
        self.rewards = []
        self.game_states = []
        self.hidden_states = []
        self.dlogps = []
        self.num_episodes = 0
        self.model = None

        if resume:
            self.model = pickle.load(open(load_file_name, 'rb'))
        else:
            # We start initializing manually
            self.model = {}
            # We will use xavier initialization to pseudo-randomly initialize the weights
            self.model['w1'] = np.random.randn(hidden, input_size) / np.sqrt(input_size)
            self.model['w2'] = np.random.randn(output_size, hidden) / np.sqrt(input_size)

        self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.items() }
        self.rmsprop_cache = { k : np.zeros_like(v) for k,v in self.model.items() }

    def sigmoid (self, n):
        return 1.0 / (1.0 + np.exp(-n))

    def discount_reward (self, r, gamma=0.99):
        discounted_r = np.zeros_like(r, dtype='f')
        running_sum = 0
        for t in reversed (range(r.size)):
            if r[t] != 0:
                running_sum = 0
            # Increment Sum
            running_sum = running_sum * gamma + r[t]
            discounted_r[t] = running_sum
        return discounted_r

    def policy_forward (self, x):
        """x must be a numpy array"""
        hidden_layer = np.dot(self.model['w1'], x)
        # Apply the relu, which just says nothing less than 0
        hidden_layer[hidden_layer < 0] = 0

        log_pred = np.dot(self.model['w2'], hidden_layer)
        pred = self.sigmoid(log_pred)
        # Converts everything into a probability distribution
        pred = pred / pred.sum()

        return pred, hidden_layer

    def policy_backward (self, eph, epdlogp):
        # We will recursively compute the chainrule
        # eph is an array of the intermediate hidden state
        dw2 = np.dot(eph.T, epdlogp).T

        dh = np.dot(self.model['w2'].T, epdlogp.T).T

        # Apply Relu
        dh[eph <= 0] = 0

        dw1 = np.dot(self.ep_game_states.T, dh).T

        return {'w1': dw1, 'w2': dw2}

    def finish_episode (self):
        self.rewards.append(0)

        self.num_episodes += 1
        # This Episode's Data gets saved
        self.ep_game_states = np.vstack(self.game_states)       # Observations
        ep_hidden_states = np.vstack(self.hidden_states)   # Hidden States
        ep_dlogps = np.vstack(self.dlogps)                  # Gradients
        ep_rewards = np.vstack(self.rewards)               # Rewards

        #the strength with which we encourage a sampled action is the weighted sum of all rewards afterwards, but later rewards are exponentially less important
        # compute the discounted reward backwards through time
        discounted_epr = self.discount_reward(ep_rewards)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        ep_dlogps *= discounted_epr
        grad = self.policy_backward(ep_hidden_states, ep_dlogps)

        # Save this episode's gradients so that we apply at the end of the batch in bulk
        for k in self.model: self.grad_buffer[k] += grad[k]

        if self.num_episodes % batch_size == 0:
            for k,v in self.mode.items():
                g = self.grad_buffer[k]
                self.rmsprop_cache[k] = decay_rate * self.rmsprop_cache[k] + (1 - decay_rate) * g**2
                self.model[k] += learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
            # Reset the Gradient buffer to accumulate again next batch
            self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.items() }

        # Clean Up After Yourself
        self.score = 0
        self.num_ticks = 0
        self.rewards = []
        self.game_states = []
        self.hidden_states = []
        self.dlogps = []

    def save (self, file_name="save.p"):
        pickle.dump(self.model, open(file_name, 'wb'))

    def next_move (self):
        """
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
        """

        # Replace with your AI

        # Records the reward of the previous decision
        if self.num_ticks > 0:
            self.rewards.append(self.get_score()-self.get_prev_score())

        # Gets the observation
        curr_piece = np.array(self.get_current_piece_map())
        curr_board = np.array(self.get_board_state())
        curr_state = curr_piece + curr_board / 2.0
        # Makes a decision
        output, hidden_state = self.policy_forward(curr_state.ravel())
        # Save the game state and hidden states so that we can do back prop later
        self.hidden_states.append(hidden_state)
        self.game_states.append(curr_state.ravel())

        # Make a decision
        pr = np.random.uniform()
        decision = None
        for i in range(output.size):
            if pr <= output[i]:
                decision = i
                break
            pr -= output[i]

        # Assign a "Fake Label" that would encourage the decision we took just now
        fake_label = np.zeros(output.size, dtype='f')
        fake_label[decision] = 1.0
        # This would encourage the decision we made.
        self.dlogps.append(fake_label-output)

        self.num_ticks += 1

        return decision

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
        return self.score

    def get_prev_score (self):
        """Returns an integer value of the previous game tick's score"""
        return self.prev_score

    def get_board_state (self):
        """
        Will Retrun 2D grid of 0s, 1s,

        0 indicates empty space
        1 indicates a locked in piece.

        current piece is not included
        """
        return self.board

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
        self.prev_score = self.score
        self.score = game_state["score"]
        self.time = game_state["allotted_time"]
        self.board = game_state["current_board"]
        self.coord = game_state["position"]
        self.current_piece_map = game_state["current_piece_map"]
        pass

if __name__ == '__main__':
    # Initialize the model
    # Initialize the seed for the blocks
    # Let the games begin
    seed(439)

    save_file_name = "save.p"

    while True:
        model = Model(True, save_file_name)
        TetrisApp().run()
        model.finish_episode()
        if model.stop:
            break
        model.save(save_file_name)

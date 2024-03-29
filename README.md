# Python Tetris AI

An implementation of Tetris in Python, with the intention of having some AI play
the game.

```
pip3 install -r requirements.txt
python3 app.py --model=<name of package>
```

For more information, use the command `python3 app.py --help`.

## Arena
Plays a round robin style tournament with specified bots and reports score for each bot

```
python3 app.py --arena <bot1> <bot2> <bot3> ...
```

## Adding Your Custom Tetris Bot

Tetris bots are implemented as python modules in the sub-directory `models`. An
example of such a module is given as the default `ai_rando` model, which makes a
random move every game tick.

In order to create your own Python module, add a directory to `models` with the
name for your Tetris bot (e.g. `ai_rando`). Within that directory, add two
Python files: `__init__.py` and `main.py`. The init file can be empty, or have
other instructions related to initializing your model.

```
tetris-python/
├── models/
│   ├── ai_rando/
│   │   ├── __init__.py
│   │   ├── main.py
```

The `main.py` file must have a `Model` class that inherits from `TetrisBot` available using
`from stubs import TetrisBot`. You then override the `next_move()` method and
the constructor as you would like.

Within your `next_move()` method, you can take advantage of getter methods
inherited from `TetrisBot`. Details on the methods you have access to are
available below.

The `next_move()` method should return a list of integers, where the integers
correspond to different keys or actions the Tetris Bot can take at each given
time step. Details on what action each integer corresponds to can be seen below.

## Tetris App Constraints

Every frame, the game state of the App is updated. Once the state of the game
is updated, a representation of the game state is provided to your model.

Your model's `next_move()` method is then queried. If your model takes too long
to generate a response in the competitive mode, frames will be skipped. For
instance, if your model takes the time of 10 frames to come up with an output,
the Tetris app will render 9 frames as if your Tetris Bot gave no input and
use the input on the 10th frame. This is to incentivize efficiency in code. This
timeout feature can be toggled by entering a casual mode (enabling the `--debug` flag).

### Tetris Game State Methods

Your tetris model has access to a number of methods that grant you access to the
game state representation. Below list each of the methods that your Tetris Bot
inherits from the `TetrisBot` class for the purpose of analyzing game state:

| Method                          | Value                                                                     |
|---------------------------------|---------------------------------------------------------------------------|    
|`get_board_width()`              | The number of cols of the board                                           |
|`get_board_height()`             | The number of rows of the board                                           |
|`get_current_piece()`            | A 2D array for the on-board tetris piece, with rotation                   |
|`get_next_piece()`               | A 2D array for the up-coming tetris piece, with rotation                  |
|`get_current_piece_id()`         | The int ID of the on-board piece, without rotation                        |
|`get_next_piece_id()`            | The int ID of the up-coming piece, without rotation                       |
|`get_current_piece_orientation()`| The int of the on-board piece's rotation, UP for default                  |
|`get_next_piece_orientation()`   | The int of the up-coming piece's rotation, UP always                      |
|`get_current_piece_map()`        | A 2D array spanning the entire board containing the current piece         |
|`get_current_piece_coord()`      | A tuple of (row, col) of the top left spot of the matrix                  |   
|`get_cleared_lines()`            | The number of lines cleared in the previous game tick                     |
|`get_score()`                    | The current score shown on screen                                         |
|`get_board_state()`              | A 2D array of the current board without the current piece                 |
|`get_allotted_time()`            | The amount of time in ms that your bot has to compute in competitive mode |


### Random Number Generator

Note, to make a reliable number generator that is consistent even in arena mode
(which uses multi-threading), you can create separate instances of random.Random
for each thread.

```
local_random = random.Random()
local_random.seed(1234)
local_random.randint(1,100) # same int for different threads
```

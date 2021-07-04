# Python Tetris AI

An implementation of Tetris in Python, with the intention of having some AI play
the game.

```
pip3 install -r requirements.txt
python3 app.py --model=<name of package>
```

For more information, use the command `python3 app.py --help`.

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
├── models
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
timeout feature can be toggled by entering a casual mode (enabling the debug flag).

### Tetris Game State Methods

Your tetris model has access to a number of methods that grant you access to the
game state representation. Below list each of the methods that your Tetris Bot
inherits from the `TetrisBot` class for the purpose of analyzing game state:

| Method                    | Value |
|---------------------------|-------|
|`next_move`                |   -   |              
|---------------------------|-------|
|`get_board_width`          |   -   |                    
|---------------------------|-------|
|`get_board_height`         |   -   |                     
|---------------------------|-------|
|`get_current_piece`        |   -   |                      
|---------------------------|-------|
|`get_next_piece`           |   -   |                   
|---------------------------|-------|
|`get_current_piece_id`     |   -   |                         
|---------------------------|-------|
|`get_next_piece_id`        |   -   |                      
|---------------------------|-------|
|`get_current_piece_map`    |   -   |                          
|---------------------------|-------|
|`get_current_piece_coord`  |   -   |                            
|---------------------------|-------|
|`get_score`                |   -   |              
|---------------------------|-------|
|`get_board_state`          |   -   |                    
|---------------------------|-------|
|`get_allotted_time`        |   -   |                      


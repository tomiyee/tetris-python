
from arena import Arena
from argparse import ArgumentParser
import importlib
from tetris import TetrisApp
from stubs import import_player

if __name__ == "__main__":

    # Parse the Arguments
    parser = ArgumentParser()
    parser.add_argument( "-a", "--arena",
        nargs='+',
        help="Use several players in arena",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        help="The name of the module with your tetris AI model. This module should be \n \
    within the `models` directory. An example is given as `ai_rando`.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode: The Tetris Engine will wait until input \
                is received from your Model before updating the frame"
    )
    parser.add_argument(
        '-s',
        '--seed',
        dest='seed',
        default=69,
        help='The seed for the random number generator. Affects block generation'
    )
    args = parser.parse_args()
    args.seed = int(args.seed)

    if args.model:

        model = import_player(args.model)
        # Let the games begin
        App = TetrisApp(model, debug=args.debug, seed=args.seed)
        

    if args.arena:
        players = args.arena
        player_models = [import_player(player) for player in players]
        Arena(player_models, debug = args.debug).run_round_robin(seed=args.seed)

    print("Nothing else to do.")


from arena import Arena
from argparse import ArgumentParser
import importlib

if __name__ == "__main__":

    # Parse the Arguments
    parser = ArgumentParser()
    parser.add_argument(
        "-a",
        "--arena",
        nargs='+',
        help="Use several players in arena",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="file_name",
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
    args = parser.parse_args()

    if "model" in args:
        # Imports the specified model for the Tetris Game
        try:
            module = importlib.import_module(f"models.{args.file_name}.main")
        except: 
            raise Exception("[Error] Could not import specified module", args.file_name)

        # Initialize the model
        model = module.Model()

        # Let the games begin
        App = TetrisApp(model, debug=args.debug)
        App.run()

    if "arena" in args:
        players = args.arena
        player_models = [importlib.import_module(f"models.{player}.main").Model() for player in players]
        for m, p in zip(player_models, players):
            m._name = p


        Arena(player_models).run_round_robin()

    print("Nothing else to do.")

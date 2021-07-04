from itertools import combinations
from multiprocessing import Pool
from tetris import TetrisApp

class Arena():

    def __init__(self, players):
        if len(players) < 2: raise Exception("[Error] Fewer than 2 players specified")
        self.player_pairs = combinations(players, 2)

    def run_game(self, player, seed=10000):
        App = TetrisApp(player)
        score = App.run(seed=seed)
        return score 

    def run_round_robin(self, seed = 1000):
        for a, b in self.player_pairs:
            with Pool(2) as p:
                scores = p.starmap(self.run_game, [(a,seed), (b,seed)])
                print(f"Player Score: {a._name} - {scores[0]}")
                print(f"Player Score: {b._name} - {scores[1]}")
        print("[+] Completed Round Robin") 

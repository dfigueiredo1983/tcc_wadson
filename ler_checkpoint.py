import pickle
from pathlib import Path
import os


# 🔥 STUBS NECESSÁRIOS
class NetFlowProblem:
    pass

class CheckpointCallback:
    pass

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def ler_checkpoint(checkpoint: str):
    with open(checkpoint, 'rb') as f:
        algo = pickle.load(f)

    print("Geração:", algo.n_gen)

    pop = algo.pop

    X = pop.get("X")
    F = pop.get("F")

    print("\nShape X:", X.shape)
    print("Shape F:", F.shape)

    nds = NonDominatedSorting()
    front = nds.do(F, only_non_dominated_front=True)

    print("\nPareto front:")
    print(F[front])

if __name__ == '__main__':
    path_base = "/mnt/d/tcc_aline_resultados"

    list_checkpoints = os.listdir(path_base)

    for checkpoint in list_checkpoints:
        print(checkpoint)
        # caminho = os.path.join(path_base, checkpoint)
        # ler_checkpoint(caminho)

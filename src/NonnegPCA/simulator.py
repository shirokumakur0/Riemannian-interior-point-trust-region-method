import hydra
from numpy import linalg

import sys
sys.path.append('./src/base')
import base_simulator

# Violation for a sphere manifold
def manviofun(problem, x):
    manvio = linalg.norm(x) - 1
    return manvio

class Simulator(base_simulator.Simulator):
    def add_solver_option(self, option):
        option["manviofun"] = manviofun
        # Can add "callbackfun" to option in the same manner.
        return option

@hydra.main(version_base=None, config_path=".", config_name="config_simulation")
def main(cfg):

    # Experiment of nonnegative PCA
    director = Simulator(cfg)
    director.run()

if __name__=='__main__':
    main()
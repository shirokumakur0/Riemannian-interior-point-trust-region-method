import hydra
from numpy import linalg

import sys
sys.path.append('./src/base')
import solver_tuner

# Violation for a sphere manifold
def manviofun(problem, x):
    manvio = linalg.norm(x) - 1
    return manvio

class Tuner(solver_tuner.Tuner):
    def add_solver_option(self, option):
        option["manviofun"] = manviofun
        # Can add "callbackfun" to option in the same manner.
        return option

@hydra.main(version_base=None, config_path=".", config_name="config_tuning")
def main(cfg):

    # Hyperparameter tuning for RALM in nonnegative PCA
    tuner = Tuner(cfg)
    tuner.run()

if __name__=='__main__':
    main()
import hydra

import sys
sys.path.append('./src/base')
import base_simulator
import numpy as np

# Tutorial for your project:
# 1. If your solver employs some callback function or similar ones (e.g., manviofun in RALM), you can integrate them in 'add_solver_option(self, option)',
#    which is assumed to receive, edit and return option. Otherwise, in most cases, you can directly run this file to conduct your experiments,
#    as everything is already implemented in 'src/base/base_simulator.py'.
# 2. If you encounter any errors or need to customize the simulator's behavior for your specific project, you can overwrite the 'simulator' in this file.
#
# Next, please proceed to the 'analyzer.ipynb'.

import os
import copy

def manviofun(problem, x):
    J = x[0]
    R = x[1]
    Q = x[2]
    manvio = 0
    # Chaeck if J is skew-symmetric
    manvio += np.linalg.norm(J + J.T)

    # Check if R and Q are symmetric
    manvio += np.linalg.norm(R - R.T)
    manvio += np.linalg.norm(Q - Q.T)

    # Check if R is positive definite
    R_eigenvalues = np.linalg.eigvalsh(R)  # Eigenvalues of R
    if not np.all(R_eigenvalues > 0):
        print("R is not positive definite.")
        manvio = np.inf
    # Check if Q is positive definite
    Q_eigenvalues = np.linalg.eigvalsh(Q)  # Eigenvalues of Q
    if not np.all(Q_eigenvalues > 0):
        print("Q is not positive definite.")
        manvio = np.inf
    return manvio

class Simulator(base_simulator.Simulator):
    def add_solver_option(self, option):
        option["manviofun"] = manviofun
        # Can add "callbackfun" to option in the same manner.
        return option

    def run(self):
        # Make directories if not exist
        os.makedirs(f'intermediate/{self.cfg.problem_name}', exist_ok=True)  # Create {problem_name} folder under 'data' folder
        os.makedirs(f'intermediate/{self.cfg.problem_name}/{self.cfg.problem_instance}/{self.cfg.problem_initialpoint}', exist_ok=True)  # Create {inst} folder under {problem_name} folder
        self.logger.info(f"Running a simulator of class {self.__class__} -- instance: {self.cfg.problem_instance}, initial point: {self.cfg.problem_initialpoint}")

        # Coordinate problem
        problem_coordinator = self.set_coordinator()
        self.logger.info(f"Running a problem coordinator of class {problem_coordinator.__class__}")
        problem = problem_coordinator.run()
        self.logger.info(f"Finished running a problem coordinator of class {problem_coordinator.__class__}")

        # Loop with respect to solver
        solver_name = self.cfg.solver_name
        for name in solver_name:
            solver = self.set_solver(name)  # set solver
            self.logger.info(f"Running a solver of class {solver.__class__}")
            output = solver.run(copy.deepcopy(problem))  # run the experiments
            # changed the following line
            self.save_output(output.name, output)
            self.logger.info(f"Finished running a solver of class {solver.__class__}")
        self.logger.info(f"Finished running a simulator of class {self.__class__} -- instance: {self.cfg.problem_instance}, initial point: {self.cfg.problem_initialpoint}")

@hydra.main(version_base=None, config_path=".", config_name="config_simulation")
def main(cfg):

    director = Simulator(cfg)
    director.run()

if __name__=='__main__':
    main()
import hydra
from numpy import linalg

import sys
sys.path.append('./src/base')
import base_simulator

import os
import copy

# Violation for a sphere manifold
def manviofun(problem, x):
    manvio = linalg.norm(x) - 1
    return manvio

class Simulator(base_simulator.Simulator):
    def add_solver_option(self, option):
        option["manviofun"] = manviofun
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

            self.save_output(output.name, output)
            self.logger.info(f"Finished running a solver of class {solver.__class__}")
        self.logger.info(f"Finished running a simulator of class {self.__class__} -- instance: {self.cfg.problem_instance}, initial point: {self.cfg.problem_initialpoint}")



@hydra.main(version_base=None, config_path=".", config_name="config_simulation")
def main(cfg):

    # Experiment of nonnegative PCA
    director = Simulator(cfg)
    director.run()

if __name__=='__main__':
    main()
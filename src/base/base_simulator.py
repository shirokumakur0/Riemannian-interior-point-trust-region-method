import logging, os, copy, importlib, csv
import numpy as np
import pandas as pd
import sys
sys.path.append('./src/solver')

class Simulator():
    def __init__(self, cfg):
        # Assertion
        assert hasattr(cfg, 'problem_name')
        assert hasattr(cfg, 'problem_instance')
        assert hasattr(cfg, 'problem_initialpoint')
        assert hasattr(cfg, 'problem_coordinator_name')
        assert hasattr(cfg, 'solver_name')
        assert hasattr(cfg, 'solver_option')

        # Set the configuration file
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

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
            # output = solver.run(problem)  # run the experiments
            self.save_output(name, output)
            self.logger.info(f"Finished running a solver of class {solver.__class__}")
        self.logger.info(f"Finished running a simulator of class {self.__class__} -- instance: {self.cfg.problem_instance}, initial point: {self.cfg.problem_initialpoint}")

    def set_coordinator(self):
        cfg = self.cfg
        problem_coordinator_name = cfg.problem_coordinator_name
        module_coordinator = importlib.import_module(problem_coordinator_name)  # dynamic importation
        coordinator = module_coordinator.Coordinator(cfg)
        return coordinator

    def set_solver(self, solver_name):
        cfg = self.cfg
        solver_option = cfg.solver_option

        # Set option for 'solver_name'
        option = copy.deepcopy(dict(solver_option.common))
        if hasattr(solver_option, solver_name):
            specific = dict(getattr(solver_option, solver_name))
            option.update(specific)  # putting common before specific

        option = self.add_solver_option(option)  # add options depending on problem structures.

        # Dynamic importation of solver
        module_solver = importlib.import_module(solver_name)
        class_solver = getattr(module_solver, solver_name)
        solver = class_solver(option)
        return solver

    # Can be overridden in the subclass.
    # Usually used to add a callback function and other specific functions depending on problem structures.
    # 'callbackfun' and 'manviofun' in RALM are typical examples.
    def add_solver_option(self, option):
        return option

    def save_output(self, solver_name, output):
        # for loop with respect to attributes in output
        for attr, content in vars(output).items():
            # Set a path where the outputs is stored
            csvpath = f'{self.cfg.output_path}/{solver_name}_{attr}.csv'

            # Store content based on its data type by pandas and numpy libraries.
            # If you require saving data in a different format, override this function to accommodate your specific needs.
            if isinstance(content, np.matrix) or isinstance(content, np.ndarray):
                np.savetxt(csvpath, content)
            elif isinstance(content, dict):
                with open(csvpath, mode="w") as csvfile:
                    for key, value in content.items():
                        if not isinstance(value, list):
                            content[f"{key}"] = [value]
                    df = pd.DataFrame(content)
                    df.to_csv(csvpath, index=False)
            else:
                with open(csvpath, 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(content)
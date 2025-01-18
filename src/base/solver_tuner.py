import hydra, copy, importlib, logging, os, csv
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('./src/solver')

class Tuner():
    def __init__(self, cfg):
        # Assertion
        assert hasattr(cfg, 'problem_name')
        assert hasattr(cfg, 'problem_instance')
        assert hasattr(cfg, 'problem_initialpoint')
        assert hasattr(cfg, 'problem_coordinator_name')
        assert hasattr(cfg, 'solver_name')
        assert hasattr(cfg, 'solver_evaluation')
        assert hasattr(cfg, 'solver_option')

        # Set instance variables
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.solver_index = 0
        self.solver_settings = {}
        self.scores = {}

    def run(self):
        # Logging
        self.logger.info(f"Running a solver tuner of class {self.__class__}")

        # Set the coordinator and the solver with the solver_option
        self.set_class_coordinator()
        self.set_class_solver()
        solver_option = dict(self.cfg.solver_option)

        # Grid search for solver settings
        self.gridsearch({}, solver_option)

        # Save solver_settings
        self.save_to_tuning(pd.DataFrame(self.solver_settings), "solver_settings")

        # Make a figure of performance profile
        self.performance_profile(self.scores)

        # Logging
        self.logger.info(f"Finished running a solver tuner of class {self.__class__}")

    # Used to save information as solver_settings.csv, score.csv, and relative_score.csv.
    def save_to_tuning(self, dataframe, file_name):
        os.makedirs(f"{self.cfg.output_path}", exist_ok=True)
        csv_path = f"{self.cfg.output_path}/{file_name}.csv"
        dataframe.to_csv(csv_path)

    # Save each experimental result if self.cfg.save_experiment is True.
    def save_experiment_output(self, solver_index, instance, output):
        # for loop with respect to attributes in output
        for attr, content in vars(output).items():
            # Set a path where the outputs is stored
            os.makedirs(self.cfg.output_path, exist_ok=True)
            os.makedirs(f"{self.cfg.output_path}/setting_{solver_index}", exist_ok=True)
            csvpath = f'{self.cfg.output_path}/setting_{solver_index}/{instance}_{attr}.csv'

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

    # Run an experiment with a solver option loaded as the argument.
    def run_experiment(self, option):
        # Set an instance of the solver with the argument (option)
        option = self.add_solver_option(option)  # add options depending on problem structures.
        solver = self.class_solver(option)
        self.solver_settings[f"setting_{self.solver_index}"] = option

        cfg = self.cfg
        evaluation_index = cfg.solver_evaluation  # index for performance profile

        # Set the configuration file for problem coordinator.
        # problem_instance, one of its attributes, will be added in the following loop.
        coordinator_cfg = {"problem_name": cfg.problem_name,
                            "problem_initialpoint": cfg.problem_initialpoint,
                            "problem_coordinator_name": cfg.problem_coordinator_name}
        coordinator_cfg = OmegaConf.create(coordinator_cfg)

        score_per_instances = {}  # stores the score of each problem_instance
        for instance in cfg.problem_instance:
            # Construct the problem with 'instance'
            coordinator_cfg.problem_instance = instance
            problem_coordinator = self.class_coordinator(coordinator_cfg)
            self.logger.info(f"Running a problem coordinator of class {problem_coordinator.__class__} -- instance: {instance}")
            problem = problem_coordinator.run()
            self.logger.info(f"Finished running a problem coordinator of class {problem_coordinator.__class__} -- instance: {instance}")

            # Solve the problem
            self.logger.info(f"Running a solver of class {solver.__class__} with setting_{self.solver_index}")
            output = solver.run(copy.deepcopy(problem))
            # output = solver.run(problem)
            self.logger.info(f"Finished running a solver of class {solver.__class__} with setting_{self.solver_index}")
            if self.cfg.save_experiment:
                self.save_experiment_output(self.solver_index, instance, output)

            # Store the value of evaluation_index from the experimental result
            score = output.log[evaluation_index][-1]
            score_per_instances[instance] = score

        self.scores[f"setting_{self.solver_index}"] = score_per_instances
        self.solver_index += 1  # update solver_index

    # Can be overridden in the subclass.
    # Usually used to add a callback function and other specific functions depending on problem structures.
    # 'callbackfun' and 'manviofun' in RALM are typical examples.
    def add_solver_option(self, option):
        return option

    def gridsearch(self, option, params):
        if params == {}:  # has a complete option
            self.logger.info(f"Running an experiment with {option}")
            self.run_experiment(option) # running an experiment with the option
            self.logger.info(f"Finished an experiments with {option}")
        else:  # recursion
            params_keys = list(params.keys())
            key = params_keys[0]  # pick up a key (one of hyperparameters)
            values = params[key]  # take its value
            for val in values:
                new_option = copy.deepcopy(option)
                new_option[key] = val
                new_params = copy.deepcopy(params)
                del new_params[key]
                self.gridsearch(new_option, new_params)

    def set_class_solver(self):
        # Dynamically importing the solver
        solver_name = self.cfg.solver_name
        module_solver = importlib.import_module(solver_name)
        self.class_solver = getattr(module_solver, solver_name)

    def set_class_coordinator(self):
        # Dynamically importing the problem coordinator
        problem_coordinator_name = self.cfg.problem_coordinator_name
        module_coordinator = importlib.import_module(problem_coordinator_name)
        self.class_coordinator = getattr(module_coordinator, 'Coordinator')

    def performance_profile(self, scores_dict):
        max_threshold = self.cfg.max_threshold
        threshold_step = self.cfg.threshold_step

        # Convert a dictionary to a dataframe and save the data
        scores = pd.DataFrame(scores_dict)
        self.save_to_tuning(scores, "score")

        # Compute and save relative scores
        min_values = scores.min(axis=1)
        relative_values = scores.div(min_values, axis=0)
        self.save_to_tuning(relative_values, "relative_score")

        numrow = relative_values.shape[0]
        solver_settings = relative_values.columns
        x_values = [i / threshold_step for i in range(threshold_step-1, threshold_step * (max_threshold + 1))]
        _, ax = plt.subplots()

        for solver in solver_settings:
            relscore = relative_values.loc[:,solver]
            y_values = [(relscore <= thrshld).sum() / numrow for thrshld in x_values]
            ax.plot(x_values, y_values, label=solver)
            ax.legend()

        # Make a figure and save it
        ax.legend(bbox_to_anchor=(1,0.05), loc='lower right')
        plt.xlabel('$t$')
        plt.ylabel('P$(r_p \\leq t: 1\\leq s \\leq n)$')
        plt.title(f'Performance profile ({self.cfg.solver_name} at {self.cfg.problem_name})')
        plt.savefig(f"{self.cfg.output_path}/performance_profile.pdf")

@hydra.main(version_base=None, config_path=".", config_name="config_tuning")
def main(cfg):
    tuner = Tuner(cfg)
    tuner.run()

if __name__=='__main__':
    main()
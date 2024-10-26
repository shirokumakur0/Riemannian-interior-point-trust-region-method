import hydra, copy, time, wandb
import numpy as np
from dataclasses import dataclass

import sys
sys.path.append('./src/base')
from base_solver import Solver, BaseOutput

# Tutorial for your project:
# 1. Before making any edits to this file, make sure to change the file name and the following class
#    from 'solver_template'. They should be the same.
# 2. Implement the dataclass 'Output'. It is a subclass of 'BaseOutput' in 'src/base/base_solver.py'
#    and includes the optimization results.
# 3. Implement the following methods in the class:
#    -- '__init__': This method is a constructor. It should receive the argument 'options' other than 'self'.
#           You can add default options for your solver, such as stopping criteria and hyperparameters in the method.
#    -- 'run': This method should receive the argument 'problem' other than 'self'
#           and return an instance of ‘Output’. Implement your algorithm in the method.
#    -- 'evaluation': This method can receive various arguments (typically, problem, current and previous points, and others)
#           and should return a dictionary that stores the current evaluations of the optimization. The method is usually called in the ‘run’ method at each step.
#           You can edit the method as needed.
#    -- 'solver status': This method can receive various arguments (typically, parameters in the solver)
#           and should return a dictionary that stores the current status of the solver. The method is usually called in the ‘run’ method at each step.
#           You can record any status of the solver in the method.

@dataclass
class Output(BaseOutput):
    # Inherited the following from BaseOutput:
    # x: field(default_factory=list)
    # option: Optional[Dict]
    # log: Optional[Dict]
    # Added other variables as needed.
    pass

class solver_template(Solver):
    def __init__(self, option):
        # Default setting for algorithm
        default_option = {
            # Stopping criteria
            'maxtime': 100,
            'maxiter': 100,

            # Display setting
            'verbosity': 0,

            # Wandb logging
            'wandb_logging': False,

            # Callback function in 'evaluation' method.
            'callbackfun': lambda problem, xCur, option, eval: eval,

            # Add other options used in your solver.

        }
        # Merge default_option and the argument
        default_option.update(option)  # putting the setting in the default_option before that in the argument
        self.option = default_option
        self.log = {}  # will be filled in self.add_log

        if self.option["wandb_logging"]:
            _ = wandb.init(project=self.option["wandb_project"],  # the project name where this run will be logged
                             config=self.option)  # save hyperparameters and metadata

    # Running an experiment
    def run(self, problem):
        # Assertion
        assert hasattr(problem, 'costfun')
        assert hasattr(problem, 'initialpoint')
        # Add assertions for other attributes in your problem.

        # Initialization
        costfun = problem.costfun
        verbosity = self.option.verbosity
        ## Set a current point. Typically used the info in 'problem.initialpoint'
        xCur = [] 
        ##
        xPrev = copy.deepcopy(xCur)
        iter = 0
        start_time = time.time()

        # The first evaluation and logging
        eval_log = self.evaluation(problem, xPrev, xCur)
        solver_log = self.solver_status()
        self.add_log(iter, start_time, eval_log, solver_log)

        # Set specific stopping criteria
        ## Add the criteria other than time and iteration.
        criterion = (False, "message")
        stopping_criteria =[criterion]
        ##

        while True:
            if verbosity:
                print(f"Iter: {iter}, Cost: {costfun(xCur)}")

            # Check stopping criteria (time, iteration and specific ones)
            stop, reason = self.check_stoppingcriterion(start_time, iter, stopping_criteria)
            if stop:
                self.option["stoppingcriterion"] = reason
                if verbosity:
                    print(reason)
                break
            # Count an iteration
            iter += 1

            ## Computation for a new iterate.
            ## Write here.

            # Update the iterate.
            # Write here
            xCur = []

            # Evaluation and logging
            eval_log = self.evaluation(problem, xPrev, xCur)
            solver_log = self.solver_status()
            self.add_log(iter, start_time, eval_log, solver_log)

            # Update previous x and stopping criteria
            xPrev = copy.deepcopy(xCur)
            criterion = (False, "message")
            stopping_criteria =[criterion]

        # After exiting while loop, we return the final output
        output = Output(x=xCur,
                        option=copy.deepcopy(self.option),
                        log=self.log)
        return output

    # Will be overridden in subclasses as needed.
    def evaluation(self, problem, xPrev, xCur , *args, **kwargs):
        # Cost evaluation
        costfun = problem.costfun
        cost = costfun(xCur)

        # Distance evaluation
        dist = np.linalg.norm(xPrev, xCur)

        # Write here. Add other evaluations as needed.

        eval = {"cost": cost,
                "distance": dist
                }

        option = self.option
        callbackfun = option["callbackfun"]
        eval = callbackfun(problem, xCur, option, eval)

        return eval

    # Will be overridden in subclasses as needed.
    def solver_status(self, *args, **kwargs):

        # Write here. Add solver status as needed.

        solver_status = {}
        return solver_status

@hydra.main(version_base=None, config_path="../NonnegPCA", config_name="config_simulation")
def main(cfg):  # Experiment of nonnegative PCA. Mainly for debugging

    # Import a problem set
    sys.path.append('./src/your-project-name')
    import coordinator

    # Call a problem coordinator
    your_coordinator = coordinator.Coordinator(cfg)
    problem = your_coordinator.run()

    # Solver option setting
    solver_option = cfg.solver_option
    option = copy.deepcopy(dict(solver_option["common"]))
    specific = dict(getattr(solver_option, "solver_template (your-solver-name)"))
    option.update(specific)

    # Run the experiment
    solver = solver_template(option)
    output = solver.run(problem)
    print(output)

if __name__=='__main__':
    main()
import time, wandb
from dataclasses import dataclass, field
from typing import Optional, Dict
from abc import ABCMeta

@dataclass
class BaseOutput:
    x: field(default_factory=list)
    option: Optional[Dict]
    log: Optional[Dict]

class Solver(object, metaclass=ABCMeta):
    # Will be overridden in subclasses as needed.
    def __init__(self, solver_option, *args, **kwargs):
        # Default setting for augmented Lagrangian method
        default_option = {
            # Stopping criteria
            'maxtime': 100,
            'maxiter': 100,

            # Wandb logging
            'wandb_logging': False,

            # Callback function at each step.
            'callbackfun': lambda problem, xCur, option, eval: eval,
        }

        # Merge default_option and the argument
        default_option.update(solver_option)  # putting the setting in the default_option before that in the argument
        self.option = default_option
        self.log = {}

    # Overridden in subclasses
    def run(self, problem):
        pass

    # Will be overridden in subclasses as needed.
    def evaluation(self, problem, xCur, *args, **kwargs):
        eval = {}
        return eval

    # Will be overridden in subclasses as needed.
    def solver_status(self, *args, **kwargs):
        solver_status = {}
        return solver_status

    # Save optimization process
    def add_log(self, iter, start_time, eval, solver_status):
        # Logging in self.log
        if iter == 0:  # only for the first evaluation
            self.log["iteration"] = [0]
            run_time = 0
            self.log["time"] = [run_time]
            for key, value in eval.items():
                self.log[f"{key}"] = [value]
            for key, value in solver_status.items():
                self.log[f"{key}"] = [value]
        else:
            self.log["iteration"].append(iter)
            run_time = time.time() - start_time
            self.log["time"].append(run_time)
            for key, value in eval.items():
                self.log[f"{key}"].append(value)
            for key, value in solver_status.items():
                self.log[f"{key}"].append(value)

        # Wandb logging
        if self.option["wandb_logging"]:
            wandblog = {"time": run_time}
            wandblog.update(eval)
            wandblog.update(solver_status)
            wandb.log(wandblog)

    def check_stoppingcriterion(self, start_time, iter, stopping_criteria):
        option = self.option
        maxtime = option["maxtime"]
        maxiter = option["maxiter"]

        stop = False
        reason = None

        run_time = time.time() - start_time
        if run_time >= maxtime:
            stop = True
            reason = (f"Max time exceeded; runtime={run_time:.2f} and maxtime={maxtime}")
        elif iter >= maxiter:
            stop = True
            reason = (f"Max iteration count reached; maxiter={maxiter} after {run_time:.2f} seconds")

        for flag, msg in stopping_criteria:
            if flag:
                stop = True
                reason = (f"{msg} after {run_time:.2f} seconds")

        return stop, reason
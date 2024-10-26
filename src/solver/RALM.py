import hydra, copy, time, pymanopt, wandb
import numpy as np
from dataclasses import dataclass, field

from utils import build_Lagrangefun, evaluation

import sys
sys.path.append('./src/base')
from base_solver import Solver, BaseOutput

@dataclass
class Output(BaseOutput):
    ineqLagmult: field(default_factory=list)
    eqLagmult: field(default_factory=list)

# Riemannian Augmented Lagrange function
def build_almfun(ineqLagmult, eqLagmult, rho, costfun, ineqconstraints, eqconstraints, manifold):
    @pymanopt.function.autograd(manifold)
    def almfun(point):
        val = costfun(point)
        violation = 0
        if ineqconstraints.has_constraint:
            for idx in range(ineqconstraints.num_constraint):
                funval = (ineqconstraints.constraint[idx])(point)
                violation += max(0, (ineqLagmult[idx]/rho) + funval) ** 2 
        if eqconstraints.has_constraint:
            for idx in range(eqconstraints.num_constraint):
                funval = (eqconstraints.constraint[idx])(point)
                violation += (funval + (eqLagmult[idx]/rho)) ** 2

        violation *= rho * 0.5
        val += violation
        return val
    return almfun

class RALM(Solver):
    def __init__(self, option):
        # Default setting for augmented Lagrangian method
        default_option = {
            # Stopping criteria
            'maxtime': 100,
            'maxiter': 100,
            'tolresid': 1e-6,

            # Outer loop setting
            'rho': 1,
            'bound': 20,
            'tau': 0.8,
            'thetarho': 0.3,
            'numOuterItertgn': 30,

            # Inner loop setting
            'innersubsolver': "SteepestDescent",
            'maxInnerIter': 200,
            'startingtolgradnorm': 1e-3,
            'endingtolgradnorm': 1e-6,
            'innerminstepsize': 1e-10,

            # Display setting
            'verbosity': 0,

            # Measuring violation for manifold constraints in 'self.compute_residual'
            'manviofun': lambda problem, x: 0,

            # Callback function at each step.
            'callbackfun': lambda problem, x, eval: eval,

            # Wandb logging
            'wandb_logging': False
        }
        # Merge default_option and the argument
        default_option.update(option)  # putting the setting in the default_option before that in the argument
        self.option = default_option
        self.log = {}  # will be filled in self.add_log

        if self.option["wandb_logging"]:
            _ = wandb.init(project=self.option["wandb_project"],  # the project name where this run will be logged
                            name = "RALM",  # the name of the run
                            config=self.option)  # save hyperparameters and metadata

    # Running an experiment
    def run(self, problem):
        # Assertion
        assert hasattr(problem, 'searchspace')
        assert hasattr(problem, 'costfun')
        assert hasattr(problem, 'eqconstraints')
        assert hasattr(problem, 'ineqconstraints')
        assert hasattr(problem, 'initialpoint')
        assert hasattr(problem, 'initialineqLagmult')
        assert hasattr(problem, 'initialeqLagmult')

        # Set the optimization problem
        costfun = problem.costfun
        ineqconstraints = problem.ineqconstraints
        eqconstraints = problem.eqconstraints
        manifold = problem.searchspace

        # Set initial points
        xCur = problem.initialpoint
        ineqLagCur = problem.initialineqLagmult
        eqLagCur = problem.initialeqLagmult
        xPrev = copy.deepcopy(xCur)
        OuterIteration = 0
        start_time = time.time()

        # Set hyperparameters
        option = self.option
        rho = option["rho"]
        oldacc = float('inf')
        bound = option["bound"]
        tau = option["tau"]
        thetarho = option["thetarho"]
        endingtolgradnorm = option["endingtolgradnorm"]
        verbosity = option["verbosity"]

        # Set the subsolver
        innersubsolver = option["innersubsolver"]
        maxInnerIter=option["maxInnerIter"]
        innerminstepsize = option["innerminstepsize"]
        tolgradnorm = option["startingtolgradnorm"]
        thetatolgradnorm = np.power(option["endingtolgradnorm"]/option["startingtolgradnorm"],
                                    1/option["numOuterItertgn"])

        # The first evaluation and logging
        manviofun = option["manviofun"]
        callbackfun = option["callbackfun"]
        eval_log = evaluation(problem, xPrev, xCur, ineqLagCur, eqLagCur, manviofun, callbackfun)
        solver_log = self.solver_status(rho, ineqLagCur, eqLagCur, ineqconstraints, eqconstraints)
        self.add_log(OuterIteration, start_time, eval_log, solver_log)

        # Preparation for check stopping criteria
        residual = eval_log["residual"]
        tolresid = option["tolresid"]
        residual_criterion = (residual <= tolresid, "KKT residual tolerance reached; current residual=" + str(residual) + " and tolresid=" + str(tolresid))
        stopping_criteria =[residual_criterion]

        while True:
            if verbosity:
                print(f"Iter: {OuterIteration}, Cost: {costfun(xCur)}, KKT residual: {residual}")

            # Check stopping criteria (time, iteration and residual)
            stop, reason = self.check_stoppingcriterion(start_time, OuterIteration, stopping_criteria)
            if stop:
                self.option["stoppingcriterion"] = reason
                if verbosity:
                    print(reason)
                break
            # Count an iteration
            OuterIteration += 1

            # Set augmented Lagrangian
            almfun = build_almfun(ineqLagmult=ineqLagCur,
                                    eqLagmult=eqLagCur,
                                    rho=rho, 
                                    costfun=costfun,
                                    ineqconstraints=ineqconstraints,
                                    eqconstraints=eqconstraints,
                                    manifold=manifold)

            # Construct the subproblem and the subsolver
            subproblem = pymanopt.Problem(manifold, almfun)
            class_subsolver = getattr(pymanopt.optimizers, innersubsolver)
            subsolver = class_subsolver(max_iterations=maxInnerIter,
                                        min_step_size=innerminstepsize,
                                        min_gradient_norm=tolgradnorm,
                                        verbosity=0)

            # Solve the subproblem
            result = subsolver.run(problem=subproblem,
                                   initial_point=xCur)
            xCur = result.point

            # Update Lagrange multipliers
            newacc = 0
            if ineqconstraints.has_constraint:
                for idx in range(ineqconstraints.num_constraint):
                    ineqcstrfun = ineqconstraints.constraint[idx]
                    ineqcost = ineqcstrfun(xCur)
                    newacc = max(newacc, abs(max(-ineqLagCur[idx]/rho, ineqcost)))
                    ineqLagCur[idx] = min(bound, max(0, ineqLagCur[idx] + rho * ineqcost))
            if eqconstraints.has_constraint:
                for idx in range(eqconstraints.num_constraint):
                    eqcstrfun = eqconstraints.constraint[idx]
                    eqcost = eqcstrfun(xCur)
                    newacc = max(newacc, abs(eqcost))
                    eqLagCur[idx] = min(bound, max(-bound, eqLagCur[idx] + rho * eqcost))

            # Update rho.
            # Attention: the following update strategy has been adopted
            # in the Matlab imprementation in Github by losangle (https://github.com/losangle/Optimization-on-manifolds-with-extra-constraints)
            # In the original paper (https://arxiv.org/abs/1901.10000),
            # the correct condition seems to be 'OuterIter != 0 and newacc > tau * oldacc'
            if OuterIteration == 0 or newacc > tau * oldacc:
                rho = rho/thetarho
            oldacc = newacc
            tolgradnorm = max(endingtolgradnorm, tolgradnorm * thetatolgradnorm)

            # Evaluation and logging
            eval_log = evaluation(problem, xPrev, xCur, ineqLagCur, eqLagCur, manviofun, callbackfun)
            solver_log = self.solver_status(rho, ineqLagCur, eqLagCur, ineqconstraints, eqconstraints)
            self.add_log(OuterIteration, start_time, eval_log, solver_log)

            # Update previous x and residual
            xPrev = copy.deepcopy(xCur)
            residual = eval_log["residual"]
            residual_criterion = (residual <= tolresid, "KKT residual tolerance reached; current residual=" + str(residual) + " and tolresid=" + str(tolresid))
            stopping_criteria =[residual_criterion]

        # After exiting while loop, we return the final output
        output = Output(x=xCur,
                        ineqLagmult=ineqLagCur,
                        eqLagmult=eqLagCur,
                        option=copy.deepcopy(self.option),
                        log=self.log)
        
        if self.option["wandb_logging"]:
            wandb.finish()
        
        return output

    # Examine the max value of Lagrange multipliers
    def solver_status(self, rho, ineqLagmult, eqLagmult, ineqconstraints, eqconstraints):
        solver_status = {}
        solver_status["rho"] = rho
        maxabsLagmult = float('-inf')

        if ineqconstraints.has_constraint:
            for Lagmult in ineqLagmult:
                maxabsLagmult = max(maxabsLagmult, abs(Lagmult))
        if eqconstraints.has_constraint:
            for Lagmult in eqLagmult:
                maxabsLagmult = max(maxabsLagmult, abs(Lagmult))

        solver_status["maxabsLagmult"] = maxabsLagmult
        return solver_status

@hydra.main(version_base=None, config_path="../NonnegPCA", config_name="config_simulation")
def main(cfg):  # Experiment of nonnegative PCA. Mainly for debugging

    # Import a problem set from NonnegPCA
    sys.path.append('./src/NonnegPCA')
    import coordinator

    # Call a problem coordinator
    nonnegPCA_coordinator = coordinator.Coordinator(cfg)
    problem = nonnegPCA_coordinator.run()

    # Solver option setting
    solver_option = cfg.solver_option
    option = copy.deepcopy(dict(solver_option["common"]))
    specific = dict(getattr(solver_option, "RALM"))
    option.update(specific)

    # Run the experiment
    ralmsolver = RALM(option)
    output = ralmsolver.run(problem)
    print(output)

if __name__=='__main__':
    main()
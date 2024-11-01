import hydra, copy, time, pymanopt, wandb
import numpy as np
from dataclasses import dataclass, field
from utils import build_Lagrangefun, evaluation, tangentorthobasis, hessianmatrix
from cvxopt import matrix, solvers

import sys
sys.path.append('./src/base')
from base_solver import Solver, BaseOutput

@dataclass
class Output(BaseOutput):
    ineqLagmult: field(default_factory=list)
    # eqLagmult: field(default_factory=list)

# Riemannian interior point trust region method
class RIPTRM(Solver):
    def __init__(self, option):
        # Default setting for augmented Lagrangian method
        default_option = {
            # Stopping criteria
            'maxtime': 100,
            'maxiter': 100,
            'tolresid': 1e-6,


            # Line search setting
            'rho': 1,
            'tau': 0.5,
            'beta': 0.9,
            'gamma': 0.25,
            'linesearch_max': 400,

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
        iteration = 0
        start_time = time.time()

        # Set hyperparameters
        option = self.option
        rho = option["rho"]
        tau = option["tau"]
        beta = option["beta"]
        gamma = option["gamma"]
        verbosity = option["verbosity"]

        # The first evaluation and logging
        manviofun = option["manviofun"]
        callbackfun = option["callbackfun"]
        eval_log = evaluation(problem, xPrev, xCur, ineqLagCur, eqLagCur, manviofun, callbackfun)
        solver_log = self.solver_status(
                            ineqLagCur,
                            eqLagCur,
                            ineqconstraints,
                            eqconstraints,
                            rho)

        self.add_log(iteration, start_time, eval_log, solver_log)

        # Preparation for check stopping criteria
        residual = eval_log["residual"]
        tolresid = option["tolresid"]
        residual_criterion = (residual <= tolresid, "KKT residual tolerance reached; current residual=" + str(residual) + " and tolresid=" + str(tolresid))
        stopping_criteria =[residual_criterion]

        while True:
            if verbosity:
                print(f"Iter: {iteration}, Cost: {costfun(xCur)}, KKT residual: {residual}")

            # Check stopping criteria (time, iteration and residual)
            stop, reason = self.check_stoppingcriterion(start_time, iteration, stopping_criteria)
            if stop:
                self.option["stoppingcriterion"] = reason
                if verbosity:
                    print(reason)
                break
            # Count an iteration
            iteration += 1

            # Preparation for constructing the subproblem
            costLag = build_Lagrangefun(ineqLagmult=ineqLagCur,
                                        eqLagmult=eqLagCur,
                                        costfun=costfun,
                                        ineqconstraints=ineqconstraints,
                                        eqconstraints=eqconstraints,
                                        manifold=manifold)
            Lagproblem = pymanopt.Problem(manifold, costLag)
            orthobasis = tangentorthobasis(manifold, xCur, manifold.dim)


            # Line search
            # f0 = ell_1penaltyfun(xCur, rho, costfun, ineqconstraints, eqconstraints)
            # gammadf0 = gamma * df0
            # stepsize = 1
            # newx = manifold.retraction(xCur, stepsize * dir)
            # newf = ell_1penaltyfun(newx, rho, costfun, ineqconstraints, eqconstraints)
            # linesearch_status = 1
            # linesearch_counter = 0
            # while newf > f0 - stepsize * gammadf0:
            #     linesearch_counter += 1
            #     if linesearch_counter >= option["linesearch_max"]:
            #         linesearch_status = 0
            #         break
            #     stepsize *= beta
            #     gammadf0 *= beta
            #     newx = manifold.retraction(xCur, stepsize * dir)
            #     newf = ell_1penaltyfun(newx, rho, costfun, ineqconstraints, eqconstraints)

            # Update variables
            # xCur = newx
            # ineqLagCur = ineqLagsol
            # eqLagCur = eqLagsol

            # Evaluation and logging
            eval_log = evaluation(problem, xPrev, xCur, ineqLagCur, eqLagCur, manviofun, callbackfun)
            solver_log = self.solver_status(
                      ineqLagCur,
                      eqLagCur,
                      ineqconstraints,
                      eqconstraints,
                      rho,
                      upsilon=upsilon,
                      sol=sol,
                      stepsize=stepsize,
                      linesearch_status=linesearch_status,
                      linesearch_counter=linesearch_counter
                      )
            self.add_log(iteration, start_time, eval_log, solver_log)

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

    # Examine the solver status
    def solver_status(self,
                      ineqLagmult,
                      eqLagmult,
                      ineqconstraints,
                      eqconstraints,
                      rho,
                      upsilon=None,
                      sol=None,
                      stepsize=None,
                      linesearch_status=None,
                      linesearch_counter=None
                      ):
        solver_status = {}
        solver_status["rho"] = rho
        solver_status["upsilon"] = upsilon

        maxabsLagmult = float('-inf')
        if ineqconstraints.has_constraint:
            for Lagmult in ineqLagmult:
                maxabsLagmult = max(maxabsLagmult, abs(Lagmult))
        if eqconstraints.has_constraint:
            for Lagmult in eqLagmult:
                maxabsLagmult = max(maxabsLagmult, abs(Lagmult))
        solver_status["maxabsLagmult"] = maxabsLagmult


        solver_status["stepsize"] = stepsize
        solver_status["linesearch_status"] = linesearch_status
        solver_status["linesearch_counter"] = linesearch_counter

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
    if hasattr(solver_option, "RIPTRM"):
        specific = dict(getattr(solver_option, "RIPTRM"))
        option.update(specific)

    # Run the experiment
    riptrmsolver = RIPTRM(option)
    output = riptrmsolver.run(problem)
    print(output)

if __name__=='__main__':
    main()
import hydra, copy, time, pymanopt, wandb
import numpy as np
from dataclasses import dataclass

from utils import evaluation, NonlinearProblem, Output
from typing import Any

import sys
sys.path.append('./src/base')
from base_solver import Solver

@dataclass
class SubProblem:
    manifold: Any
    cost: Any
    riemannian_gradient: Any
    preconditioner: Any

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
            'LagmultUnbdUpdate': False,  # True if RALM aims to find AKKT points else False.

            # Inner loop setting
            'innersubsolver':"SteepestDescent",  # "ConjugateGradient",
            'maxInnerIter': 200,
            'startingtolgradnorm': 1e-3,
            'endingtolgradnorm': 1e-6,
            'innerminstepsize': 1e-10,

            # Display setting
            'verbosity': 1,

            # Measuring violation for manifold constraints in 'self.compute_residual'
            'manviofun': lambda problem, x: 0,

            # Callback function at each step.
            'callbackfun': lambda problem, x, y, z, eval: eval,

            # Wandb logging
            'wandb_logging': False,

            # Exit on error
            'do_exit_on_error': True,
        }
        # Merge default_option and the argument
        default_option.update(option)  # putting the setting in the default_option before that in the argument
        self.option = default_option
        self.log = {}  # will be filled in self.add_log
        self.name = f"RALM_{self.option['innersubsolver']}"
        self.initialize_wandb()

        # if self.option["wandb_logging"]:
        #     wandb.finish()
        #     _ = wandb.init(project=self.option["wandb_project"],  # the project name where this run will be logged
        #                     name = f"RALM_{self.option['innersubsolver']}",  # the name of the run
        #                     config=self.option)  # save hyperparameters and metadata

    def set_LagEvals(self, problem):
        # Set ineqLagCurUnbd and eqLagCurUnbd if RALM aims to find AKKT points (the update is based on the paper by Yamakawa and Sato.)
        # Otherwise, use ineqLagCur and eqLagCur for the evaluation based on the paper by Liu and Boumal.
        has_ineqconstraints = problem.has_ineqconstraints
        has_eqconstraints = problem.has_eqconstraints
        LagmultUnbdUpdate = self.option["LagmultUnbdUpdate"]

        if has_ineqconstraints and LagmultUnbdUpdate:
            ineqLagEval = self.ineqLagCurUnbd
        else:
            ineqLagEval = self.ineqLagCur
        if has_eqconstraints and LagmultUnbdUpdate:
            eqLagEval = self.eqLagCurUnbd
        else:
            eqLagEval = self.eqLagCur
        return ineqLagEval, eqLagEval

    # def assert_hasattr(self, obj, *args):
    #     for arg in args:
    #         assert hasattr(obj, arg), f"Missing attribute: {arg}"

    def preprocess(self, problem):
        # Set initial points
        xCur = copy.deepcopy(problem.initialpoint)
        self.ineqLagCur = copy.deepcopy(problem.initialineqLagmult)
        self.eqLagCur = copy.deepcopy(problem.initialeqLagmult)

        # Set parameters
        option = self.option
        self.rho = copy.deepcopy(self.option["rho"])
        self.oldacc = float('inf')


        # Construct ineqLagCurUnbd and eqLagCurUnbd if RALM aims to find AKKT points.
        # The update is based on the paper by Yamakawa and Sato (https://link.springer.com/article/10.1007/s10589-021-00336-w).
        verbosity = option["verbosity"]
        LagmultUnbdUpdate = option["LagmultUnbdUpdate"]
        if LagmultUnbdUpdate:  # set the unbounded initial Lagrange multipliers
            if verbosity:
                print("Enabled unbounded updates for the Lagrange multipliers.")
            self.ineqLagCurUnbd = copy.deepcopy(problem.initialineqLagmult)
            self.eqLagCurUnbd = copy.deepcopy(problem.initialeqLagmult)

        # Set paramters fo the subsolver
        self.thetatolgradnorm = np.power(option["endingtolgradnorm"]/option["startingtolgradnorm"],
                                    1/option["numOuterItertgn"])

        # Compute ineqLagEval and eqLagEval
        ineqLagEval, eqLagEval = self.set_LagEvals(problem)

        return xCur, ineqLagEval, eqLagEval

    def step(self, problem, xCur, OuterIteration):
        # Set functions
        costfun = problem.cost
        ineqconstraints = problem.ineqconstraints_all
        eqconstraints = problem.eqconstraints_all
        manifold = problem.manifold

        has_ineqconstraints = problem.has_ineqconstraints
        has_eqconstraints = problem.has_eqconstraints
        num_ineqconstraints = problem.num_ineqconstraints
        num_eqconstraints = problem.num_eqconstraints

        gradcostfun = problem.riemannian_gradient
        gradineqconstraints = problem.ineqconstraints_riemannian_gradient_all
        gradeqconstraints = problem.eqconstraints_riemannian_gradient_all

        # Set parameters
        option = self.option
        bound = option["bound"]
        tau = option["tau"]
        thetarho = option["thetarho"]
        endingtolgradnorm = option["endingtolgradnorm"]
        LagmultUnbdUpdate = option["LagmultUnbdUpdate"]

        # Set the subsolver
        innersubsolver = option["innersubsolver"]
        maxInnerIter=option["maxInnerIter"]
        innerminstepsize = option["innerminstepsize"]
        tolgradnorm = option["startingtolgradnorm"]
        thetatolgradnorm = self.thetatolgradnorm

        # @pymanopt.function.autograd(manifold)
        def costalmfun(point):
            val = costfun(point)
            violation = 0
            for i in range(len(ineqconstraints)):
                violation = violation + max(0, (self.ineqLagCur[i]/self.rho) + ineqconstraints[i](point))**2
            for j in range(len(eqconstraints)):
                violation = violation + ((self.eqLagCur[j]/self.rho) + eqconstraints[j](point))**2
            violation = violation * self.rho * 0.5
            val = val + violation
            return val
        
        # @pymanopt.function.autograd(manifold)
        def gradalmfun(point):
            vec = gradcostfun(point)
            for i in range(len(gradineqconstraints)):
                if (ineqconstraints[i](point) + self.ineqLagCur[i]/self.rho) > 0:
                    vec = vec + (self.ineqLagCur[i] + self.rho * ineqconstraints[i](point)) * gradineqconstraints[i](point)
            for j in range(len(gradeqconstraints)):
                vec = vec + (self.eqLagCur[j] + self.rho * eqconstraints[j](point)) * gradeqconstraints[j](point)
            return vec
        
        """
        Note: The functions costalmfun and gradalmfun are defined without using @pymanopt.function.autograd() 
        to ensure compatibility with cases where the manifold is a product manifold.
        This approach prevents the direct use of pymanopt.Problem() to construct the subproblem, 
        as pymanopt.Problem() requires both cost and riemannian_gradient to be wrapped with @pymanopt.function.autograd().
        To address this limitation, we symptomatically introduce a custom class, SubProblem, which can handle functions of any type.
        """
        subproblem = SubProblem(
            manifold=manifold,
            cost=costalmfun,
            riemannian_gradient=gradalmfun,
            preconditioner=problem.preconditioner
            )

        # Construct the subproblem and the subsolver
        # subproblem = pymanopt.Problem(manifold=manifold, cost=costalmfun, riemannian_gradient=gradalmfun)
        class_subsolver = getattr(pymanopt.optimizers, innersubsolver)
        subsolver = class_subsolver(max_iterations=maxInnerIter,
                                    min_step_size=innerminstepsize,
                                    min_gradient_norm=tolgradnorm,
                                    verbosity=0)

        # Solve the subproblem
        result = subsolver.run(problem=subproblem,
                                initial_point=xCur)
        xCur = result.point

        # Construct ineqLagCurUnbd and eqLagCurUnbd if RALM aims to find AKKT points.
        # The update is based on the paper by Yamakawa and Sato.
        if LagmultUnbdUpdate:
            if has_ineqconstraints:
                for idx in range(num_ineqconstraints):
                    ineqcstrfun = ineqconstraints[idx]
                    ineqcost = ineqcstrfun(xCur)
                    self.ineqLagCurUnbd[idx] = max(0, self.ineqLagCur[idx] + self.rho * ineqcost)
            if has_eqconstraints:
                for idx in range(num_eqconstraints):
                    eqcstrfun = eqconstraints[idx]
                    eqcost = eqcstrfun(xCur)
                    self.eqLagCurUnbd[idx] = self.eqLagCur[idx] + self.rho * eqcost

        # Update Lagrange multipliers
        newacc = 0
        if has_ineqconstraints:
            for idx in range(num_ineqconstraints):
                ineqcstrfun = ineqconstraints[idx]
                ineqcost = ineqcstrfun(xCur)
                newacc = max(newacc, abs(max(-self.ineqLagCur[idx]/self.rho, ineqcost)))
                self.ineqLagCur[idx] = min(bound, max(0, self.ineqLagCur[idx] + self.rho * ineqcost))
        if has_eqconstraints:
            for idx in range(num_eqconstraints):
                eqcstrfun = eqconstraints[idx]
                eqcost = eqcstrfun(xCur)
                newacc = max(newacc, abs(eqcost))
                self.eqLagCur[idx] = min(bound, max(-bound, self.eqLagCur[idx] + self.rho * eqcost))

        # Update rho.
        # Attention: the following update strategy has been adopted
        # in the Matlab imprementation in Github by losangle (https://github.com/losangle/Optimization-on-manifolds-with-extra-constraints)
        # In the original paper by Liu and Boumal (https://arxiv.org/abs/1901.10000),
        # the correct condition seems to be 'OuterIter != 0 and newacc > tau * oldacc'
        if OuterIteration == 0 or newacc > tau * self.oldacc:
            self.rho = self.rho/thetarho
        self.oldacc = newacc
        tolgradnorm = max(endingtolgradnorm, tolgradnorm * thetatolgradnorm)

        # Compute ineqLagEval and eqLagEval
        ineqLagEval, eqLagEval = self.set_LagEvals(problem)

        return xCur, ineqLagEval, eqLagEval

    def postprocess(self, xfinal, ineqLagfinal, eqLagfinal):
        output = Output(name = self.name,
                        x=xfinal,
                        ineqLagmult=ineqLagfinal,
                        eqLagmult=eqLagfinal,
                        option=copy.deepcopy(self.option),
                        log=self.log)
        return output

    # Running an experiment
    def run(self, problem):
        assert isinstance(problem, NonlinearProblem), "Input problem must be an instance of NonlinearProblem"
        # self.assert_hasattr(self.option, 'LagmultUnbdUpdate', 'verbosity', 'do_exit_on_error', 'manviofun', 'callbackfun', 'wandb_logging')

        xCur, ineqLagEval, eqLagEval = self.preprocess(problem)
        xPrev = copy.deepcopy(xCur)
        # self.assert_hasattr(self, 'ineqLagCur', 'eqLagCur', 'rho', 'oldacc', 'thetatolgradnorm')
        # if self.option['LagmultUnbdUpdate']:
        #     self.assert_hasattr(self, 'ineqLagCurUnbd', 'eqLagCurUnbd')

        OuterIteration = 0
        start_time = time.time()

        # Variables for evaluation and logging
        manviofun = self.option["manviofun"]
        callbackfun = self.option["callbackfun"]
        tolresid = self.option["tolresid"]

        verbosity = self.option['verbosity']
        do_exit_on_error = self.option['do_exit_on_error']

        excluded_time = 0

        while True:
            # Evaluation and logging
            log_start_time = time.time()

            if do_exit_on_error:
                try:
                    eval_log = evaluation(problem, xPrev, xCur, ineqLagEval, eqLagEval, manviofun, callbackfun)
                    solver_log = self.solver_status(self.rho, ineqLagEval, eqLagEval)
                except Exception as e:
                    print(f"Error: {e}")
                    break
            else:
                eval_log = evaluation(problem, xPrev, xCur, ineqLagEval, eqLagEval, manviofun, callbackfun)
                solver_log = self.solver_status(self.rho, ineqLagEval, eqLagEval)
            # eval_log = evaluation(problem, xPrev, xCur, ineqLagEval, eqLagEval, manviofun, callbackfun)
            # solver_log = self.solver_status(self.rho, ineqLagEval, eqLagEval)
            log_end_time = time.time()
            excluded_time += log_end_time - log_start_time
            self.add_log(OuterIteration, start_time, eval_log, solver_log, excluded_time)

            # Preparation for check stopping criteria
            xPrev = copy.deepcopy(xCur)
            residual = eval_log["residual"]
            residual_criterion = (residual <= tolresid, "KKT residual tolerance reached; current residual=" + str(residual) + " and tolresid=" + str(tolresid))
            stopping_criteria =[residual_criterion]

            # Display current status
            if verbosity:
                print(f"Iter: {OuterIteration}, Cost: {problem.cost(xCur)}, KKT residual: {residual}")

            # Check stopping criteria (time, iteration and residual)
            stop, reason = self.check_stoppingcriterion(start_time, OuterIteration, stopping_criteria, excluded_time)
            if stop:
                self.option["stoppingcriterion"] = reason
                if verbosity:
                    print(reason)
                break

            # # Count an iteration
            OuterIteration += 1

            if do_exit_on_error:
                try:
                    xCur, ineqLagEval, eqLagEval = self.step(problem, xCur, OuterIteration)
                except Exception as e:
                    print(f"Error: {e}")
                    break
            else:
                xCur, ineqLagEval, eqLagEval = self.step(problem, xCur, OuterIteration)

        # After exiting while loop, we return the final output
        output = self.postprocess(xCur, ineqLagEval, eqLagEval)

        # Finish wandb logging if working
        if self.option["wandb_logging"]:
            wandb.finish()

        return output

    # Examine the max value of Lagrange multipliers
    def solver_status(self, rho, ineqLagmult, eqLagmult):
        solver_status = {}
        solver_status["rho"] = rho
        maxabsLagmult = float('-inf')

        # if ineqconstraints.has_constraint:
        for Lagmult in ineqLagmult:
            maxabsLagmult = max(maxabsLagmult, abs(Lagmult))
        # if eqconstraints.has_constraint:
        for Lagmult in eqLagmult:
            maxabsLagmult = max(maxabsLagmult, abs(Lagmult))

        solver_status["maxabsLagmult"] = maxabsLagmult
        return solver_status

@hydra.main(version_base=None, config_path="../PackingCircles", config_name="config_simulation")
def main(cfg):  # Experiment of nonnegative PCA. Mainly for debugging

    # Import a problem set from NonnegPCA
    sys.path.append('./src/PackingCircles')
    import coordinator

    # Call a problem coordinator
    coordinator = coordinator.Coordinator(cfg)
    problem = coordinator.run()

    # Solver option setting
    solver_option = cfg.solver_option
    option = copy.deepcopy(dict(solver_option["common"]))
    if hasattr(solver_option, "RALM"):
        specific = dict(getattr(solver_option, "RALM"))
        option.update(specific)

    # Run the experiment
    solver = RALM(option)
    output = solver.run(problem)
    print(output)

if __name__=='__main__':
    main()
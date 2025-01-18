import scipy.optimize
import hydra, copy, time, pymanopt, wandb, scipy
import numpy as np
from dataclasses import dataclass, field

import sys
sys.path.append('./src/base')
from base_solver import Solver, BaseOutput

@dataclass
class Output(BaseOutput):
    ineqLagmult: field(default_factory=list)
    eqLagmult: field(default_factory=list)
    maniLagmult: field(default_factory=list)

class EuclideanIPTRM(Solver):
    def __init__(self, option):
        # Default setting for Euclidean interior point trust region method
        default_option = {
            # Stopping criteria
            'maxtime': 100,
            # 'maxiter': 100,  # unavailable in scipy.optimize.minimize
            'tolresid': 1e-6,

            # scipt.optimize.minimize setting
            'jac': None,
            'hess': None,
            'hessp': None,
            # 'bounds': None,
            # 'tol': None,

            # scipy.optimize.NonlinearConstraint setting
            # 'constraints_jac': '2-point',
            # 'constraints_hess': scipy.optimize.BFGS(),
            # 'constraints_keep_feasible': False,
            # 'constraints_finite_diff_rel_step': None,
            # 'constraints_finite_diff_jac_sparsity': None,

            # scipy.optimize.minimize.trust-constr setting
            # 'gtol': 1e-8,
            # 'xtol': 1e-8,
            # 'barrier_tol': 1e-8,
            # 'sparse_jacobian': None,
            'initial_tr_radius': 1,
            'initial_constr_penalty': 1,
            'initial_barrier_parameter': 0.1,
            # 'factorization_method': None,
            # 'finite_diff_rel_step': None,

            # Display setting
            'disp': False,
            'verbosity': 0,

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
            wandb.finish()
            _ = wandb.init(project=self.option["wandb_project"],  # the project name where this run will be logged
                            name = "scipy-trust-constr",  # the name of the run
                            config=self.option)  # save hyperparameters and metadata

    # Running an experiment
    def run(self, problem):
        # Assertion
        assert hasattr(problem, 'cost')
        assert hasattr(problem, 'eqconstraints')
        assert hasattr(problem, 'ineqconstraints')
        assert hasattr(problem, 'initialpoint')
        assert hasattr(problem, 'maniconstraints')
        # assert hasattr(problem, 'initialineqLagmult')
        # assert hasattr(problem, 'initialeqLagmult')

        # Set the optimization problem
        costfun = problem.cost
        ineqconstraints = problem.ineqconstraints
        eqconstraints = problem.eqconstraints
        maniconstraints = problem.maniconstraints

        # Set initial points
        xCur = problem.initialpoint
        self.iter = 0
        self.start_time = time.time()
        
        # Set hyperparameters
        option = self.option
        maxiter = option["maxiter"]
        tolresid = option["tolresid"]
        
        jac = option["jac"]
        hess = option["hess"]
        hessp = option["hessp"]
        # bounds = option["bounds"]
        # tol = option["tol"]
        
        # constraint_jac = option["constraints_jac"]
        # constraint_hess = option["constraints_hess"]
        # constraint_keep_feasible = option["constraints_keep_feasible"]
        # constraint_finite_diff_rel_step = option["constraints_finite_diff_rel_step"]
        # constraint_finite_diff_jac_sparsity = option["constraints_finite_diff_jac_sparsity"]

        constraints = []
        for ineqcstrfun in ineqconstraints:
            fun = scipy.optimize.NonlinearConstraint(fun=ineqcstrfun,
                                                        lb=-np.inf,
                                                        ub=0,
                                                    #  jac=constraint_jac,
                                                    #  hess=constraint_hess,
                                                    #  keep_feasible=constraint_keep_feasible,
                                                    #  finite_diff_rel_step=constraint_finite_diff_rel_step,
                                                    #  finite_diff_jac_sparsity=constraint_finite_diff_jac_sparsity
                                                        )
            constraints.append(fun)

        for eqcstrfun in eqconstraints:
            fun = scipy.optimize.NonlinearConstraint(fun=eqcstrfun,
                                                        lb=0,
                                                        ub=0,
                                                    #  jac=constraint_jac,
                                                    #  hess=constraint_hess,
                                                    #  keep_feasible=constraint_keep_feasible,
                                                    #  finite_diff_rel_step=constraint_finite_diff_rel_step,
                                                    #  finite_diff_jac_sparsity=constraint_finite_diff_jac_sparsity
                                                        )
            constraints.append(fun)
        
        for idx in range(len(maniconstraints.constraints)):
            if maniconstraints.type[idx] == 'eq':
                lb = 0
            elif maniconstraints.type[idx] == 'ineq':
                lb = -np.inf
            else:
                raise ValueError("maniconstraints.type should be 'eq' or 'ineq'")
            manicstrfun = maniconstraints.constraints[idx]
            fun = scipy.optimize.NonlinearConstraint(fun=manicstrfun,
                                                        lb=lb,
                                                        ub=0,
                                                    #  jac=constraint_jac,
                                                    #  hess=constraint_hess,
                                                    #  keep_feasible=constraint_keep_feasible,
                                                    #  finite_diff_rel_step=constraint_finite_diff_rel_step,
                                                    #  finite_diff_jac_sparsity=constraint_finite_diff_jac_sparsity
                                                        )
            
            constraints.append(fun)

        # constraints = []
        # if ineqconstraints.has_constraint:
        #     for ineqcstrfun in ineqconstraints.constraint:
        #         fun = scipy.optimize.NonlinearConstraint(fun=ineqcstrfun,
        #                                                  lb=-np.inf,
        #                                                  ub=0,
        #                                                 #  jac=constraint_jac,
        #                                                 #  hess=constraint_hess,
        #                                                 #  keep_feasible=constraint_keep_feasible,
        #                                                 #  finite_diff_rel_step=constraint_finite_diff_rel_step,
        #                                                 #  finite_diff_jac_sparsity=constraint_finite_diff_jac_sparsity
        #                                                  )
        #         constraints.append(fun)

        # if eqconstraints.has_constraint:
        #     for eqcstrfun in eqconstraints.constraint:
        #         fun = scipy.optimize.NonlinearConstraint(fun=eqcstrfun,
        #                                                  lb=0,
        #                                                  ub=0,
        #                                                 #  jac=constraint_jac,
        #                                                 #  hess=constraint_hess,
        #                                                 #  keep_feasible=constraint_keep_feasible,
        #                                                 #  finite_diff_rel_step=constraint_finite_diff_rel_step,
        #                                                 #  finite_diff_jac_sparsity=constraint_finite_diff_jac_sparsity
        #                                                  )
        #         constraints.append(fun)
        
        # if maniconstraints.has_constraint:
        #     for idx in range(maniconstraints.num_constraint):
        #         if maniconstraints.type[idx] == 'eq':
        #             lb = 0
        #         elif maniconstraints.type[idx] == 'ineq':
        #             lb = -np.inf
        #         else:
        #             raise ValueError("maniconstraints.type should be 'eq' or 'ineq'")
        #         manicstrfun = maniconstraints.constraint[idx]
        #         fun = scipy.optimize.NonlinearConstraint(fun=manicstrfun,
        #                                                  lb=lb,
        #                                                  ub=0,
        #                                                 #  jac=constraint_jac,
        #                                                 #  hess=constraint_hess,
        #                                                 #  keep_feasible=constraint_keep_feasible,
        #                                                 #  finite_diff_rel_step=constraint_finite_diff_rel_step,
        #                                                 #  finite_diff_jac_sparsity=constraint_finite_diff_jac_sparsity
        #                                                  )
                
        #         constraints.append(fun)
                

        inner_option = {}
        inner_option["gtol"] = tolresid  # option["gtol"]
        inner_option["xtol"] = tolresid  # option["xtol"]
        inner_option["barrier_tol"] = option['initial_barrier_parameter']  # option["barrier_tol"]
        # inner_option["sparse_jacobian"] = option["sparse_jacobian"]
        inner_option["initial_tr_radius"]= option["initial_tr_radius"]
        inner_option["initial_constr_penalty"] = option["initial_constr_penalty"]
        inner_option["initial_barrier_parameter"] = option["initial_barrier_parameter"]
        # inner_option["factorization_method"] = option["factorization_method"]
        # inner_option["finite_diff_rel_step"] = option["finite_diff_rel_step"]
        inner_option["maxiter"] = maxiter+1  # since trust-constr save the (inner_option["maxiter"]-1) iterations
        inner_option["verbose"] = option["verbosity"]
        inner_option["disp"] = option["disp"]

        self.callbackfun = option["callbackfun"]
        self.ineqnum = len(ineqconstraints)
        self.eqnum = len(eqconstraints)
        self.maninum = len(maniconstraints.constraints)

        def inner_callbackfun(xCur, status):
            eval = self.evaluation(problem, self.xPrev, xCur, status, self.callbackfun)
            solver_status = self.solver_status(status)
            self.add_log(self.iter, self.start_time, eval, solver_status)
            self.iter += 1
            self.xPrev = copy.deepcopy(xCur)

        xCurshape = copy.deepcopy(xCur.shape)
        xCur = xCur.reshape(-1)
        self.xPrev = copy.deepcopy(xCur)
        res = scipy.optimize.minimize(fun=costfun,
                                x0=xCur,
                                # args=(),
                                method='trust-constr',
                                jac=jac,
                                hess=hess,
                                hessp=hessp,
                                # bounds=None,
                                constraints=constraints,
                                # tol=tol,
                                options=inner_option,
                                callback=inner_callbackfun,
                                )
        xCur = res["x"].reshape(xCurshape)
        v = res["v"]
        
        ineqLagmult = np.concatenate(v[:self.ineqnum]) if self.ineqnum > 0 else np.array([])
        eqLagmult = np.concatenate(v[self.ineqnum:self.ineqnum+self.eqnum]) if self.eqnum > 0 else np.array([])
        maniLagmult = np.concatenate(v[self.ineqnum+self.eqnum:]) if self.maninum > 0 else np.array([])
        
        output = Output(name="EuclideanIPTRM",
                        x=xCur,
                        ineqLagmult=ineqLagmult,
                        eqLagmult=eqLagmult,
                        maniLagmult=maniLagmult,
                        option=copy.deepcopy(self.option),
                        log=self.log)
        
        if self.option["wandb_logging"]:
            wandb.finish()
        
        return output

    def evaluation(self, problem, xPrev, xCur, status, callbackfun):
        cost = status["fun"]
        dist = np.linalg.norm(xCur - xPrev)
        
        lagrangian_grad = status["lagrangian_grad"]
        gradnorm = np.linalg.norm(lagrangian_grad)
        
        constr = status["constr"]
        ineqconstrval = np.concatenate(constr[:self.ineqnum]) if self.ineqnum > 0 else np.array([])
        eqconstrval = np.concatenate(constr[self.ineqnum:self.ineqnum+self.eqnum]) if self.eqnum > 0 else np.array([])
        maniconstrval = np.concatenate(constr[self.ineqnum+self.eqnum:]) if self.maninum > 0 else np.array([])
        
        vioineqconstrval = np.maximum(ineqconstrval, 0)
        vioeqconstrval = np.abs(eqconstrval)
        viomaniconstrval = np.array([None] * self.maninum)
        if self.maninum > 0:
            for idx in range(self.maninum):
                if problem.maniconstraints.type[idx] == 'eq':
                    viomaniconstrval[idx] = np.abs(maniconstrval[idx])
                elif problem.maniconstraints.type[idx] == 'ineq':
                    viomaniconstrval[idx] = np.maximum(maniconstrval[idx], 0)
                else:
                    raise ValueError("maniconstraints.type should be 'eq' or 'ineq'")
        maxviolation = status["constr_violation"]
        
        totalvio = np.sum(vioineqconstrval) + np.sum(vioeqconstrval) + np.sum(viomaniconstrval)
        totalconstrnum = self.ineqnum + self.eqnum + self.maninum
        meanviolation = totalvio / totalconstrnum

        
        v = status["v"]
        ineqLagmult = np.concatenate(v[:self.ineqnum]) if self.ineqnum > 0 else np.array([])
        # eqLagmult = np.concatenate(v[self.ineqnum:self.ineqnum+self.eqnum]) if self.eqnum > 0 else np.array([])
        maniLagmult = np.concatenate(v[self.ineqnum+self.eqnum:]) if self.maninum > 0 else np.array([])

        maniconstrtype = problem.maniconstraints.type
        maniineqindices = np.where(np.atleast_1d(maniconstrtype == 'ineq'))[0]
        maniineqLagmult = maniLagmult[maniineqindices]
        maniineqconstrval = maniconstrval[maniineqindices]
        
        vioineqLagmult = np.maximum(0, -ineqLagmult)
        viomaniineqLagmult = np.maximum(0, -maniineqLagmult)
        
        # residial of KKT conditions
        residual = np.sqrt(gradnorm**2
                           + np.sum(vioineqconstrval**2) + np.sum(vioeqconstrval**2) + np.sum(viomaniconstrval**2)
                           + np.sum(vioineqLagmult**2) + np.sum(viomaniineqLagmult**2)
                           + np.sum((ineqLagmult*ineqconstrval)**2) + np.sum((maniineqLagmult*maniineqconstrval)**2)
                )

        eval = {"cost": cost,
                "distance": dist,
                "residual": residual,
                "gradnorm": gradnorm,
                "maxviolation": maxviolation,
                "meanviolation": meanviolation
                }

        eval = callbackfun(problem, xCur, eval)
        return eval
    
    def solver_status(self, status):
        solver_status = {}
        solver_status["tr_radius"] = status["tr_radius"]
        solver_status["barrier_parameter"] = status["barrier_parameter"]
        solver_status["maxabsLagmult"] = np.max(np.abs(status["v"]))
        return solver_status

@hydra.main(version_base=None, config_path="../Model_Ob", config_name="config_euclidean_simulation")
def main(cfg):  # Experiment of nonnegative PCA. Mainly for debugging

    # Import a problem set from NonnegPCA
    sys.path.append('./src/Model_Ob')
    import importlib

    # Call a problem coordinator
    problem_coordinator_name = cfg.problem_coordinator_name
    module_coordinator = importlib.import_module(problem_coordinator_name)  # dynamic importation
    coordinator = module_coordinator.Coordinator(cfg)
    problem = coordinator.run()

    # Solver option setting
    solver_option = cfg.solver_option
    option = copy.deepcopy(dict(solver_option["common"]))
    if hasattr(solver_option, "EuclideanIPTRM"):
        specific = dict(getattr(solver_option, "EuclideanIPTRM"))
        option.update(specific)

    # Run the experiment
    solver = EuclideanIPTRM(option)
    output = solver.run(problem)
    print(output)

if __name__=='__main__':
    main()
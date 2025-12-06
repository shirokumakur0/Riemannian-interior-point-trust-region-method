import hydra, copy, time, wandb
import numpy as np
from utils import evaluation, tangentorthobasis, selfadj_operator2matrix, operatorspectrum, NonlinearProblem, Output
from cvxopt import spmatrix, matrix, solvers

import sys
sys.path.append('./src/base')
from base_solver import Solver

# ell_1 penalty function for line search
def ell_1penaltyfun(point, rho, costfun, ineqconstraints, eqconstraints):
    val = costfun(point)
    cstrvio = 0
    for idx in range(len(ineqconstraints)):
        funval = (ineqconstraints[idx])(point)
        cstrvio = cstrvio + max(0, funval)
    for idx in range(len(eqconstraints)):
        funval = (eqconstraints[idx])(point)
        cstrvio = cstrvio + abs(funval)
    val = val + rho * cstrvio
    return val

class RSQO(Solver):
    def __init__(self, option):
        # Default setting for sequential quadratic optimization
        default_option = {
            # Stopping criteria
            'maxtime': 100,
            'maxiter': 100,
            'tolresid': 1e-6,

            # Quadratic optimization setting
            'quadoptim_type': 'reghess',  # 'reghess', 'reghess_operator', 'eye'
            'quadoptim_eigvalcorr': 1e-8,
            'quadoptim_eigvalthld': 1e-5,
            'quadoptim_maxiter': 400,
            'quadoptim_abstol': 1e-12,
            'quadoptim_reltol': 1e-12,
            'quadoptim_feastol': 1e-12,

            'quadoptim_basisfun': lambda manifold, x, dim: tangentorthobasis(manifold, x, dim),

            # Line search setting
            'rho': 1,
            'tau': 0.5,
            'beta': 0.9,
            'gamma': 0.25,
            'linesearch_max': 10000,
            'linesearch_threshold': 1e-8,

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
        self.name = f"RSQO_{self.option['quadoptim_type']}_corr{self.option['quadoptim_eigvalcorr']:.0e}"
        self.initialize_wandb()

    def preprocess(self, problem):
        xCur = copy.deepcopy(problem.initialpoint)
        ineqLagCur = copy.deepcopy(problem.initialineqLagmult)
        eqLagCur = copy.deepcopy(problem.initialeqLagmult)
        rho = copy.deepcopy(self.option["rho"])
        return xCur, ineqLagCur, eqLagCur, rho

    def step(self, problem, xCur, ineqLagCur, eqLagCur, rho):
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

        hesscostfun = problem.riemannian_hessian
        hessineqconstraints = problem.ineqconstraints_riemannian_hessian_all
        hesseqconstraints = problem.eqconstraints_riemannian_hessian_all
        
        option = self.option
        tau = option["tau"]
        beta = option["beta"]
        gamma = option["gamma"]
        verbosity = option["verbosity"]

        # Set parameters for quadratic optimization
        quadoptim_eigvalthld = option["quadoptim_eigvalthld"]
        quadoptim_eigvalcorr = option["quadoptim_eigvalcorr"]
        quadoptim_basisfun = option["quadoptim_basisfun"]
        tolresid = option["tolresid"]
        quadoptim_abstol = max(option['quadoptim_abstol'], tolresid)
        quadoptim_reltol = max(option['quadoptim_reltol'], tolresid)
        quadoptim_feastol = max(option['quadoptim_feastol'], tolresid)

        if option["quadoptim_type"] == 'reghess':
            # Preparation for constructing the subproblem
            def hessLagfunCur(tangent_vector):
                vec = hesscostfun(xCur, tangent_vector)
                for i in range(len(hessineqconstraints)):
                    vec = vec + ineqLagCur[i] * hessineqconstraints[i](xCur, tangent_vector)
                for j in range(len(hesseqconstraints)):
                    vec = vec + eqLagCur[j]  * hesseqconstraints[j](xCur, tangent_vector)
                return vec
            
            orthobasis = quadoptim_basisfun(manifold, xCur, manifold.dim)
            Q = selfadj_operator2matrix(manifold, xCur, hessLagfunCur, orthobasis)
            eigenvalues, eigenvectors = np.linalg.eigh(Q)
            for i in range(len(eigenvalues)):
                if eigenvalues[i] < quadoptim_eigvalthld:
                    eigenvalues[i] = quadoptim_eigvalcorr
            eigenvalues = np.real(eigenvalues)
            Q = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            Q = 0.5 * (Q + Q.T)
            Q = np.real(Q)
            Q = matrix(Q)
        elif option["quadoptim_type"] == 'reghess_operator':
            def hessLagfun(x, tangent_vector):
                vec = hesscostfun(x, tangent_vector)
                for i in range(len(hessineqconstraints)):
                    vec = vec + ineqLagCur[i] * hessineqconstraints[i](x, tangent_vector)
                for j in range(len(hesseqconstraints)):
                    vec = vec + eqLagCur[j]  * hesseqconstraints[j](x, tangent_vector)
                return vec
            w, orthobasis = operatorspectrum(manifold, hessLagfun, xCur)
            w = np.where(w < quadoptim_eigvalthld, quadoptim_eigvalcorr, w)
            Q = spmatrix(w, range(len(w)), range(len(w)))
        elif option["quadoptim_type"] == 'eye':
            orthobasis = quadoptim_basisfun(manifold, xCur, manifold.dim)
            Q = np.eye(manifold.dim)
            Q = matrix(Q)
        else:
            raise ValueError("quadoptim_type must be 'reghess', 'reghess_operator', or 'eye'.")

        # Compute the first-order term in the objective function
        gradobjxCur = gradcostfun(xCur)
        p = np.empty(len(orthobasis))
        for i in range(len(orthobasis)):
            p[i] = manifold.inner_product(xCur, gradobjxCur, orthobasis[i])
        p = matrix(p)

        # Compute the inequality constraints
        G = None
        h = None
        if has_ineqconstraints:
            G = np.empty((num_ineqconstraints, len(orthobasis)))
            h = np.empty(num_ineqconstraints)
            for i in range(num_ineqconstraints):
                ineqcstrfun = ineqconstraints[i]
                h[i] = -ineqcstrfun(xCur)
                gradineqcstrfun = gradineqconstraints[i]
                gradineqxCur = gradineqcstrfun(xCur)
                for j in range(len(orthobasis)):
                    G[i,j] = manifold.inner_product(xCur, gradineqxCur, orthobasis[j])
            G = matrix(G)
            h = matrix(h)

        # Compute the equality constraints
        A = None
        b = None
        if has_eqconstraints:
            A = np.empty((num_eqconstraints, len(orthobasis)))
            b = np.empty(num_eqconstraints)

            for i in range(num_eqconstraints):
                eqcstrfun = eqconstraints[i]
                b[i] = -eqcstrfun(xCur)
                gradeqcstrfun = gradeqconstraints[i]
                gradeqxCur = gradeqcstrfun(xCur)
                for j in range(len(orthobasis)):
                    A[i,j] = manifold.inner_product(xCur, gradeqxCur, orthobasis[j])
            A = matrix(A)
            b = matrix(b)

        if verbosity <=1:
            solvers.options['show_progress'] = False

        solvers.options['abstol'] =  quadoptim_abstol
        solvers.options['reltol'] =  quadoptim_reltol
        solvers.options['feastol'] =  quadoptim_feastol

        # Solve the subproblem
        sol = solvers.qp(P=Q, q=p, G=G, h=h, A=A, b=b)
        coeff = np.array(sol['x']).T[0]
        ineqLagsol = np.array(sol['z']).T[0]
        eqLagsol = np.array(sol['y']).T[0]
        # Assertion for the shapes of solution
        assert coeff.shape == (len(orthobasis),)
        assert ineqLagsol.shape == (num_ineqconstraints,)
        assert eqLagsol.shape == (num_eqconstraints,)

        # Compute the df0 (the inner product of B_k[d] and d) and dir (the search direction)
        # for line search
        Q = np.array(matrix(Q))
        df0 = coeff @ Q @ coeff
        dir = manifold.zero_vector(xCur)
        for i in range(len(coeff)):
            dir = dir + coeff[i] * orthobasis[i]

        normdx = manifold.norm(xCur, dir)

        # Update rho if necessary
        upsilon = 0
        if has_ineqconstraints:
            upsilon = max([upsilon, max(ineqLagsol)])
        if has_eqconstraints:
            upsilon = max([upsilon, max(abs(eqLagsol))])
        if rho < upsilon:
            rho = upsilon + tau

        # Line search
        f0 = ell_1penaltyfun(xCur, rho, costfun, ineqconstraints, eqconstraints)
        gammadf0 = gamma * df0
        stepsize = 1
        newx = manifold.retraction(xCur, stepsize * dir)
        newf = ell_1penaltyfun(newx, rho, costfun, ineqconstraints, eqconstraints)
        linesearch_status = 1
        linesearch_counter = 0

        while newf > (f0 - gammadf0) and np.abs(newf - (f0 - gammadf0)) > option["linesearch_threshold"]:
            linesearch_counter += 1
            if linesearch_counter >= option["linesearch_max"]:
                linesearch_status = 0
                break
            stepsize *= beta
            gammadf0 *= beta
            newx = manifold.retraction(xCur, stepsize * dir)
            newf = ell_1penaltyfun(newx, rho, costfun, ineqconstraints, eqconstraints)

        # Update vriables
        xCur = newx
        ineqLagCur = ineqLagsol
        eqLagCur = eqLagsol
        return xCur, ineqLagCur, eqLagCur, rho, upsilon, sol, normdx, stepsize, df0, linesearch_status, linesearch_counter

    def postprocess(self, xfinal, ineqLagfinal, eqLagfinal):
        output = Output(name=self.name,
                        x=xfinal,
                        ineqLagmult=ineqLagfinal,
                        eqLagmult=eqLagfinal,
                        option=copy.deepcopy(self.option),
                        log=self.log)
        return output

    # Running an experiment
    # @profile
    def run(self, problem):
        assert isinstance(problem, NonlinearProblem), "Input problem must be an instance of NonlinearProblem"

        xCur, ineqLagCur, eqLagCur, rho = self.preprocess(problem)
        xPrev = copy.deepcopy(xCur)
        upsilon = None
        sol = None
        stepsize = None
        normdx = None
        df0 = None
        linesearch_status = None
        linesearch_counter = None

        iteration = 0
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
            eval_log = evaluation(problem, xPrev, xCur, ineqLagCur, eqLagCur, manviofun, callbackfun)
            solver_log = self.solver_status(ineqLagCur, eqLagCur, rho, upsilon, sol, normdx, stepsize, df0, linesearch_status, linesearch_counter)
            log_end_time = time.time()
            excluded_time += log_end_time - log_start_time
            self.add_log(iteration, start_time, eval_log, solver_log, excluded_time)

            # Preparation for check stopping criteria
            xPrev = copy.deepcopy(xCur)
            residual = eval_log["residual"]
            residual_criterion = (residual <= tolresid, "KKT residual tolerance reached; current residual=" + str(residual) + " and tolresid=" + str(tolresid))
            stopping_criteria =[residual_criterion]

            # Display current status
            if verbosity:
                print(f"Iter: {iteration}, Cost: {problem.cost(xCur)}, KKT residual: {residual}")

            # Check stopping criteria (time, iteration and residual)
            stop, reason = self.check_stoppingcriterion(start_time, iteration, stopping_criteria, excluded_time)
            if stop:
                self.option["stoppingcriterion"] = reason
                if verbosity:
                    print(reason)
                break

            # Count an iteration
            iteration += 1

            if do_exit_on_error:
                try:
                    xCur, ineqLagCur, eqLagCur, rho, upsilon, sol, normdx, stepsize, df0, linesearch_status, linesearch_counter = self.step(problem, xCur, ineqLagCur, eqLagCur, rho)
                except Exception as e:
                    print(f"Error: {e}")
                    break
            else:
                xCur, ineqLagCur, eqLagCur, rho, upsilon, sol, normdx, stepsize, df0, linesearch_status, linesearch_counter = self.step(problem, xCur, ineqLagCur, eqLagCur, rho)

        # After exiting while loop, we return the final output
        output = self.postprocess(xCur, ineqLagCur, eqLagCur)

        # Finish wandb logging if working
        if self.option["wandb_logging"]:
            wandb.finish()

        return output

    # Examine the solver status
    def solver_status(self,
                      ineqLagmult,
                      eqLagmult,
                      rho,
                      upsilon=None,
                      sol=None,
                      normdx=None,
                      stepsize=None,
                      df0=None,
                      linesearch_status=None,
                      linesearch_counter=None
                      ):
        solver_status = {}
        solver_status["rho"] = rho
        solver_status["upsilon"] = upsilon

        maxabsLagmult = float('-inf')
        for Lagmult in ineqLagmult:
            maxabsLagmult = max(maxabsLagmult, abs(Lagmult))
        for Lagmult in eqLagmult:
            maxabsLagmult = max(maxabsLagmult, abs(Lagmult))
        solver_status["maxabsLagmult"] = maxabsLagmult

        if sol is not None:
            solver_status["quadoptim_status"] = sol["status"]
            solver_status["quadoptim_iter"] = sol["iterations"]
            solver_status["quadoptim_gap"] = sol["gap"]
            # solver_status["quadoptim_relativegap"] = sol["relative gap"]
            solver_status["quadoptim_primalobjective"] = sol["primal objective"]
            solver_status["quadoptim_dualobjective"] = sol["dual objective"]
            solver_status["quadoptim_primalinfeasibility"] = sol["primal infeasibility"]
            solver_status["quadoptim_dualinfeasibility"] = sol["dual infeasibility"]
            # solver_status["quadoptim_primalslack"] = sol["primal slack"]
            # solver_status["quadoptim_dualslack"] = sol["dual slack"]
        else:
            solver_status["quadoptim_status"] = None
            solver_status["quadoptim_iter"] = None
            solver_status["quadoptim_gap"] = None
            # solver_status["quadoptim_relativegap"] = None
            solver_status["quadoptim_primalobjective"] = None
            solver_status["quadoptim_dualobjective"] = None
            solver_status["quadoptim_primalinfeasibility"] = None
            solver_status["quadoptim_dualinfeasibility"] = None
            # solver_status["quadoptim_primalslack"] = None
            # solver_status["quadoptim_dualslack"] = None

        solver_status["normdx"] = normdx
        solver_status["stepsize"] = stepsize
        solver_status["df0"] = df0
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
    if hasattr(solver_option, "RSQO"):
        specific = dict(getattr(solver_option, "RSQO"))
        option.update(specific)

    # Run the experiment
    rsqosolver = RSQO(option)
    
    output = rsqosolver.run(problem)
    print(output)

if __name__=='__main__':
    main()
    # cProfile.run("main()", sort="tottime")
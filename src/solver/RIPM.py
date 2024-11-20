import hydra, copy, time, pymanopt, wandb, warnings
import numpy as np
from dataclasses import dataclass, field
from utils import evaluation, tangentorthobasis, operator2matrix, tangent2vec, TangentSpaceConjResMethod, selfadj_operator2matrix
from scipy import linalg

import sys
sys.path.append('./src/base')
from base_solver import Solver, BaseOutput

warnings.filterwarnings("ignore", message="Output seems independent of input.")

@dataclass
class Output(BaseOutput):
    ineqLagmult: field(default_factory=list)
    eqLagmult: field(default_factory=list)

def barGx(x, z, ineqconstraints):
    vec = 0  # will be converted to a zero embedded vector by broadcasting
    for idx in range(ineqconstraints.num_constraint):
        egrad = ineqconstraints.constraint[idx].get_gradient_operator()
        vec = vec + z[idx] * egrad(x)
    return vec

def ehess_barGx(x, z, dx, ineqconstraints):
    vec = 0  # will be converted to a zero embedded vector by broadcasting
    for idx in range(ineqconstraints.num_constraint):
        ehess = ineqconstraints.constraint[idx].get_hessian_operator()
        vec = vec + z[idx] * ehess(x, dx)
    return vec

def barGxaj(x, dx, ineqconstraints):
    val = np.zeros(ineqconstraints.num_constraint)
    for idx in range(ineqconstraints.num_constraint):
        egrad = ineqconstraints.constraint[idx].get_gradient_operator()
        val[idx] = egrad(x).reshape(-1) @ dx.reshape(-1)
    return val

def barHx(x, y, eqconstraints):
    vec = 0   # will be converted to a zero embedded vector by broadcasting
    for idx in range(eqconstraints.num_constraint):
        egrad = eqconstraints.constraint[idx].get_gradient_operator()
        vec = vec + y[idx] * egrad(x)
    return vec

def ehess_barHx(x, y, dx, eqconstraints):
    vec = 0  # will be converted to a zero embedded vector by broadcasting
    for idx in range(eqconstraints.num_constraint):
        ehess = eqconstraints.constraint[idx].get_hessian_operator()
        vec = vec + y[idx] * ehess(x, dx)
    return vec

def barHxaj(x, dx, eqconstraints):
    val = np.zeros(eqconstraints.num_constraint)
    for idx in range(eqconstraints.num_constraint):
        egrad = eqconstraints.constraint[idx].get_gradient_operator()
        val[idx] = egrad(x).reshape(-1) @ dx.reshape(-1)
    return val

def build_KKTVectorField(gradLagrangian, ineqconstraints, eqconstraints):
    def KKTVectorField(xyzs):
        x, y, z, s = xyzs
        dx = gradLagrangian(x, y, z)
        dy = np.empty(eqconstraints.num_constraint, dtype=object)
        for idx in range(eqconstraints.num_constraint):
            dy[idx] = (eqconstraints.constraint[idx])(x)
        dz = np.empty(ineqconstraints.num_constraint, dtype=object)
        for idx in range(ineqconstraints.num_constraint):
            dz[idx] = (ineqconstraints.constraint[idx])(x) + s[idx]
        ds = z * s
        dxdydzds = pymanopt.manifolds.product._ProductTangentVector([dx, dy, dz, ds])
        return dxdydzds
    return KKTVectorField

def build_egradLagrangian(costegradfun, ineqconstraints, eqconstraints):
    if eqconstraints.has_constraint:
        def egradLagrangian(x, y, z):
            val = costegradfun(x) + barGx(x, z, ineqconstraints) + barHx(x, y, eqconstraints)
            return val
    else:
        def egradLagrangian(x, y, z):
            val = costegradfun(x) + barGx(x, z, ineqconstraints)
            return val
    return egradLagrangian

def build_ehessLagrangian(costehessfun, ineqconstraints, eqconstraints):
    if eqconstraints.has_constraint:
        def ehessLagrangian(x, y, z, dx):
            val = costehessfun(x, dx) + ehess_barGx(x, z, dx, ineqconstraints) + ehess_barHx(x, y, dx, eqconstraints)
            return val
    else:
        def ehessLagrangian(x, y, z, dx):
            val = costehessfun(x, dx) + ehess_barGx(x, z, dx, ineqconstraints)
            return val
    return ehessLagrangian

def RepresentMatMethod(Aw, Hxaj, cq, xy_manifold, xy, xbasis, ybasis, RepMat_exit_warning_triggered):
    x_manifold = xy_manifold.manifolds[0]
    y_manifold = xy_manifold.manifolds[1]
    # xdim = x_manifold.dim
    ydim = y_manifold.dim
    x = xy[0]
    y = xy[1]
    basis = [pymanopt.manifolds.product._ProductTangentVector([xb, y_manifold.zero_vector(y)]) for xb in xbasis]\
        + [pymanopt.manifolds.product._ProductTangentVector([x_manifold.zero_vector(x), yb]) for yb in ybasis]

    # Under the basis, the following codes return a saddle-point
    # linear system whose matrix has the form
    #
    #           [ HessLag_mat + THETA_mat | Hx_mat]
    #   T_mat = -----------------------------------
    #           [Hx_mat.T                 | 0     ]
    # where
    # - n:= dim of manifold, l:= dim of equality constraints,
    # - HessLag_mat and THETA_mat are symmetric n x n,
    # - Hx_mat is l x n, with l <= n,
    # - 0 is l x l zero matrix.
    
    # The next code is equal to run:
    # HessLag_mat = SelfAdj_operator2matrix(M, x, x, HessLag, Bx, Bx);
    # THETA_mat = SelfAdj_operator2matrix(M, x, x, THETA, Bx, Bx);
    # Aw_mat = HessLag_mat + THETA_mat;
    Aw_mat = selfadj_operator2matrix(x_manifold, x, Aw, xbasis)

    # The next code is equal to run:
    # Hx_mat = operator2matrix(y_manifold, y, x, Hx, ybasis, xbasis, x_manifold);
    # T_mat =[Aw_mat,  Hx_mat; ...
    #        Hx_mat', zeros((ydim, ydim))];
    # But, Hxaj is cheaper, because Hx needs to call
    # orthogonal projection onto tangent space.
    Hxaj_mat = operator2matrix(x_manifold, x, y, Hxaj, xbasis, ybasis, y_manifold); # Hx_mat.T=Hxaj_mat.
    T_mat = np.block([
        [Aw_mat, Hxaj_mat.T],
        [Hxaj_mat, np.zeros((ydim, ydim))]
    ])
    cq_vec = tangent2vec(xy_manifold, xy, basis, cq)
    if RepMat_exit_warning_triggered:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", linalg.LinAlgWarning)
            sol_vec = linalg.solve(T_mat, cq_vec, assume_a='sym')
            for warning in caught_warnings:
                    if issubclass(warning.category, linalg.LinAlgWarning):
                        return None, None, None
    else:
        sol_vec = linalg.solve(T_mat, cq_vec, assume_a='sym')
    NTdir = xy_manifold.zero_vector(xy)
    for i in range(len(basis)):
        NTdir = NTdir + sol_vec[i] * basis[i]
    RepresentMat = T_mat
    RepresentMatOrder = xy_manifold.dim

    return NTdir, RepresentMat, RepresentMatOrder
    


class RIPM(Solver):
    def __init__(self, option):
        # Default setting for interior point method
        default_option = {
            # Stopping criteria
            'maxtime': 100,
            'maxiter': 100,
            'tolresid': 1e-6,

            # Iteration
            'KrylovIterMethod': False,
            'KrylovTolrelresid': 1e-9,
            'KrylovMaxIteration': 1000,
            'checkNTequation': False,
            'RepMat_basisfun': lambda manifold, x: tangentorthobasis(manifold, x, manifold.dim),
            'RepMat_exit_warning_triggered': False,

            # Line search setting
            'gamma': 0.9,
            'linesearch_execute_fun2': False,
            'linesearch_beta': 1e-4,
            'linesearch_theta': 0.5,
            'linesearch_max_steps': 50,

            # Other parameters
            'heuristic_z_s': False,
            'desired_tau_1': 0.5,
            'important': 1,

            # Display setting
            'verbosity': 1,

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
            wandb.finish()
            if self.option["KrylovIterMethod"]:
                name = "RIPM_Krylov"
            else:
                name = "RIPM_RepMat"
            _ = wandb.init(project=self.option["wandb_project"],  # the project name where this run will be logged
                             name = name,
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

        # Set the operators
        Gx = lambda x, z: manifold.euclidean_to_riemannian_gradient(x, barGx(x, z, ineqconstraints))
        Gxaj = lambda x, dx: barGxaj(x, manifold.embedding(x, dx), ineqconstraints)
        Hx = lambda x, y: manifold.euclidean_to_riemannian_gradient(x, barHx(x, y, eqconstraints))
        Hxaj = lambda x, dx: barHxaj(x, manifold.embedding(x, dx), eqconstraints)
        costegradfun = costfun.get_gradient_operator()
        costehessfun = costfun.get_hessian_operator()
        egradLagrangian = build_egradLagrangian(costegradfun, ineqconstraints, eqconstraints)
        ehessLagrangian = build_ehessLagrangian(costehessfun, ineqconstraints, eqconstraints)
        gradLagrangian = lambda x, y, z: manifold.euclidean_to_riemannian_gradient(x, egradLagrangian(x, y, z))
        hessLagrangian = lambda x, y, z, dx: manifold.euclidean_to_riemannian_hessian(x, egradLagrangian(x, y, z), ehessLagrangian(x, y, z, dx), dx)
        KKTVectorField = build_KKTVectorField(gradLagrangian, ineqconstraints, eqconstraints)

        # Set the product manifolds
        xyzs_manifold = pymanopt.manifolds.Product([manifold,
                                                    pymanopt.manifolds.Euclidean(eqconstraints.num_constraint),
                                                    pymanopt.manifolds.Euclidean(ineqconstraints.num_constraint),
                                                    pymanopt.manifolds.Euclidean(ineqconstraints.num_constraint)])
        xy_manifold = pymanopt.manifolds.Product([manifold,
                                                  pymanopt.manifolds.Euclidean(eqconstraints.num_constraint)])
        
        # Set the parameters
        option = self.option
        heuristic_z_s =  option["heuristic_z_s"]
        desired_tau_1 = option["desired_tau_1"]
        important = option["important"]
        gamma = option["gamma"]
        KrylovIterMethod = option["KrylovIterMethod"]
        KrylovTolrelresid = option["KrylovTolrelresid"]
        KrylovMaxIteration = option["KrylovMaxIteration"]
        checkNTequation = option["checkNTequation"]
        RepMat_basisfun = option["RepMat_basisfun"]
        RepMat_exit_warning_triggered = option["RepMat_exit_warning_triggered"]
        ls_beta = option["linesearch_beta"]
        ls_execute_fun2 = option["linesearch_execute_fun2"]
        ls_theta = option["linesearch_theta"]
        ls_max_steps = option["linesearch_max_steps"]
        verbosity = option["verbosity"]

        # Set initial points
        xCur = problem.initialpoint
        yCur = problem.initialeqLagmult
        ineqnum = ineqconstraints.num_constraint
        if heuristic_z_s:
            zCur = np.ones(ineqnum)
            zCur[0] =np.real(np.sqrt((ineqnum - 1)/(ineqnum/desired_tau_1 - 1)));
            sCur = important * zCur
        else:
            zCur = problem.initialineqLagmult
            sCur = problem.initialineqLagmult
        xPrev = copy.deepcopy(xCur)
        iteration = 0

        # Set initial points on xyzs_manifold
        xyzsCur = [xCur, yCur, zCur, sCur]
        Ehat = xyzs_manifold.zero_vector(xyzsCur)
        ehat = np.ones(ineqconstraints.num_constraint)
        Ehat[3] = ehat
        KKTvec = KKTVectorField(xyzsCur)
        PhiCur = xyzs_manifold.norm(xyzsCur, KKTvec)**2

        # Set inintial points on xy_manifold
        xyCur = [xCur, yCur]
        v0 = xy_manifold.zero_vector(xyCur)
        start_time = time.time()

        # Set constants for centrality conditions
        tau_1 = min(zCur * sCur) * ineqnum/ (zCur @ sCur)  # min(zCur * sCur) / ((zCur @ sCur) / ineqnum)
        tau_2 = (zCur @ sCur) / np.real(np.sqrt(PhiCur))

        # Construct parameters sigma to controls the final convergence rate
        sigma = min(0.5, np.real(np.sqrt(PhiCur)))
        rho = (zCur @ sCur) / ineqnum

        # The first evaluation and logging
        manviofun = option["manviofun"]
        callbackfun = option["callbackfun"]
        eval_log = evaluation(problem, xPrev, xCur, zCur, yCur, manviofun, callbackfun)
        solver_log = self.solver_status(
                            zCur,
                            yCur,
                            ineqconstraints,
                            eqconstraints,
                            PhiCur,
                            sigma,
                            rho,
                            KrylovIterMethod=KrylovIterMethod,
                            checkNTequation=checkNTequation
                            )

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

            # Get Newton direction NTdir by solving condensed NT equation.
            # Right-hand side (cq) of condensed NT equation is (c, q).
            c = - KKTvec[0] - Gx(xCur, (zCur * KKTvec[2] + sigma * rho * ehat - KKTvec[3]) / sCur)
            q = - KKTvec[1]
            cq = pymanopt.manifolds.product._ProductTangentVector([c,q])

            # Define the operators.
            OperatorHessLag = lambda dx: hessLagrangian(xCur, yCur, zCur, dx)
            OperatorTHETA = lambda dx: Gx(xCur, Gxaj(xCur, dx) * (zCur / sCur))
            OperatorAw = lambda dx: OperatorHessLag(dx) + OperatorTHETA(dx)
            OperatorHx = lambda dy: Hx(xCur, dy)
            OperatorHxaj = lambda dx: Hxaj(xCur, dx)
            
            def build_operatorT(OperatorAw, OperatorHx, OperatorHxaj, eqconstraints):
                if eqconstraints.has_constraint:
                    def OperatorT(dxdy):
                        vec = pymanopt.manifolds.product._ProductTangentVector([OperatorAw(dxdy[0]) + OperatorHx(dxdy[1]), OperatorHxaj(dxdy[0])])
                        return vec
                else:
                    def OperatorT(dxdy):
                        vec = pymanopt.manifolds.product._ProductTangentVector([OperatorAw(dxdy[0]), np.array([])])
                        return vec
                return OperatorT
            OperatorT = build_operatorT(OperatorAw, OperatorHx, OperatorHxaj, eqconstraints)

            # Solve the condensed NT equation T(dxdy) = cq.
            solve_info = []
            if KrylovIterMethod:
                NTdirdxdy, t, rel_res, _ = TangentSpaceConjResMethod(OperatorT, cq, v0, xy_manifold, xyCur, KrylovTolrelresid, KrylovMaxIteration)
                solve_info = [t, rel_res]
            else:
                xbasis = RepMat_basisfun(manifold, xCur)
                ybasis = np.eye(eqconstraints.num_constraint)
                NTdirdxdy, _, _ = RepresentMatMethod(OperatorAw, OperatorHxaj, cq, xy_manifold, xyCur, xbasis, ybasis, RepMat_exit_warning_triggered)
                if RepMat_exit_warning_triggered and NTdirdxdy is None:
                    print("np.linalg.LinAlgWarning was detected within the RpresentMatMethod, terminating the algorithm.")
                    break
            # Recovery dz and ds.
            Ntdirdz = (zCur * (Gxaj(xCur, NTdirdxdy[0]) + KKTvec[2]) + sigma * rho * ehat - KKTvec[3]) / sCur
            NTdirds = (sigma * rho * ehat - KKTvec[3] - sCur * Ntdirdz) / zCur
            NTdir = pymanopt.manifolds.product._ProductTangentVector([NTdirdxdy[0], NTdirdxdy[1], Ntdirdz, NTdirds])
            
            # Check Newton direction NTdir
            # DEBUG ONLY
            NTdir_info = []
            if checkNTequation:
                # Covariant derivative of KKT vector field
                # DEBUG ONLY
                def CovarDerivKKT(x, y, z, s, dw):
                    dx, dy, dz, ds = dw
                    nablaFdx = hessLagrangian(x, y, z, dx) + Gx(x, dz)
                    if eqconstraints.has_constraint:
                        nablaFdx = nablaFdx + Hx(x, dy)
                    nablaFdy = Hxaj(x, dx)
                    nablaFdz = Gxaj(x, dx) + ds
                    nablaFds = z * ds + s * dz
                    nablaF = pymanopt.manifolds.product._ProductTangentVector([nablaFdx, nablaFdy, nablaFdz, nablaFds])
                    return nablaF

                # Adjoint of covariant Derivative of KKT vector field
                # DEBUG ONLY
                def CovarDerivKKTaj(x, y, z, s, dw):
                    dx, dy, dz, ds = dw
                    nablaFajdx = hessLagrangian(x, y, z, dx) + Gx(x, dz)
                    if eqconstraints.has_constraint:
                        nablaFajdx = nablaFajdx + Hx(x, dy)
                    nablaFajdy = Hxaj(x, dx)
                    nablaFajdz = Gxaj(x, dx) + s * ds
                    nablaFajds = z * ds + dz
                    nablaFaj = pymanopt.manifolds.product._ProductTangentVector([nablaFajdx, nablaFajdy, nablaFajdz, nablaFajds])
                    return nablaFaj
                
                nablaF = lambda dw: CovarDerivKKT(xCur, yCur, zCur, sCur, dw)
                nablaFaj = lambda dw: CovarDerivKKTaj(xCur, yCur, zCur, sCur, dw)
                
                # Check Item#1: The residual of the non-condensed Newton equation.
                # Note that the right-hand side of original NT equation is
                # -KKTvec+sigma*rho*Ehat
                # NTdir_error1 should be zero.
                NTeq_rhs = -KKTvec + sigma * rho * Ehat
                nablaF_NTdir = nablaF(NTdir)
                NTdir_error1 = xyzs_manifold.norm(xyzsCur, nablaF_NTdir - NTeq_rhs)
                # print("NTdir_error1", NTdir_error1)
                
                # Check Item#2: If NTdir is correct solution, then
                # <grad phi, NTdir> = 2(|F(w)|^{2}+sigma*rho*InnerProduct(z,s)) holds,
                # where grad phi = 2*nablaFaj(KKTvec).
                # NTdir_error2 should be zero.
                gradphi = 2 * nablaFaj(KKTvec)
                val_innerproduct = xyzs_manifold.inner_product(xyzsCur, gradphi, NTdir)
                NTdir_error2 = abs(val_innerproduct - 2*(sigma*rho*(zCur @ sCur)-PhiCur))
                # print("NTdir_error2", NTdir_error2)
                
                # Record Item: record norm of NTdirl; angle between - grad phi and NTdir.
                Norm_gradphi = xyzs_manifold.norm(xyzsCur, gradphi)
                NTdir_norm = xyzs_manifold.norm(xyzsCur, NTdir)
                NTdir_angle = - val_innerproduct / (Norm_gradphi * NTdir_norm)
                NTdir_info = [NTdir_error1, NTdir_error2, NTdir_norm, NTdir_angle]
            
            # Backtracking line search and update.
            # Central functions
            fun_1 = lambda z, s: min(z * s) - gamma * tau_1 * (z @ s / ineqnum)
            fun_2 = lambda z, s, Phi: z @ s - gamma * tau_2 * np.sqrt(Phi)
            
            # Note that <grad phi, NTdir> = ls_RightItem, if NTdir is a correct solution.
            ls_RightItem = 2 * (sigma * rho * (zCur @ sCur) - PhiCur)
            
            stepsize = 1
            ls_status = True
            r = 0
            while True:
                xyzsNew = xyzs_manifold.retraction(xyzsCur, stepsize * NTdir)
                KKTvec = KKTVectorField(xyzsNew)
                PhiNew = xyzs_manifold.norm(xyzsNew, KKTvec)**2
                zNew = xyzsNew[2]
                sNew = xyzsNew[3]
                if PhiNew - PhiCur <= ls_beta * stepsize * ls_RightItem and fun_1(zNew, sNew) >= 0:
                    if ls_execute_fun2 and fun_2(zNew, sNew, PhiNew) >= 0:
                        break
                    else:
                        break
                r += 1
                if r > ls_max_steps:
                    ls_status = False
                    break
                stepsize = ls_theta * stepsize

            # Update the current point.
            xCur, yCur, zCur, sCur = xyzsNew
            # KKT vec is already updated
            PhiCur = PhiNew
            # Update points on xyzs_manifold
            xyzsCur = [xCur, yCur, zCur, sCur]
            Ehat = xyzs_manifold.zero_vector(xyzsCur)
            ehat = np.ones(ineqconstraints.num_constraint)
            Ehat[3] = ehat
            KKTvec = KKTVectorField(xyzsCur)
            PhiCur = xyzs_manifold.norm(xyzsCur, KKTvec)**2
            # Update points on xy_manifold
            xyCur = [xCur, yCur]
            v0 = xy_manifold.zero_vector(xyCur)

            # Update parameters
            sigma = min(0.5, np.sqrt(PhiCur)**0.5)
            rho = (zCur @ sCur) / ineqnum
            gamma = 0.5 * (gamma + 0.5)

            # Evaluation and logging
            eval_log = evaluation(problem, xPrev, xCur, zCur, yCur, manviofun, callbackfun)
            solver_log = self.solver_status(
                      zCur,
                      yCur,
                      ineqconstraints,
                      eqconstraints,
                      PhiCur,
                      sigma,
                      rho,
                      stepsize=stepsize,
                      linesearch_status=ls_status,
                      linesearch_counter=r,
                      KrylovIterMethod=KrylovIterMethod,
                      solve_info=solve_info,
                      checkNTequation=checkNTequation,
                      NTdir_info=NTdir_info
                      )
            
            self.add_log(iteration, start_time, eval_log, solver_log)

            # Update previous x and residual
            xPrev = copy.deepcopy(xCur)
            residual = eval_log["residual"]
            residual_criterion = (residual <= tolresid, "KKT residual tolerance reached; current residual=" + str(residual) + " and tolresid=" + str(tolresid))
            stopping_criteria =[residual_criterion]

        # After exiting while loop, we return the final output
        output = Output(x=xCur,
                        ineqLagmult=zCur,
                        eqLagmult=yCur,
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
                      Phi,
                      sigma,
                      rho,
                      stepsize=None,
                      linesearch_status=None,
                      linesearch_counter=None,
                      KrylovIterMethod=None,
                      solve_info=None,
                      checkNTequation=False,
                      NTdir_info=None
                      ):
        solver_status = {}
        solver_status["Phi"] = Phi
        solver_status["sigma"] = sigma
        solver_status["rho"] = rho

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
        
        if KrylovIterMethod:
            if solve_info is not None:
                t, rel_res = solve_info
                solver_status["KrylovIterMethod"] = KrylovIterMethod
                solver_status["KrylovIterMethod_Iter"] = t
                solver_status["KrylovIterMethod_RelRes"] = rel_res
            else:
                solver_status["KrylovIterMethod"] = None
                solver_status["KrylovIterMethod_Iter"] = None
                solver_status["KrylovIterMethod_RelRes"] = None
        
        if checkNTequation:
            if NTdir_info is not None:
                NTdir_error1, NTdir_error2, NTdir_norm, NTdir_angle = NTdir_info
                solver_status["NTdir_error1"] = NTdir_error1
                solver_status["NTdir_error2"] = NTdir_error2
                solver_status["NTdir_norm"] = NTdir_norm
                solver_status["NTdir_angle"] = NTdir_angle
            else:
                solver_status["NTdir_error1"] = None
                solver_status["NTdir_error2"] = None
                solver_status["NTdir_norm"] = None
                solver_status["NTdir_angle"] = None

        return solver_status

@hydra.main(version_base=None, config_path="../NonnegPCA", config_name="config_simulation")
def main(cfg):  # Experiment of nonnegative PCA. Mainly for debugging

    # Import a problem set from NonnegPCA
    sys.path.append('./src/NonnegPCA')
    import coordinator

    # Call a problem coordinator
    problem_coordinator = coordinator.Coordinator(cfg)
    problem = problem_coordinator.run()

    # Solver option setting
    solver_option = cfg.solver_option
    option = copy.deepcopy(dict(solver_option["common"]))
    if hasattr(solver_option, "RIPM"):
        specific = dict(getattr(solver_option, "RIPM"))
        option.update(specific)

    # Run the experiment
    solver = RIPM(option)
    output = solver.run(problem)
    print(output)

if __name__=='__main__':
    main()
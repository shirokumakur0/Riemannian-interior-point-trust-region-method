import hydra, copy, time, pymanopt, wandb
import numpy as np
from dataclasses import dataclass, field
from utils import  tangentorthobasis, operator2matrix, evaluation
import warnings
import scipy


import sys
sys.path.append('./src/base')
from base_solver import Solver, BaseOutput

@dataclass
class Output(BaseOutput):
    ineqLagmult: field(default_factory=list)
    eqLagmult: field(default_factory=list)

from scipy.sparse.linalg import cg, LinearOperator, eigs #, eigsh

def TRSgep(A, a, B, Del):
    """
    Solves the trust-region subproblem by a generalized eigenproblem without iterations.

    minimize (x^T A x) / 2 + a^T x
    subject to x^T B x <= Del^2

    Parameters:
        A (ndarray): Symmetric nxn matrix.
        a (ndarray): nx1 vector.
        B (ndarray): Symmetric positive definite nxn matrix.
        Del (float): Radius constraint.

    Returns:
        x (ndarray): Solution vector.
        lam1 (float): Lagrange multiplier.
    """
    n = A.shape[0]
    tolhardcase = 1e-4  # tolerance for hard-case

    # Construct the block matrix MM1
    M0 = np.block([[- B, A], [A, - np.outer(a, a) / (Del**2)]])
    MM1 = np.block([[np.zeros((n, n)), B], [B, np.zeros((n, n))]])

    # Possible interior solution
    # p1, _ = scipy.sparse.linalg.cg(A, -a, tol=1e-12, maxiter=500)
    p1 = scipy.linalg.solve(A, -a, assume_a='sym')
    # print("p1", p1)
    """p1 = scipy.linalg.solve(A, -a, assume_a='sym')やnp.linalg.solve, scipy.sparse.linalg.cg(A, -a, tol=1e-12, maxiter=500)と比較"""
    # print(p1.T @ B @ p1, p1 @ B @ p1)
    # input()
    
    if np.linalg.norm(A @ p1 + a) / np.linalg.norm(a) < 1e-5:
        if p1 @ B @ p1 >= Del**2:  # outside of the trust region
            p1 = np.full_like(p1, np.nan)  # ineligible
    else:  # numerically incorrect
        p1 = np.full_like(p1, np.nan)

    # Core of the code: generalized eigenproblem
    lams, vecs = scipy.linalg.eig(a=M0, b=-MM1)

    # print("eig", lams, vecs)
    # print("repmat", lams)
    # lams = np.real(lams)

    rmidx = np.argmax(np.real(lams))
    lam1 = np.real(lams[rmidx])
    V = vecs[:, rmidx]
    V = np.real(V) if np.linalg.norm(np.real(V)) >= 1e-3 else np.imag(V)
    # print("lam1, V", lam1, V)
    # print("val", M0 @ V - lam1 * -MM1 @ V)

    # print("V",V.shape)
    # print("rm", rmeigval, rmvec)
    # print("M0v-lamM1v", M0 @ rmvec - rmeigval * -MM1 @ rmvec)
    # randvec = np.random.rand(2*n)
    # print("Moone repmat", M0 @  randvec)

    # Afun = LinearOperator((2*n, 2*n), matvec=lambda x: MM0timesx(A, B, a, Del, x))
    # # print("fun", Afun(randvec))
    # minusMM1fun = LinearOperator((2*n, 2*n), matvec=lambda x: -MM1 @ x)
    # lam1, V = eigsh(Afun, k=1, M=minusMM1fun, which='LA', mode='buckling') # , v0=np.random.rand(2*n))
    # print("lam1, V", lam1, V)
    
    x = V[:n]  # extract solution component
    
    # print("Vhere", V.reshape(-1))
    # print("eig", Afun(V.reshape(-1))-lam1*minusMM1fun(V.reshape(-1)))

    normx = np.sqrt(x @ B @ x)
    x = x / normx * Del  # in the easy case, this naive normalization improves accuracy
    if x @ a > 0:
        x = -x  # take correct sign
    if normx < tolhardcase:  # enter hard case
        # print("hard case")
        x1 = V[n:]
        Pvect = x1  # first try only k=1, almost always enough
        # fun = LinearOperator((2*n, 2*n), matvec=lambda x: MM0timesx(A, B, a, Del, x))
        alpha1 = copy.deepcopy(lam1)
        Alam1B = A + lam1 * B
        # print("Pvect", Pvect)
        BPvecti = B @ Pvect
        H = Alam1B + alpha1 * np.outer(BPvecti, BPvecti)
        """
        for i in range(Pvect.shape[1]):
            BPvecti = B @ Pvect[:, i]
            H += alpha1 * np.outer(BPvecti, BPvecti)
        """
        x2 = scipy.linalg.solve(H, -a, assume_a='sym')
        # print("x2.solve", x2)
        # pcgfun = LinearOperator((n, n),matvec=lambda x: pcgforAtilde(A, B, lam1, Pvect.reshape(-1,1), lam1, x))
        # x2, _ = cg(pcgfun, -a, tol=1e-12, maxiter=500)
        # print("x2.cg", x2)


        # Residual check for hard case refinement
        if np.linalg.norm((A + lam1 * B) @ x2 + a) / np.linalg.norm(a) > tolhardcase:
            _, v = scipy.linalg.eigh(A, B)  # ascending order
            # print("w,v",A @ v[:,0]- w[0]* B @ v[:,0])
            for ii in [3,6,9]:  # Iteratively refine solution if needed
                Pvect = v[:,:ii]
                BPvecti = B @ Pvect
                H = Alam1B + alpha1 * BPvecti @ BPvecti.T
                x2 = scipy.linalg.solve(H, -a, assume_a='sym')
                # print("x2Pvect.scipy", x2)
                # pcgfun = LinearOperator((n, n),matvec=lambda x: pcgforAtilde(A, B, lam1, Pvect, lam1, x))
                # x2, _ = cg(pcgfun, -a, tol=1e-8, maxiter=500)
                # print("x2Pvect.scipy.sparse", x2)
                if np.linalg.norm((A + lam1 * B) @ x2 + a) / np.linalg.norm(a) < tolhardcase:
                    break

        Bx = B @ x1
        Bx2 = B @ x2
        aa = x1 @ Bx
        bb = 2 * x2 @ Bx
        cc = x2 @ Bx2 - Del**2
        alp = (-bb + np.sqrt(bb**2 - 4 * aa * cc)) / (2 * aa)  #norm(x2+alp*x)-Delta
        x = x2 + alp * x1
        # print("x2", x2)
        # print("xbefore", x)

    # Choose between interior and boundary solution
    if not np.isnan(p1).any():
        # print("p1", p1)
        # print("x", x)
        # print("entered")
        # x = x.reshape(-1)
        p1objval = 0.5 * (p1 @ A @ p1)  + a @ p1
        xobjval = 0.5 * (x @ A @ x)  + a @ x
        if p1objval < xobjval:
            x = p1
            lam1 = 0

    return x, lam1

# def pcgforAtilde(A, B, lamA, Pvect, alpha1, x):
#     """
#     Helper function for the hard case of TRSgep.
#     """
#     y = A @ x + lamA * (B @ x)
#     for i in range(Pvect.shape[1]):
#         y += alpha1 * (x.T @ B @ Pvect[:, i]) * (B @ Pvect[:, i])
#     return y
    

# Riemannian interior point trust region method
class RIPTRM(Solver):
    def __init__(self, option):
        # Default setting for augmented Lagrangian method
        default_option = {
            # Stopping criteria
            'maxtime': 100,
            'maxiter': 100,
            'tolresid': 1e-6,

            # Trust region setting
            'initial_TR_radius': 1,
            'maximal_TR_radius': 10,
            'rho': 0.2,
            'gamma': 0.9,
            'forcing_function_Lagrangian': lambda mu: mu,
            'forcing_function_complementarity': lambda mu: mu,
            'forcing_function_second_order': lambda mu: mu,
            'second_order_stationarity': False,
            'TRS_solver': 'exact_RepMat',  # 'exact_RepMat', 'exact', 'exact_implicit', or 'Cauchy'
            'initial_barrier_parameter': 0.99,
            'barrier_parameter_r': 0.2,
            'barrier_parameter_c': 0.95,
            
            'const_left': 0.5,
            'const_right': 1e+10,  # 1e+20
            'basisfun': lambda manifold, x: tangentorthobasis(manifold, x, manifold.dim),
            'save_inner_iteration': True,
            
            # Display setting
            'verbosity': 2,

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
        assert hasattr(problem, 'ineqconstraints')
        assert hasattr(problem, 'eqconstraints')
        assert hasattr(problem, 'initialpoint')
        assert hasattr(problem, 'initialineqLagmult')
        assert hasattr(problem, 'initialeqLagmult')

        # if hasattr(problem, 'eqconstraints'):
        if problem.eqconstraints.has_constraint:
            warnings.warn("Equality constraints detecred. Currently, RIPTRM does not support equality constraints and will completely ignore them.", Warning)

        # Set the optimization problem
        manifold = problem.searchspace
        costfun = problem.costfun
        original_ineqconstraints = problem.ineqconstraints
        ineqconstraints = copy.deepcopy(problem.ineqconstraints)
        # eqconstraints = problem.eqconstraints

        def set_inverse_ineqconstraints(idx):
            costineqfun = original_ineqconstraints.constraint[idx]
            @pymanopt.function.autograd(manifold)
            def invcostineqfun(x):
                return -1 * costineqfun(x)
            return invcostineqfun
        for idx in range(original_ineqconstraints.num_constraint):
            ineqconstraints.constraint[idx] = set_inverse_ineqconstraints(idx)

        # Set initial points
        option = self.option
        xCur = problem.initialpoint
        yCur = problem.initialineqLagmult
        muCur = option["initial_barrier_parameter"]
        # eqLagCur = problem.initialeqLagmult
        xPrev = copy.deepcopy(xCur)
        iteration = 0
        start_time = time.time()
        
        # Set parameters
        initial_TR_radius = option['initial_TR_radius']
        maximal_TR_radius = option['maximal_TR_radius']
        rho = option['rho']
        gamma = option['gamma']
        forcing_function_Lagrangian = option['forcing_function_Lagrangian']
        forcing_function_complementarity = option['forcing_function_complementarity']
        forcing_function_second_order = option['forcing_function_second_order']
        second_order_stationarity = option['second_order_stationarity']
        TRS_solver = option['TRS_solver']
        const_left = option['const_left']
        const_right = option['const_right']
        barrier_parameter_r = option['barrier_parameter_r']
        barrier_parameter_c = option['barrier_parameter_c']
        save_inner_iteration = option['save_inner_iteration']
        basisfun = option['basisfun']
        verbosity = option["verbosity"]

        # The first evaluation and logging
        manviofun = option["manviofun"]
        callbackfun = option["callbackfun"]
        eval_log = evaluation(problem, xPrev, xCur, yCur, [], manviofun, callbackfun)
        
        solver_log = self.solver_status(
                      yCur,
                      muCur,
                      ineqconstraints,
                      )
        
        self.add_log(iteration, start_time, eval_log, solver_log)
        
        # Preparation for check stopping criteria
        residual = eval_log["residual"]
        tolresid = option["tolresid"]
        residual_criterion = (residual <= tolresid, "KKT residual tolerance reached; current residual=" + str(residual) + " and tolresid=" + str(tolresid))
        stopping_criteria =[residual_criterion]

        while True:
            if verbosity == 1:
                print(f"Outer itertion: {iteration}, Cost: {costfun(xCur)}, KKT residual: {residual}")

            # Check stopping criteria (time, iteration and residual)
            stop, reason = self.check_stoppingcriterion(start_time, iteration, stopping_criteria)
            if stop:
                self.option["stoppingcriterion"] = reason
                if verbosity:
                    print(reason)
                break
            # Count an iteration
            iteration += 1
            
            # Set the inner option
            inner_option = {}
            inner_option["second_order_stationarity"] = second_order_stationarity
            inner_option["stopping_criterion_Lagrangian"] = forcing_function_Lagrangian(muCur)
            inner_option["stopping_criterion_complementarity"] = forcing_function_complementarity(muCur)
            if second_order_stationarity:
                inner_option["stopping_criterion_second_order"] = forcing_function_second_order(muCur)
            inner_option["TRS_solver"] = TRS_solver
            inner_option["gamma"] = gamma
            inner_option["maximal_TR_radius"] = maximal_TR_radius
            inner_option["rho"] = rho
            inner_option["const_left"] = const_left
            inner_option["const_right"] = const_right
            inner_option["save_inner_iteration"] = save_inner_iteration
            
            def InnerIteration(x_initial, y_initial, mu, initial_TR_radius, costfun, ineqconstraints, manifold, inner_option):
                assert 'stopping_criterion_Lagrangian' in inner_option, "Key 'stopping_criterion_Lagrangian' not found in inner_option"
                assert 'stopping_criterion_complementarity' in inner_option, "Key ''stopping_criterion_complementarity'' not found in inner_option"
                assert 'second_order_stationarity' in inner_option, "Key 'second_order_stationarity' not found in inner_option"
                if inner_option["second_order_stationarity"]:
                    assert 'stopping_criterion_second_order' in inner_option, "Key 'stopping_criterion_second_order' not found in inner_option"
                assert 'TRS_solver' in inner_option, "Key 'TRS_solver' not found in inner_option"
                assert 'gamma' in inner_option, "Key 'gamma' not found in inner_option"
                assert 'maximal_TR_radius' in inner_option, "Key 'maximal_TR_radius' not found in inner_option"
                assert 'rho' in inner_option, "Key 'rho' not found in inner_option"
                assert 'const_left' in inner_option, "Key 'const_left' not found in inner_option"
                assert 'const_right' in inner_option, "Key 'const_right' not found in inner_option"
                
                stopping_criterion_Lagrangian = inner_option["stopping_criterion_Lagrangian"]
                stopping_criterion_complementarity = inner_option["stopping_criterion_complementarity"]
                second_order_stationarity = inner_option["second_order_stationarity"]
                if second_order_stationarity:
                    stopping_criterion_second_order = inner_option["stopping_criterion_second_order"]
                TRS_solver = inner_option["TRS_solver"]
                gamma = inner_option["gamma"]
                maximal_TR_radius = inner_option["maximal_TR_radius"]
                rho = inner_option["rho"]
                const_left = inner_option["const_left"]
                const_right = inner_option["const_right"]
                
                xCur = x_initial
                yCur = y_initial
                inner_xPrev = copy.deepcopy(xCur)
                TR_radius = initial_TR_radius
                inner_iteration = 0

                egradcostfun = costfun.get_gradient_operator()
                ehesscostfunoprator = costfun.get_hessian_operator()
                costineqconstvecfun = lambda x: np.array([ineqfun(x) for ineqfun in ineqconstraints.constraint])
                egradineqconstvec = [ineqfun.get_gradient_operator() for ineqfun in ineqconstraints.constraint]
                ehessineqconstoperatorvec = [ineqfun.get_hessian_operator() for ineqfun in ineqconstraints.constraint]
                
                # costLagrangefun = lambda x, y: costfun(x) - y @ costineqconstvecfun(x)
                egradLagrangefun = lambda x, y: egradcostfun(x) - y @ np.array([egradineqfun(x) for egradineqfun in egradineqconstvec])
                gradLagrangefun = lambda x, y: manifold.euclidean_to_riemannian_gradient(x, egradLagrangefun(x, y))
                ehessLagrangefun = lambda x, y, dx: ehesscostfunoprator(x, dx) - y @ np.array([ehessineqconstoperator(x, dx) for ehessineqconstoperator in ehessineqconstoperatorvec])
                hessLagrangefun = lambda x, y, dx: manifold.euclidean_to_riemannian_hessian(x, egradLagrangefun(x, y), ehessLagrangefun(x, y, dx), dx)
                
                
                while True:
                    if verbosity > 1:
                        print(f"Itertion: {iteration}-{inner_iteration}, Cost: {costfun(xCur)}, KKT residual: {residual}, TR_radius: {TR_radius}")
                        inner_iteration += 1
                    # egradcostfun = costfun.get_gradient_operator()
                    
                    # costineqfuns = ineqconstraints.constraint
                    # costineqvecfun = lambda x: np.array([ineqfun(x) for ineqfun in costineqfuns])
                    # print("TR_radius", TR_radius)
                    # print("xCur", xCur)
                    # print("yCur", yCur)
                    # print("costineqvecxCur", type(costineqvecxCur))
                    # costineqvecxCur = np.array([ineqfun(xCur) for ineqfun in costineqfuns])
                    # egradineqfuns = [ineqfun.get_gradient_operator() for ineqfun in costineqfuns]
                    # gradineqconstvecxCur = [manifold.euclidean_to_riemannian_gradient(xCur, egradineqfun(xCur)) for egradineqfun in egradineqconstvec]
                    
                    inner_info = {}
                    inner_info["inner_status"] = None
                    inner_info["num_inner"] = inner_iteration
                    inner_info["TR_radius"] = TR_radius
                    inner_info["normdx"] = None
                    inner_info["min_primal_feasibility"] = None
                    inner_info["min_dual_feasibility"] = None
                    # inner_info["normgradLagfun"] = None
                    inner_info["complementarity"] = None
                    inner_info["mineigval_TRS_quadratic"] = None
                    inner_info["ared"] = None
                    inner_info["pred"] = None
                    inner_info["ared/pred"] = None
                    inner_info["radius_update"] = None
                    inner_info["dual_clipping"] = None
                    
                    gradcostfunxCur = manifold.euclidean_to_riemannian_gradient(xCur, egradcostfun(xCur))
                    costineqconstvecxCur = costineqconstvecfun(xCur)
                    gradineqconstvecxCur = [manifold.euclidean_to_riemannian_gradient(xCur, egradineqfun(xCur)) for egradineqfun in egradineqconstvec]
                    
                    Gx = lambda v: np.sum([v[idx] * gradineqconstvecxCur[idx] for idx in range(len(gradineqconstvecxCur))], axis=0)
                    Gxaj = lambda dx: np.array([manifold.inner_product(xCur, gradineqvec, dx) for gradineqvec in gradineqconstvecxCur])
                    
                    TRS_quadratic = lambda dx: hessLagrangefun(xCur, yCur, dx) + Gx((yCur * Gxaj(dx)) / costineqconstvecxCur)
                    TRS_linear = gradcostfunxCur -  Gx(mu / costineqconstvecxCur)
                    
                    if TRS_solver == 'Cauchy':
                        inner_product_TRSquadlin_lin = manifold.inner_product(xCur, TRS_quadratic(TRS_linear), TRS_linear)
                        norm_TRS_linear = manifold.norm(xCur, TRS_linear)
                        if inner_product_TRSquadlin_lin <= 0:
                            tau = TR_radius / norm_TRS_linear
                        else:
                            tau = min((norm_TRS_linear**2)/inner_product_TRSquadlin_lin, (TR_radius / norm_TRS_linear))
                        # print("tau", tau, "inner_product_TRSquadlin_lin", inner_product_TRSquadlin_lin, "norm_TRS_linear", norm_TRS_linear)
                        dxCur = - tau * TRS_linear
                    elif TRS_solver == 'exact_RepMat':
                        basis = basisfun(manifold, xCur)
                        
                        def selfadj_operator2matrix(manifold, operator, x, basis):
                            n = len(basis)
                            Hbasis = np.empty_like(basis)
                            for k in range(n):
                                Hbasis[k] = operator(basis[k])
                            H = np.zeros((n, n))
                            for i in range(n):
                                H[i, i] = manifold.inner_product(x, basis[i], Hbasis[i])
                                for j in range(i+1, n):
                                    H[i, j] = manifold.inner_product(x, basis[i], Hbasis[j])
                                    H[j, i] = H[i, j]
                            return H
                        
                        Hmatrix = selfadj_operator2matrix(manifold, TRS_quadratic, xCur, basis)
                        
                        cvector = np.empty(len(basis))
                        for i in range(len(basis)):
                            cvector[i] = manifold.inner_product(xCur, TRS_linear, basis[i])
                        
                        n = len(basis)
                        # print(Hmatrix)
                        coeff, _ = TRSgep(Hmatrix, cvector, np.eye(n), TR_radius)
                        
                        dxCur = np.sum([coeff[i] * basis[i] for i in range(n)], axis=0)

                        # p0coeff = scipy.linalg.solve(Hmatrix, -cvector, assume_a='sym')
                        # p0 = np.sum([p0coeff[i] * basis[i] for i in range(n)], axis=0)
                        # p0_eligibility = manifold.norm(xCur, p0) < TR_radius
                        
                        # p0 = scipy.linalg.solve(Hmatrix, -cvector, assume_a='sym')
                        # p0_eligibility = np.linalg.norm(p0) < TR_radius
                        
                        # eyemat = np.eye(n)
                        # zeromat = np.zeros((n, n))
                        # M0 = np.block([
                        #     [- eyemat, Hmatrix],
                        #     [Hmatrix, - np.outer(cvector, cvector) / (TR_radius**2)]
                        #    ])
                        # M1 = np.block([
                        #     [zeromat, eyemat],
                        #     [eyemat, zeromat]
                        #    ])
                        # # lam = scipy.linalg.eig(a=M0, b=-M1,right=False)
                        # lams, vecs = scipy.linalg.eig(a=M0, b=-M1)
                        # # print("lams", lams)
                        
                        # lams = np.real(lams)
                        # rmidx = np.argmax(lams)
                        # rmeigval = lams[rmidx]
                        # rmvec = vecs[:, rmidx]
                        # u = 1.1 * 1e-16
                        
                        
                        # secondrmlam = np.real(np.sort(lams)[-2])
                        # # print("secondrmlam", secondrmlam)
                        # y1 = rmvec[:n]
                        # y2 = rmvec[n:]
                        
                        # print(rmvec, y1, y2)
                        
                        # gap = rmeigval - secondrmlam
                        # normy1 = np.linalg.norm(y1)
                        # is_hardcase = normy1 <= np.sqrt(u / gap)
                        # if is_hardcase:
                        #     print("hardcase")
                        #     V = scipy.null_space(Hmatrix+rmeigval*eyemat)
                            
                        #     v = V[:, 0]
                        #     print(v)
                        #     print("Null?", (Hmatrix+rmeigval*eyemat) @ v)
                            
                            
                        #     dxCur = np.sum([y1[i] * basis[i] for i in range(n)], axis=0)
                        # else:
                        #     p1 = - np.sign(cvector @ y2) * TR_radius * y1 / normy1
                        
                    else:
                        raise ValueError(f"TRS_solver {TRS_solver} is not supported.")
                    
                    dyCur = - yCur + mu * (1 / costineqconstvecxCur) - yCur * Gxaj(dxCur) / costineqconstvecxCur
                    
                    xNew = manifold.retraction(xCur, dxCur)
                    yNew = yCur + dyCur
                    costineqconstvecxNew = costineqconstvecfun(xNew)
                    # print("xNew", xNew)
                    # print("costineqvecxNew", costineqconstvecxNew)
                    
                    
                    primal_feasibility_criterion = np.all(costineqconstvecxNew > 0)
                    dual_feasibility_criterion = np.all(yNew > 0)
                    normgradLagfun = manifold.norm(xNew, gradLagrangefun(xNew, yNew))
                    normgradLagfun_criterion = normgradLagfun <= stopping_criterion_Lagrangian
                    complementarity = np.linalg.norm(yNew * costineqconstvecxNew - mu)
                    complementary_criterion = complementarity <= stopping_criterion_complementarity
                    if second_order_stationarity:
                        pass
                        """ここは後で書く"""
                        # mineigval_TRS_quadratic = mineigval <= stopping_criterion_second_order
                    else:
                        mineigval_TRS_quadratic = None  # ignore as the stopping criterion
                        mineigval_criterion = True  # ignore as the stopping criterion
                    
                    inner_info["normdx"] = manifold.norm(xCur, dxCur)
                    inner_info["min_primal_feasibility"] = min(costineqconstvecxNew)
                    inner_info["min_dual_feasibility"] = min(yNew)
                    # inner_info["normgradLagfun"] = normgradLagfun
                    inner_info["complementarity"] = complementarity
                    inner_info["mineigval_TRS_quadratic"] = mineigval_TRS_quadratic
                    
                    if primal_feasibility_criterion and dual_feasibility_criterion and normgradLagfun_criterion and complementary_criterion and mineigval_criterion:
                        inner_info["inner_status"] = "converged"
                        return xNew, yNew, inner_info, inner_xPrev
                    
                    normdxCur = manifold.norm(xCur, dxCur)
                    if not primal_feasibility_criterion:
                        inner_info["inner_status"] = "primal_infeasible"
                        TR_radius = gamma * normdxCur
                        
                        if save_inner_iteration:
                            eval_log = evaluation(problem, xPrev, inner_xPrev, yCur, [], manviofun, callbackfun)
                            solver_log = self.solver_status(
                                    yCur,
                                    muCur,
                                    ineqconstraints,
                                    inner_info = inner_info,
                                    )
                            self.add_log(iteration, start_time, eval_log, solver_log)
                        
                        continue
                    
                    logbarrierfun = lambda x: costfun(x) - mu * np.sum(np.log(costineqconstvecfun(x)))
                    # ared = logbarrierfun(xCur) - logbarrierfun(manifold.retraction(xCur, dxCur))
                    ared = logbarrierfun(xCur) - logbarrierfun(xNew)
                    pred = - 0.5 * manifold.inner_product(xCur, TRS_quadratic(dxCur), dxCur) - manifold.inner_product(xCur, TRS_linear, dxCur)
                    
                    inner_info["ared"] = ared
                    inner_info["pred"] = pred
                    inner_info["ared/pred"] = ared / pred
                    
                    if ared < 0.25 * pred:
                        inner_info["radius_update"] = "reduced"
                        TR_radius = 0.25 * TR_radius
                    elif ared > 0.75 * pred and normdxCur == TR_radius:
                        inner_info["radius_update"] = "expanded"
                        TR_radius = min(2 * TR_radius, maximal_TR_radius)
                    else:
                        inner_info["radius_update"] = "unchanged"
                        TR_radius = TR_radius
                    
                    if ared > rho * pred:
                        inner_info["inner_status"] = "successful"
                        xCur = xNew
                        I_left = const_left * np.minimum(np.minimum(yCur, mu / costineqconstvecxNew), 1)
                        I_right = np.maximum(const_right, const_right / mu, np.maximum(yCur, const_right / costineqconstvecxNew))
                        clippingfun = lambda yNew: np.minimum(np.maximum(yNew, I_left), I_right)
                        clippedyNew = clippingfun(yNew)
                        
                        if np.array_equal(yNew, clippedyNew):
                            inner_info["dual_clipping"] = False
                        else:
                            inner_info["dual_clipping"] = True
                        yCur = clippedyNew
                    else:
                        inner_info["inner_status"] = "unsuccessful"
                        xCur = xCur
                        yCur = yCur
                        
                    if save_inner_iteration:
                        eval_log = evaluation(problem, xPrev, inner_xPrev, yCur, [], manviofun, callbackfun)
                        solver_log = self.solver_status(
                                yCur,
                                muCur,
                                ineqconstraints,
                                inner_info = inner_info,
                                )
                        self.add_log(iteration, start_time, eval_log, solver_log)
                    inner_xPrev = copy.deepcopy(xCur)
                
            xNew, yNew, inner_info, inner_xPrev = InnerIteration(xCur, yCur, muCur, initial_TR_radius, costfun, ineqconstraints, manifold, inner_option)
            muCur = barrier_parameter_c * (muCur ** (1 + barrier_parameter_r))
            # Update variables
            xCur = xNew
            yCur = yNew
            
            if save_inner_iteration:
                xPrev = inner_xPrev
            
            # Evaluation and logging
            eval_log = evaluation(problem, xPrev, xCur, yCur, [], manviofun, callbackfun)
            # print("upto here")
            # print(eval_log)
            # print("compl-2", np.sum((yCur * xCur)**2) )
            # print("manvio", manviofun(problem, xCur))
            solver_log = self.solver_status(
                      yCur,
                      muCur,
                      ineqconstraints,
                      inner_info = inner_info,
                      )
            self.add_log(iteration, start_time, eval_log, solver_log)
            
            # Update previous x and residual
            xPrev = copy.deepcopy(xCur)
            residual = eval_log["residual"]
            residual_criterion = (residual <= tolresid, "KKT residual tolerance reached; current residual=" + str(residual) + " and tolresid=" + str(tolresid))
            stopping_criteria =[residual_criterion]

        # After exiting while loop, we return the final output
        output = Output(x=xCur,
                        ineqLagmult=yCur,
                        eqLagmult=[],
                        option=copy.deepcopy(self.option),
                        log=self.log)

        if self.option["wandb_logging"]:
            wandb.finish()

        return output

    # Examine the solver status
    def solver_status(self,
                      yCur,
                      mu,
                      ineqconstraints,
                      inner_info = None,
                      ):
        solver_status = {}
        solver_status["mu"] = mu

        if inner_info is not None:
            solver_status["num_inner"] = inner_info["num_inner"]
            solver_status["inner_status"] = inner_info["inner_status"]
            solver_status["TR_radius"] = inner_info["TR_radius"]
            solver_status["normdx"] = inner_info["normdx"]
            solver_status["min_primal_feasibility"] = inner_info["min_primal_feasibility"]
            solver_status["min_dual_feasibility"] = inner_info["min_dual_feasibility"]
            # solver_status["normgradLagfun"] = inner_info["normgradLagfun"]
            solver_status["complementarity"] = inner_info["complementarity"]
            solver_status["mineigval_TRS_quadratic"] = inner_info["mineigval_TRS_quadratic"]
            solver_status["ared"] = inner_info["ared"]
            solver_status["pred"] = inner_info["pred"]
            solver_status["ared/pred"] = inner_info["ared/pred"]
            solver_status["radius_update"] = inner_info["radius_update"]
            solver_status["dual_clipping"] = inner_info["dual_clipping"]
            
        else:
            solver_status["num_inner"] = None
            solver_status["inner_status"] = None
            solver_status["TR_radius"] = None
            solver_status["normdx"] = None
            solver_status["min_primal_feasibility"] = None
            solver_status["min_dual_feasibility"] = None
            # solver_status["normgradLagfun"] = None
            solver_status["complementarity"] = None
            solver_status["mineigval_TRS_quadratic"] = None
            solver_status["ared"] = None
            solver_status["pred"] = None
            solver_status["ared/pred"] = None
            solver_status["radius_update"] = None
            solver_status["dual_clipping"] = None

        maxabsLagmult = float('-inf')
        if ineqconstraints.has_constraint:
            for Lagmult in yCur:
                maxabsLagmult = max(maxabsLagmult, abs(Lagmult))
        solver_status["maxabsLagmult"] = maxabsLagmult
        
        # solver_status["stepsize"] = stepsize
        # solver_status["linesearch_status"] = linesearch_status
        # solver_status["linesearch_counter"] = linesearch_counter

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
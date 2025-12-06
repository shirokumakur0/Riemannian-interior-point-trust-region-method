import hydra, copy, time, pymanopt, wandb, warnings
import numpy as np
from utils import  tangentorthobasis, evaluation, selfadj_operator2matrix, Output, compute_residual, NonlinearProblem
import warnings
import scipy
import cProfile

import sys
sys.path.append('./src/base')
from base_solver import Solver

from pymanopt.tools import ndarraySequenceMixin, return_as_class_instance
class _ProductAmbientVector(ndarraySequenceMixin, list):
    @return_as_class_instance(unpack=False)
    def __add__(self, other):
        if len(self) != len(other):
            raise ValueError("Arguments must be same length")
        return [v + other[k] for k, v in enumerate(self)]

    @return_as_class_instance(unpack=False)
    def __sub__(self, other):
        if len(self) != len(other):
            raise ValueError("Arguments must be same length")
        return [v - other[k] for k, v in enumerate(self)]

    @return_as_class_instance(unpack=False)
    def __mul__(self, other):
        return [other * val for val in self]

    __rmul__ = __mul__

    @return_as_class_instance(unpack=False)
    def __truediv__(self, other):
        return [val / other for val in self]

    @return_as_class_instance(unpack=False)
    def __neg__(self):
        return [-val for val in self]

# From Pymanopt's source code
def truncated_conjugate_gradient(
    manifold, hess, x, fgradx, eta, Delta, theta, kappa, mininner, maxinner, preconditioner, use_rand=False
    ):
    inner = manifold.inner_product

    if not use_rand:  # and therefore, eta == 0
        Heta = manifold.zero_vector(x)
        r = fgradx
        e_Pe = 0
    else:  # and therefore, no preconditioner
        # eta (presumably) ~= 0 was provided by the caller.
        Heta = hess(x, eta)
        r = fgradx + Heta
        e_Pe = inner(x, eta, eta)

    r_r = inner(x, r, r)
    norm_r = np.sqrt(r_r)
    norm_r0 = norm_r

    # Precondition the residual
    if not use_rand:
        z = preconditioner(x, r)
    else:
        z = r

    # Compute z'*r
    z_r = inner(x, z, r)
    d_Pd = z_r

    # Initial search direction
    delta = -z
    if not use_rand:
        e_Pd = 0
    else:
        e_Pd = inner(x, eta, delta)

    # If the Hessian or a linear Hessian approximation is in use, it is
    # theoretically guaranteed that the model value decreases strictly with
    # each iteration of tCG. Hence, there is no need to monitor the model
    # value. But, when a nonlinear Hessian approximation is used (such as
    # the built-in finite-difference approximation for example), the model
    # may increase. It is then important to terminate the tCG iterations
    # and return the previous (the best-so-far) iterate. The variable below
    # will hold the model value.

    def model_fun(eta, Heta):
        return inner(x, eta, fgradx) + 0.5 * inner(x, eta, Heta)

    if not use_rand:
        model_value = 0
    else:
        model_value = model_fun(eta, Heta)

    # Pre-assume termination because j == end.
    stop_tCG = "MAX_INNER_ITER"

    # Begin inner/tCG loop.
    for j in range(int(maxinner)):
        # This call is the computationally intensive step
        Hdelta = hess(x, delta)

        # Compute curvature (often called kappa)
        d_Hd = inner(x, delta, Hdelta)

        # Note that if d_Hd == 0, we will exit at the next "if" anyway.
        if d_Hd != 0:
            alpha = z_r / d_Hd
            # <neweta,neweta>_P =
            # <eta,eta>_P
            # + 2*alpha*<eta,delta>_P
            # + alpha*alpha*<delta,delta>_P
            e_Pe_new = e_Pe + 2 * alpha * e_Pd + alpha**2 * d_Pd
        else:
            e_Pe_new = e_Pe

        # Check against negative curvature and trust-region radius
        # violation. If either condition triggers, we bail out.
        if d_Hd <= 0 or e_Pe_new >= Delta**2:
            # want
            #  ee = <eta,eta>_prec,x
            #  ed = <eta,delta>_prec,x
            #  dd = <delta,delta>_prec,x
            tau = (
                -e_Pd + np.sqrt(e_Pd * e_Pd + d_Pd * (Delta**2 - e_Pe))
            ) / d_Pd

            eta = eta + tau * delta

            # If only a nonlinear Hessian approximation is available, this
            # is only approximately correct, but saves an additional
            # Hessian call.
            Heta = Heta + tau * Hdelta

            # Technically, we may want to verify that this new eta is
            # indeed better than the previous eta before returning it (this
            # is always the case if the Hessian approximation is linear,
            # but I am unsure whether it is the case or not for nonlinear
            # approximations.) At any rate, the impact should be limited,
            # so in the interest of code conciseness (if we can still hope
            # for that), we omit this.

            if d_Hd <= 0:
                stop_tCG = "NEGATIVE_CURVATURE"
            else:
                stop_tCG = "EXCEEDED_TR"
            break

        # No negative curvature and eta_prop inside TR: accept it.
        e_Pe = e_Pe_new
        new_eta = eta + alpha * delta

        # If only a nonlinear Hessian approximation is available, this is
        # only approximately correct, but saves an additional Hessian call.
        new_Heta = Heta + alpha * Hdelta

        # Verify that the model cost decreased in going from eta to
        # new_eta. If it did not (which can only occur if the Hessian
        # approximation is nonlinear or because of numerical errors), then
        # we return the previous eta (which necessarily is the best reached
        # so far, according to the model cost). Otherwise, we accept the
        # new eta and go on.
        new_model_value = model_fun(new_eta, new_Heta)
        if new_model_value >= model_value:
            stop_tCG = "MODEL_INCREASED"
            break

        eta = new_eta
        Heta = new_Heta
        model_value = new_model_value

        # Update the residual.
        r = r + alpha * Hdelta

        # Compute new norm of r.
        r_r = inner(x, r, r)
        norm_r = np.sqrt(r_r)

        # Check kappa/theta stopping criterion.
        # Note that it is somewhat arbitrary whether to check this stopping
        # criterion on the r's (the gradients) or on the z's (the
        # preconditioned gradients). [CGT2000], page 206, mentions both as
        # acceptable criteria.
        if j >= mininner and norm_r <= norm_r0 * min(
            norm_r0**theta, kappa
        ):
            # Residual is small enough to quit
            if kappa < norm_r0**theta:
                stop_tCG = "REACHED_TARGET_LINEAR"
            else:
                stop_tCG = "REACHED_TARGET_SUPERLINEAR"
            break

        # Precondition the residual.
        if not use_rand:
            z = preconditioner(x, r)
        else:
            z = r

        # Save the old z'*r.
        zold_rold = z_r
        # Compute new z'*r.
        z_r = inner(x, z, r)

        # Compute new search direction
        beta = z_r / zold_rold
        delta = -z + beta * delta

        # Re-tangentialize delta to make sure it remains within the tangent
        # space.
        delta = manifold.to_tangent_space(x, delta)

        # Update new P-norms and P-dots [CGT2000, eq. 7.5.6 & 7.5.7].
        e_Pd = beta * (e_Pd + alpha * d_Pd)
        d_Pd = z_r + beta * beta * d_Pd

    return eta, Heta, j, stop_tCG

def TRSgep(A, a, B, Del, tolhardcase=1e-4):
    """
    Solves the trust-region subproblem by a generalized eigenproblem without iterations.

    minimize (x^T A x) / 2 + a^T x
    subject to x^T B x <= Del^2

    Parameters:
        A (ndarray): Symmetric nxn matrix.
        a (ndarray): nx1 vector.
        B (ndarray): Symmetric positive definite nxn matrix.
        Del (float): Radius constraint.
        tolhardcase (float): Tolerance for the hard case.

    Returns:
        x (ndarray): Solution vector.
        lam1 (float): Lagrange multiplier.
        info (str): Information about the solution.
    """
    n = A.shape[0]

    # Construct the block matrix MM1
    MM0 = np.block([[- B, A], [A, - np.outer(a, a) / (Del**2)]])
    MM1 = np.block([[np.zeros((n, n)), B], [B, np.zeros((n, n))]])
    # Possible interior solution
    p1, _ = scipy.sparse.linalg.cg(A, -a)
    if np.linalg.norm(A @ p1 + a) / np.linalg.norm(a) < 1e-5:
        if p1 @ B @ p1 >= Del**2:  # outside of the trust region
            p1 = np.full_like(p1, np.nan)  # ineligible
    else:  # numerically incorrect
        p1 = np.full_like(p1, np.nan)

    # Core of the code: generalized eigenproblem
    lams, vecs = scipy.linalg.eig(a=MM0, b=-MM1)
    rmidx = np.argmax(np.real(lams))
    lam1 = np.real(lams[rmidx])  # rightmost eigenvalue
    V = vecs[:, rmidx]  # corresponding rightmost eigenvector
    V = np.real(V) # if np.linalg.norm(np.real(V)) >= 1e-3 else np.imag(V)  # sometimes complex
    x = V[:n]  # extract solution component
    normx = np.sqrt(x @ (B @ x))
    x = x / normx * Del  # in the easy case, this naive normalization improves accuracy
    if x @ a > 0:
        x = -x  # take correct sign
    type = "boundary"
    if normx < tolhardcase:  # enter hard case
        x1 = V[n:]
        alpha1 = lam1 # copy.deepcopy(lam1)
        Pvect = x1  # first try only k=1, almost always enough
        Alam1B = A + lam1 * B
        BPvecti = B @ Pvect
        H = Alam1B + alpha1 * np.outer(BPvecti, BPvecti)
        x2 = scipy.linalg.solve(H, -a, assume_a='sym')
        type = "hardcase_1"

        # Residual check for hard case refinement
        if np.linalg.norm(Alam1B @ x2 + a) / np.linalg.norm(a) > tolhardcase:
            _, v = scipy.linalg.eigh(A, B)
            for ii in [3,6,9]:  # Iteratively refine solution if needed
                Pvect = v[:,:ii]  # Slices returns only the portion within the actual size of the array if the specified range exceeds the bounds.
                BPvecti = B @ Pvect
                H = Alam1B + alpha1 * BPvecti @ BPvecti.T
                x2 = scipy.linalg.solve(H, -a, assume_a='sym')
                type = f"hardcase_{ii}"
                if np.linalg.norm(Alam1B @ x2 + a) / np.linalg.norm(a) < tolhardcase:
                    break
        Bx = B @ x1
        Bx2 = B @ x2
        aa = x1 @ Bx
        bb = 2 * x2 @ Bx
        cc = x2 @ Bx2 - Del**2
        alp = (-bb + np.sqrt(bb**2 - 4 * aa * cc)) / (2 * aa)  # norm(x2+alp*x)-Delta
        x = x2 + alp * x1

    # Choose between interior and boundary solution
    if not np.isnan(p1).any():
        p1objval = 0.5 * (p1 @ A @ p1)  + a @ p1
        xobjval = 0.5 * (x @ A @ x)  + a @ x
        if p1objval <= xobjval:
            x = p1
            lam1 = 0
            type = "interior"
    return x, lam1, type

# Riemannian interior point trust region method
class RIPTRM(Solver):
    def __init__(self, option):
        # Default setting for interior point trust region method
        default_option = {
            # Stopping criteria
            'maxtime': 240,
            'maxiter': 100,
            'tolresid': 1e-15,
            'inner_maxiter': None,
            'inner_maxtime': None,

            # Inner iteration setting
            'initial_TR_radius': None,
            'minimal_initial_TR_radius': 1e-15,
            'maximal_TR_radius': 10,
            'rho': 0.1,  # threshold for the acceptance of the trial point
            'reduction_regularization': 1e3,
            'gamma': 0.25,  # the factor to shrink the trust region radius if the primal point is infeasible
            'forcing_function_Lagrangian': lambda mu: max(mu, 1e-14),
            'forcing_function_complementarity': lambda mu: max(1e-3 * mu, 1e-14),
            'forcing_function_second_order': lambda mu: mu,
            'min_barrier_parameter': 1e-15,
            'TRS_solver': 'Exact_RepMat',  # 'Exact_RepMat' or 'tCG'
            'second_order_stationarity': True,
            'do_euclidean_lincomb': False,
            'is_euclidean_embedded': False,
            'TRS_tolresid': 1e-12,
            'TRS_tolhardcase': 1e-8,
            'tCG_theta': 1,
            'tCG_kappa': 0.1,
            'tCG_mininner': 1,
            'checkTRSoptimality': False,
            'initial_barrier_parameter': 0.1,
            'barrier_parameter_update_r': 0.01,
            'barrier_parameter_update_c': 0.5,
            'barrier_parameter_update_b': 0.8,
            'do_simple_barrier_parameter_update': True,
            'const_left': 0.5,
            'const_right': 1e+20,
            'basisfun': lambda manifold, x: tangentorthobasis(manifold, x, manifold.dim),

            # Display setting
            'verbosity': 0,

            # Measuring violation for manifold constraints in 'self.compute_residual'
            'manviofun': lambda problem, x: 0,

            # Callback function at each step.
            'callbackfun': lambda problem, x, y, z, eval: eval,

            # logging
            'save_inner_iteration': True,
            'wandb_logging': False,

            # Exit on error
            'do_exit_on_error': True,
        }
        # Merge default_option and the argument
        default_option.update(option)  # putting the setting in the default_option before that in the argument
        self.option = default_option
        self.excluded_time = 0
        self.log = {}  # will be filled in self.add_log
        self.name = f"RIPTRM_{self.option['TRS_solver']}"
        self.initialize_wandb()

    def check_TRS_optimality(self, xCur, TR_radius, dxCur, lam1, HwCur, cxCur, manifold):
        basisfun = self.option["basisfun"]
        pred = 0 - 0.5 * manifold.inner_product(xCur, HwCur(dxCur), dxCur) - manifold.inner_product(xCur, cxCur, dxCur)
        dxCurnorm = manifold.norm(xCur, dxCur)
        cxCurnorm = manifold.norm(xCur, cxCur)
        basisxCur = basisfun(manifold, xCur)
        HwCurmatrix = selfadj_operator2matrix(manifold, xCur, HwCur, basisxCur)
        maxeigvalHwCur = scipy.sparse.linalg.eigsh(HwCurmatrix, k=1, which='LA', return_eigenvectors=False)[0]
        mineigvalHwCur = scipy.sparse.linalg.eigsh(HwCurmatrix, k=1, which='SA', return_eigenvectors=False)[0]
        Cauchydiff = pred - 0.5 * manifold.norm(xCur, cxCur)* min(TR_radius, cxCurnorm/maxeigvalHwCur)
        Eigendiff = pred + 0.5 *(TR_radius**2)*(mineigvalHwCur)
        Cauchycond = True if Cauchydiff >= 0 else Cauchydiff
        Eigencond = True if Eigendiff >= 0 or mineigvalHwCur >= 0 else Eigendiff
        print("Cauchy", Cauchycond, "Eigen", Eigencond)
        if lam1 is not None:
            TRS_KKTresid = HwCur(dxCur) + lam1 * dxCur + cxCur
            TRS_compl = lam1 * (TR_radius - dxCurnorm)
            TRS_normconst = TR_radius - dxCurnorm
            TRS_normcond = True if TRS_normconst >= 0 else TRS_normconst
            TRS_succeq = mineigvalHwCur + lam1
            TRS_succeqcond = True if TRS_succeq >= 0 else TRS_succeq
            print("TRS_KKTresid", manifold.norm(xCur, TRS_KKTresid), "TRS_compl", np.linalg.norm(TRS_compl), "TRS_normconst", TRS_normcond, "TRS_succeq", TRS_succeqcond)

    def set_initial_inner_info(self, inner_iteration, TR_radius):
        inner_info = {}
        inner_info["inner_status"] = None
        inner_info["num_inner"] = inner_iteration
        inner_info["TR_radius"] = TR_radius
        inner_info["normdx"] = None
        inner_info["dxtype"] = None
        inner_info["minxfeasi"] = None
        inner_info["minyfeasi"] = None
        inner_info["compl"] = None
        inner_info["mineigvalHw"] = None
        inner_info["ared/pred"] = None
        inner_info["radius_update"] = None
        inner_info["dual_clipping"] = None
        return inner_info

    def inner_preprocess(self, x_initial, y_initial, initial_TR_radius):
        xCur = x_initial
        yCur = y_initial
        inner_xPrev = copy.deepcopy(xCur)
        TR_radius = initial_TR_radius

        self.is_RepMat_available = False
        self.basisxCur = None
        self.HwCurmatrix = None
        self.cxCurvector = None
        self.basisxNew = None
        self.HwNewmatrix = None
        self.cxNewvector = None

        self.costxCur = None
        self.costineqconstvecxCur = None
        self.costxNew = None
        self.costineqconstvecxNew = None
        return xCur, yCur, inner_xPrev, TR_radius

    def compute_direction(self, problem, xCur, HwCur, cxCur, TR_radius, manifold):
        TRS_solver = self.option["TRS_solver"]
        TRS_tolhardcase = self.option["TRS_tolhardcase"]
        basisfun = self.option["basisfun"]
        theta = self.option["tCG_theta"]
        kappa = self.option["tCG_kappa"]
        mininner = self.option["tCG_mininner"]
        if TRS_solver == 'Exact_RepMat':
            xdim = manifold.dim
            if not self.is_RepMat_available:
                self.basisxCur = basisfun(manifold, xCur)
                self.HwCurmatrix = selfadj_operator2matrix(manifold, xCur, HwCur, self.basisxCur)
                self.cxCurvector = np.empty(xdim)
                for i in range(xdim):
                    self.cxCurvector[i] = manifold.inner_product(xCur, cxCur, self.basisxCur[i])
            coeff, lam1, type = TRSgep(self.HwCurmatrix, self.cxCurvector, np.eye(xdim), TR_radius, TRS_tolhardcase)
            dxCur = manifold.zero_vector(xCur)
            for i in range(xdim):
                dxCur = dxCur + coeff[i] * self.basisxCur[i]
        elif TRS_solver == 'tCG':
            eta = manifold.zero_vector(xCur)
            maxinner = manifold.dim
            tCG_HwCur = lambda x, dx: HwCur(dx)
            preconditioner = problem.preconditioner
            dxCur, _, _, stop_tCG = truncated_conjugate_gradient(manifold, tCG_HwCur, xCur, cxCur, eta, TR_radius, theta, kappa, mininner, maxinner, preconditioner, use_rand=False)
            type = f"tCG_{stop_tCG}"
            lam1 = None
        else:
            raise ValueError(f"TRS_solver {TRS_solver} is not supported.")
        return dxCur, lam1, type

    def egradLagrangefun(self, problem, x, y):
        manifold = problem.manifold
        egradineqconstraints = problem.ineqconstraints_euclidean_gradient_all
        if isinstance(manifold, pymanopt.manifolds.Product):
            egradcostfun = lambda x: _ProductAmbientVector(problem.euclidean_gradient(x))
        else:
            egradcostfun = problem.euclidean_gradient
        if isinstance(manifold, pymanopt.manifolds.Product):
            egradineqconstvecfun = lambda x: [- _ProductAmbientVector(egrad(x)) for egrad in egradineqconstraints]
        else:
            egradineqconstvecfun = lambda x: [-egrad(x) for egrad in egradineqconstraints]
        egradcostx = egradcostfun(x)
        egradineqconstvecx = egradineqconstvecfun(x)
        vec = egradcostx
        for i in range(len(y)):
            vec = vec - y[i] * egradineqconstvecx[i]
        return vec

    def gradLagrangefun(self, problem, x, y, do_euclidean_lincomb):
        manifold = problem.manifold
        gradcostfun = problem.riemannian_gradient
        gradineqconstraints = problem.ineqconstraints_riemannian_gradient_all
        gradineqconstvecfun = lambda x: [-grad(x) for grad in gradineqconstraints]
        if do_euclidean_lincomb:
            vec = self.egradLagrangefun(problem, x, y)
            vec = manifold.euclidean_to_riemannian_gradient(x, vec)
        else:
            gradcostx = gradcostfun(x)
            gradineqconstvecx = gradineqconstvecfun(x)
            vec = gradcostx
            for i in range(len(y)):
                vec = vec - y[i] * gradineqconstvecx[i]
        return vec

    def hessLagrangefun(self, problem, x, y, dx, do_euclidean_lincomb):
        hesscostfun = problem.riemannian_hessian
        hessineqconstraints = problem.ineqconstraints_riemannian_hessian_all
        hessineqconstvecfun = lambda x, dx: [-hess(x, dx) for hess in hessineqconstraints]
        manifold = problem.manifold
        ehessineqconstraints = problem.ineqconstraints_euclidean_hessian_all
        if isinstance(manifold, pymanopt.manifolds.Product):
            ehesscostfun = lambda x, dx: _ProductAmbientVector(problem.euclidean_hessian(x, dx))
        else:
            ehesscostfun = problem.euclidean_hessian
        if isinstance(manifold, pymanopt.manifolds.Product):
            ehessineqconstvecfun = lambda x, dx: [-_ProductAmbientVector(ehess(x, dx)) for ehess in ehessineqconstraints]
        else:
            ehessineqconstvecfun = lambda x, dx: [-ehess(x, dx) for ehess in ehessineqconstraints]

        def ehessLagrangefun(x, y, dx):
            ehesscost = ehesscostfun(x, dx)
            ehessineqconstvec = ehessineqconstvecfun(x, dx)
            vec = ehesscost
            for i in range(len(y)):
                vec = vec - y[i] * ehessineqconstvec[i]
            return vec

        if do_euclidean_lincomb:
            egrad = self.egradLagrangefun(problem, x, y)
            vec = ehessLagrangefun(x, y, dx)
            vec = manifold.euclidean_to_riemannian_hessian(x, egrad, vec, dx)
        else:
            vec = hesscostfun(x, dx)
            hessineqconstvec = hessineqconstvecfun(x, dx)
            for i in range(len(y)):
                vec = vec - y[i] * hessineqconstvec[i]
        return vec

    def Gxfun(self, problem, x, v, do_euclidean_lincomb):
        manifold = problem.manifold
        gradineqconstraints = problem.ineqconstraints_riemannian_gradient_all
        gradineqconstvecfun = lambda x: [-grad(x) for grad in gradineqconstraints]
        egradineqconstraints = problem.ineqconstraints_euclidean_gradient_all

        if isinstance(manifold, pymanopt.manifolds.Product):
            egradineqconstvecfun = lambda x: [- _ProductAmbientVector(egrad(x)) for egrad in egradineqconstraints]
        else:
            egradineqconstvecfun = lambda x: [-egrad(x) for egrad in egradineqconstraints]

        def eGxfun(x, v):
            egradineqconstvecx = egradineqconstvecfun(x)
            vec = manifold.zero_vector(x)
            for idx in range(len(egradineqconstvecx)):
                vec = vec + v[idx] * egradineqconstvecx[idx]
            return vec

        if do_euclidean_lincomb:
            vec = eGxfun(x, v)
            vec = manifold.euclidean_to_riemannian_gradient(x, vec)
        else:
            gradineqconstvecx = gradineqconstvecfun(x)
            vec = manifold.zero_vector(x)
            for idx in range(len(gradineqconstvecx)):
                vec = vec + v[idx] * gradineqconstvecx[idx]
        return vec

    def Gxajfun(self, problem, x, dx, is_euclidean_embedded):
        manifold = problem.manifold
        egradineqconstraints = problem.ineqconstraints_euclidean_gradient_all
        gradineqconstraints = problem.ineqconstraints_riemannian_gradient_all
        gradineqconstvecfun = lambda x: [-grad(x) for grad in gradineqconstraints]
        if isinstance(manifold, pymanopt.manifolds.Product):
            egradineqconstvecfun = lambda x: [- _ProductAmbientVector(egrad(x)) for egrad in egradineqconstraints]
        else:
            egradineqconstvecfun = lambda x: [-egrad(x) for egrad in egradineqconstraints]

        def eGxajfun(x, dx):
            egradineqconstvecx = egradineqconstvecfun(x)
            return np.array([manifold.inner_product(x, egradineq, dx) for egradineq in egradineqconstvecx])

        if is_euclidean_embedded:
            return eGxajfun(x, dx)
        else:
            gradineqconstvecx = gradineqconstvecfun(x)
            return np.array([manifold.inner_product(x, gradineq, dx) for gradineq in gradineqconstvecx])

    # Check stopping criteria
    def compute_inner_stoppingcriteria(self, problem, xNew, yNew, mu, inner_option):
        ineqconstraints = problem.ineqconstraints_all
        costineqconstvecfun = lambda x: np.array([-ineqfun(x) for ineqfun in ineqconstraints])
        costineqconstvecxNew = costineqconstvecfun(xNew)
        gradcostfun = problem.riemannian_gradient
        manifold = problem.manifold
        xdim = manifold.dim
        basisfun = self.option["basisfun"]
        stopping_criterion_Lagrangian = inner_option["stopping_criterion_Lagrangian"]
        stopping_criterion_complementarity = inner_option["stopping_criterion_complementarity"]
        second_order_stationarity = self.option["second_order_stationarity"]
        if second_order_stationarity:
            stopping_criterion_second_order = inner_option["stopping_criterion_second_order"]
        TRS_solver = self.option["TRS_solver"]
        do_euclidean_lincomb = self.option["do_euclidean_lincomb"]
        is_euclidean_embedded = self.option["is_euclidean_embedded"]

        xfeasi_criterion = np.all(costineqconstvecxNew > 0)
        yfeasi_criterion = np.all(yNew > 0)
        normgradLagfun = manifold.norm(xNew, self.gradLagrangefun(problem, xNew, yNew, do_euclidean_lincomb))
        normgradLagfun_criterion = normgradLagfun <= stopping_criterion_Lagrangian
        complementarity = np.linalg.norm(yNew * costineqconstvecxNew - mu)
        complementary_criterion = complementarity <= stopping_criterion_complementarity
        mineigvalHwNew = None
        mineigval_criterion = True
        if TRS_solver == 'Exact_RepMat' and second_order_stationarity:
            GxNew = lambda v: self.Gxfun(problem, xNew, v, do_euclidean_lincomb)
            GxajNew = lambda dx: self.Gxajfun(problem, xNew, dx, is_euclidean_embedded)
            HwNew = lambda dx: self.hessLagrangefun(problem, xNew, yNew, dx, do_euclidean_lincomb) + GxNew((yNew * GxajNew(dx)) / costineqconstvecxNew)
            basisxNew = basisfun(manifold, xNew)
            HwNewmatrix = selfadj_operator2matrix(manifold, xNew, HwNew, basisxNew)
            gradcostfunxNew = gradcostfun(xNew)
            cxNew = gradcostfunxNew -  GxNew(mu / costineqconstvecxNew)
            cxNewvector = np.empty(xdim)
            for i in range(xdim):
                cxNewvector[i] = manifold.inner_product(xNew, cxNew, basisxNew[i])

            eigval = scipy.linalg.eigh(HwNewmatrix, eigvals_only=True)  # ascending order
            mineigvalHwNew = eigval[0]
            mineigval_criterion = True if mineigvalHwNew >= -stopping_criterion_second_order else False

            self.basisxNew = basisxNew
            self.HwNewmatrix = HwNewmatrix
            self.cxNewvector = cxNewvector

        output = {}
        output["xfeasi_criterion"] = xfeasi_criterion
        output["yfeasi_criterion"] = yfeasi_criterion
        output["normgradLagfun_criterion"] = normgradLagfun_criterion
        output["complementary_criterion"] = complementary_criterion
        output["mineigval_criterion"] = mineigval_criterion
        output["minxfeasi"] = min(costineqconstvecxNew)
        output["minyfeasi"] = min(yNew)
        output["compl"] = complementarity
        output["mineigvalHw"] = mineigvalHwNew
        return output

    def update_xy_TR_radius(self, problem, xCur, yCur, HwCur, cxCur, dxCur, normdxCur, xNew, yNew, mu, TR_radius):
        costfun = problem.cost
        ineqconstraints = problem.ineqconstraints_all
        manifold = problem.manifold
        costineqconstvecfun = lambda x: np.array([-ineqfun(x) for ineqfun in ineqconstraints])
        reduction_regularization = self.option["reduction_regularization"]
        maximal_TR_radius = self.option["maximal_TR_radius"]
        rho = self.option["rho"]
        const_left = self.option["const_left"]
        const_right = self.option["const_right"]
        TRS_solver = self.option["TRS_solver"]
        second_order_stationarity = self.option["second_order_stationarity"]

        def logbarrfun(x, mu, costfunx=None, costineqconstvecx=None):
            if costfunx is None:
                costfunx = costfun(x)
            if costineqconstvecx is None:
                costineqconstvecx = costineqconstvecfun(x)
            return costfun(x) - mu * np.sum(np.log(costineqconstvecx))

        # Compute the cost and inequality constraints at the new point
        self.costxNew = costfun(xNew)
        self.costineqconstvecxNew = costineqconstvecfun(xNew)

        # Compute the acutual and predicted reductions
        logbarrxCur = logbarrfun(xCur, mu, costfunx=self.costxCur, costineqconstvecx=self.costineqconstvecxCur)
        logbarrxNew = logbarrfun(xNew, mu, costfunx=self.costxNew, costineqconstvecx=self.costineqconstvecxNew)
        ared =  logbarrxCur - logbarrxNew
        pred = 0 - 0.5 * manifold.inner_product(xCur, HwCur(dxCur), dxCur) - manifold.inner_product(xCur, cxCur, dxCur)
        red_reg = max(1, abs(logbarrxCur)) * np.spacing(1) * reduction_regularization
        ared = ared + red_reg
        pred = pred + red_reg

        # Update the trust region radius
        output = {}
        output["ared/pred"] = ared / pred
        if ared < 0.25 * pred:
            output["radius_update"] = "reduced"
            TR_radiusNext = 0.25 * TR_radius
        elif ared >= 0.75 * pred and np.abs(normdxCur - TR_radius) <= 1e-15:
            output["radius_update"] = "expanded"
            TR_radiusNext = min(2 * TR_radius, maximal_TR_radius)
        else:
            output["radius_update"] = "unchanged"
            TR_radiusNext = TR_radius

        if ared > rho * pred:
            inner_status = "successful"
            output["inner_status"] = inner_status
            xNext = copy.deepcopy(xNew)
            I_left = const_left * np.minimum(np.minimum(yCur, mu / self.costineqconstvecxNew), 1)
            I_right = np.maximum(const_right, const_right / mu, np.maximum(yCur, const_right / self.costineqconstvecxNew))
            clippingfun = lambda yNew: np.minimum(np.maximum(yNew, I_left), I_right)
            clippedyNew = clippingfun(yNew)
            if np.array_equal(yNew, clippedyNew):
                output["dual_clipping"] = False
                self.is_RepMat_available = False
                if TRS_solver == 'Exact_RepMat' and second_order_stationarity:
                    self.basisxCur = copy.deepcopy(self.basisxNew)
                    self.HwCurmatrix = copy.deepcopy(self.HwNewmatrix)
                    self.cxCurvector = copy.deepcopy(self.cxNewvector)
                    self.is_RepMat_available = True
            else:
                output["dual_clipping"] = True
                self.is_RepMat_available = False
            yNext = clippedyNew
        else:
            inner_status = "unsuccessful"
            output["inner_status"] = inner_status
            output["dual_clipping"] = None
            xNext = xCur
            yNext = yCur
            self.is_RepMat_available = True

        return xNext, yNext, TR_radiusNext, output

    def inner_step(self, problem, xCur, yCur, mu, TR_radius, inner_iteration, inner_option):
        inner_info = self.set_initial_inner_info(inner_iteration, TR_radius)
        gamma = self.option["gamma"]
        checkTRSoptimality = self.option["checkTRSoptimality"]
        do_euclidean_lincomb = self.option["do_euclidean_lincomb"]
        is_euclidean_embedded = self.option["is_euclidean_embedded"]

        # Set function components
        costfun = problem.cost
        ineqconstraints = problem.ineqconstraints_all
        manifold = problem.manifold
        gradcostfun = problem.riemannian_gradient

        # Set a vector-valued function
        costineqconstvecfun = lambda x: np.array([-ineqfun(x) for ineqfun in ineqconstraints])

        # Set functions at xCur
        self.costxCur = costfun(xCur)
        self.costineqconstvecxCur = costineqconstvecfun(xCur)
        gradcostfunxCur = gradcostfun(xCur)
        GxCur = lambda v: self.Gxfun(problem, xCur, v, do_euclidean_lincomb)
        GxajCur = lambda dx: self.Gxajfun(problem, xCur, dx, is_euclidean_embedded)
        HwCur = lambda dx: self.hessLagrangefun(problem, xCur, yCur, dx, do_euclidean_lincomb) + GxCur((yCur * GxajCur(dx)) / self.costineqconstvecxCur)
        cxCur = gradcostfunxCur -  GxCur(mu / self.costineqconstvecxCur)

        # Compute the direction
        dxCur, lam1, type = self.compute_direction(problem, xCur, HwCur, cxCur, TR_radius, manifold)
        inner_info["dxtype"] = type
        normdxCur = manifold.norm(xCur, dxCur)
        inner_info["normdx"] = normdxCur

        # Check TRS optimality
        if checkTRSoptimality:
            self.check_TRS_optimality(xCur, TR_radius, dxCur, lam1, HwCur, cxCur, manifold)

        # Compute xNew and yNew
        dyCur = - yCur + mu * (1 / self.costineqconstvecxCur) - yCur * GxajCur(dxCur) / self.costineqconstvecxCur
        xNew = manifold.retraction(xCur, dxCur)
        yNew = yCur + dyCur

        # Check stopping criteria of the inner iteration
        inner_stopcriteria_result = self.compute_inner_stoppingcriteria(problem, xNew, yNew, mu, inner_option)
        xfeasi_criterion = inner_stopcriteria_result["xfeasi_criterion"]
        yfeasi_criterion = inner_stopcriteria_result["yfeasi_criterion"]
        normgradLagfun_criterion = inner_stopcriteria_result["normgradLagfun_criterion"]
        complementary_criterion = inner_stopcriteria_result["complementary_criterion"]
        mineigval_criterion = inner_stopcriteria_result["mineigval_criterion"]

        # Set inner_info
        inner_info["minxfeasi"] = inner_stopcriteria_result["minxfeasi"]
        inner_info["minyfeasi"] = inner_stopcriteria_result["minyfeasi"]
        inner_info["compl"] = inner_stopcriteria_result["compl"]
        inner_info["mineigvalHw"] = inner_stopcriteria_result["mineigvalHw"]

        # Return if all stopping criteria are satisfied
        if xfeasi_criterion and yfeasi_criterion and normgradLagfun_criterion and complementary_criterion and mineigval_criterion:
            inner_status = "converged"
            inner_info["inner_status"] = inner_status
            exitflag = True
            return exitflag, xNew, yNew, TR_radius, inner_info

        # Return if x is infeasible. Unsuccessful iteration.
        if not xfeasi_criterion:
            inner_status = "primal_infeasible"
            inner_info["inner_status"] = inner_status
            TR_radiusShrink = gamma * normdxCur
            self.is_RepMat_available = True
            exitflag = False
            return exitflag, xCur, yCur, TR_radiusShrink, inner_info

        xNext, yNext, TR_radiusNext, update_result = self.update_xy_TR_radius(problem, xCur, yCur, HwCur, cxCur, dxCur, normdxCur, xNew, yNew, mu, TR_radius)
        inner_info["ared/pred"] = update_result["ared/pred"]
        inner_info["radius_update"] = update_result["radius_update"]
        inner_info["dual_clipping"] = update_result["dual_clipping"]
        inner_info["inner_status"] = update_result["inner_status"]
        exitflag = False
        return exitflag, xNext, yNext, TR_radiusNext, inner_info

    def inner_run(self, problem, outer_iteration, outer_start_time, x_initial, y_initial, mu, initial_TR_radius, inner_option):
        # Set initial point
        xCur, yCur, inner_xPrev, TR_radius = self.inner_preprocess(x_initial, y_initial, initial_TR_radius)
        inner_xPrev = copy.deepcopy(xCur)
        inner_iteration = 0
        inner_start_time = time.time()
        verbosity = self.option["verbosity"]
        manviofun = self.option["manviofun"]
        callbackfun = self.option["callbackfun"]
        inner_maxiter = self.option["inner_maxiter"]
        save_inner_iteration =  self.option["save_inner_iteration"]
        exitflag = False
        inner_info = self.set_initial_inner_info(inner_iteration, TR_radius)
        inner_info["inner_status"] = "initial"

        while True:
            # Set initial inner_info
            if verbosity > 1:
                costprint = "{:.3e}".format(problem.cost(xCur))
                KKTresidprint, _, _, _, _ = compute_residual(problem, xCur, yCur, [], manviofun)
                TRradiusprint = "{:.3e}".format(TR_radius)
                print(f"Iter: {outer_iteration}-{inner_iteration}, Cost: {costprint}, KKT resid: {KKTresidprint}, TR: {TRradiusprint}, Dir: {inner_info["dxtype"]}, Stat: {inner_info["inner_status"]}")

            inner_iteration += 1

            exitflag, xCur, yCur, TR_radius, inner_info = self.inner_step(problem, xCur, yCur, mu, TR_radius, inner_iteration, inner_option)

            if save_inner_iteration:
                log_start_time = time.time()
                eval_log = evaluation(problem, inner_xPrev, xCur, yCur, [], manviofun, callbackfun)
                solver_log = self.solver_status(yCur, mu, save_inner_iteration, inner_info = inner_info)
                log_end_time = time.time()
                self.excluded_time += log_end_time - log_start_time
                self.add_log(outer_iteration, outer_start_time, eval_log, solver_log, self.excluded_time)
            inner_xPrev = copy.deepcopy(xCur)

            # Check stopping criteria (time and iteration)
            if self.option["inner_maxtime"] is None:
                inner_maxtime = self.option["maxtime"] # outer's maxtime
                run_time = time.time() - outer_start_time - self.excluded_time
            else:
                inner_maxtime = self.option["inner_maxtime"]
                run_time = time.time() - inner_start_time
            if run_time >= inner_maxtime:
                inner_info["inner_status"] = "max-time-exceeded"
                exitflag = True
                xCur = x_initial
                yCur = y_initial
                TR_radius = initial_TR_radius
                inner_xPrev = copy.deepcopy(x_initial)
            if inner_maxiter is not None:
                if inner_iteration >= inner_maxiter:
                    inner_info["inner_status"] = "max-iter-exceeded"
                    exitflag = True
                    xCur = x_initial
                    yCur = y_initial
                    TR_radius = initial_TR_radius
                    inner_xPrev = copy.deepcopy(x_initial)
            # Exit if the stopping criteria are satisfied
            if exitflag:
                break

        return xCur, yCur, TR_radius, inner_info

    def outer_preprocess(self, problem):
        xCur = copy.deepcopy(problem.initialpoint)
        yCur = copy.deepcopy(problem.initialineqLagmult)
        muCur = self.option["initial_barrier_parameter"]

        # Set the trust region radius
        if self.option['initial_TR_radius'] is None:
            try:
                Delta_bar = problem.manifold.typical_dist
            except NotImplementedError:
                Delta_bar = np.sqrt(problem.manifold.dim)
            initial_TR_radius = Delta_bar / 8
        else:
            initial_TR_radius = self.option['initial_TR_radius']

        return xCur, yCur, muCur, initial_TR_radius

    def outer_step(self, problem, xCur, yCur, muCur, TR_radius, iteration, start_time):
        # Set parameters for outer iteration
        option = self.option
        second_order_stationarity = option['second_order_stationarity']
        forcing_function_Lagrangian = option['forcing_function_Lagrangian']
        forcing_function_complementarity = option['forcing_function_complementarity']
        forcing_function_second_order = option['forcing_function_second_order']
        minimal_initial_TR_radius = self.option["minimal_initial_TR_radius"]
        min_barrier_parameter = option['min_barrier_parameter']
        barrier_parameter_update_r = option['barrier_parameter_update_r']
        barrier_parameter_update_c = option['barrier_parameter_update_c']
        barrier_parameter_update_b = option['barrier_parameter_update_b']
        do_simple_barrier_parameter_update = option['do_simple_barrier_parameter_update']

        # Update the inner option
        inner_option = {}
        inner_option["stopping_criterion_Lagrangian"] = forcing_function_Lagrangian(muCur)
        inner_option["stopping_criterion_complementarity"] = forcing_function_complementarity(muCur)
        if second_order_stationarity:
            inner_option["stopping_criterion_second_order"] = forcing_function_second_order(muCur)

        xCur, yCur, TR_radius, inner_info = self.inner_run(problem, iteration, start_time, xCur, yCur, muCur, TR_radius, inner_option)

        # Update the barrier parameter and TR radius
        if do_simple_barrier_parameter_update:
            muCur = max(min_barrier_parameter, barrier_parameter_update_c * (muCur ** (1 + barrier_parameter_update_r)))
        else:
            muCur = max(min_barrier_parameter, min(barrier_parameter_update_b * muCur, barrier_parameter_update_c  * (muCur**(1 + barrier_parameter_update_r))))
        TR_radius = max(TR_radius, minimal_initial_TR_radius)

        return xCur, yCur, muCur, TR_radius, inner_info

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
        if problem.has_eqconstraints:
            warnings.warn("Equality constraints detecred. Currently, RIPTRM does not support equality constraints and will completely ignore them.", Warning)

        xCur, yCur, muCur, TR_radius = self.outer_preprocess(problem)
        xPrev = copy.deepcopy(xCur)
        inner_info = None
        iteration = 0
        start_time = time.time()

        # Set parameters for outer iteration
        option = self.option
        verbosity = option["verbosity"]
        save_inner_iteration = option["save_inner_iteration"]

        # The first evaluation and logging
        manviofun = option["manviofun"]
        callbackfun = option["callbackfun"]
        do_exit_on_error = self.option['do_exit_on_error']

        # Outer iteration
        while True:
            log_eval_start_time = time.time()
            eval_log = evaluation(problem, xPrev, xCur, yCur, [], manviofun, callbackfun)
            log_eval_end_time = time.time()
            self.excluded_time += log_eval_end_time - log_eval_start_time
            if iteration == 0 or not save_inner_iteration:
                log_status_start_time = time.time()
                solver_log = self.solver_status(yCur, muCur, save_inner_iteration, inner_info)
                log_status_end_time = time.time()
                self.excluded_time += log_status_end_time - log_status_start_time
                self.add_log(iteration, start_time, eval_log, solver_log, self.excluded_time)
            residual = eval_log["residual"]
            tolresid = option["tolresid"]
            residual_criterion = (residual <= tolresid, "KKT residual tolerance reached; current residual=" + str(residual) + " and tolresid=" + str(tolresid))
            stopping_criteria =[residual_criterion]
            xPrev = copy.deepcopy(xCur)

            if verbosity == 1:
                print(f"Outer iteration: {iteration}, Cost: {problem.cost(xCur)}, KKT residual: {residual}, mu: {muCur}")

            # Check stopping criteria (time, iteration and residual)
            stop, reason = self.check_stoppingcriterion(start_time, iteration, stopping_criteria, self.excluded_time)
            if stop:
                self.option["stoppingcriterion"] = reason
                if verbosity:
                    print(reason)
                break
            # Count an iteration
            iteration += 1

            if do_exit_on_error:
                try:
                    xCur, yCur, muCur, TR_radius, inner_info = self.outer_step(problem, xCur, yCur, muCur, TR_radius, iteration, start_time)
                except Exception as e:
                    print(f"Error: {e}")
                    break
            else:
                xCur, yCur, muCur, TR_radius, inner_info = self.outer_step(problem, xCur, yCur, muCur, TR_radius, iteration, start_time)

        # After exiting while loop, we return the final output
        output = self.postprocess(xCur, yCur, [])

        if self.option["wandb_logging"]:
            wandb.finish()

        return output


    # Examine the solver status
    def solver_status(self,
                      yCur,
                      mu,
                      save_inner_iteration,
                      inner_info = None,
                      ):
        solver_status = {}
        solver_status["mu"] = mu

        if inner_info is not None:
            solver_status["num_inner"] = inner_info["num_inner"]
            solver_status["inner_status"] = inner_info["inner_status"]
            solver_status["TR_radius"] = inner_info["TR_radius"]
        else:
            solver_status["num_inner"] = None
            solver_status["inner_status"] = None
            solver_status["TR_radius"] = None

        if  save_inner_iteration:
            if inner_info is not None:
                solver_status["dxtype"] = inner_info["dxtype"]
                solver_status["normdx"] = inner_info["normdx"]
                solver_status["minxfeasi"] = inner_info["minxfeasi"]
                solver_status["minyfeasi"] = inner_info["minyfeasi"]
                solver_status["compl"] = inner_info["compl"]
                solver_status["mineigvalHw"] = inner_info["mineigvalHw"]
                solver_status["ared/pred"] = inner_info["ared/pred"]
                solver_status["radius_update"] = inner_info["radius_update"]
                solver_status["dual_clipping"] = inner_info["dual_clipping"]
            else:
                solver_status["dxtype"] = None
                solver_status["normdx"] = None
                solver_status["minxfeasi"] = None
                solver_status["minyfeasi"] = None
                solver_status["compl"] = None
                solver_status["mineigvalHw"] = None
                solver_status["ared/pred"] = None
                solver_status["radius_update"] = None
                solver_status["dual_clipping"] = None

        maxabsLagmult = float('-inf')
        for Lagmult in yCur:
            maxabsLagmult = max(maxabsLagmult, abs(Lagmult))
        solver_status["maxabsLagmult"] = maxabsLagmult
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
    if hasattr(solver_option, "RIPTRM"):
        specific = dict(getattr(solver_option, "RIPTRM"))
        option.update(specific)

    # Run the experiment
    riptrmsolver = RIPTRM(option)
    output = riptrmsolver.run(problem)
    print(output)

if __name__=='__main__':
    main()
    # cProfile.run("main()", sort="tottime")
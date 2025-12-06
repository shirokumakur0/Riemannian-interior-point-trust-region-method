import hydra, copy, time, pymanopt, wandb, warnings
import numpy as np
from dataclasses import dataclass, field
from utils import evaluation, tangentorthobasis, operator2matrix, tangent2vec, selfadj_operator2matrix, NonlinearProblem, Output
from scipy import linalg

import sys
sys.path.append('./src/base')
from base_solver import Solver, BaseOutput

warnings.filterwarnings("ignore", message="Output seems independent of input.")

def barFx(x, c, gradfuns, manifold):
    vec = manifold.zero_vector(x)
    for idx in range(len(gradfuns)):
        g = gradfuns[idx]
        vec = vec + c[idx] * g(x)
    return vec

def barFxaj(x, dx, gradfuns, manifold):
    val = np.zeros(len(gradfuns))
    for idx in range(len(gradfuns)):
        g = gradfuns[idx]
        val[idx] = manifold.inner_product(x, g(x), dx)
    return val

def hess_Fx(x, c, dx, hessfuns, manifold):
    vec =  manifold.zero_vector(x)
    for idx in range(len(hessfuns)):
        h = hessfuns[idx]
        vec = vec + c[idx] * h(x, dx)
    return vec

def barGx(x, z, egradineqconstraints):
    vec = 0  # will be converted to a zero embedded vector by broadcasting
    for idx in range(len(egradineqconstraints)):
        egrad = egradineqconstraints[idx]
        vec = vec + z[idx] * egrad(x)
    return vec

def ehess_barGx(x, z, dx, ehessineqconstraints):
    vec = 0  # will be converted to a zero embedded vector by broadcasting
    for idx in range(len(ehessineqconstraints)):
        ehess = ehessineqconstraints[idx]
        vec = vec + z[idx] * ehess(x, dx)
    return vec

def barGxaj(x, dx, egradineqconstraints):
    val = np.zeros(len(egradineqconstraints))
    for idx in range(len(egradineqconstraints)):
        egrad = egradineqconstraints[idx]
        val[idx] = egrad(x).reshape(-1) @ dx.reshape(-1)
    return val

def barHx(x, y, egradeqconstraints):
    vec = 0   # will be converted to a zero embedded vector by broadcasting
    for idx in range(len(egradeqconstraints)):
        egrad = egradeqconstraints[idx]
        vec = vec + y[idx] * egrad(x)
    return vec

def ehess_barHx(x, y, dx, ehesseqconstraints):
    vec = 0  # will be converted to a zero embedded vector by broadcasting
    for idx in range(len(ehesseqconstraints)):
        ehess = ehesseqconstraints[idx]
        vec = vec + y[idx] * ehess(x, dx)
    return vec

def barHxaj(x, dx, egradeqconstraints):
    val = np.zeros(len(egradeqconstraints))
    for idx in range(len(egradeqconstraints)):
        egrad = egradeqconstraints[idx]
        val[idx] = egrad(x).reshape(-1) @ dx.reshape(-1)
    return val

def build_KKTVectorField(gradLagrangian, ineqconstraints, eqconstraints):
    def KKTVectorField(xyzs):
        x, y, z, s = xyzs
        dx = gradLagrangian(x, y, z)
        dy = np.empty(len(eqconstraints), dtype=object)
        for idx in range(len(eqconstraints)):
            dy[idx] = (eqconstraints[idx])(x)
        dz = np.empty(len(ineqconstraints), dtype=object)
        for idx in range(len(ineqconstraints)):
            dz[idx] = (ineqconstraints[idx])(x) + s[idx]
        ds = z * s
        dxdydzds = pymanopt.manifolds.product._ProductTangentVector([dx, dy, dz, ds])
        return dxdydzds
    return KKTVectorField

def build_egradLagrangian(costegradfun, egradineqconstraints, egradeqconstraints):
    if len(egradeqconstraints) > 0:
        def egradLagrangian(x, y, z):
            val = costegradfun(x) + barGx(x, z, egradineqconstraints) + barHx(x, y, egradeqconstraints)
            return val
    else:
        def egradLagrangian(x, y, z):
            val = costegradfun(x) + barGx(x, z, egradineqconstraints)
            return val
    return egradLagrangian

def build_ehessLagrangian(costehessfun, ehessineqconstraints, ehesseqconstraints):
    if len(ehesseqconstraints) > 0:
        def ehessLagrangian(x, y, z, dx):
            val = costehessfun(x, dx) + ehess_barGx(x, z, dx, ehessineqconstraints) + ehess_barHx(x, y, dx, ehesseqconstraints)
            return val
    else:
        def ehessLagrangian(x, y, z, dx):
            val = costehessfun(x, dx) + ehess_barGx(x, z, dx, ehessineqconstraints)
            return val
    return ehessLagrangian

def build_gradLagrangian(costegradfun, gradineqconstraints, gradeqconstraints, manifold):
    if len(gradeqconstraints) > 0:
        def gradLagrangian(x, y, z):
            val = costegradfun(x) + barFx(x, z, gradineqconstraints, manifold) + barFx(x, y, gradeqconstraints, manifold)
            return val
    else:
        def gradLagrangian(x, y, z):
            val = costegradfun(x) + barFx(x, z, gradineqconstraints, manifold)
            return val
    return gradLagrangian

def build_hessLagrangian(costehessfun, hessineqconstraints, hesseqconstraints, manifold):
    if len(hesseqconstraints) > 0:
        def hessLagrangian(x, y, z, dx):
            val = costehessfun(x, dx) + hess_Fx(x, z, dx, hessineqconstraints, manifold) + hess_Fx(x, y, dx, hesseqconstraints, manifold)
            return val
    else:
        def hessLagrangian(x, y, z, dx):
            val = costehessfun(x, dx) + hess_Fx(x, z, dx, hessineqconstraints, manifold)
            return val
    return hessLagrangian

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
            'basisfun': lambda manifold, x: tangentorthobasis(manifold, x, manifold.dim),
            'do_euclidean_lincomb': False,

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
            'verbosity': 2,

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
        type = "Krylov" if self.option["KrylovIterMethod"] else "RepMat"
        self.name = f"RIPM_{type}_gamma{self.option['gamma']}_beta{self.option['linesearch_beta']}_theta{self.option['linesearch_theta']}"
        self.initialize_wandb()

    def xyzs_manifold_zero_vector(self, xyzs):
        x, y, z, s = xyzs
        dx = self.x_manifold.zero_vector(x)
        dy = self.y_manifold.zero_vector(y)
        dz = self.z_manifold.zero_vector(z)
        ds = self.s_manifold.zero_vector(s)
        return [dx, dy, dz, ds]

    def xyzs_manifold_norm(self, xyzs, dxdydzds):
        norm = np.sqrt(self.xyzs_manifold_inner_product(xyzs, dxdydzds, dxdydzds))
        return norm

    def xy_manifold_norm(self, xy, dxdy):
        norm = np.sqrt(self.xy_manifold_inner_product(xy, dxdy, dxdy))
        return norm

    def xy_manifold_zero_vector(self, xyCur):
        x, y = xyCur
        dx = self.x_manifold.zero_vector(x)
        dy = self.y_manifold.zero_vector(y)
        return [dx, dy]

    def xyzs_manifold_retraction(self, xyzs, dxdydzds):
        x, y, z, s = xyzs
        dx, dy, dz, ds = dxdydzds
        xNew = self.x_manifold.retraction(x, dx)
        yNew = self.y_manifold.retraction(y, dy)
        zNew = self.z_manifold.retraction(z, dz)
        sNew = self.s_manifold.retraction(s, ds)
        xyzsNew = [xNew, yNew, zNew, sNew]
        return xyzsNew

    def xyzs_manifold_inner_product(self, xyzs, dxdydzds1, dxdydzds2):
        x, y, z, s = xyzs
        dx1, dy1, dz1, ds1 = dxdydzds1
        dx2, dy2, dz2, ds2 = dxdydzds2
        x_inner_product = self.x_manifold.inner_product(x, dx1, dx2)
        y_inner_product = self.y_manifold.inner_product(y, dy1, dy2)
        z_inner_product = self.z_manifold.inner_product(z, dz1, dz2)
        s_inner_product = self.s_manifold.inner_product(s, ds1, ds2)
        xyzs_inner_product = x_inner_product + y_inner_product + z_inner_product + s_inner_product
        return xyzs_inner_product

    def xy_manifold_inner_product(self, xy, dxdy1, dxdy2):
        x, y = xy
        dx1, dy1 = dxdy1
        dx2, dy2 = dxdy2
        x_inner_product = self.x_manifold.inner_product(x, dx1, dx2)
        y_inner_product = self.y_manifold.inner_product(y, dy1, dy2)
        xy_inner_product = x_inner_product + y_inner_product
        return xy_inner_product

    def RepresentMatMethod(self, Aw, Hxaj, Hx, cq, xy, xbasis, ybasis):
        x_manifold = self.x_manifold
        y_manifold = self.y_manifold
        do_euclidean_lincomb = self.option['do_euclidean_lincomb']
        xdim = x_manifold.dim
        ydim = y_manifold.dim
        x, y  = xy
        c, q = cq

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
        Hxaj_mat = operator2matrix(x_manifold, x, y, Hxaj, xbasis, ybasis, y_manifold) # Hx_mat.T=Hxaj_mat.
        if do_euclidean_lincomb:
            T_mat = np.block([
                [Aw_mat, Hxaj_mat.T],
                [Hxaj_mat, np.zeros((ydim, ydim))]
            ])
        else:
            Hx_mat = operator2matrix(y_manifold, y, x, Hx, ybasis, xbasis, x_manifold)
            if ydim > 0:
                T_mat = np.block([
                    [Aw_mat, Hx_mat],
                    [Hxaj_mat, np.zeros((ydim, ydim))]
                ])
            else:
                T_mat = Aw_mat
        c_vec = tangent2vec(x_manifold, x, xbasis, c)
        q_vec = tangent2vec(y_manifold, y, ybasis, q)
        cq_vec = np.concatenate([c_vec, q_vec])
        sol_vec = linalg.solve(T_mat, cq_vec, assume_a='sym')
        NTdirx = x_manifold.zero_vector(x)
        NTdiry = y_manifold.zero_vector(y)

        for i in range(len(xbasis)):
            NTdirx = NTdirx + sol_vec[i] * xbasis[i]
        for j in range(len(ybasis)):
            NTdiry = NTdiry + sol_vec[xdim+j] * ybasis[j]
        NTdir = [NTdirx, NTdiry]

        RepresentMat = T_mat
        RepresentMatOrder = x_manifold.dim + y_manifold.dim

        return NTdir, RepresentMat, RepresentMatOrder
    
    def TangentSpaceConjResMethod(self, A, b, v0, xy, tol, maxiter):
        # Conjugate residual method for solving linear operator equation: A(v)=b,
        # where A is some self-adjoint operator to and from some linear space E,
        # b is an element in E. Assume the existence of solution v.
        
        # Yousef Saad - Iterative methods for sparse linear systems,
        # 2nd edition-SIAM (2003) P203. ALGORITHM 6.20
        
        v = v0  # initialization
        r = b - A(v)  # r are residuals.
        p = copy.deepcopy(r)  # p are conjugate directions.
        b_norm = self.xy_manifold_norm(xy, b)
        r_norm = self.xy_manifold_norm(xy, r)
        rel_res = r_norm / b_norm
        Ar = A(r)
        Ap = A(p)
        rAr = self.xy_manifold_inner_product(xy, r, Ar)
        t = 0  # at t-th iteration
        info =  np.zeros((maxiter, 2))
        while True:
            info[t] = [t, rel_res]
            t += 1
            a = rAr / self.xy_manifold_inner_product(xy, Ap, Ap)  # step length
            v = v + a * p  # update x # v + a*p
            r = r - a * Ap  # residual # r - a*Ap
            r_norm = self.xy_manifold_norm(xy, r)
            rel_res = r_norm / b_norm
            if rel_res < tol or t == maxiter:
                break
            Ar = A(r)
            old_rAr = rAr
            rAr = self.xy_manifold_inner_product(xy, r, Ar)
            beta = rAr / old_rAr  # improvement this step
            p = r + beta*p  # search direction # r + beta*p
            Ap = Ar + beta*Ap  # Ar + beta*Ap
        vfinal = v
        return vfinal, t, rel_res, info
    
    def preprocess(self, problem):
        # Set the parameters
        option = self.option
        heuristic_z_s =  option["heuristic_z_s"]
        desired_tau_1 = option["desired_tau_1"]
        important = option["important"]

        ineqconstraints = problem.ineqconstraints_all
        eqconstraints = problem.eqconstraints_all
        num_ineqconstraints = problem.num_ineqconstraints
        num_eqconstraints = problem.num_eqconstraints
        manifold = problem.manifold
        # egradcostfun = problem.euclidean_gradient
        egradineqconstraints = problem.ineqconstraints_euclidean_gradient_all
        egradeqconstraints = problem.eqconstraints_euclidean_gradient_all
        gradineqconstraints = problem.ineqconstraints_riemannian_gradient_all
        gradeqconstraints = problem.eqconstraints_riemannian_gradient_all
        # ehesscostfun = problem.euclidean_hessian
        ehessineqconstraints = problem.ineqconstraints_euclidean_hessian_all
        ehesseqconstraints = problem.eqconstraints_euclidean_hessian_all
        hessineqconstraints = problem.ineqconstraints_riemannian_hessian_all
        hesseqconstraints = problem.eqconstraints_riemannian_hessian_all

        # Set initial points
        xCur = copy.deepcopy(problem.initialpoint)
        yCur = copy.deepcopy(problem.initialeqLagmult)
        if heuristic_z_s:
            zCur = np.ones(num_ineqconstraints)
            zCur[0] =np.real(np.sqrt((num_ineqconstraints - 1)/(num_ineqconstraints/desired_tau_1 - 1)))
            sCur = important * zCur
        else:
            zCur = copy.deepcopy(problem.initialineqLagmult)
            sCur = copy.deepcopy(problem.initialineqLagmult)

        do_euclidean_lincomb = option['do_euclidean_lincomb']
        # Set the operators
        """
        The original RIPM implementation by Lai-Yoshise is restricted to the embedded manifold.
        There, we can efficiently compute the Riemannian gradient and Hessian of composite functions
        by summing up the euclidean ones of the objective function and constraints and projecting them onto the tangent space.
        We follow the same approach where do_euclidean_lincomb is True.
        On the other hands, we also compute the Riemannian gradient and Hessian of the composite functions
        by calculating the Euclidean gradient and Hessian of each component, projecting them onto the tangent space, and then summing them up.
        Although this approach is less efficient than the former, it is more general and can be applied to any manifold.
        Due to the compatibility with the pymanopt, we use the latter approach when considering product manifolds, for example.
        """
        if do_euclidean_lincomb:
            Gx = lambda x, z: manifold.euclidean_to_riemannian_gradient(x, barGx(x, z, egradineqconstraints))
            Gxaj = lambda x, dx: barGxaj(x, manifold.embedding(x, dx), egradineqconstraints)
            Hx = lambda x, y: manifold.euclidean_to_riemannian_gradient(x, barHx(x, y, egradeqconstraints))
            Hxaj = lambda x, dx: barHxaj(x, manifold.embedding(x, dx), egradeqconstraints)
            costegradfun = problem.euclidean_gradient
            costehessfun = problem.euclidean_hessian
            egradLagrangian = build_egradLagrangian(costegradfun, egradineqconstraints, egradeqconstraints)
            ehessLagrangian = build_ehessLagrangian(costehessfun, ehessineqconstraints, ehesseqconstraints)
            gradLagrangian = lambda x, y, z: manifold.euclidean_to_riemannian_gradient(x, egradLagrangian(x, y, z))
            hessLagrangian = lambda x, y, z, dx: manifold.euclidean_to_riemannian_hessian(x, egradLagrangian(x, y, z), ehessLagrangian(x, y, z, dx), dx)
        else:
            Gx = lambda x, z: barFx(x, z, gradineqconstraints, manifold)
            Gxaj = lambda x, dx: barFxaj(x, manifold.embedding(x, dx), gradineqconstraints, manifold)
            Hx = lambda x, y: barFx(x, y, gradeqconstraints, manifold)
            Hxaj = lambda x, dx: barFxaj(x, manifold.embedding(x, dx), gradeqconstraints, manifold)
            costgradfun = problem.riemannian_gradient
            costhessfun = problem.riemannian_hessian
            gradLagfun = build_gradLagrangian(costgradfun, gradineqconstraints, gradeqconstraints, manifold)
            hessLagfun = build_hessLagrangian(costhessfun, hessineqconstraints, hesseqconstraints, manifold)
            gradLagrangian = lambda x, y, z: gradLagfun(x, y, z)
            hessLagrangian = lambda x, y, z, dx: hessLagfun(x, y, z, dx)
        self.Gx = Gx
        self.Gxaj = Gxaj
        self.Hx = Hx
        self.Hxaj = Hxaj
        self.gradLagrangian = gradLagrangian
        self.hessLagrangian = hessLagrangian

        self.KKTVectorField = build_KKTVectorField(gradLagrangian, ineqconstraints, eqconstraints)

        # Set the product manifolds
        self.x_manifold = manifold
        self.y_manifold = pymanopt.manifolds.Euclidean(num_eqconstraints)
        self.z_manifold = pymanopt.manifolds.Euclidean(num_ineqconstraints)
        self.s_manifold = pymanopt.manifolds.Euclidean(num_ineqconstraints)

        # Set initial points on xyzs_manifold
        xyzsCur = [xCur, yCur, zCur, sCur]
        Ehat = pymanopt.manifolds.product._ProductTangentVector(self.xyzs_manifold_zero_vector(xyzsCur))
        ehat = np.ones(num_ineqconstraints)
        Ehat[3] = ehat
        KKTvec = self.KKTVectorField(xyzsCur)
        PhiCur = self.xyzs_manifold_norm(xyzsCur, KKTvec)**2

        # Construct parameters sigma to controls the final convergence rate
        sigma = min(0.5, np.real(np.sqrt(PhiCur)**0.5))
        rho = (zCur @ sCur) / num_ineqconstraints
        self.gamma = copy.deepcopy(option["gamma"])
        xyCur = [xCur, yCur]
        self.v0 = pymanopt.manifolds.product._ProductTangentVector(self.xy_manifold_zero_vector(xyCur))

        # Set constants for centrality conditions
        self.tau_1 = min(zCur * sCur) * num_ineqconstraints/ (zCur @ sCur)  # min(zCur * sCur) / ((zCur @ sCur) / ineqnum)
        self.tau_2 = (zCur @ sCur) / np.real(np.sqrt(PhiCur))

        return xCur, yCur, zCur, sCur, KKTvec, PhiCur, Ehat, sigma, rho

    def step(self, problem, xCur, yCur, zCur, sCur, KKTvec, PhiCur, Ehat, sigma, rho):
        has_eqconstraints = problem.has_eqconstraints
        num_ineqconstraints = problem.num_ineqconstraints
        num_eqconstraints = problem.num_eqconstraints

        ehat = Ehat[3]
        Gx = self.Gx
        Gxaj = self.Gxaj
        Hx = self.Hx
        Hxaj = self.Hxaj
        hessLagrangian = self.hessLagrangian
        KKTVectorField = self.KKTVectorField
        manifold = self.x_manifold

        v0 = self.v0
        gamma = self.gamma
        tau_1 = self.tau_1
        tau_2 = self.tau_2
        xyCur = [xCur, yCur]
        xyzsCur = [xCur, yCur, zCur, sCur]

        option = self.option
        KrylovIterMethod = option["KrylovIterMethod"]
        KrylovTolrelresid = option["KrylovTolrelresid"]
        KrylovMaxIteration = option["KrylovMaxIteration"]
        checkNTequation = option["checkNTequation"]
        basisfun = option["basisfun"]
        ls_beta = option["linesearch_beta"]
        ls_theta = option["linesearch_theta"]
        ls_max_steps = option["linesearch_max_steps"]
        ls_execute_fun2 = option["linesearch_execute_fun2"]

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

        def build_operatorT(OperatorAw, OperatorHx, OperatorHxaj):
            if has_eqconstraints:
                def OperatorT(dxdy):
                    vec = pymanopt.manifolds.product._ProductTangentVector([OperatorAw(dxdy[0]) + OperatorHx(dxdy[1]), OperatorHxaj(dxdy[0])])
                    return vec
            else:
                def OperatorT(dxdy):
                    vec = pymanopt.manifolds.product._ProductTangentVector([OperatorAw(dxdy[0]), np.array([])])
                    return vec
            return OperatorT
        OperatorT = build_operatorT(OperatorAw, OperatorHx, OperatorHxaj)

        # Solve the condensed NT equation T(dxdy) = cq.
        solve_info = []
        if KrylovIterMethod:
            NTdirdxdy, t, rel_res, _ = self.TangentSpaceConjResMethod(OperatorT, cq, v0, xyCur, KrylovTolrelresid, KrylovMaxIteration)
            solve_info = [t, rel_res]
        else:
            xbasis = basisfun(manifold, xCur)
            ybasis = np.eye(num_eqconstraints)
            NTdirdxdy, _, _ = self.RepresentMatMethod(OperatorAw, OperatorHxaj, OperatorHx, cq, xyCur, xbasis, ybasis)

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
                if has_eqconstraints:
                    nablaFdx = nablaFdx + Hx(x, dy)
                nablaFdy = Hxaj(x, dx)
                nablaFdz = Gxaj(x, dx) + ds
                nablaFds = z * ds + s * dz
                nablaF = pymanopt.manifolds.product._ProductTangentVector([nablaFdx, nablaFdy, nablaFdz, nablaFds])
                return nablaF

            def xyzs_operator2matrix(x, y, z, s, F, xbasis, ybasis, zbasis, sbasis):
                n = len(xbasis) + len(ybasis) + len(zbasis) + len(sbasis)
                A = np.zeros((n, n))
                xyzsCur = [x, y, z, s]
                zeroxbasis = np.zeros_like(xbasis[0])
                if ybasis.shape[0] > 0:
                    zeroybasis = np.zeros_like(ybasis[0])
                else:
                    zeroybasis = np.array([])
                zerozbasis = np.zeros_like(zbasis[0])
                zerosbasis = np.zeros_like(sbasis[0])

                xyzsbasis = (
                    [pymanopt.manifolds.product._ProductTangentVector([xbasis[i], zeroybasis, zerozbasis, zerosbasis]) for i in range(len(xbasis))]
                    + [pymanopt.manifolds.product._ProductTangentVector([zeroxbasis, ybasis[j], zerozbasis, zerosbasis]) for j in range(len(ybasis))]
                    + [pymanopt.manifolds.product._ProductTangentVector([zeroxbasis, zeroybasis, zbasis[k], zerosbasis]) for k in range(len(zbasis))]
                    + [pymanopt.manifolds.product._ProductTangentVector([zeroxbasis, zeroybasis, zerozbasis, sbasis[l]]) for l in range(len(sbasis))]
                )
                for j in range(n):
                    Fj = F(x, y, z, s, xyzsbasis[j])
                    for i in range(n):  # Non-symmetric
                        A[i, j] = self.xyzs_manifold_inner_product(xyzsCur, xyzsbasis[i], Fj)
                return A, xyzsbasis, n
            xbasis = basisfun(manifold, xCur)
            ybasis = np.eye(num_eqconstraints)
            zbasis = np.eye(num_ineqconstraints)
            sbasis = np.eye(num_ineqconstraints)
            CovarDerivKKT_mat, CovarDerivKKT_basis, xyzsdim = xyzs_operator2matrix(xCur, yCur, zCur, sCur, CovarDerivKKT, xbasis, ybasis, zbasis, sbasis)
            eigvals = np.linalg.eigvals(CovarDerivKKT_mat)  # 一般の正方行列
            idxmin = np.argmin(np.abs(eigvals))
            CovDerivKKT_minabseigval = eigvals[idxmin]

            # # DEBUG for checking the correctness of CovarDerivKKT_mat
            # def lincomb_basis(coeffs):
            #     v = 0 * CovarDerivKKT_basis[0]
            #     for alpha, b in zip(coeffs, CovarDerivKKT_basis):
            #         v = v + alpha * b
            #     return v

            # for trial in range(10):
            #     c = np.random.randn(xyzsdim)
            #     dw = lincomb_basis(c)

            #     F1 = CovarDerivKKT(xCur, yCur, zCur, sCur, dw)

            #     c2 = CovarDerivKKT_mat @ c
            #     F2 = lincomb_basis(c2)

            #     diff = F1 - F2
            #     err = self.xyzs_manifold_inner_product(xyzsCur, diff, diff)**0.5
            #     print(f"trial {trial}, error = {err}")

            # Adjoint of covariant Derivative of KKT vector field
            # DEBUG ONLY
            def CovarDerivKKTaj(x, y, z, s, dw):
                dx, dy, dz, ds = dw
                nablaFajdx = hessLagrangian(x, y, z, dx) + Gx(x, dz)
                if has_eqconstraints:
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
            NTdir_error1 = self.xyzs_manifold_norm(xyzsCur, nablaF_NTdir - NTeq_rhs)
            verbosity = self.option["verbosity"]
            if verbosity >= 2:
                print("NTdir_error1", NTdir_error1)

            # Check Item#2: If NTdir is correct solution, then
            # <grad phi, NTdir> = 2(|F(w)|^{2}+sigma*rho*InnerProduct(z,s)) holds,
            # where grad phi = 2*nablaFaj(KKTvec).
            # NTdir_error2 should be zero.
            gradphi = 2 * nablaFaj(KKTvec)
            val_innerproduct = self.xyzs_manifold_inner_product(xyzsCur, gradphi, NTdir)
            NTdir_error2 = abs(val_innerproduct - 2*(sigma*rho*(zCur @ sCur)-PhiCur))
            if verbosity >= 2:
                print("NTdir_error2", NTdir_error2)

            # Record Item: record norm of NTdirl; angle between - grad phi and NTdir.
            Norm_gradphi = self.xyzs_manifold_norm(xyzsCur, gradphi)
            NTdir_norm = self.xyzs_manifold_norm(xyzsCur, NTdir)
            NTdir_angle = - val_innerproduct / (Norm_gradphi * NTdir_norm)
            NTdir_info = [NTdir_error1, NTdir_error2, NTdir_norm, NTdir_angle, CovDerivKKT_minabseigval]

        # Backtracking line search and update.
        # Central functions
        fun_1 = lambda z, s: min(z * s) - gamma * tau_1 * (z @ s / num_ineqconstraints)
        fun_2 = lambda z, s, Phi: z @ s - gamma * tau_2 * np.sqrt(Phi)

        # Note that <grad phi, NTdir> = ls_RightItem, if NTdir is a correct solution.
        ls_RightItem = 2 * (sigma * rho * (zCur @ sCur) - PhiCur)

        # Also compute the inner product of NTdir and gradient of f
        costgradfun = problem.riemannian_gradient
        gradfCur = costgradfun(xCur)
        gradfNTdir = problem.manifold.inner_product(xCur, gradfCur, NTdir[0])

        normNTdirx = self.x_manifold.norm(xCur, NTdir[0])
        normNTdirw = self.xyzs_manifold_norm(xyzsCur, NTdir)

        stepsize = 1
        ls_status = True
        r = 0
        while True:
            xyzsNew = self.xyzs_manifold_retraction(xyzsCur, stepsize * NTdir)
            KKTvec = KKTVectorField(xyzsNew)
            PhiNew = self.xyzs_manifold_norm(xyzsNew, KKTvec)**2
            zNew = xyzsNew[2]
            sNew = xyzsNew[3]
            if (PhiNew - PhiCur <= ls_beta * stepsize * ls_RightItem
                    and fun_1(zNew, sNew) >= 0
                    and (not ls_execute_fun2 or fun_2(zNew, sNew, PhiNew) >= 0)):
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
        Ehat = pymanopt.manifolds.product._ProductTangentVector(self.xyzs_manifold_zero_vector(xyzsCur))
        ehat = np.ones(num_ineqconstraints)
        Ehat[3] = ehat
        KKTvec = KKTVectorField(xyzsCur)
        PhiCur = self.xyzs_manifold_norm(xyzsCur, KKTvec)**2
        xyCur = [xCur, yCur]
        self.v0 = pymanopt.manifolds.product._ProductTangentVector(self.xy_manifold_zero_vector(xyCur))

        # Update parameters
        sigma = min(0.5, np.sqrt(PhiCur)**0.5)
        rho = (zCur @ sCur) / num_ineqconstraints
        gamma = 0.5 * (gamma + 0.5)
        self.gamma = gamma

        output = {}
        output["xCur"] = xCur
        output["yCur"] = yCur
        output["zCur"] = zCur
        output["sCur"] = sCur
        output["KKTvec"] = KKTvec
        output["PhiCur"] = PhiCur
        output["sigma"] = sigma
        output["rho"] = rho
        output["normNTdirx"] = normNTdirx
        output["normNTdirw"] = normNTdirw
        output["stepsize"] = stepsize
        output["ls_status"] = ls_status
        output["ls_RightItem"] = ls_RightItem
        output["gradfNTdir"] = gradfNTdir
        output["linesearch_counter"] = r
        output["solve_info"] = solve_info
        output["NTdir_info"] = NTdir_info
        return output

    def run(self, problem):
        assert isinstance(problem, NonlinearProblem), "Input problem must be an instance of NonlinearProblem"

        xCur, yCur, zCur, sCur, KKTvec, PhiCur, Ehat, sigma, rho = self.preprocess(problem)
        xPrev = copy.deepcopy(xCur)
        normNTdirx = None
        normNTdirw = None
        stepsize = None
        ls_status = None
        ls_RightItem = None
        gradfNTdir = None
        r = None
        KrylovIterMethod = self.option["KrylovIterMethod"]
        solve_info = None
        checkNTequation = self.option["checkNTequation"]
        NTdir_info = None

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
            eval_log = evaluation(problem, xPrev, xCur, zCur, yCur, manviofun, callbackfun)
            solver_log = self.solver_status(zCur, yCur, PhiCur, sigma, rho, normNTdirx=normNTdirx, normNTdirw=normNTdirw,
                      stepsize=stepsize, linesearch_status=ls_status, linesearch_counter=r, linesearch_RightItem=ls_RightItem, gradfNTdir=gradfNTdir,
                      KrylovIterMethod=KrylovIterMethod, solve_info=solve_info, checkNTequation=checkNTequation, NTdir_info=NTdir_info)
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

            # # Count an iteration
            iteration += 1

            if do_exit_on_error:
                try:
                    output = self.step(problem, xCur, yCur, zCur, sCur, KKTvec, PhiCur, Ehat, sigma, rho)
                except Exception as e:
                    print(f"Error: {e}")
                    break
            else:
                output = self.step(problem, xCur, yCur, zCur, sCur, KKTvec, PhiCur, Ehat, sigma, rho)

            xCur = output["xCur"]
            yCur = output["yCur"]
            zCur = output["zCur"]
            sCur = output["sCur"]
            KKTvec = output["KKTvec"]
            PhiCur = output["PhiCur"]
            sigma = output["sigma"]
            rho = output["rho"]
            normNTdirx = output["normNTdirx"]
            normNTdirw = output["normNTdirw"]
            stepsize = output["stepsize"]
            ls_status = output["ls_status"]
            ls_RightItem = output["ls_RightItem"]
            gradfNTdir = output["gradfNTdir"]
            r = output["linesearch_counter"]
            solve_info = output["solve_info"]
            NTdir_info = output["NTdir_info"]

        # After exiting while loop, we return the final output
        output = self.postprocess(xCur, zCur, yCur)
        if self.option["wandb_logging"]:
            wandb.finish()
        return output

    def postprocess(self, xfinal, ineqLagfinal, eqLagfinal):
        output = Output(name=self.name,
                        x=xfinal,
                        ineqLagmult=ineqLagfinal,
                        eqLagmult=eqLagfinal,
                        option=copy.deepcopy(self.option),
                        log=self.log)
        return output

    # Examine the solver status
    def solver_status(self,
                      ineqLagmult,
                      eqLagmult,
                      Phi,
                      sigma,
                      rho,
                      normNTdirx=None,
                      normNTdirw=None,
                      stepsize=None,
                      linesearch_status=None,
                      linesearch_counter=None,
                      linesearch_RightItem=None,
                      gradfNTdir=None,
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
        for Lagmult in ineqLagmult:
            maxabsLagmult = max(maxabsLagmult, abs(Lagmult))
        for Lagmult in eqLagmult:
            maxabsLagmult = max(maxabsLagmult, abs(Lagmult))
        solver_status["maxabsLagmult"] = maxabsLagmult

        solver_status["normNTdirx"] = normNTdirx
        solver_status["normNTdirw"] = normNTdirw

        solver_status["stepsize"] = stepsize
        solver_status["linesearch_status"] = linesearch_status
        solver_status["linesearch_counter"] = linesearch_counter
        solver_status["linesearch_RightItem"] = linesearch_RightItem
        solver_status["gradfNTdir"] = gradfNTdir

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
                NTdir_error1, NTdir_error2, NTdir_norm, NTdir_angle, CovDerivKKT_minabseigval = NTdir_info
                solver_status["NTdir_error1"] = NTdir_error1
                solver_status["NTdir_error2"] = NTdir_error2
                solver_status["NTdir_norm"] = NTdir_norm
                solver_status["NTdir_angle"] = NTdir_angle
                solver_status["CovDerivKKT_minabseigval"] = CovDerivKKT_minabseigval
            else:
                solver_status["NTdir_error1"] = None
                solver_status["NTdir_error2"] = None
                solver_status["NTdir_norm"] = None
                solver_status["NTdir_angle"] = None
                solver_status["CovDerivKKT_minabseigval"] = None

        return solver_status

@hydra.main(version_base=None, config_path="../NonnegPCA", config_name="config_simulation")
def main(cfg):  # Mainly for debugging

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
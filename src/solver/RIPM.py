import hydra, copy, time, pymanopt, wandb
import numpy as np
from dataclasses import dataclass, field
from utils import evaluation, tangentorthobasis, operator2matrix, tangent2vec
from scipy import linalg

import sys
sys.path.append('./src/base')
from base_solver import Solver, BaseOutput

@dataclass
class Output(BaseOutput):
    ineqLagmult: field(default_factory=list)
    eqLagmult: field(default_factory=list)

def barGx(x, z, ineqconstraints):
    val = 0  # 本当はzero vector
    for idx in range(ineqconstraints.num_constraint):
        egrad = ineqconstraints.constraint[idx].get_gradient_operator()
        # print("z, egrad", z, egrad(x))
        val += z[idx] * egrad(x)
    # print("barGx", val)
    return val

def ehess_barGx(x, z, dx, ineqconstraints):
    val = 0
    for idx in range(ineqconstraints.num_constraint):
        ehess = ineqconstraints.constraint[idx].get_hessian_operator()
        val += z[idx] * ehess(x, dx)
    # print("ehess_barGx", val)
    return val

def barGxaj(x, dx, ineqconstraints):
    val = np.zeros(ineqconstraints.num_constraint)
    for idx in range(ineqconstraints.num_constraint):
        egrad = ineqconstraints.constraint[idx].get_gradient_operator()
        # print(f"barGxaj {idx}:", "egrad", egrad(x).reshape(-1), "dx", dx.reshape(-1))
        val[idx] = egrad(x).reshape(-1) @ dx.reshape(-1)
    # print("barGxaj", val)
    return val

def barHx(x, y, eqconstraints):
    val = 0  # 本当はzero vector
    for idx in range(eqconstraints.num_constraint):
        egrad = eqconstraints.constraint[idx].get_gradient_operator()
        # print("y, egrad", y, egrad(x))
        val += y[idx] * egrad(x)
    # print("barHx:", val)
    return val

def ehess_barHx(x, y, dx, eqconstraints):
    val = 0
    for idx in range(eqconstraints.num_constraint):
        ehess = eqconstraints.constraint[idx].get_hessian_operator()
        val += y[idx] * ehess(x, dx)
    # print("ehess_barHx:", val)
    return val

def barHxaj(x, dx, eqconstraints):
    val = np.zeros(eqconstraints.num_constraint)
    for idx in range(eqconstraints.num_constraint):
        egrad = eqconstraints.constraint[idx].get_gradient_operator()
        val[idx] = egrad(x).reshape(-1) @ dx.reshape(-1)
        # print(f"barHxaj {idx}:", val[idx], egrad(x), dx)
    # print("barHxaj:", val)
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
        # dxdydzds = [dx, dy, dz, ds]
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

def TangentSpaceConjResMethod(A, b, v0, M, x, tol, maxiter):
    # Conjugate residual method for solving linear opeartor equation: A(v)=b,
    # where A is some self-adjoint operator to and from some linear space E,
    # b is an element in E. Assume the existence of solution v.
    
    # Yousef Saad - Iterative methods for sparse linear systems,
    # 2nd edition-SIAM (2003) P203. ALGORITHM 6.20
    
    v = v0  # initialization
    r = b - A(v)  # r are residuals.
    p = r  # p are conjugate directions.
    b_norm = M.norm(x, b)
    r_norm = M.norm(x, r)
    rel_res = r_norm / b_norm
    Ar = A(r)
    Ap = A(p)
    rAr = M.inner_product(x, r, Ar)
    t = 0  # at t-th iteration
    info =  np.zeros((maxiter, 2))
    while True:
        # print("iter", t)
        info[t] = [t, rel_res]
        t += 1
        a = rAr / M.inner_product(x, Ap, Ap)  # step length
        v = v + a * p  # update x # v + a*p
        r = r - a * Ap  # residual # r - a*Ap
        r_norm = M.norm(x, r)
        rel_res = r_norm / b_norm
        if rel_res < tol or t == maxiter:
            break
        Ar = A(r)
        old_rAr = copy.deepcopy(rAr)
        rAr = M.inner_product(x, r, Ar)
        # print("iter", t, "rAr", rAr, "old_rAr", old_rAr)
        beta = rAr / old_rAr  # improvement this step
        p = r + beta*p  # search direction # r + beta*p
        Ap = Ar + beta*Ap  # Ar + beta*Ap
    vfinal = v
    return vfinal, t, rel_res, info

def RepresentMatMethod(Aw, Hxaj, cq, xy_manifold, xy, xbasis, ybasis):
    x_manifold = xy_manifold.manifolds[0]
    y_manifold = xy_manifold.manifolds[1]
    xdim = x_manifold.dim
    ydim = y_manifold.dim
    x = xy[0]
    y = xy[1]
    # print("basis", basis)
    # xbasis = basis[:,0][:xdim]
    # print("xbasis", xbasis)
    # ybasis = basis[xdim:][1]
    basis = [pymanopt.manifolds.product._ProductTangentVector([xb, y_manifold.zero_vector(y)]) for xb in xbasis]\
        + [pymanopt.manifolds.product._ProductTangentVector([x_manifold.zero_vector(x), yb]) for yb in ybasis]
        
    # print("basis", basis)

    
    def SelfAdj_operator2matrix(M, x, F, Bx):
        n = len(Bx)
        A_mat = np.zeros((n, n))
        for j in range(n):
            FBxj = F(Bx[j])
            for i in range(j+1):
                A_mat[i, j] = M.inner_product(x, FBxj, (Bx[i]))
        # print("A_mat, before:", A_mat)
        A_mat = A_mat + np.triu(A_mat, 1).T
        # print("A_mat, after :", A_mat - A_mat.T)
        return A_mat
    
    # Under the basis, the following codes return a saddle-point
    # linear system whose matrix has the form
    #
    #           [ HessLag_mat + THETA_mat | Hx_mat]
    #   T_mat = -----------------------------------
    #           [Hx_mat.T                 | 0     ]
    # where
    # - n:= dim of manifod, l:= dim of equality constraints,
    # - HessLag_mat and THETA_mat are symmetric n x n,
    # - Hx_mat is l x n, with l <= n,
    # - 0 is l x l zero matrix.
    
    # the next code is equal to run:
    # HessLag_mat = SelfAdj_operator2matrix(M, x, x, HessLag, Bx, Bx);
    # THETA_mat = SelfAdj_operator2matrix(M, x, x, THETA, Bx, Bx);
    # Aw_mat = HessLag_mat + THETA_mat;
    Aw_mat = SelfAdj_operator2matrix(x_manifold, x, Aw, xbasis)

    # the next code is equal to run:
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
    # print("Basis!", basis)
    # print("end!")
    cq_vec = tangent2vec(xy_manifold, xy, basis, cq)

    # print("T_mat", T_mat)
    # print("cq_vec", cq_vec)
    # direct method
    # sol_vec = np.linalg.solve(T_mat, cq_vec)
    sol_vec = linalg.solve(T_mat, cq_vec, assume_a='sym')
    # print("solvec   ", sol_vec)
    """scipy.linalg.solveを使っているが, np.linalg.solveでも良い
    どちらが良いかは検討"""
    # print("diff", T_mat @ sol_vec - cq_vec)
    NTdir = xy_manifold.zero_vector(xy)
    print(type(NTdir))
    for i in range(len(basis)):
        # print("sol_vec[i]", sol_vec[i], "basis[i]", basis[i])
        # print(type(basis[i]))
        NTdir = NTdir + sol_vec[i] * basis[i]
    # NTdir= lincomb(M_Euc_eq, xy, Bxy, sol_vec);
    RepresentMat = T_mat
    RepresentMatOrder = xy_manifold.dim
    # print("NTdir now", NTdir)
    return NTdir, RepresentMat, RepresentMatOrder
    


class RIPM(Solver):
    def __init__(self, option):
        # Default setting for augmented Lagrangian method
        default_option = {
            # Stopping criteria
            'maxtime': 100,
            'maxiter': 100,
            'tolresid': 1e-6,

            # Iteration
            'KrylovIterMethod': False, # Lai-Yoshise 実装ではFalseがdefault"""
            'KrylovTolrelresid': 1e-9,
            'KrylovMaxIteration': 100,  # 1000 is default in Lai-Yoshise's implementation
            'checkNTequation': True,  # Lai-Entryの実装ではFalseがdefault"""
            'RepMat_basisfun': lambda manifold, x: tangentorthobasis(manifold, x, manifold.dim),

            # Line search setting
            'gamma': 0.9,
            'linesearch_execute_fun2': True,  # Lai-Entryの実装ではFalseがdefault"""
            'linesearch_beta': 1e-4,
            'linesearch_theta': 0.5,
            'linesearch_max_steps': 50,

            # Other parameters
            'heuristic_z_s': False,
            'desired_tau_1': 0.5,
            'important': 1,

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
        xyzs_manifold = pymanopt.manifolds.Product([manifold, pymanopt.manifolds.Euclidean(eqconstraints.num_constraint),
                                                    pymanopt.manifolds.Euclidean(ineqconstraints.num_constraint),
                                                    pymanopt.manifolds.Euclidean(ineqconstraints.num_constraint)])
        xy_manifold = pymanopt.manifolds.Product([manifold, pymanopt.manifolds.Euclidean(eqconstraints.num_constraint)])
        
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
            # print("heuristic_z_s", zCur)
            zCur[0] =np.real(np.sqrt(((ineqnum - 1)/(ineqnum/desired_tau_1 - 1))));
            sCur = important * zCur
        else:
            zCur = problem.initialineqLagmult
            sCur = problem.initialineqLagmult
        # print("yCur", yCur, "zCur", zCur)
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
        tau_2 = (zCur @ sCur) / np.sqrt(PhiCur)

        # Construct parameters sigma to controls the final convergence rate
        sigma = min(0.5, np.sqrt(PhiCur))
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
            cq = [c, q]
            # print("cq", cq , type(cq))
            cq = pymanopt.manifolds.product._ProductTangentVector(cq)
            # print("cq after", cq, type(cq))
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
                # print(xy_manifold.dim)
                # print(tangentorthobasis(xy_manifold, xyCur, xy_manifold.dim))
                # if xbasis is None:
                #     xbasis = tangentorthobasis(xy_manifold, x, xdim)
                #     print("xbasis", xbasis)
                # if ybasis is None:
                #     ybasis = np.eye(ydim)
                #     print(ybasis)
                xbasis = RepMat_basisfun(manifold, xCur)
                ybasis = np.eye(eqconstraints.num_constraint)
                # basis = [[xb, np.zeros(eqconstraints.num_constraint)] for xb in xbasis] + [[np.zeros(manifold.dim), yb] for yb in ybasis]
                # print("Basis", basis)
                NTdirdxdy, _, _ = RepresentMatMethod(OperatorAw, OperatorHxaj, cq, xy_manifold, xyCur, xbasis, ybasis)
                # print(NTdir)
                """ここなんとかする"""
            
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
                        nablaFdx += Hx(x, dy)
                    nablaFdy = Hxaj(x, dx)
                    nablaFdz = Gxaj(x, dx) + ds
                    nablaFds = z * ds + s * dz
                    nablaF = pymanopt.manifolds.product._ProductTangentVector([nablaFdx, nablaFdy, nablaFdz, nablaFds])
                    return nablaF

                # Adjoint of ovariant Derivative of KKT vector field
                # DEBUG ONLY
                def CovarDerivKKTaj(x, y, z, s, dw):
                    dx, dy, dz, ds = dw
                    nablaFajdx = hessLagrangian(x, y, z, dx) + Gx(x, dz)
                    if eqconstraints.has_constraint:
                        nablaFajdx += Hx(x, dy)
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
                print("NTdir_error1", NTdir_error1)
                
                # Check Item#2: If NTdir is correct solution, then
                # <grad phi, NTdir> = 2(|F(w)|^{2}+sigma*rho*InnerProduct(z,s)) holds,
                # where grad phi = 2*nablaFaj(KKTvec).
                # NTdir_error2 should be zero.
                gradphi = 2 * nablaFaj(KKTvec)
                val_innerproduct = xyzs_manifold.inner_product(xyzsCur, gradphi, NTdir)
                NTdir_error2 = abs(val_innerproduct - 2*(sigma*rho*(zCur @ sCur)-PhiCur))
                print("NTdir_error2", NTdir_error2)
                
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
            ls_max_steps_flag = True
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
                    ls_max_steps_flag = False
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
                      linesearch_status=ls_max_steps_flag,
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

@hydra.main(version_base=None, config_path="../Model_Ob", config_name="config_simulation")
def main(cfg):  # Experiment of nonnegative PCA. Mainly for debugging

    # Import a problem set from NonnegPCA
    sys.path.append('./src/Model_Ob')
    import coordinator

    # Call a problem coordinator
    nonnegPCA_coordinator = coordinator.Coordinator(cfg)
    problem = nonnegPCA_coordinator.run()

    # Solver option setting
    solver_option = cfg.solver_option
    option = copy.deepcopy(dict(solver_option["common"]))
    if hasattr(solver_option, "RIPM"):
        specific = dict(getattr(solver_option, "RIPM"))
        option.update(specific)

    # Run the experiment
    ripmsolver = RIPM(option)
    output = ripmsolver.run(problem)
    print(output)

if __name__=='__main__':
    main()
import pymanopt.manifolds
import pymanopt.optimizers
import sys
import hydra
import numpy as np
import copy

# def compute_KKTVectorField(xyzs, ineqconstraints, eqconstraints, gradLagrangian):
#     x, y, z, s = xyzs
#     dx = gradLagrangian(x, y, z)
#     dy = np.empty(eqconstraints.num_constraint, dtype=object)
#     for idx in range(eqconstraints.num_constraint):
#         dy[idx] = (eqconstraints.constraint[idx])(x)
#     dz = np.empty(ineqconstraints.num_constraint, dtype=object)
#     for idx in range(ineqconstraints.num_constraint):
#         dz[idx] = (ineqconstraints.constraint[idx])(x) + s[idx]
#     ds = z * s
#     # dxdydzds = [dx, dy, dz, ds]
#     dxdydzds = pymanopt.manifolds.product._ProductTangentVector([dx, dy, dz, ds])
#     return dxdydzds

def barGx(x, z, ineqconstraints):
    val = 0  # 本当はzero vector
    for idx in range(ineqconstraints.num_constraint):
        egrad = ineqconstraints.constraint[idx].get_gradient_operator()
        val += z[idx] * egrad(x)
    print("barGx", val)
    return val

def ehess_barGx(x, z, dx, ineqconstraints):
    val = 0
    for idx in range(ineqconstraints.num_constraint):
        ehess = ineqconstraints.constraint[idx].get_hessian_operator()
        val += z[idx] * ehess(x, dx)
    print("ehess_barGx", val)
    return val

def barGxaj(x, dx, ineqconstraints):
    val = np.zeros(ineqconstraints.num_constraint)
    for idx in range(ineqconstraints.num_constraint):
        egrad = ineqconstraints.constraint[idx].get_gradient_operator()
        val[idx] = egrad(x) @ dx
        # print(f"barGxaj {idx}", val[idx], egrad(x), dx)
    # print("barGxaj", val)
    return val

def barHx(x, y, eqconstraints):
    val = 0  # 本当はzero vector
    for idx in range(eqconstraints.num_constraint):
        egrad = eqconstraints.constraint[idx].get_gradient_operator()
        val += y[idx] * egrad(x)
    print("barHx:", val)
    return val

def ehess_barHx(x, y, dx, eqconstraints):
    val = 0
    for idx in range(eqconstraints.num_constraint):
        ehess = eqconstraints.constraint[idx].get_hessian_operator()
        val += y[idx] * ehess(x, dx)
    print("ehess_barHx:", val)
    return val

def barHxaj(x, dx, eqconstraints):
    val = np.zeros(eqconstraints.num_constraint)
    for idx in range(eqconstraints.num_constraint):
        egrad = eqconstraints.constraint[idx].get_gradient_operator()
        val[idx] = egrad(x) @ dx
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
        print("iter", t)
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
        beta = rAr / old_rAr  # improvement this step
        p = r + beta*p  # search direction # r + beta*p
        Ap = Ar + beta*Ap  # Ar + beta*Ap
    vfinal = v
    return vfinal, t, rel_res, info

@hydra.main(version_base=None, config_path="../NonnegPCA", config_name="config_simulation")
def main(cfg):  # Experiment of nonnegative PCA. Mainly for debugging

    # Import a problem set from NonnegPCA
    sys.path.append('./src/NonnegPCA')
    import coordinator

    # Call a problem coordinator
    nonnegPCA_coordinator = coordinator.Coordinator(cfg)
    problem = nonnegPCA_coordinator.run()

    costfun = problem.costfun
    ineqconstraints = problem.ineqconstraints
    eqconstraints = problem.eqconstraints
    manifold = problem.searchspace

    # Option setting
    option = {}
    option["heuristic_z_s"] = True
    option["desired_tau_1"] = 0.5
    option["important"] = 1
    option["gamma"] = 0.9
    option["KrylovIterMethod"] = True
    """Lai-Yoshise 実装でのdefaultはFalse"""
    option["KrylovTolrelres"] = 1e-9
    option["KrylovMaxiter"] = 1000
    option["checkNTequation"] = True
    option["ls_beta"] = 1e-4
    option["ls_execute_fun2"] = True
    """Lai-Yoshise 実装でのdefaultはFalse"""
    option["ls_theta"] = 0.5
    option["ls_max_steps"] = 50
    
    Gx = lambda x, z: manifold.euclidean_to_riemannian_gradient(x, barGx(x, z, ineqconstraints))
    Gxaj = lambda x, dx: barGxaj(x, manifold.embedding(x, dx), ineqconstraints)
    Hx = lambda x, y: manifold.euclidean_to_riemannian_gradient(x, barHx(x, y, eqconstraints))
    Hxaj = lambda x, dx: barHxaj(x, manifold.embedding(x, dx), eqconstraints)
    
    heuristic_z_s =  option["heuristic_z_s"]
    desired_tau_1 = option["desired_tau_1"]
    important = option["important"]
    gamma = option["gamma"]
    KrylovIterMethod = option["KrylovIterMethod"]
    KrylovTolrelres = option["KrylovTolrelres"]
    KrylovMaxiter = option["KrylovMaxiter"]
    checkNTequation = option["checkNTequation"]
    ls_beta = option["ls_beta"]
    ls_execute_fun2 = option["ls_execute_fun2"]
    ls_theta = option["ls_theta"]
    ls_max_steps = option["ls_max_steps"]
    
    # Set initial points
    xCur = problem.initialpoint
    yCur = problem.initialeqLagmult
    
    # print("type", type(xCur), type(yCur))
    
    ineqnum = ineqconstraints.num_constraint
    if heuristic_z_s:
        zCur = np.ones(ineqnum)
        zCur[0] =np.real(np.sqrt(((ineqnum - 1)/(ineqnum/desired_tau_1 - 1))));
        sCur = important * zCur
    else:
        zCur = problem.initialineqLagmult
        sCur = problem.initialineqLagmult
    
    xyzs_manifold = pymanopt.manifolds.Product([manifold, pymanopt.manifolds.Euclidean(eqconstraints.num_constraint),
                                                pymanopt.manifolds.Euclidean(ineqconstraints.num_constraint),
                                                pymanopt.manifolds.Euclidean(ineqconstraints.num_constraint)])
    xy_manifold = pymanopt.manifolds.Product([manifold, pymanopt.manifolds.Euclidean(eqconstraints.num_constraint)])
    
    # print("problem", problem)
    costegradfun = costfun.get_gradient_operator()
    costehessfun = costfun.get_hessian_operator()


    egradLagrangian = build_egradLagrangian(costegradfun, ineqconstraints, eqconstraints)
    ehessLagrangian = build_ehessLagrangian(costehessfun, ineqconstraints, eqconstraints)
    # egradLagrangian = lambda x, y, z: costegradfun(x) + barGx(x, z, ineqconstraints) + barHx(x, y, eqconstraints)
    # ehessLagrangian = lambda x, y, z, dx: costehessfun(x, dx) + ehess_barGx(x, z, dx, ineqconstraints) # + ehess_barHx(x, y, dx, eqconstraints)
    gradLagrangian = lambda x, y, z: manifold.euclidean_to_riemannian_gradient(x, egradLagrangian(x, y, z))
    hessLagrangian = lambda x, y, z, dx: manifold.euclidean_to_riemannian_hessian(x, egradLagrangian(x, y, z), ehessLagrangian(x, y, z, dx), dx)
    
    KKTVectorField = build_KKTVectorField(gradLagrangian, ineqconstraints, eqconstraints)
    # fun = problem.costfun.get_gradient_operator()
    # print(fun(xCur))
    # print(fun)
    # input()

    xyzsCur = [xCur, yCur, zCur, sCur]
    # print(xyzsCur)
    # dim = manifold.dim
    Ehat = xyzs_manifold.zero_vector(xyzsCur)
    ehat = np.ones(ineqconstraints.num_constraint)
    Ehat[3] = ehat
    # print(dxyzs)
    
    xyCur = [xCur, yCur]
    v0 = xy_manifold.zero_vector(xyCur)
    # print(v0)
    # print("pt", manifold.random_point(), type(manifold.random_point()))
    # print("pt", xyzs_manifold.random_point())
    # print("vec", xyzs_manifold.random_tangent_vector(xCur))

    iter = 0
    KKTvec = KKTVectorField(xyzsCur)
    # KKTvec = compute_KKTVectorField(xyzsCur, ineqconstraints, eqconstraints, gradLagrangian)  # [dx, dy, dz, ds]
    # print(xyzs_manifold.random_point())
    # print("xyzsCur", xyzsCur)
    # print("KKTvec", KKTvec)
    PhiCur = xyzs_manifold.norm(xyzsCur, KKTvec)**2
    # print(PhiCur)
    """costCurとKKTresidualCurを計算する, timeticする.パスした"""
    
    # Set constants for centrality conditions
    tau_1 = min(zCur * sCur) * ineqnum/ (zCur @ sCur)  # min(zCur * sCur) / ((zCur @ sCur) / ineqnum)
    tau_2 = (zCur @ sCur) / np.sqrt(PhiCur)
    
    # Construct parameters sigma to controls the final convergence rate
    sigma = min(0.5, np.sqrt(PhiCur))
    rho = (zCur @ sCur) / ineqnum
    # gamma = option["gamma"]
    
    
    """Save stats in a struct array info, and preallocation.
    Stopping flag, finally it should be true
    """
    
    # while True:
    #     """Displayをする"""
    
    iter += 1
    
    # Get Newton direction NTdir by solving condensed NT equation.
    
    # Right-hand side (cq) of condensed NT equation is (c, q).
    c = - KKTvec[0] - Gx(xCur, (zCur * KKTvec[2] + sigma * rho * ehat - KKTvec[3]) / sCur)
    q = - KKTvec[1]
    cq = [c, q]
    print("cq", cq , type(cq))
    # cq = xy_manifold.to_tangent_space(xyCur, cq)
    cq = pymanopt.manifolds.product._ProductTangentVector(cq)
    print("cq after", cq, type(cq))
    # Define some operators.
    OperatorHessLag = lambda dx: hessLagrangian(xCur, yCur, zCur, dx)
    OperatorTHETA = lambda dx: Gx(xCur, Gxaj(xCur, dx) * (zCur / sCur))
    OperatorAw = lambda dx: OperatorHessLag(dx) + OperatorTHETA(dx)
    OperatorHx = lambda dy: Hx(xCur, dy)
    OperatorHxaj = lambda dx: Hxaj(xCur, dx)
    # OperatorT = lambda dxdy: [OperatorAw(dxdy[0]) + OperatorHx(dxdy[1]), OperatorHxaj(dxdy[0])]
    
    def build_operatorT(OperatorAw, OperatorHx, OperatorHxaj, eqconstraints):
        if eqconstraints.has_constraint:
            def OperatorT(dxdy):
                # vec = xy_manifold.to_tangent_space(xyCur, [OperatorAw(dxdy[0]) + OperatorHx(dxdy[1]), OperatorHxaj(dxdy[0])])
                vec = pymanopt.manifolds.product._ProductTangentVector([OperatorAw(dxdy[0]) + OperatorHx(dxdy[1]), OperatorHxaj(dxdy[0])])
                return vec
        else:
            def OperatorT(dxdy):
                # vec = xy_manifold.to_tangent_space(xyCur, [OperatorAw(dxdy[0]), np.array([])])
                vec = pymanopt.manifolds.product._ProductTangentVector([OperatorAw(dxdy[0]), np.array([])])
                return vec
        return OperatorT
    OperatorT = build_operatorT(OperatorAw, OperatorHx, OperatorHxaj, eqconstraints)

    # Solve the condensed NT equation T(dxdy) = cq.
    # print("xy_manifold.dim", xy_manifold.num_values)
    if KrylovIterMethod:
        NTdirdxdy, t, rel_res, info = TangentSpaceConjResMethod(OperatorT, cq, v0, xy_manifold, xyCur, KrylovTolrelres, KrylovMaxiter)
        # print(NTdirdxdy, t, rel_res, info)
    else:
        pass
    
    # Recovery dz and ds.
    Ntdirdz = (zCur * (Gxaj(xCur, NTdirdxdy[0]) + KKTvec[2]) + sigma * rho * ehat - KKTvec[3]) / sCur
    NTdirds = (sigma * rho * ehat - KKTvec[3] - sCur * Ntdirdz) / zCur
    NTdir = pymanopt.manifolds.product._ProductTangentVector([NTdirdxdy[0], NTdirdxdy[1], Ntdirdz, NTdirds])
    
    # Check Newton direction NTdir
    # DEBUG ONLY
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
    
    # Backtracking line search and update.
    # Central functions
    fun_1 = lambda z, s: min(z * s) - gamma * tau_1 * (z @ s / ineqnum)
    fun_2 = lambda z, s, Phi: z @ s - gamma * tau_2 * np.sqrt(Phi)
    
    # Note that <grad phi, NTdir> = ls_RightItem, if NTdir is a correct solution.
    ls_RightItem = 2 * (sigma * rho * (zCur @ sCur) - PhiCur)
    
    stepsize = 1
    ls_max_steps_flag = False
    r = 0
    while True:
        xyzsNew = xyzs_manifold.retraction(xyzsCur, stepsize * NTdir)
        # KKTvec = compute_KKTVectorField(xyzsNew, ineqconstraints, eqconstraints, gradLagrangian)
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
            ls_max_steps_flag = True
            break
        stepsize = ls_theta * stepsize
    
    # Update the current point.
    xCur, yCur, zCur, sCur = xyzsNew
    PhiCur = PhiNew
    """costcurとkktresidualcur"""
        
    # Update parameters
    sigma = min(0.5, np.sqrt(PhiCur)**0.5)
    rho = (zCur @ sCur) / ineqnum
    gamma = 0.5 * (gamma + 0.5)
    
    print("xCur", xCur)
        # xNew = manifold.retraction(xCur, stepsize * NTdir[0])
        # yNew = yCur + stepsize * NTdir[1]
        # zNew = zCur + stepsize * NTdir[2]
        # sNew = sCur + stepsize * NTdir[3]
        # KKTvec = 

    print("attained!")
    input()

if __name__ == "__main__":
    main()

# import numpy as np
# dz = [None] * 5
# for i in range(5):
#     dz[i] = i
# print(dz)

# print(np.random.rand(0))

# # import pymanopt
# # mani = pymanopt.manifolds.euclidean.Euclidean(0)
# # print(mani.random_point())
# import pymanopt.manifolds
# import pymanopt.optimizers
# from pymanopt.manifolds import Product, Sphere, Stiefel
# # 各マニフォールドを定義
# sphere1 = Sphere(3)    # 3次元球面
# sphere2 = Sphere(3)    # 別の3次元球面
# stiefel = Stiefel(3, 2)  # Stiefel manifold（3x2行列）

# # Product Manifoldの作成
# product_manifold = Product([sphere1, sphere2, stiefel])

# import pymanopt
# from pymanopt.optimizers import SteepestDescent
# from pymanopt import Problem

# # 目的関数の定義
# def cost_function(point):
#     x, y, z = point  # それぞれのマニフォールドのポイント
#     return x @ y + z[0, 0]  # 簡単な例としてxとyの内積とzの要素を使う

# # 問題の設定
# problem = Problem(manifold=product_manifold, cost=cost_function)

# # ソルバーの選択
# solver = SteepestDescent()

# # 最適化の実行
# result = solver.solve(problem)
# print("Optimal point:", result)

# # condensed_product_manifold
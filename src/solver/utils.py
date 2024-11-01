import numpy as np
import pymanopt
import copy
from scipy.sparse.linalg import LinearOperator, eigs
from scipy.linalg import eig

# Lagrange function
def build_Lagrangefun(ineqLagmult, eqLagmult, costfun, ineqconstraints, eqconstraints, manifold):
    @pymanopt.function.autograd(manifold)
    def Lagrangefun(point):
        val = costfun(point)
        if ineqconstraints.has_constraint:
            for idx in range(ineqconstraints.num_constraint):
                funval = (ineqconstraints.constraint[idx])(point)
                val += ineqLagmult[idx] * funval
        if eqconstraints.has_constraint:
            for idx in range(eqconstraints.num_constraint):
                funval = (eqconstraints.constraint[idx])(point)
                val += eqLagmult[idx] * funval
        return val
    return Lagrangefun

def compute_maxmeanviolations(problem, x):
    # Setting
    ineqconstraints = problem.ineqconstraints
    eqconstraints = problem.eqconstraints

    maxviolation = 0
    meanviolation = 0

    # Compute the errors of ineqiality/equality constraints
    if ineqconstraints.has_constraint:
        for idx in range(ineqconstraints.num_constraint):
            ineqcstrfun = ineqconstraints.constraint[idx]
            violation = max(ineqcstrfun(x), 0)
            maxviolation = max(maxviolation, violation)
            meanviolation += violation

    if eqconstraints.has_constraint:
        for idx in range(eqconstraints.num_constraint):
            eqcstrfun = eqconstraints.constraint[idx]
            violation = abs(eqcstrfun(x))
            maxviolation = max(maxviolation, violation)
            meanviolation += violation

    if ineqconstraints.num_constraint + eqconstraints.num_constraint > 0:
        meanviolation = meanviolation / (ineqconstraints.num_constraint + eqconstraints.num_constraint)

    return maxviolation, meanviolation

def compute_residual(problem, x, ineqLagmult, eqLagmult, manviofun):
    # Setting
    costfun = problem.costfun
    ineqconstraints = problem.ineqconstraints
    eqconstraints = problem.eqconstraints
    manifold = problem.searchspace
    # Set Lagrange function
    Lagfun = build_Lagrangefun(ineqLagmult=ineqLagmult,
                                    eqLagmult=eqLagmult,
                                    costfun=costfun,
                                    ineqconstraints=ineqconstraints,
                                    eqconstraints=eqconstraints,
                                    manifold=manifold)
    Lagproblem = pymanopt.Problem(manifold, Lagfun)

    # Compute violation of gradient of the Lagrange function
    gradient = Lagproblem.riemannian_gradient

    gradnorm = manifold.norm(x, gradient(x))
    squared_gradLagvio =  gradnorm ** 2

    # Compute violation of the complementary condition
    squared_complvio = 0
    if ineqconstraints.has_constraint:
        for idx in range(ineqconstraints.num_constraint):
            ineqcstrfun = ineqconstraints.constraint[idx]
            violation = ineqLagmult[idx] * ineqcstrfun(x)
            squared_complvio += violation ** 2

    # Compute violation of the nonnegativity condition for Lagrange multipliers for inequality
    squared_nonnegvio = 0
    if ineqconstraints.has_constraint:
        for valLag in ineqLagmult:
            violation = max(-valLag, 0)
            squared_nonnegvio += violation ** 2

    # Compute the errors of ineqiality/equality constraints
    squared_ineqvio = 0
    if ineqconstraints.has_constraint:
        for idx in range(ineqconstraints.num_constraint):
            ineqcstrfun = ineqconstraints.constraint[idx]
            violation = max(ineqcstrfun(x), 0)
            squared_ineqvio += violation ** 2

    squared_eqvio = 0
    if eqconstraints.has_constraint:
        for idx in range(eqconstraints.num_constraint):
            eqcstrfun = eqconstraints.constraint[idx]
            violation = abs(eqcstrfun(x))
            squared_ineqvio += violation ** 2

    # Sum of the above violations
    KKTresid = np.sqrt(squared_gradLagvio
                        + squared_complvio
                        + squared_nonnegvio
                        + squared_ineqvio
                        + squared_eqvio)

    # Compute manifold violation
    manvio =  manviofun(problem, x)
    residual = KKTresid + manvio

    return residual, gradnorm

def evaluation(problem, xPrev, xCur, ineqLagmult, eqLagmult, manviofun, callbackfun):
    # Cost evaluation
    costfun = problem.costfun
    cost = costfun(xCur)

    # distance evaluation
    manifold = problem.searchspace
    dist = manifold.dist(xPrev, xCur)

    # residial of KKT conditions with manifold violation
    residual, gradnorm = compute_residual(problem, xCur, ineqLagmult, eqLagmult, manviofun)

    maxviolation, meanviolation = compute_maxmeanviolations(problem, xCur)

    eval = {"cost": cost,
            "distance": dist,
            "residual": residual,
            "gradnorm": gradnorm,
            "maxviolation": maxviolation,
            "meanviolation": meanviolation}

    eval = callbackfun(problem, xCur, eval)

    return eval

def orthogonalize(manifold, x, A):
    """
    Orthogonalize the basis vectors with respect to the metric at x.
    """
    n = len(A)
    Q = np.empty_like(A)  # Note: np.empty_like is not suited for product manifolds.
    R = np.zeros((n, n))
    for j in range(n):
        v = copy.deepcopy(A[j])
        for i in range(j):
            qi = Q[i]
            R[i,j] = manifold.inner_product(x, qi, v)
            v -= R[i,j] * qi
        R[j,j] = manifold.norm(x, v)
        Q[j] = v / R[j,j]
    return Q, R

def tangentorthobasis(manifold, x, n):
    """
    Return an orthonormal basis of the tangent space at x.
    """
    basis = []
    for _ in range(n):
        v = manifold.random_tangent_vector(x)
        basis.append(v)
    orthobasis, _ = orthogonalize(manifold, x, basis)
    return orthobasis

def hessianmatrix(problem, x, basis=None):
    if basis is None:
        n = problem.manifold.dim
        basis = tangentorthobasis(problem.manifold, x, n)
    else:
        n = len(basis)
    Hbasis = np.empty_like(basis)
    for k in range(len(basis)):
        Hbasis[k] = problem.riemannian_hessian(x, basis[k])

    H = np.zeros((n, n))
    for i in range(n):
        H[i, i] = problem.manifold.inner_product(x, basis[i], Hbasis[i])
        for j in range(i+1, n):
            H[i, j] = problem.manifold.inner_product(x, basis[i], Hbasis[j])
            H[j, i] = H[i, j]
    return H, basis

def hessianspectrum(problem, x, threshold=1e-9):
    tgtvec_shape = problem.manifold.zero_vector(x).shape
    n = len(problem.manifold.zero_vector(x).reshape(-1))
    dim = problem.manifold.dim

    if dim == 0:
        w = []
        v = []
        return w,v

    tgtfun = lambda v: problem.manifold.to_tangent_space(x, v)
    riemhess = lambda dir: problem.riemannian_hessian(x, dir)

    reshapefun = lambda v: v.reshape(tgtvec_shape)
    vecfun = lambda v: v.reshape(-1)
    hessopr = LinearOperator((n, n), matvec=lambda dir: vecfun(tgtfun(riemhess(tgtfun(reshapefun(dir))))))

    # For speeding up (to do)
    # tanvec_size = len(tgtvec_shape)
    # if tanvec_size == 1:
    #     hessopr = LinearOperator((n, n), matvec=lambda dir: tgtfun(riemhess(tgtfun(dir))))
    # else:
    #     reshapefun = lambda v: v.reshape(tgtvec_shape)
    #     vecfun = lambda v: v.reshape(-1)
    #     hessopr = LinearOperator((n, n), matvec=lambda dir: vecfun(tgtfun(riemhess(tgtfun(reshapefun(dir))))))

    if dim <= n-2:
        w, v_vector = eigs(hessopr, k=dim, which='LM')
        v = []
        for j in range(v_vector.shape[1]):  # column loop
            vec = v_vector[:,j]
            tgtvec = tgtfun(reshapefun(vec))
            v.append(tgtvec)
        v = np.array(v)
    else:
        if n <= 2:
            H, basis = hessianmatrix(problem, x)
            w, coeff = eig(H)
            v = []
            for j in range(coeff.shape[1]):
                d = problem.manifold.zero_vector(x)
                for i in range(len(basis)):
                    d += basis[i] * coeff[i,j]
                v.append(d)
        else:
            w_lm, v_lm = eigs(hessopr, k=n-2, which='LR')
            w_sm, v_sm = eigs(hessopr, k=2, which='SR')
            w = np.hstack((w_lm, w_sm))
            v_vector = np.hstack((v_lm, v_sm))
            v = []
            for j in range(v_vector.shape[1]):  # column loop
                vec = v_vector[:,j]
                tgtvec = tgtfun(reshapefun(vec))
                v.append(tgtvec)

        sort_indices = np.argsort(np.abs(w))
        sort_indices = sort_indices[::-1]
        w = w[sort_indices]
        v = np.array(v)
        v = v[sort_indices]
        w = w[:dim]
        v = v[:dim]

    w = w.real  # eigenvalues for Hermite matrix are always real.
    # Reduce the numerical error: change the imaginary part to zero if it is sufficiently small.
    tgtvec_type = problem.manifold.zero_vector(x).dtype
    # v = np.where(np.abs(v.imag) <= threshold, v.real, v)
    v = v.astype(tgtvec_type)
    return w, v

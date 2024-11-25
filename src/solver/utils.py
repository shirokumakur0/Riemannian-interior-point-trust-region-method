import numpy as np
import pymanopt
import copy
from scipy.sparse.linalg import LinearOperator, eigs
from scipy.linalg import eig
from dataclasses import dataclass, field
from typing import Any


import sys
sys.path.append('./src/base')
from base_solver import BaseOutput

@dataclass
class Output(BaseOutput):
    ineqLagmult: field(default_factory=list)
    eqLagmult: field(default_factory=list)

@dataclass
class ManifoldConstraints:
    constraints: list = field(default_factory=list)
    type: list = field(default_factory=list)

@dataclass
class EuclideanNonlinearProblem:
    cost: Any
    ineqconstraints: Any
    eqconstraints: Any
    initialpoint: Any
    maniconstraints: ManifoldConstraints
    # initialineqLagmult: Any
    # initialeqLagmult: Any

class NonlinearProblem(pymanopt.Problem):
    def __init__(
        self,
        manifold,
        cost,
        ineqconstraints=[],
        eqconstraints=[],
        initialpoint=None,
        initialineqLagmult = np.array([]),
        initialeqLagmult = np.array([]),
        ):
        super().__init__(manifold, cost)
        
        # ineqconstraints: list of inequality constraints
        self._original_ineqconstraints = ineqconstraints
        self.num_ineqconstraints = len(ineqconstraints)
        self.has_ineqconstraints = self.num_ineqconstraints > 0
        _ineqconstraints = [None] * self.num_ineqconstraints
        for i in range(self.num_ineqconstraints):
            _ineqconstraints[i] = self._wrap_function(ineqconstraints[i])
        self._ineqconstraints = _ineqconstraints
        
        # eqconstraints: list of equality constraints
        self._original_eqconstraints = eqconstraints
        self.num_eqconstraints = len(eqconstraints)
        self.has_eqconstraints = self.num_eqconstraints > 0
        _eqconstraints = [None] * self.num_eqconstraints
        for i in range(self.num_eqconstraints):
            _eqconstraints[i] = self._wrap_function(eqconstraints[i])
        self._eqconstraints = _eqconstraints

        self._ineqconstraints_euclidean_gradient = [None] * self.num_ineqconstraints
        self._ineqconstraints_riemannian_gradient = [None] * self.num_ineqconstraints
        self._eqconstraints_euclidean_gradient = [None] * self.num_eqconstraints
        self._eqconstraints_riemannian_gradient = [None] * self.num_eqconstraints

        self._ineqconstraints_euclidean_hessian = [None] * self.num_ineqconstraints
        self._ineqconstraints_riemannian_hessian = [None] * self.num_ineqconstraints
        self._eqconstraints_euclidean_hessian = [None] * self.num_eqconstraints
        self._eqconstraints_riemannian_hessian = [None] * self.num_eqconstraints

        # initialpoint and Lagrange multipliers
        self.initialpoint = initialpoint
        self.initialineqLagmult = initialineqLagmult
        self.initialeqLagmult = initialeqLagmult

    def ineqconstraints(self, index):
        return self._ineqconstraints[index]

    @property
    def ineqconstraints_all(self):
        return [self.ineqconstraints(i) for i in range(self.num_ineqconstraints)]
        
    def eqconstraints(self, index):
        return self._eqconstraints[index]

    @property
    def eqconstraints_all(self):
        return [self.eqconstraints(i) for i in range(self.num_eqconstraints)]

    def ineqconstraints_euclidean_gradient(self, index):
        if self._ineqconstraints_euclidean_gradient[index] is None:
            self._ineqconstraints_euclidean_gradient[index] = self._wrap_gradient_operator(
                self._original_ineqconstraints[index].get_gradient_operator()
            )
        return self._ineqconstraints_euclidean_gradient[index]
    
    @property
    def ineqconstraints_euclidean_gradient_all(self):
        return [self.ineqconstraints_euclidean_gradient(i) for i in range(self.num_ineqconstraints)]

    def ineqconstraints_riemannian_gradient(self, index):
        if self._ineqconstraints_riemannian_gradient[index] is None:
            def build_riemannian_gradient(index):
                def riemannian_gradient(point):
                    return self.manifold.euclidean_to_riemannian_gradient(
                        point, self.ineqconstraints_euclidean_gradient(index)(point)
                    )
                return riemannian_gradient
            self._ineqconstraints_riemannian_gradient[index] = build_riemannian_gradient(index)
        return self._ineqconstraints_riemannian_gradient[index]

    @property
    def ineqconstraints_riemannian_gradient_all(self):
        return [self.ineqconstraints_riemannian_gradient(i) for i in range(self.num_ineqconstraints)]

    def eqconstraints_euclidean_gradient(self, index):
        if self._eqconstraints_euclidean_gradient[index] is None:
            self._eqconstraints_euclidean_gradient[index] = self._wrap_gradient_operator(
                self._original_eqconstraints[index].get_gradient_operator()
            )
        return self._eqconstraints_euclidean_gradient[index]

    @property
    def eqconstraints_euclidean_gradient_all(self):
        return [self.eqconstraints_euclidean_gradient(i) for i in range(self.num_eqconstraints)]

    def eqconstraints_riemannian_gradient(self, index):
        if self._eqconstraints_riemannian_gradient[index] is None:
            def build_riemannian_gradient(index):
                def riemannian_gradient(point):
                    return self.manifold.euclidean_to_riemannian_gradient(
                        point, self.eqconstraints_euclidean_gradient(index)(point)
                    )
                return riemannian_gradient
            self._eqconstraints_riemannian_gradient[index] = build_riemannian_gradient(index)
        return self._eqconstraints_riemannian_gradient[index]

    @property
    def eqconstraints_riemannian_gradient_all(self):
        return [self.eqconstraints_riemannian_gradient(i) for i in range(self.num_eqconstraints)]

    def ineqconstraints_euclidean_hessian(self, index):
        if self._ineqconstraints_euclidean_hessian[index] is None:
            self._ineqconstraints_euclidean_hessian[index] = self._wrap_hessian_operator(
                self._original_ineqconstraints[index].get_hessian_operator(),
                embed_tangent_vectors=True,
            )
        return self._ineqconstraints_euclidean_hessian[index]
    
    @property
    def ineqconstraints_euclidean_hessian_all(self):
        return [self.ineqconstraints_euclidean_hessian(i) for i in range(self.num_ineqconstraints)]

    def ineqconstraints_riemannian_hessian(self, index):
        if self._ineqconstraints_riemannian_hessian[index] is None:
            def build_riemannian_hessian(index):
                def riemannian_hessian(point, tangent_vector):
                    return self.manifold.euclidean_to_riemannian_hessian(
                        point,
                        self.ineqconstraints_euclidean_gradient(index)(point),
                        self.ineqconstraints_euclidean_hessian(index)(point, tangent_vector),
                        tangent_vector,
                    )
                return riemannian_hessian
            self._ineqconstraints_riemannian_hessian[index] = build_riemannian_hessian(index)
        return self._ineqconstraints_riemannian_hessian[index]
    
    @property
    def ineqconstraints_riemannian_hessian_all(self):
        return [self.ineqconstraints_riemannian_hessian(i) for i in range(self.num_ineqconstraints)]

    def eqconstraints_euclidean_hessian(self, index):
        if self._eqconstraints_euclidean_hessian[index] is None:
            self._eqconstraints_euclidean_hessian[index] = self._wrap_hessian_operator(
                self._original_eqconstraints[index].get_hessian_operator(),
                embed_tangent_vectors=True,
            )
        return self._eqconstraints_euclidean_hessian[index]

    @property
    def eqconstraints_euclidean_hessian_all(self):
        return [self.eqconstraints_euclidean_hessian(i) for i in range(self.num_eqconstraints)]

    def eqconstraints_riemannian_hessian(self, index):
        if self._eqconstraints_riemannian_hessian[index] is None:
            def build_riemannian_hessian(index):
                def riemannian_hessian(point, tangent_vector):
                    return self.manifold.euclidean_to_riemannian_hessian(
                        point,
                        self.eqconstraints_euclidean_gradient(index)(point),
                        self.eqconstraints_euclidean_hessian(index)(point, tangent_vector),
                        tangent_vector,
                    )
                return riemannian_hessian
            self._eqconstraints_riemannian_hessian[index] = build_riemannian_hessian(index)
        return self._eqconstraints_riemannian_hessian[index]
    
    @property
    def eqconstraints_riemannian_hessian_all(self):
        return [self.eqconstraints_riemannian_hessian(i) for i in range(self.num_eqconstraints)]

def tgtvecshapefun(manifold, x, vec):
    """
    Note: need to add elif isinstance(manifold, pymanopt.manifolds.FixedRankEmbeeded)
    if we deal with the fixed-rank embeeded manifold.
    """
    if isinstance(manifold, pymanopt.manifolds.Product):
        product_tgtvec = []
        for mani, point in zip(manifold.manifolds, x):
            tgtvec = tgtvecshapefun(mani, point, vec)
            tgtvec_len = len(tgtvec.reshape(-1))
            vec = vec[tgtvec_len:]
            product_tgtvec.append(tgtvec)
        return pymanopt.manifolds.product._ProductTangentVector(product_tgtvec)
    else:
        zerovec = manifold.zero_vector(x)
        tgtvec_shape = zerovec.shape
        tgtvec_len = len(zerovec.reshape(-1))
        return vec[:tgtvec_len].reshape(tgtvec_shape)

def vectorizefun(manifold, x, tgtvec):
    """
    Note: need to add elif isinstance(manifold, pymanopt.manifolds.FixedRankEmbeeded)
    if we deal with the fixed-rank embeeded manifold.
    """
    if isinstance(manifold, pymanopt.manifolds.Product):
        product_vec = np.array([])
        for mani, point, elem in zip(manifold.manifolds, x, tgtvec):
            product_vec = np.concatenate((product_vec,  vectorizefun(mani, point, elem)))
        return product_vec
    else:
        return tgtvec.reshape(-1)


def compute_maxmeanviolations(problem, x):
    # Setting
    ineqconstraints = problem.ineqconstraints_all
    eqconstraints = problem.eqconstraints_all
    has_ineqconstraints = problem.has_ineqconstraints
    num_ineqconstraints = problem.num_ineqconstraints
    has_eqconstraints = problem.has_eqconstraints
    num_eqconstraints = problem.num_eqconstraints

    maxviolation = 0
    meanviolation = 0

    # Compute the errors of ineqiality/equality constraints
    if has_ineqconstraints:
        for idx in range(num_ineqconstraints):
            ineqcstrfun = ineqconstraints[idx]
            violation = max(ineqcstrfun(x), 0)
            maxviolation = max(maxviolation, violation)
            meanviolation += violation

    if has_eqconstraints:
        for idx in range(num_eqconstraints):
            eqcstrfun = eqconstraints[idx]
            violation = abs(eqcstrfun(x))
            maxviolation = max(maxviolation, violation)
            meanviolation += violation

    if num_ineqconstraints + num_eqconstraints > 0:
        meanviolation = meanviolation / (num_ineqconstraints + num_eqconstraints)

    return maxviolation, meanviolation

def compute_residual(problem, x, ineqLagmult, eqLagmult, manviofun):
    # Setting
    ineqconstraints = problem.ineqconstraints_all
    eqconstraints = problem.eqconstraints_all
    manifold = problem.manifold
    has_ineqconstraints = problem.has_ineqconstraints
    num_ineqconstraints = problem.num_ineqconstraints
    has_eqconstraints = problem.has_eqconstraints
    num_eqconstraints = problem.num_eqconstraints
    
    # Set Lagrange function
    gradcost = problem.riemannian_gradient
    gradineqconstraints = problem.ineqconstraints_riemannian_gradient_all
    gradeqconstraints = problem.eqconstraints_riemannian_gradient_all
    
    def gradLagfun(x):
        vec = gradcost(x)
        for i in range(num_ineqconstraints):
            vec = vec + ineqLagmult[i] * gradineqconstraints[i](x)
        for j in range(num_eqconstraints):
            vec = vec + eqLagmult[j] * gradeqconstraints[j](x)
        return vec
    gradientx = gradLagfun(x)
    gradnorm = manifold.norm(x, gradientx)
    squared_gradLagvio =  gradnorm ** 2

    # Compute violation of the complementary condition
    squared_complvio = 0
    if has_ineqconstraints:
        for idx in range(num_ineqconstraints):
            ineqcstrfun = ineqconstraints[idx]
            violation = ineqLagmult[idx] * ineqcstrfun(x)
            squared_complvio += violation ** 2

    # Compute violation of the nonnegativity condition for Lagrange multipliers for inequality
    squared_nonnegvio = 0
    if has_ineqconstraints:
        for valLag in ineqLagmult:
            violation = max(-valLag, 0)
            squared_nonnegvio += violation ** 2

    # Compute the errors of inequality/equality constraints
    squared_ineqvio = 0
    if has_ineqconstraints:
        for idx in range(num_ineqconstraints):
            ineqcstrfun = ineqconstraints[idx]
            violation = max(ineqcstrfun(x), 0)
            squared_ineqvio += violation ** 2

    squared_eqvio = 0
    if has_eqconstraints:
        for idx in range(num_eqconstraints):
            eqcstrfun = eqconstraints[idx]
            violation = abs(eqcstrfun(x))
            squared_ineqvio += violation ** 2

    # Compute manifold violation
    manvio =  manviofun(problem, x)
    squared_manvio = manvio ** 2
    
    # Sum of the above violations
    residual = np.sqrt(squared_gradLagvio
                        + squared_complvio
                        + squared_nonnegvio
                        + squared_ineqvio
                        + squared_eqvio
                        + squared_manvio
                        )

    return residual, gradnorm

def evaluation(problem, xPrev, xCur, ineqLagmult, eqLagmult, manviofun, callbackfun):
    # Cost evaluation
    costfun = problem.cost
    cost = costfun(xCur)

    # distance evaluation
    manifold = problem.manifold
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
    Q = [None] * n
    # Q = np.empty_like(A)  # Note: np.empty_like is not suited for product manifolds.
    R = np.zeros((n, n))
    for j in range(n):
        v = copy.deepcopy(A[j])
        for i in range(j):
            qi = Q[i]
            R[i,j] = manifold.inner_product(x, qi, v)
            v = v - R[i,j] * qi
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

def hessianspectrum(problem, x):
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

def operatorspectrum(manifold, operator, x):
    dim = manifold.dim
    if dim == 0:
        w = []
        v = []
        return w,v

    tgtfun = lambda v: manifold.to_tangent_space(x, v)
    operatorx = lambda dir: operator(x, dir)

    # reshapefun = lambda v: v.reshape(tgtvec_shape)
    reshapefun = lambda vec: tgtvecshapefun(manifold, x, vec)
    vecfun = lambda tgtvec: vectorizefun(manifold, x, tgtvec)
    
    n = len(vecfun(manifold.zero_vector(x)))
    # vecfun = lambda v: v.reshape(-1)
    linopr = LinearOperator((n, n), matvec=lambda dir: vecfun(tgtfun(operatorx(tgtfun(reshapefun(dir))))))

    # For speeding up (to do)
    # tanvec_size = len(tgtvec_shape)
    # if tanvec_size == 1:
    #     hessopr = LinearOperator((n, n), matvec=lambda dir: tgtfun(riemhess(tgtfun(dir))))
    # else:
    #     reshapefun = lambda v: v.reshape(tgtvec_shape)
    #     vecfun = lambda v: v.reshape(-1)
    #     hessopr = LinearOperator((n, n), matvec=lambda dir: vecfun(tgtfun(riemhess(tgtfun(reshapefun(dir))))))

    if dim <= n-2:
        w, v_vector = eigs(linopr, k=dim, which='LM')
        v = []
        for j in range(v_vector.shape[1]):  # column loop
            vec = v_vector[:,j].real
            tgtvec = tgtfun(reshapefun(vec))
            v.append(tgtvec)
        # v = np.array(v)
    else:
        if n <= 2:
            basis = tangentorthobasis(manifold, x, manifold.dim)
            H, basis = selfadj_operator2matrix(manifold, operator, x, basis)
            w, coeff = eig(H)
            v = []
            for j in range(coeff.shape[1]):
                d = manifold.zero_vector(x)
                for i in range(len(basis)):
                    d = d + basis[i] * coeff[i,j]
                v.append(d)
        else:
            w_lm, v_lm = eigs(linopr, k=n-2, which='LR')
            w_sm, v_sm = eigs(linopr, k=2, which='SR')
            w = np.hstack((w_lm, w_sm))
            v_vector = np.hstack((v_lm, v_sm))
            v = []
            for j in range(v_vector.shape[1]):  # column loop
                vec = v_vector[:,j].real
                tgtvec = tgtfun(reshapefun(vec))
                v.append(tgtvec)

        sort_indices = np.argsort(np.abs(w))
        sort_indices = sort_indices[::-1]
        w = w[sort_indices]
        # v = np.array(v)
        # v = v[sort_indices]
        v = [v[i] for i in sort_indices]
        w = w[:dim]
        v = v[:dim]

    w = w.real  # eigenvalues for Hermite matrix are always real.
    # Reduce the numerical error: change the imaginary part to zero if it is sufficiently small.
    # tgtvec_type = manifold.zero_vector(x).dtype
    # v = v.astype(tgtvec_type)
    return w, v

def operator2matrix(Mx, x, y, F, Bx=None, By=None, My=None):
    # Forms a matrix representing a linear operator between two tangent spaces
    #
    # Given a manifold structure M, two points x and y on that manifold, a
    # function F encoding a linear operator from the tangent space T_x M to the
    # tangent space T_y M, and the orthonomal vectors in T_x M and T_y M,
    # this tool forms the matrix A which
    # represents the operator F in those bases. In particular, the singular
    # values of A are equal to the singular values of F. If two manifold
    # structures are passed, then x is a point on Mx and y is a point on My.
    #
    # The matrix A represents the linear operator F restricted to the span of Bx, composed
    # with orthogonal projection to the span of By. Of course, if Bx and By are
    # orthonormal bases of T_x M and T_y M, then this is simply a
    # representation of F. Same comment if two manifolds are passed.

    if My is None:
        My = Mx
    
    if Bx is None:
        Bx = tangentorthobasis(Mx, x, Mx.dim)
    if By is None:
        By = tangentorthobasis(My, y, My.dim)

    n_in = len(Bx)
    n_out = len(By)
    A = np.zeros((n_out, n_in))
    for j in range(n_in):
        FBxj = F(Bx[j])
        A[:, j] = tangent2vec(My, y, By, FBxj)
    return A

def selfadj_operator2matrix(M, x, F, Bx):
    n = len(Bx)
    A_mat = np.zeros((n, n))
    for j in range(n):
        FBxj = F(Bx[j])
        for i in range(j+1):
            A_mat[i, j] = M.inner_product(x, FBxj, (Bx[i]))
    A_mat = A_mat + np.triu(A_mat, 1).T
    return A_mat

def tangent2vec(M, x, basis, u):
    n = len(basis)
    vec =np.zeros(n)
    for k in range(n):
        vec[k] = M.inner_product(x, basis[k], u)
    return vec

def TangentSpaceConjResMethod(A, b, v0, M, x, tol, maxiter):
    # Conjugate residual method for solving linear operator equation: A(v)=b,
    # where A is some self-adjoint operator to and from some linear space E,
    # b is an element in E. Assume the existence of solution v.
    
    # Yousef Saad - Iterative methods for sparse linear systems,
    # 2nd edition-SIAM (2003) P203. ALGORITHM 6.20
    
    v = v0  # initialization
    r = b - A(v)  # r are residuals.
    p = copy.deepcopy(r)  # p are conjugate directions.
    b_norm = M.norm(x, b)
    r_norm = M.norm(x, r)
    rel_res = r_norm / b_norm
    Ar = A(r)
    Ap = A(p)
    rAr = M.inner_product(x, r, Ar)
    t = 0  # at t-th iteration
    info =  np.zeros((maxiter, 2))
    while True:
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
        old_rAr = rAr
        rAr = M.inner_product(x, r, Ar)
        beta = rAr / old_rAr  # improvement this step
        p = r + beta*p  # search direction # r + beta*p
        Ap = Ar + beta*Ap  # Ar + beta*Ap

    vfinal = v
    return vfinal, t, rel_res, info
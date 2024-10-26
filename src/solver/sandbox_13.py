import autograd.numpy as anp
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
import numpy as np
import time
anp.random.seed(42)
import copy
from utils import hessianspectrum, build_Lagrangefun
import scipy
import hydra
from scipy.sparse.linalg import LinearOperator, eigs
from dataclasses import dataclass, field
from scipy.linalg import eig
import sys
from cvxopt import spmatrix, matrix, solvers

@dataclass
class DXYZSVector:
    dx: list = field(default_factory=list)
    dy: list = field(default_factory=list)
    dz: list = field(default_factory=list)
    ds: list = field(default_factory=list)

# def Gx(x, z, ineqconstraints, manifold):
#     val = 0
#     for idx in range(ineqconstraints.num_constraint):
#         egrad = ineqconstraints.constraint[idx].get_gradient_operator()
#         rgrad = manifold.euclidean_to_riemannian_gradient(x, egrad(x))
#         val += z[idx] * rgrad(x)
#     return val

# def rhess_Gx(x, z, dx, ineqconstraints, manifold):
#     val = 0
#     for idx in range(ineqconstraints.num_constraint):
#         egrad = ineqconstraints.constraint[idx].get_gradient_operator()
#         ehess = ineqconstraints.constraint[idx].get_hessian_operator()
#         rhess = manifold.euclidean_to_riemannian_hessian(x, egrad(x), ehess(x, dx), dx)
#         val += z[idx] * rhess(x, dx)
#     return val

def compute_KKTVectorField(x, y, z, s, ineqconstraints, eqconstraints, gradLagrangian):
    F = DXYZSVector
    F.dx = gradLagrangian(x, y, z)
    if eqconstraints.has_constraint:
        dy = []
        for idx in range(eqconstraints.num_constraint):
            val = (eqconstraints.constraint[idx])(x)
            dy.append(val)
        F.dy = dy
    else:
        F.dy = []
    dz = []
    for idx in range(ineqconstraints.num_constraint):
        val = (ineqconstraints.constraint[idx])(x) + s[idx]
        dz.append(val)
    F.dz = dz
    F.ds = z * s
    return F

def barGx(x, z, ineqconstraints):
    val = 0
    for idx in range(ineqconstraints.num_constraint):
        egrad = ineqconstraints.constraint[idx].get_gradient_operator()
        val += z[idx] * egrad(x)
    return val

def ehess_barGx(x, z, dx, ineqconstraints):
    val = 0
    for idx in range(ineqconstraints.num_constraint):
        ehess = ineqconstraints.constraint[idx].get_hessian_operator()
        val += z[idx] * ehess(x, dx)
    return val

def barGxaj(x, dx, ineqconstraints):
    val = np.zeros(ineqconstraints.num_constraint)
    for idx in range(ineqconstraints.num_constraint):
        egrad = ineqconstraints.constraint[idx].get_gradient_operator()
        val[idx] = egrad(x) @ dx
        """↑これでいいのか検討"""
    return val

def barHx(x, y, eqconstraints):
    val = 0
    for idx in range(eqconstraints.num_constraint):
        egrad = eqconstraints.constraint[idx].get_gradient_operator()
        val += y[idx] * egrad(x)
    return val

def ehess_barHx(x, y, dx, eqconstraints):
    val = 0
    for idx in range(eqconstraints.num_constraint):
        ehess = eqconstraints.constraint[idx].get_hessian_operator()
        val += y[idx] * ehess(x, dx)
    return val

def barHxaj(x, dx, eqconstraints):
    val = np.zeros(eqconstraints.num_constraint)
    for idx in range(eqconstraints.num_constraint):
        egrad = eqconstraints.constraint[idx].get_gradient_operator()
        val[idx] = egrad(x) @ dx
        """↑これでいいのか検討"""
    return val

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

    # Set initial points
    xCur = problem.initialpoint
    ineqLagCur = problem.initialineqLagmult
    eqLagCur = problem.initialeqLagmult
    
    # Option setting
    option = {}
    option["heuristic_z_s"] = True
    option["desired_tau_1"] = 0.5
    option["important"] = 1
    
    Gx = lambda x, z: manifold.euclidean_to_riemannian_gradient(x, barGx(x, z, ineqconstraints))
    Gxaj = lambda x, dx: barGxaj(x, manifold.embedding(x, dx), ineqconstraints)
    
    if eqconstraints.has_constraint:
        Hx = lambda x, y: manifold.egrad2rgrad(x, barHx(x, y, eqconstraints))
        Hxaj = lambda x, dx: barHxaj(x, manifold.embedding(x, dx), eqconstraints)

        egradLagrangian = lambda x, y, z: problem.euclidean_gradient(x) + barGx(x, z, ineqconstraints) + barHx(x, y, eqconstraints)
        ehessLagrangian = lambda x, y, z, dx: problem.euclidean_hessian(x, dx) + ehess_barGx(x, z, dx, ineqconstraints) + ehess_barHx(x, y, dx, eqconstraints)
        
    else:
        egradLagrangian = lambda x, y, z: problem.euclidean_gradient(x) + barGx(x, z, ineqconstraints)
        ehessLagrangian = lambda x, y, z, dx: problem.euclidean_hessian(x, dx) + ehess_barGx(x, z, dx, ineqconstraints)
        
    gradLagrangian = lambda x, y, z: manifold.euclidean_to_riemannian_gradient(x, egradLagrangian(x, y, z))
    hessLagrangian = lambda x, y, z, dx: manifold.euclidean_to_riemannian_hessian(x, egradLagrangian(x, y, z), ehessLagrangian(x, y, z, dx), dx)
    
    if eqconstraints.has_constraint:
        eqnum = eqconstraints.num_constraint
        y = np.zeros(eqnum)
    else:
        y = []
        
    ineqnum = ineqconstraints.num_constraint
    if option["heuristic_z_s"]:
        z = np.ones(ineqnum)
        z[1] =np.real(np.sqrt(((ineqnum - 1)/(ineqnum/option["desired_tau_1"] - 1))));
        s = option["important"] * z
    else:
        z = np.random.rand(ineqnum)
        s = np.random.rand(ineqnum)
    
    dim = manifold.dim
    zeroTxM = manifold.zero_vector(xCur)
    if eqconstraints.has_constraint:
        dimeq = eqconstraints.num_constraint
        zeroTyEucli = np.zeros(dimeq)
    else:
        dimeq = 0
        zeroTyEucli = []
    
    dimineq = ineqconstraints.num_constraint
    zeroTzEucli = np.zeros(dimineq)
    ehat = np.ones(dimineq)
    Ehat = DXYZSVector(dx=zeroTxM, dy=zeroTyEucli, dz=zeroTzEucli, ds=ehat)
    v0 = DXYZSVector(dx=zeroTxM, dy=zeroTyEucli)
    
    iter = 0
    KKTVec = compute_KKTVectorField(xCur, y, z, s, ineqconstraints, eqconstraints, gradLagrangian)
    
    """ここまで書いて力尽きた(2024.1.3)"""
    
    # tgtvec = manifold.random_tangent_vector(xCur)
    # print("tgtvec", tgtvec)
    # val = barGxaj(xCur, tgtvec, ineqconstraints)
    # print("val", val)
        
        
    # tgtvec = manifold.random_tangent_vector(xCur)
    # print("tgtvec", tgtvec)
    # val = barHxaj(xCur, tgtvec, eqconstraints)
    # print("val", val)
    
    
    
    
    
    
    
    # costLag = build_Lagrangefun(ineqLagmult=ineqLagCur,
    #                             eqLagmult=eqLagCur,
    #                             costfun=costfun,
    #                             ineqconstraints=ineqconstraints,
    #                             eqconstraints=eqconstraints,
    #                             manifold=manifold)
    # Lagproblem = pymanopt.Problem(manifold, costLag)


    # dim = 20
    # # manifold = pymanopt.manifolds.Sphere(dim)
    # manifold = pymanopt.manifolds.positive_definite.SymmetricPositiveDefinite(dim)

    # matrix = anp.random.normal(size=(dim, dim))
    # matrix = 0.5 * (matrix + matrix.T) - 0.5 * np.diag(np.ones(dim))

    # @pymanopt.function.autograd(manifold)
    # def cost(point):
    #     # return -point @ matrix @ point
    #     return np.trace(-point @ matrix @ point)

    # problem = pymanopt.Problem(manifold, cost)
    # # n = problem.manifold.dim
    # x = problem.manifold.random_point()
    # ineqLagmult = []
    # eqLagmult = []


if __name__=='__main__':
    main()


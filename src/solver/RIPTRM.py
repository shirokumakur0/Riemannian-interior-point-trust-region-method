import hydra, copy, time, pymanopt, wandb, warnings
import numpy as np
from dataclasses import dataclass, field
from utils import  tangentorthobasis, evaluation, tgtvecshapefun, vectorizefun, selfadj_operator2matrix, Output
import warnings
import scipy

import sys, slepc4py
slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc
Print = PETSc.Sys.Print

import sys
sys.path.append('./src/base')
from base_solver import Solver


# np.__config__.show()
# import warnings
# import traceback

# # スタックトレースを含む警告の処理を設定
# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#     log = f"{filename}:{lineno}: {category.__name__}: {message}\n"
#     log += "".join(traceback.format_stack(limit=5))  # トレースの深さを調整
#     print(log)

# warnings.showwarning = warn_with_traceback

warnings.filterwarnings("ignore", message="Output seems independent of input.")

def TRSgep(A, a, B, Del, tolhardcase=1e-4, exit_warning_triggered=False):
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
        exit_warning_triggered (bool): Whether to exit when a warning is triggered.

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
    if exit_warning_triggered:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", scipy.linalg.LinAlgWarning)
            p1 = scipy.linalg.solve(A, -a, assume_a='sym')
            for warning in caught_warnings:
                if issubclass(warning.category, scipy.linalg.LinAlgWarning):
                    # print(warning.message)
                    return None, None, "solve failed"
    else:
        p1 = scipy.linalg.solve(A, -a, assume_a='sym')
    if np.linalg.norm(A @ p1 + a) / np.linalg.norm(a) < 1e-5:
        if p1 @ B @ p1 >= Del**2:  # outside of the trust region
            p1 = np.full_like(p1, np.nan)  # ineligible
    else:  # numerically incorrect
        p1 = np.full_like(p1, np.nan)

    # Core of the code: generalized eigenproblem
    
    # MM0_norm = scipy.linalg.norm(MM0, ord=2)
    # MM1_norm = scipy.linalg.norm(MM1, ord=2)
    # print(MM0_norm, MM1_norm)
    # print("cond", np.linalg.cond(MM0), np.linalg.cond(MM1))
    if exit_warning_triggered:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            lams, vecs = scipy.linalg.eig(a=MM0, b=-MM1)
            # lams, vecs = scipy.linalg.eig(a=MM0/MM0_norm, b=-MM1/MM1_norm)
            for warning in caught_warnings:
                if issubclass(warning.category, scipy.linalg.LinAlgWarning):
                    # print(warning.message)
                    return None, None, "eig failed"
    else:
        lams, vecs = scipy.linalg.eig(a=MM0, b=-MM1)
        # lams, vecs = scipy.linalg.eig(a=MM0/MM0_norm, b=-MM1/MM1_norm)
    # print("before", lams)
    # lams = lams * MM0_norm / MM1_norm
    # print("after", lams)
    rmidx = np.argmax(np.real(lams))
    lam1 = np.real(lams[rmidx])  # rightmost eigenvalue
    V = vecs[:, rmidx]  # corresponding rightmost eigenvector
    
    # print("resid", np.linalg.norm(MM0 @ V + MM1 @ V * lam1))
    # print("lam1, V", lam1, V, np.linalg.norm(np.real(V)) >= 1e-3)
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
        if exit_warning_triggered:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", scipy.linalg.LinAlgWarning)
                x2 = scipy.linalg.solve(H, -a, assume_a='sym')
                for warning in caught_warnings:
                    if issubclass(warning.category, scipy.linalg.LinAlgWarning):
                        print(warning.message)
                        return None, None, "solve failed"
        else:
            x2 = scipy.linalg.solve(H, -a, assume_a='sym')
        type = "hardcase_1"

        # Residual check for hard case refinement
        if np.linalg.norm(Alam1B @ x2 + a) / np.linalg.norm(a) > tolhardcase:
            if exit_warning_triggered:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    _, v = scipy.linalg.eigh(A, B)  # ascending order
                    for warning in caught_warnings:
                        if issubclass(warning.category, scipy.linalg.LinAlgWarning):
                            # print(warning.message)
                            return None, None, "eig failed"
            else:
                _, v = scipy.linalg.eigh(A, B)  # ascending order

            for ii in [3,6,9]:  # Iteratively refine solution if needed
                Pvect = v[:,:ii]  # Slices returns only the portion within the actual size of the array if the specified range exceeds the bounds.
                BPvecti = B @ Pvect
                H = Alam1B + alpha1 * BPvecti @ BPvecti.T
                if exit_warning_triggered:
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        warnings.simplefilter("always", scipy.linalg.LinAlgWarning)
                        x2 = scipy.linalg.solve(H, -a, assume_a='sym')
                        for warning in caught_warnings:
                            if issubclass(warning.category, scipy.linalg.LinAlgWarning):
                                # print(warning.message)
                                return None, None, "solve failed"
                else:
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

def basefun(x, y, A, tgtfun, reshapefun, vecfun):
    y[:] = vecfun(tgtfun(A(tgtfun(reshapefun(x)))))

class PETSc_operator(object):
    def __init__(self, fun):
        self.fun = fun

    def mult(self, mat, xx, yy):
        x = xx.getArray(readonly=1).reshape(-1)
        y = yy.getArray(readonly=0).reshape(-1)
        self.fun(x, y)

def PETSc_solve(n, eignum, Afun, Bfun=None, tol=1e-12, maxiter=1000, eigval_type="LARGEST_REAL"):
    Acontext = PETSc_operator(Afun)
    Bcontext = PETSc_operator(Bfun)
    
    PETSc_A = PETSc.Mat().createPython([n,n], Acontext)
    PETSc_A.setUp()
    if Bfun is not None:
        PETSc_B = PETSc.Mat().createPython([n,n], Bcontext)
        PETSc_B.setUp()
    
    xr, _ = PETSc_A.getVecs()
    xi, _ = PETSc_A.getVecs()
    
    E = SLEPc.EPS().create()
    E.setTolerances(tol, maxiter)
    if Bfun is not None:
        E.setOperators(PETSc_A, PETSc_B)  # Set operators A and B for the generalized eigenproblem
    else:
        E.setOperators(PETSc_A)
    E.setDimensions(eignum, PETSc.DECIDE)  # Number of eigenvalues to compute
    E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)  # Generalized non-Hermitian eigenvalue problem
    if eigval_type == "LARGEST_REAL":
        E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    elif eigval_type == "SMALLEST_REAL":
        E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    else:
        raise ValueError("eigval_type must be 'LARGEST_REAL' or 'SMALLEST_REAL'")
    st = E.getST()  # Spectral transformation
    ksp = st.getKSP()  # Krylov subspace solver (avoid LU decomposition)
    ksp.setType("gmres")  # GMRES solver
    pc = ksp.getPC()
    pc.setType("none")  # No preconditioner
    E.solve()
    return E, xr, xi

# Riemannian interior point trust region method
class RIPTRM(Solver):
    def __init__(self, option):
        # Default setting for interior point trust region method
        default_option = {
            # Stopping criteria
            'maxtime': 100,
            'maxiter': 100,
            'tolresid': 1e-6,
            'inner_maxiter': None,
            'inner_maxtime': 120,

            # Inner iteration setting
            'initial_TR_radius': None,
            'maximal_TR_radius': 10,
            'rho': 0.2,  # threshold for the acceptance of the trial point
            'gamma': 0.25,  # the factor to shrink the trust region radius if the primal point is infeasible
            'forcing_function_Lagrangian': lambda mu: mu,
            'forcing_function_complementarity': lambda mu: mu,
            'forcing_function_second_order': lambda mu: 100 * mu,
            'min_barrier_parameter': 1e-15,
            'TRS_solver': 'Exact_RepMat',  # 'Exact_RepMat', 'Exact_Operator', or 'Cauchy'
            'second_order_stationarity': True,
            'TRS_tolresid': 1e-12,
            'TRS_tolhardcase': 1e-8,
            'exit_warning_triggered': False,
            'checkTRSoptimality': False,
            'initial_barrier_parameter': 0.1,
            'barrier_parameter_update_r': 0.8, # 0.8,
            'barrier_parameter_update_c': 0.8, #0.8,
            'const_left': 0.5,
            'const_right': 1e+20,
            'basisfun': lambda manifold, x: tangentorthobasis(manifold, x, manifold.dim),
            
            # Display setting
            'verbosity': 2,

            # Measuring violation for manifold constraints in 'self.compute_residual'
            'manviofun': lambda problem, x: 0,

            # Callback function at each step.
            'callbackfun': lambda problem, x, eval: eval,

            # logging
            'save_inner_iteration': True,
            'wandb_logging': False
        }
        # Merge default_option and the argument
        default_option.update(option)  # putting the setting in the default_option before that in the argument
        self.option = default_option
        self.log = {}  # will be filled in self.add_log

        if self.option["wandb_logging"]:
            wandb.finish()
            _ = wandb.init(project=self.option["wandb_project"],  # the project name where this run will be logged
                             name = f"RIPTRM_{self.option['TRS_solver']}",  # the name of the run
                             config=self.option)  # save hyperparameters and metadata

    def set_inner_option(self, option):
        inner_option = {}
        inner_option["second_order_stationarity"] = option["second_order_stationarity"]
        inner_option["TRS_solver"] = option['TRS_solver']
        inner_option["gamma"] = option['gamma']
        inner_option["maximal_TR_radius"] = option['maximal_TR_radius']
        inner_option["inner_maxiter"] = option['inner_maxiter']
        inner_option["inner_maxtime"] = option['inner_maxtime']
        inner_option["rho"] = option['rho']
        inner_option["const_left"] = option['const_left']
        inner_option["const_right"] =  option['const_right']
        inner_option["save_inner_iteration"] = ["save_inner_iteration"]
        inner_option["exit_warning_triggered"] = option['exit_warning_triggered']
        inner_option["TRS_tolhardcase"] = option['TRS_tolhardcase']
        inner_option["checkTRSoptimality"] = option['checkTRSoptimality']
        inner_option["exit_warning_triggered"] = option['exit_warning_triggered']
        inner_option["TRS_tolhardcase"] = option['TRS_tolhardcase']
        inner_option["TRS_tolresid"] = option['TRS_tolresid']
        inner_option["checkTRSoptimality"] = option['checkTRSoptimality']
        inner_option["exit_warning_triggered"] = option['exit_warning_triggered']
        inner_option["TRS_tolhardcase"] = option['TRS_tolhardcase']
        inner_option["checkTRSoptimality"] = option['checkTRSoptimality']
        inner_option["verbosity"] = option["verbosity"]
        inner_option["basisfun"] = option["basisfun"]
        inner_option["save_inner_iteration"] = option["save_inner_iteration"]
        inner_option["manviofun"] = option["manviofun"]
        inner_option["callbackfun"] = option["callbackfun"]
        inner_option["stopping_criterion_Lagrangian"] = None
        inner_option["stopping_criterion_complementarity"] = None
        if inner_option["second_order_stationarity"]:
            inner_option["stopping_criterion_second_order"] = None
        return inner_option

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

    def PETSc_solve_GEPTangentSpace(self, n, dim, eignum, Afun, Bfun=None, tol=1e-12, maxiter=1000, eigval_type="LARGEST_REAL"):
        E, xr, xi = PETSc_solve(n, (n-dim)+eignum, Afun, Bfun=Bfun, tol=tol, maxiter=maxiter, eigval_type=eigval_type)
        kvec = [None] * ((n-dim)+eignum)
        xrarrayvec = [None] * ((n-dim)+eignum)
        for i in range((n-dim)+eignum):
            k = E.getEigenpair(i, xr, xi)
            k = k.real
            kvec[i] = k
            xrarrayvec[i] = copy.deepcopy(np.real(xr.getArray()))
        indices = np.argsort(-np.abs(kvec))[:eignum]  # sort by absolute value
        indices = indices[np.argsort(np.array(kvec)[indices])]  # sort by value
        kvects = [kvec[i] for i in indices]
        xrvects = [xrarrayvec[i] for i in indices]
        return kvects, xrvects

    def solve_linear_equations(self, A, b, tol, exit_warning_triggered):
        if exit_warning_triggered:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", scipy.linalg.LinAlgWarning)
                sol, _ = scipy.sparse.linalg.lgmres(A, b, rtol=tol)
                for warning in caught_warnings:
                    if issubclass(warning.category, scipy.linalg.LinAlgWarning):
                        sol = None
                return sol
        else:
            sol, _ = scipy.sparse.linalg.lgmres(A, b, rtol=tol)
        return sol

    def MM0fun(self, x, y, tgtfun, reshapefun, vecfun, n, Afun, Bfun, g, Delta, manifold):
        x1 = tgtfun(reshapefun(x[:n]))
        x2 = tgtfun(reshapefun(x[n:]))
        y1 = -Bfun(x1) + Afun(x2)
        y2 = Afun(x1) - g * manifold.inner_product(x, g, x2) / Delta**2
        y1 = vecfun(tgtfun(y1))
        y2 = vecfun(tgtfun(y2))
        y[:] = np.concatenate([y1, y2])

    def MM1fun(self, x, y, tgtfun, reshapefun, vecfun, n, Bfun):
        x1 = tgtfun(reshapefun(x[:n]))
        x2 = tgtfun(reshapefun(x[n:]))
        y1 = -Bfun(x2)
        y2 = -Bfun(x1)
        y1 = vecfun(tgtfun(y1))
        y2 = vecfun(tgtfun(y2))
        y[:] = np.concatenate([y1, y2])

    def TRSgep_matrixfree(self, A, a, B, Del, x, manifold, tolresid, tolhardcase=1e-4, exit_warning_triggered=False):
        """
        Solves the trust-region subproblem by a generalized eigenproblem without iterations.

        minimize (x^T A x) / 2 + a^T x
        subject to x^T B x <= Del^2

        Parameters:
            A (LinearOperator): Symmetric operator.
            a (tangent vector): dimx1 vector.
            B (LinearOperator): Symmetric positive definite operator.
            Del (float): Radius constraint.
            x: current point
            manifold: manifold
            tolresid (float): Tolerance for the residual.
            tolhardcase (float): Tolerance for the hard case.
            exit_warning_triggered (bool): Whether to exit when a warning is triggered.

        Returns:
            tgtv (ndarray): Solution vector.
            lam1 (float): Lagrange multiplier.
            type (str): Information about the solution.
        """

        # Construct the block matrix MM1
        n = len(vectorizefun(manifold, x, manifold.zero_vector(x)))
        dim = manifold.dim
        tgtfun = lambda v: manifold.to_tangent_space(x, v)
        reshapefun = lambda v: tgtvecshapefun(manifold, x, v)
        vecfun = lambda tanvec: vectorizefun(manifold, x, tanvec)

        # Possible interior solution
        Aoperator = scipy.sparse.linalg.LinearOperator((n, n), matvec=lambda dir: vecfun(A(tgtfun(reshapefun(dir)))))
        avec = vecfun(tgtfun(a))

        p1 = self.solve_linear_equations(Aoperator, -avec, tolhardcase, exit_warning_triggered)
        if p1 is None:
            return None, None, "solve failed"
        p1 = np.real(p1)
        p1 = tgtfun(reshapefun(p1))
        is_p1_available = True
        if manifold.norm(x, A(p1) + a) / manifold.norm(x, a) < 1e-5:
            if manifold.inner_product(x, p1, B(p1)) >= Del**2:  # outside of the trust region
                is_p1_available = False  # ineligible
        else:  # numerically incorrect
            is_p1_available = False

        lambdaMM0 = lambda x, y: self.MM0fun(x, y, tgtfun, reshapefun, vecfun, n, A, B, a, Del, manifold)
        lambdaMM1 = lambda x, y: self.MM1fun(x, y, tgtfun, reshapefun, vecfun, n, B)

        k, xr = self.PETSc_solve_GEPTangentSpace(2*n, 2*dim, 1, lambdaMM0, lambdaMM1, tol=tolresid, eigval_type="LARGEST_REAL")
        lam1 = k[0]
        V = xr[0].real
        v = V[:n]  # extract solution component

        tgtv = tgtfun(reshapefun(v))
        normv = np.sqrt(manifold.inner_product(x,  B(tgtv), tgtv))
        tgtv = tgtv / normv * Del  # in the easy case, this naive normalization improves accuracy
        if manifold.inner_product(x, tgtv, a) > 0:
            tgtv = -tgtv  # take correct sign
        type = "boundary"

        if normv < tolhardcase:  # enter hard case
            x1 = np.real(tgtfun(reshapefun(V[n:])))
            Pvect = x1  # first try only k=1, almost always enough
            alpha1 = copy.deepcopy(lam1)
            Alam1B = lambda vec: vecfun(A(tgtfun(reshapefun(vec)))) + lam1 * vecfun(B(tgtfun(reshapefun(vec))))
            # Alam1B = lambda vec: vecfun(tgtfun(A(tgtfun(reshapefun(vec))))) + lam1 * vecfun(tgtfun(B(tgtfun(reshapefun(vec)))))
            BPvecti = B(Pvect)
            alpha1Bpvecti2= lambda vec: alpha1 * manifold.inner_product(x, BPvecti, tgtfun(reshapefun(vec))) * BPvecti
            H = scipy.sparse.linalg.LinearOperator((n, n), matvec=lambda vec: Alam1B(vec) + alpha1Bpvecti2(vec))

            x2 = self.solve_linear_equations(H, -avec, tolhardcase, exit_warning_triggered)
            if x2 is None:
                return None, None, "solve failed"
            x2 = np.real(tgtfun(reshapefun(x2)))
            type = "hardcase_1"

            # Residual check for hard case refinement
            if manifold.norm(x, Alam1B(x2) + a) / manifold.norm(x, a) > tolhardcase:
                maxii = min(dim, 9)
                lambdaA = lambda x, y: basefun(x, y, A, tgtfun, reshapefun, vecfun)
                lambdaB = lambda x, y: basefun(x, y, B, tgtfun, reshapefun, vecfun)
                _, Pvects = self.PETSc_solve_GEPTangentSpace(n, dim, maxii, lambdaA, lambdaB, tol=tolresid, eigval_type="SMALLEST_REAL")
                for ii in [3, 6, 9]:
                    if maxii < ii:
                        break
                    type = f"hardcase_{ii}"
                    Pvectii = Pvects[:ii]
                    BPvects = [B(tgtfun(reshapefun(Pvect))) for Pvect in Pvectii]
                    alpha1Bpvecti2= lambda vec: alpha1 * np.sum([manifold.inner_product(x, BPvect, tgtfun(reshapefun(vec))) * BPvect for BPvect in BPvects], axis=0)
                    H = scipy.sparse.linalg.LinearOperator((n, n), matvec=lambda vec: Alam1B(vec) + alpha1Bpvecti2(vec))
                    x2 = self.solve_linear_equations(H, -avec, tolhardcase, exit_warning_triggered)

                    if x2 is None:
                        return None, None, "solve failed"
                    x2 = np.real(tgtfun(reshapefun(x2)))
                    if manifold.norm(x, Alam1B(x2) + a) / manifold.norm(x, a) < tolhardcase:
                        break
            Bx = B(x1)
            Bx2 = B(x2)
            aa = manifold.inner_product(x, x1, Bx)
            bb = 2 * manifold.inner_product(x, x2, Bx)
            cc = manifold.inner_product(x, x2, Bx2) - Del**2
            alp = (-bb + np.sqrt(bb**2 - 4 * aa * cc)) / (2 * aa)  #norm(x2+alp*x)-Delta
            tgtv = x2 + alp * x1

        # Choose between interior and boundary solution
        if is_p1_available:
            p1objval = manifold.inner_product(x, 0.5 * A(p1) + a, p1)
            tgtvobjval = manifold.inner_product(x, 0.5 * A(tgtv) + a, tgtv)
            if p1objval < tgtvobjval:
                tgtv = p1
                lam1 = 0
                type = "interior"
        return tgtv, lam1, type

    def compute_Cauchy_step(self, xCur, TR_radius, TRS_quadratic, TRS_linear, manifold):
        inner_product_TRSquadlin_lin = manifold.inner_product(xCur, TRS_quadratic(TRS_linear), TRS_linear)
        norm_TRS_linear = manifold.norm(xCur, TRS_linear)
        if inner_product_TRSquadlin_lin <= 0:
            tau = TR_radius / norm_TRS_linear
            type = "Cauchy_boundary"
        else:
            tau = min((norm_TRS_linear**2)/inner_product_TRSquadlin_lin, (TR_radius / norm_TRS_linear))
            type = "Cauchy_interior" if tau < (TR_radius / norm_TRS_linear) else "Cauchy_boundary"
        dxCur = - tau * TRS_linear
        lam1 = None
        return dxCur, lam1, type

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

    def inner_iteration(self, problem, outer_iteration, outer_start_time, x_initial, y_initial, mu, initial_TR_radius, inner_option):
        # Set parameters
        stopping_criterion_Lagrangian = inner_option["stopping_criterion_Lagrangian"]
        stopping_criterion_complementarity = inner_option["stopping_criterion_complementarity"]
        second_order_stationarity = self.option["second_order_stationarity"]
        if second_order_stationarity:
            stopping_criterion_second_order = inner_option["stopping_criterion_second_order"]
        TRS_solver = self.option["TRS_solver"]
        gamma = self.option["gamma"]
        maximal_TR_radius = self.option["maximal_TR_radius"]
        inner_maxiter = self.option["inner_maxiter"]
        inner_maxtime = self.option["inner_maxtime"]
        TRS_tolresid = self.option["TRS_tolresid"]
        TRS_tolhardcase = self.option["TRS_tolhardcase"]
        rho = self.option["rho"]
        const_left = self.option["const_left"]
        const_right = self.option["const_right"]
        exit_warning_triggered = self.option["exit_warning_triggered"]
        checkTRSoptimality = self.option["checkTRSoptimality"]
        verbosity = self.option["verbosity"]
        basisfun = self.option["basisfun"]
        save_inner_iteration =  self.option["save_inner_iteration"]
        manviofun = self.option["manviofun"]
        callbackfun = self.option["callbackfun"]

        # Set initial point
        xCur = x_initial
        yCur = y_initial
        inner_xPrev = copy.deepcopy(xCur)
        TR_radius = initial_TR_radius
        inner_iteration = 0

        # Set function components
        costfun = problem.cost
        ineqconstraints = problem.ineqconstraints_all
        manifold = problem.manifold
        gradcostfun = problem.riemannian_gradient
        gradineqconstraints = problem.ineqconstraints_riemannian_gradient_all
        hesscostfun = problem.riemannian_hessian
        hessineqconstraints = problem.ineqconstraints_riemannian_hessian_all

        # Set vector-valued functions
        costineqconstvecfun = lambda x: np.array([-ineqfun(x) for ineqfun in ineqconstraints])
        gradineqconstvecfun = lambda x: [-grad(x) for grad in gradineqconstraints]
        hessineqconstvecfun = lambda x, dx: [-hess(x, dx) for hess in hessineqconstraints]

        # Define functions
        def gradLagrangefun(x, y, gradcostx=None, gradineqconstvecx=None):
            if gradcostx is None:
                gradcostx = gradcostfun(x)
            if gradineqconstvecx is None:
                gradineqconstvecx = gradineqconstvecfun(x)
            vec = gradcostx
            for i in range(len(y)):
                vec = vec - y[i] * gradineqconstvecx[i]
            return vec

        def hessLagrangefun(x, y, dx):
            vec = hesscostfun(x, dx)
            hessineqconstvec = hessineqconstvecfun(x, dx)
            for i in range(len(y)):
                vec = vec + y[i] * hessineqconstvec[i]
            return vec
        
        def Gxfun(x, v, gradineqconstvecx=None):
            if gradineqconstvecx is None:
                gradineqconstvecx = gradineqconstvecfun(x)
            vec = manifold.zero_vector(x)
            for idx in range(len(gradineqconstvecx)):
                vec = vec + v[idx] * gradineqconstvecx[idx]
            return vec

        def Gxajfun(x, dx, gradineqconstvecx=None):
            if gradineqconstvecx is None:
                gradineqconstvecx = gradineqconstvecfun(x)
            return np.array([manifold.inner_product(x, gradineq, dx) for gradineq in gradineqconstvecx])
        
        def logbarrfun(x, mu, costfunx=None, costineqconstvecx=None):
            if costfunx is None:
                costfunx = costfun(x)
            if costineqconstvecx is None:
                costineqconstvecx = costineqconstvecfun(x)
            return costfun(x) - mu * np.sum(np.log(costineqconstvecx))

        inner_start_time = time.time()
        inner_status = "initial"
        while True:
            costxCur = costfun(xCur)
            costineqconstvecxCur = costineqconstvecfun(xCur)
            gradcostfunxCur = gradcostfun(xCur)
            gradineqconstvecxCur = gradineqconstvecfun(xCur)
            normgradLagfun = manifold.norm(xCur, gradLagrangefun(
                xCur,
                yCur,
                gradcostx=gradcostfunxCur,
                gradineqconstvecx=gradineqconstvecxCur))

            # def inner_check_stopping_criteria(self):
            if verbosity > 1:
                costprint = "{:.3e}".format(costfun(xCur))
                KKTresidprint = "{:.3e}".format(normgradLagfun)
                TRradiusprint = "{:.3e}".format(TR_radius)
                if "inner_info" in locals():
                    dxtype = inner_info["dxtype"]
                else:
                    dxtype = "initial"
                print(f"Iter: {outer_iteration}-{inner_iteration}, Cost: {costprint}, KKT resid: {KKTresidprint}, TR: {TRradiusprint}, Dir: {dxtype}, Stat: {inner_status}")
            inner_iteration += 1

            # Set initial inner_info
            inner_info = self.set_initial_inner_info(inner_iteration, TR_radius)

            # Check stopping criteria (time and iteration)
            if inner_maxtime is not None:
                run_time = time.time() - inner_start_time
                if run_time >= inner_maxtime:
                    inner_info["inner_status"] = "max-time-exceeded"
                    return x_initial, y_initial, initial_TR_radius, inner_info, x_initial
            if inner_maxiter is not None:
                if inner_iteration >= inner_maxiter:
                    inner_info["inner_status"] = "max-iter-exceeded"
                    return x_initial, y_initial, initial_TR_radius, inner_info, x_initial

            # Set functions at xCur
            GxCur = lambda v: Gxfun(xCur, v, gradineqconstvecx=gradineqconstvecxCur)
            GxajCur = lambda dx: Gxajfun(xCur, dx, gradineqconstvecx=gradineqconstvecxCur)
            HwCur = lambda dx: hessLagrangefun(xCur, yCur, dx) + GxCur((yCur * GxajCur(dx)) / costineqconstvecxCur)
            cxCur = gradcostfunxCur -  GxCur(mu / costineqconstvecxCur)

            # Compute the step
            if TRS_solver == 'Cauchy':
                dxCur, lam1, type = self.compute_Cauchy_step(xCur, TR_radius, HwCur, cxCur, manifold)
                inner_info["dxtype"] = type
            elif TRS_solver == 'Exact_RepMat':
                xdim = manifold.dim
                basisxCur = basisfun(manifold, xCur)
                HwCurmatrix = selfadj_operator2matrix(manifold, xCur, HwCur, basisxCur)
                cxCurvector = np.empty(xdim)
                for i in range(xdim):
                    cxCurvector[i] = manifold.inner_product(xCur, cxCur, basisxCur[i])
                coeff, lam1, type = TRSgep(HwCurmatrix, cxCurvector, np.eye(xdim), TR_radius, TRS_tolhardcase, exit_warning_triggered)
                inner_info["dxtype"] = type
                if exit_warning_triggered and coeff is None:
                    inner_info["inner_status"] = "TRS-failed"
                    return x_initial, y_initial, initial_TR_radius, inner_info, x_initial
                dxCur = manifold.zero_vector(xCur)
                for i in range(xdim):
                    dxCur = dxCur + coeff[i] * basisxCur[i]
            elif TRS_solver == 'Exact_Operator':
                idfun = lambda vec: vec
                dxCur, lam1, type = self.TRSgep_matrixfree(HwCur, cxCur, idfun, TR_radius, xCur, manifold, TRS_tolresid, TRS_tolhardcase, exit_warning_triggered)
                inner_info["dxtype"] = type
                if exit_warning_triggered and dxCur is None:
                    inner_info["inner_status"] = "TRS-failed"
                    return x_initial, y_initial, initial_TR_radius, inner_info, x_initial
            else:
                raise ValueError(f"TRS_solver {TRS_solver} is not supported.")

            # Check TRS optimality
            if checkTRSoptimality:
                self.check_TRS_optimality(xCur, TR_radius, dxCur, lam1, HwCur, cxCur, manifold)

            # Update x and y
            dyCur = - yCur + mu * (1 / costineqconstvecxCur) - yCur * GxajCur(dxCur) / costineqconstvecxCur
            xNew = manifold.retraction(xCur, dxCur)
            yNew = yCur + dyCur
            costxNew = costfun(xNew)
            costineqconstvecxNew = costineqconstvecfun(xNew)

            # Check stopping criteria
            xfeasi_criterion = np.all(costineqconstvecxNew > 0)
            yfeasi_criterion = np.all(yNew > 0)
            normgradLagfun = manifold.norm(xNew, gradLagrangefun(xNew, yNew))
            normgradLagfun_criterion = normgradLagfun <= stopping_criterion_Lagrangian
            complementarity = np.linalg.norm(yNew * costineqconstvecxNew - mu)
            complementary_criterion = complementarity <= stopping_criterion_complementarity
            mineigvalHwNew = None
            mineigval_criterion = True
            if second_order_stationarity:
                gradineqconstvecxNew = gradineqconstvecfun(xNew)
                GxNew = lambda v: Gxfun(xNew, v, gradineqconstvecx=gradineqconstvecxNew)
                GxajNew = lambda dx: Gxajfun(xNew, dx, gradineqconstvecx=gradineqconstvecxNew)
                HwNew = lambda dx: hessLagrangefun(xNew, yNew, dx) + GxNew((yNew * GxajNew(dx)) / costineqconstvecxNew)
                if TRS_solver == 'Exact_RepMat':
                    basisxNew = basisfun(manifold, xNew)
                    HwNewmatrix = selfadj_operator2matrix(manifold, xNew, HwNew, basisxNew)
                    mineigvalHwNew = scipy.sparse.linalg.eigsh(HwNewmatrix, k=1, which='SA', return_eigenvectors=False)[0]
                    mineigval_criterion = True if mineigvalHwNew >= -stopping_criterion_second_order else False
                elif TRS_solver == 'Exact_Operator':
                    tgtfun = lambda v: manifold.to_tangent_space(xNew, v)
                    reshapefun = lambda v: tgtvecshapefun(manifold, xNew, v)
                    vecfun = lambda tanvec: vectorizefun(manifold, xNew, tanvec)
                    n = len(vecfun(manifold.zero_vector(xNew)))
                    lambdaHwNew = lambda x, y: basefun(x, y, HwNew, tgtfun, reshapefun, vecfun)
                    dim = manifold.dim
                    k, _ = self.PETSc_solve_GEPTangentSpace(n, dim, 1, lambdaHwNew, tol=TRS_tolresid, eigval_type="SMALLEST_REAL")
                    mineigvalHwNew = k[0].real
                    mineigval_criterion = True if mineigvalHwNew >= -stopping_criterion_second_order else False
                else:
                    raise ValueError(f"TRS_solver {TRS_solver} is not supported.")

            # Set inner_info
            inner_info["normdx"] = manifold.norm(xCur, dxCur)
            inner_info["minxfeasi"] = min(costineqconstvecxNew)
            inner_info["minyfeasi"] = min(yNew)
            # inner_info["normgradLagfun"] = normgradLagfun
            inner_info["compl"] = complementarity
            inner_info["mineigvalHw"] = mineigvalHwNew

            # Return if all stopping criteria are satisfied
            if xfeasi_criterion and yfeasi_criterion and normgradLagfun_criterion and complementary_criterion and mineigval_criterion:
                inner_status = "converged"
                inner_info["inner_status"] = inner_status
                return xNew, yNew, TR_radius, inner_info, inner_xPrev

            # 
            normdxCur = manifold.norm(xCur, dxCur)
            if not xfeasi_criterion:
                inner_status = "primal_infeasible"
                inner_info["inner_status"] = inner_status
                TR_radius = gamma * normdxCur
                if save_inner_iteration:
                    eval_log = evaluation(problem, inner_xPrev, xCur, yCur, [], manviofun, callbackfun)
                    solver_log = self.solver_status(
                            yCur,
                            mu,
                            save_inner_iteration,
                            inner_info = inner_info,
                            )
                    self.add_log(outer_iteration, outer_start_time, eval_log, solver_log)
                continue

            ared = logbarrfun(xCur, mu, costfunx=costxCur, costineqconstvecx=costineqconstvecxCur) - logbarrfun(xNew, mu, costfunx=costxNew, costineqconstvecx=costineqconstvecxNew)
            pred = 0 - 0.5 * manifold.inner_product(xCur, HwCur(dxCur), dxCur) - manifold.inner_product(xCur, cxCur, dxCur)
            # inner_info["ared"] = ared
            # inner_info["pred"] = pred
            inner_info["ared/pred"] = ared / pred
            if ared < 0.25 * pred:
                inner_info["radius_update"] = "reduced"
                TR_radius = 0.25 * TR_radius
            elif ared >= 0.75 * pred and np.abs(normdxCur - TR_radius) <= 1e-15:
                inner_info["radius_update"] = "expanded"
                TR_radius = min(2 * TR_radius, maximal_TR_radius)
            else:
                inner_info["radius_update"] = "unchanged"
                TR_radius = TR_radius
            if ared > rho * pred:
                inner_status = "successful"
                inner_info["inner_status"] = inner_status
                xCur = copy.deepcopy(xNew)
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
                inner_status = "unsuccessful"
                inner_info["inner_status"] = inner_status
                xCur = xCur
                yCur = yCur

            if save_inner_iteration:
                eval_log = evaluation(problem, inner_xPrev, xCur, yCur, [], manviofun, callbackfun)
                solver_log = self.solver_status(
                        yCur,
                        mu,
                        save_inner_iteration,
                        inner_info = inner_info,
                        )
                self.add_log(outer_iteration, outer_start_time, eval_log, solver_log)

            inner_xPrev = copy.deepcopy(xCur)

    # Running an experiment
    def run(self, problem):
        if problem.has_eqconstraints:
            warnings.warn("Equality constraints detecred. Currently, RIPTRM does not support equality constraints and will completely ignore them.", Warning)

        # Set initial points
        option = self.option
        costfun = problem.cost
        xCur = problem.initialpoint
        yCur = problem.initialineqLagmult
        muCur = option["initial_barrier_parameter"]
        xPrev = copy.deepcopy(xCur)
        iteration = 0
        start_time = time.time()

        # Set the trust region radius
        if option['initial_TR_radius'] is None:
            try:
                Delta_bar = problem.manifold.typical_dist
            except NotImplementedError:
                Delta_bar = np.sqrt(problem.manifold.dim)
            initial_TR_radius = Delta_bar / 8
        else:
            initial_TR_radius = option['initial_TR_radius']

        # Set parameters for outer iteration
        second_order_stationarity = option['second_order_stationarity']
        forcing_function_Lagrangian = option['forcing_function_Lagrangian']
        forcing_function_complementarity = option['forcing_function_complementarity']
        forcing_function_second_order = option['forcing_function_second_order']
        min_barrier_parameter = option['min_barrier_parameter']
        barrier_parameter_update_r = option['barrier_parameter_update_r']
        barrier_parameter_update_c = option['barrier_parameter_update_c']
        verbosity = option["verbosity"]
        save_inner_iteration = option["save_inner_iteration"]

        # The first evaluation and logging
        manviofun = option["manviofun"]
        callbackfun = option["callbackfun"]
        eval_log = evaluation(problem, xPrev, xCur, yCur, [], manviofun, callbackfun)
        solver_log = self.solver_status(
                      yCur,
                      muCur,
                      save_inner_iteration,
                      inner_info=None,
                      )
        self.add_log(iteration, start_time, eval_log, solver_log)

        # Preparation for check stopping criteria
        residual = eval_log["residual"]
        tolresid = option["tolresid"]
        residual_criterion = (residual <= tolresid, "KKT residual tolerance reached; current residual=" + str(residual) + " and tolresid=" + str(tolresid))
        stopping_criteria =[residual_criterion]

        # Set the inner option
        inner_option = self.set_inner_option(option)

        # Outer iteration
        while True:
            if verbosity == 1:
                print(f"Outer iteration: {iteration}, Cost: {costfun(xCur)}, KKT residual: {residual}, mu: {muCur}")

            # Check stopping criteria (time, iteration and residual)
            stop, reason = self.check_stoppingcriterion(start_time, iteration, stopping_criteria)
            if stop:
                self.option["stoppingcriterion"] = reason
                if verbosity:
                    print(reason)
                break
            # Count an iteration
            iteration += 1

            # Update the inner option (part 2)
            inner_option = {}
            inner_option["stopping_criterion_Lagrangian"] = forcing_function_Lagrangian(muCur)
            inner_option["stopping_criterion_complementarity"] = forcing_function_complementarity(muCur)
            if second_order_stationarity:
                inner_option["stopping_criterion_second_order"] = forcing_function_second_order(muCur)

            xNew, yNew, _, inner_info, inner_xPrev = self.inner_iteration(problem, iteration, start_time, xCur, yCur, muCur, initial_TR_radius, inner_option)

            # Update variables
            xCur = copy.deepcopy(xNew)
            yCur = copy.deepcopy(yNew)
            muCur = max(min_barrier_parameter, barrier_parameter_update_c * (muCur ** (1 + barrier_parameter_update_r)))
            # Set evalxPrev only for evaluation if necessary
            evalxPrev = inner_xPrev if save_inner_iteration or inner_info["inner_status"] != "converged" else xPrev
            # Evaluation and logging
            eval_log = evaluation(problem, evalxPrev, xCur, yCur, [], manviofun, callbackfun)
            solver_log = self.solver_status(
                      yCur,
                      muCur,
                      save_inner_iteration,
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

@hydra.main(version_base=None, config_path="../NonnegPCA/", config_name="config_simulation")
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
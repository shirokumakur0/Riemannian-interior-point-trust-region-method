import hydra
import numpy as np
import pymanopt

import sys
sys.path.append('./src/base')
import problem_coordinator

sys.path.append('./src/solver')
import utils

# Problem coordinator for stable linear system identification
class Coordinator(problem_coordinator.Coordinator):
    def run(self):
        manifold = self.set_manifold()
        costfun = self.set_costfun()
        ineqconstraints = self.set_ineqconstraints()
        eqconstraints = self.set_eqconstraints()
        initialpoint = self.set_initialpoint()
        initialineqLagmult = self.set_initialineqLagmult()
        initialeqLagmult = self.set_initialeqLagmult()
        problem = utils.NonlinearProblem(
            manifold=manifold,
            cost=costfun,
            ineqconstraints=ineqconstraints,
            eqconstraints=eqconstraints,
            initialpoint=initialpoint,
            initialineqLagmult=initialineqLagmult,
            initialeqLagmult=initialeqLagmult,
        )
        return problem

    # Set sphere manifold as a search space
    def set_manifold(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/dim.csv'
        dim = int(np.loadtxt(path))
        J_manifold = pymanopt.manifolds.SkewSymmetric(dim)
        R_manifold = pymanopt.manifolds.SymmetricPositiveDefinite(dim)
        Q_manifold = pymanopt.manifolds.SymmetricPositiveDefinite(dim)

        mani = pymanopt.manifolds.product.Product([J_manifold,
                                                   R_manifold,
                                                   Q_manifold])
        self.mani = mani
        return mani

    # Set a cost function
    def set_costfun(self):
        mani = self.mani
        is_X_noisy = self.cfg.is_X_noisy
        Xset = self.cfg.Xset
        h = self.cfg.h
        def splitXXP(Xori):
            """
            Splits the input matrix Xori into X and XP.

            Parameters:
                Xori (numpy.ndarray): Input matrix of size (M, N).

            Returns:
                tuple: Two matrices X and XP.
                    - X: The matrix excluding the last column of Xori.
                    - XP: The matrix excluding the first column of Xori (X prime).
            """
            N = Xori.shape[1]
            X = Xori[:, :N-1]
            XP = Xori[:, 1:N]
            return X, XP

        X = None
        XP = None
        for Xindex in Xset:
            dataset_path = self.dataset_path
            if is_X_noisy:
                path = f'{dataset_path}/noisyX_{Xindex}.csv'
            else:
                path = f'{dataset_path}/X_{Xindex}.csv'
            Xori_comp = np.loadtxt(path)
            Xcomp, XPcomp = splitXXP(Xori_comp)
            if X is None:
                X = Xcomp
            else:
                X = np.hstack((X, Xcomp))
            if XP is None:
                XP = XPcomp
            else:
                XP = np.hstack((XP, XPcomp))

        dim = X.shape[0]
        N = X.shape[1]
        @pymanopt.function.autograd(mani)
        def costfun(J, R, Q):
            A = (J - R) @ Q
            Atilde = np.eye(dim) + h * A
            XPminusAtildeX = XP - Atilde @ X
            val= np.trace(XPminusAtildeX @ XPminusAtildeX.T) / N
            return val

        return costfun

    def set_ineqconstraints(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/constset.csv'
        constset = np.loadtxt(path)
        mani = self.mani

        def build_onebox_constfuns(row, col, ls, rs):
            row = int(row)
            col = int(col)
            @pymanopt.function.autograd(mani)
            def onebox_lsconstfun(J, R, Q):
                A = (J-R) @ Q
                return -A[row, col] + ls

            @pymanopt.function.autograd(mani)
            def onebox_rsconstfun(J, R, Q):
                A = (J-R) @ Q
                return A[row, col] - rs
            return onebox_lsconstfun, onebox_rsconstfun

        def build_twobox_constfun(row, col, c, k):
            row = int(row)
            col = int(col)
            sk = k **2
            @pymanopt.function.autograd(mani)
            def twobox_constfun(J, R, Q):
                A = (J-R) @ Q
                return  -(A[row, col] -c)**2 + sk
            return twobox_constfun

        constraint = []
        for idx in range(constset.shape[0]):
            type = constset[idx, 0]
            if type == 0 or type == 1:
                row = constset[idx, 1]
                col = constset[idx, 2]
                ls = constset[idx, 3]
                rs = constset[idx, 4]
                onebox_lsconstfun, onebox_rsconstfun = build_onebox_constfuns(row, col, ls, rs)
                constraint.append(onebox_lsconstfun)
                constraint.append(onebox_rsconstfun)
            elif type == 2:
                row = constset[idx, 1]
                col = constset[idx, 2]
                c = constset[idx, 3]
                k = constset[idx, 4]
                twobox_constfun = build_twobox_constfun(row, col, c, k)
                constraint.append(twobox_constfun)
            else:
                raise ValueError("Invalid constraint type")
        return constraint

    # Set equality constraints, which are empty in this problem
    def set_eqconstraints(self):
        return []

    # Set Lagrange multipliers for inequality constraints
    def set_initialineqLagmult(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/initineqLagmult.csv'
        initineqLagmult = np.loadtxt(path)
        return initineqLagmult

    # Set Lagrange multipliers for equality constraints
    def set_initialeqLagmult(self):
        return np.array([])

    # Set initial points with initial Lagrange multipliers
    def set_initialpoint(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/initJ_{self.cfg.problem_initialpoint}.csv'
        init_J = np.loadtxt(path)
        path = f'{dataset_path}/initR_{self.cfg.problem_initialpoint}.csv'
        init_R = np.loadtxt(path)
        path = f'{dataset_path}/initQ_{self.cfg.problem_initialpoint}.csv'
        init_Q = np.loadtxt(path)
        initialpoint = [init_J, init_R, init_Q]
        return initialpoint

@hydra.main(version_base=None, config_path=".", config_name="config_simulation")
def main(cfg):
    coordinator = Coordinator(cfg)
    problem = coordinator.run()

if __name__=='__main__':
    main()
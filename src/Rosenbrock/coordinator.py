import hydra
import numpy as np
import pymanopt

import sys
sys.path.append('./src/base')
import problem_coordinator

sys.path.append('./src/solver')
import utils

# Problem coordinator for Rosenbrock function minimization
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

    def set_manifold(self):
        n = self.cfg.n
        k = self.cfg.k
        mani = pymanopt.manifolds.Grassmann(n, k)
        self.mani = mani
        return mani

    # Set a cost function
    def set_costfun(self):
        mani = self.mani
        alpha = self.cfg.alpha
        @pymanopt.function.autograd(mani)
        def matrixrosenbrockfun(point):
            vectorized = point.flatten()
            num = len(vectorized)
            val = 0
            for i in range(num-1):
                val = val + alpha * (vectorized[i+1] - vectorized[i])**2 + (1 - vectorized[i])**2
            return val

        return matrixrosenbrockfun

    def set_ineqconstraints(self):

        mani = self.mani
        def indexdecorated_nonnegfun(idx):
            @pymanopt.function.autograd(mani)
            def nonnegfun(point):
                vectorized = point.flatten()
                return -vectorized[idx] - 0.01
            return nonnegfun

        constraint = []
        length = len(mani.random_point().flatten())
        for idx in range(length):
            nonnegfun = indexdecorated_nonnegfun(idx)
            constraint.append(nonnegfun)

        return constraint

    # Set equality constraints, which are empty in this problem
    def set_eqconstraints(self):
        return []

    # Set initial points
    def set_initialpoint(self):
        n = self.cfg.n
        k = self.cfg.k
        initialpoint = np.eye(n)
        initialpoint = initialpoint[:, :k]
        initialpoint = np.abs(initialpoint)
        return initialpoint

    # Set Lagrange multipliers for inequality constraints
    def set_initialineqLagmult(self):
        n = self.cfg.n
        k = self.cfg.k
        initialineqLagmult = np.ones(n*k)
        return initialineqLagmult

    # Set Lagrange multipliers for equality constraints
    def set_initialeqLagmult(self):
        return np.array([])

@hydra.main(version_base=None, config_path=".", config_name="config_simulation")
def main(cfg):
    rosenbrock_coordinator = Coordinator(cfg)
    problem = rosenbrock_coordinator.run()

if __name__=='__main__':
    main()
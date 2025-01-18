import hydra
import numpy as np
import pymanopt

import sys
sys.path.append('./src/base')
import problem_coordinator

sys.path.append('./src/solver')
import utils

# Tutorial for your project:
# 1. Add the class named 'Coordinator', which will be a subclass of 'problem_coordinator.Coordinator'.
#    The name 'Coordinator' must not be changed, as it will be used in other files.
# 2. Implement the following methods in the 'Coordinator' class:
#    -- 'set_costfun(self)': This method should return the cost function (the objective function) for the optimization problem.
#    -- 'set_initialpoint(self)': This method should return the initial x and Lagrange multipliers for the optimization problem.
# 3. Implement other methods as needed such as 'set_searchspace(self)', 'set_ineqconstraints(self)', and 'set_eqconstraints(self)'.
#
# Next, please proceed to the 'config_simulation.yaml' file.
# If you implement your own solvers, you can go through 'src/solver/solver_template.py'.
# Then, proceed to the 'config_simulation.yaml' file.

# Problem coordinator for nonnegative principal component analysis
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
        # dataset_path = self.dataset_path
        # path = f'{dataset_path}/dim.csv'
        # dim = int(np.loadtxt(path))
        n = self.cfg.n
        k = self.cfg.k
        # mani = pymanopt.manifolds.SpecialOrthogonalGroup(dim)
        mani = pymanopt.manifolds.Grassmann(n, k)
        self.mani = mani
        return mani

    # Set a cost function
    def set_costfun(self):
        mani = self.mani
        alpha = self.cfg.alpha
        @pymanopt.function.autograd(mani)
        def matrixrosenbrockfun(point):  # edit here
            vectorized = point.flatten()
            num = len(vectorized)
            val = 0
            for i in range(num-1):
                val = val + alpha * (vectorized[i+1] - vectorized[i])**2 + (1 - vectorized[i])**2
            # print(val)
            return val

        return matrixrosenbrockfun

    def set_ineqconstraints(self):
        # dataset_path = self.dataset_path
        # path = f'{dataset_path}/dim.csv'
        # dim = int(np.loadtxt(path))
        # dim = self.cfg.dim

        mani = self.mani
        # Nonnegativity function as an inequality constraint
        def indexdecorated_nonnegfun(idx):
            @pymanopt.function.autograd(mani)
            def nonnegfun(point):
                vectorized = point.flatten()
                return -vectorized[idx] - 0.01 #0.001
            return nonnegfun

        constraint = []
        length = len(mani.random_point().flatten())
        for idx in range(length):
            nonnegfun = indexdecorated_nonnegfun(idx)
            constraint.append(nonnegfun)

        return constraint

    # Set equality constraints, which are empty in this problem
    def set_eqconstraints(self):
        # return Constraints()
        return []

    # Set initial points
    def set_initialpoint(self):
        # dataset_path = self.dataset_path
        # path = f'{dataset_path}/initx.csv'
        # initialpoint = np.loadtxt(path)
        # initialpoint = self.mani.random_point()
        n = self.cfg.n
        k = self.cfg.k
        initialpoint = np.eye(n)
        # print(initialpoint)
        initialpoint = initialpoint[:, :k]
        # print(initialpoint)
        initialpoint = np.abs(initialpoint)
        return initialpoint

    # Set Lagrange multipliers for inequality constraints
    def set_initialineqLagmult(self):
        # dataset_path = self.dataset_path
        # path = f'{dataset_path}/initineqLagmult.csv'
        # initialineqLagmult = np.loadtxt(path)
        n = self.cfg.n
        k = self.cfg.k
        initialineqLagmult = np.ones(n*k)
        return initialineqLagmult

    # Set Lagrange multipliers for equality constraints
    def set_initialeqLagmult(self):
        return np.array([])

    # Add other set_*** methods as needed
    # def set_***(self):
    # *** = []  # Write here
    # return ***

@hydra.main(version_base=None, config_path=".", config_name="config_simulation")
def main(cfg):
    nonnegPCA_coordinator = Coordinator(cfg)
    problem = nonnegPCA_coordinator.run()

if __name__=='__main__':
    main()
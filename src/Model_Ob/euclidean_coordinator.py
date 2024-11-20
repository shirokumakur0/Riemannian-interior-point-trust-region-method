import hydra
import numpy as np
import pymanopt
from dataclasses import dataclass, field
from typing import Any

import sys
sys.path.append('./src/base')
import problem_coordinator

@dataclass
class Constraints:
    has_constraint: bool = False  # whether having a type of constraints or not
    num_constraint: int = 0  # the number of constraints. must be compatible with the length of 'constraint' list
    constraint: list = field(default_factory=list)

@dataclass
class ManifoldConstraints(Constraints):
    type: list = field(default_factory=list)

@dataclass
class Problem(problem_coordinator.BaseProblem):
    # initialineqLagmult: field(default_factory=list)
    # initialeqLagmult: field(default_factory=list)
    eqconstraints: Constraints
    ineqconstraints: Constraints
    maniconstraints: ManifoldConstraints


# Problem coordinator for nonnegative principal component analysis
class Coordinator(problem_coordinator.Coordinator):

    def run(self):
        costfun = self.set_costfun()
        ineqconstraints = self.set_ineqconstraints()
        eqconstraints = self.set_eqconstraints()
        initialpoint = self.set_initialpoint()
        # initialineqLagmult = self.set_initialineqLagmult()
        # initialeqLagmult = self.set_initialeqLagmult()
        maniconstraints = self.set_maniconstraints()
        problem = Problem(costfun=costfun,
                          ineqconstraints=ineqconstraints,
                          eqconstraints=eqconstraints,
                          maniconstraints=maniconstraints,
                          initialpoint=initialpoint,
                          # initialineqLagmult=initialineqLagmult,
                          # initialeqLagmult=initialeqLagmult
                          )
        return problem

    # Set an inner product of x and the vectorized C as a cost function
    def set_costfun(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/C.csv'
        C = np.loadtxt(path)
        Cvec = C.reshape(-1)

        def build_costfun(Cvec):
            def costfun(X):
                return -2 * X @ Cvec
            return costfun

        fun = build_costfun(Cvec)
        return fun

    # Set nonnegativity of each element as an inequality function
    def set_ineqconstraints(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/rdim.csv'
        rdim = int(np.loadtxt(path))

        dataset_path = self.dataset_path
        path = f'{dataset_path}/cdim.csv'
        cdim = int(np.loadtxt(path))

        def build_nonnegfun(row, col, cdim):
            def nonnegfun(point):
                return -point[row*cdim+col]
            return nonnegfun

        constraint = []
        for row in range(rdim):
            for col in range(cdim):
                nonnegfun = build_nonnegfun(row, col, cdim)
                constraint.append(nonnegfun)

        ineqconstraints = Constraints(has_constraint = True if rdim > 0 and cdim > 0 else False,
                                      num_constraint = rdim * cdim,
                                      constraint = constraint)
        return ineqconstraints

    # Set the unit norm constraint on the product of X and V as an equality function
    def set_eqconstraints(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/V.csv'
        V = np.loadtxt(path)

        dataset_path = self.dataset_path
        path = f'{dataset_path}/rdim.csv'
        rdim = int(np.loadtxt(path))

        dataset_path = self.dataset_path
        path = f'{dataset_path}/cdim.csv'
        cdim = int(np.loadtxt(path))

        def build_eqfun(V, rdim, cdim):
            def eqfun(point):
                pointmat = point.reshape(rdim, cdim)
                vec = pointmat @ V
                val = vec.T @ vec - 1
                return val
            return eqfun

        fun = build_eqfun(V, rdim, cdim)
        constraint = [fun]

        eqconstraints = Constraints(has_constraint = True if rdim > 0 and cdim > 0 else False,
                                      num_constraint = 1 if rdim > 0 and cdim > 0 else 0,
                                      constraint = constraint)
        return eqconstraints

    # Set nonlinear constraints that originally form a manifold
    def set_maniconstraints(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/rdim.csv'
        rdim = int(np.loadtxt(path))

        dataset_path = self.dataset_path
        path = f'{dataset_path}/cdim.csv'
        cdim = int(np.loadtxt(path))

        def build_colnormalfun(col, rdim, cdim):
            def colnormalfun(point):
                pointmat = point.reshape(rdim, cdim)
                x = pointmat[:, col]
                return x @ x - 1
            return colnormalfun

        constraint = []
        for col in range(cdim):
            nonnegfun = build_colnormalfun(col, rdim, cdim)
            constraint.append(nonnegfun)
        type = ['eq'] * cdim

        manifoldconstraints = ManifoldConstraints(has_constraint = True if rdim > 0 else False,
                                      num_constraint = cdim,
                                      constraint = constraint,
                                      type = type)
        return manifoldconstraints

    # Set initial points with initial Lagrange multipliers
    def set_initialpoint(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/C.csv'
        C = np.loadtxt(path)
        # Projection of C onto the feasible region
        U, _, Vh = np.linalg.svd(C, full_matrices=False)
        # print("C",C)
        # print(U.shape, s.shape, Vh.shape)
        initialpoint = U @ Vh
        # print("initialpoint",initialpoint)
        return initialpoint

    # # Set Lagrange multipliers for inequality constraints
    # def set_initialineqLagmult(self):
    #     dataset_path = self.dataset_path
    #     path = f'{dataset_path}/initineqLagmult.csv'
    #     initialineqLagmult = np.loadtxt(path)
    #     initialineqLagmult = np.array(initialineqLagmult)
    #     return initialineqLagmult

    # # Set Lagrange multipliers for equality constraints
    # def set_initialeqLagmult(self):
    #     dataset_path = self.dataset_path
    #     path = f'{dataset_path}/initeqLagmult.csv'
    #     initialeqLagmult = np.loadtxt(path)
    #     initialeqLagmult = np.array([initialeqLagmult])
    #     return initialeqLagmult

@hydra.main(version_base=None, config_path=".", config_name="config_simulation")
def main(cfg):
    model_Ob_coordinator = Coordinator(cfg)
    problem = model_Ob_coordinator.run()

if __name__=='__main__':
    main()
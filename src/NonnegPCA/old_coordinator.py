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
    # costfun: Any
    # initialpoint: Any
    initialineqLagmult: field(default_factory=list)
    initialeqLagmult: field(default_factory=list)
    searchspace: Any
    eqconstraints: Constraints
    ineqconstraints: Constraints
    maniconstraints: ManifoldConstraints

# Problem coordinator for nonnegative principal component analysis
class Coordinator(problem_coordinator.Coordinator):

    def run(self):
        searchspace = self.set_searchspace()
        costfun = self.set_costfun()
        ineqconstraints = self.set_ineqconstraints()
        eqconstraints = self.set_eqconstraints()
        initialpoint = self.set_initialpoint()
        initialineqLagmult = self.set_initialineqLagmult()
        initialeqLagmult = self.set_initialeqLagmult()
        maniconstraints = self.set_maniconstraints()
        problem = Problem(searchspace=searchspace,
                          costfun=costfun,
                          ineqconstraints=ineqconstraints,
                          eqconstraints=eqconstraints,
                          initialpoint=initialpoint,
                          initialineqLagmult=initialineqLagmult,
                          initialeqLagmult=initialeqLagmult,
                          maniconstraints=maniconstraints
                          )
        return problem

    # Set sphere manifold as a search space
    def set_searchspace(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/dim.csv'
        dim = int(np.loadtxt(path))
        mani = pymanopt.manifolds.sphere.Sphere(dim)
        self.mani = mani
        return mani

    # Set a quadratic form with Z as a cost function
    def set_costfun(self):
        mani = self.mani
        dataset_path = self.dataset_path
        path = f'{dataset_path}/Z.csv'
        Z = np.loadtxt(path)

        @pymanopt.function.autograd(mani)
        def costfun(point):
            return - point @ Z @ point

        return costfun

    # Set nonnegativity of each element as an inequality function
    def set_ineqconstraints(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/dim.csv'
        dim = int(np.loadtxt(path))
        
        mani = self.mani
        # Nonnegativity function as an inequality constraint
        def indexdecorated_nonnegfun(idx):
            @pymanopt.function.autograd(mani)
            def nonnegfun(point):
                return -point[idx]
            return nonnegfun

        constraint = []
        for idx in range(dim):
            nonnegfun = indexdecorated_nonnegfun(idx)
            constraint.append(nonnegfun)

        ineqconstraints = Constraints(has_constraint = True if dim > 0 else False,
                                      num_constraint = dim,
                                      constraint = constraint)
        return ineqconstraints

    # Set equality constraints, which are empty in this problem
    def set_eqconstraints(self):
        return Constraints()

    # Set initial points
    def set_initialpoint(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/initx_{self.cfg.problem_initialpoint}.csv'
        initialpoint = np.loadtxt(path)
        return initialpoint

    # Set Lagrange multipliers for inequality constraints
    def set_initialineqLagmult(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/initineqLagmult.csv'
        initialineqLagmult = np.loadtxt(path)
        return initialineqLagmult

    # Set Lagrange multipliers for equality constraints
    def set_initialeqLagmult(self):
        return np.array([])

    # Set nonlinear constraints expressing the sphere manifold
    # Used for the Euclidean solvers. Not used for the Riemannian solvers.
    def set_maniconstraints(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/dim.csv'
        dim = int(np.loadtxt(path))
        manifun = lambda point: np.sum(point**2) - 1
        manifoldconstraints = ManifoldConstraints(has_constraint = True if dim > 0 else False,
                                      num_constraint = 1,
                                      constraint = [manifun],
                                      type = ['eq'])
        return manifoldconstraints


@hydra.main(version_base=None, config_path=".", config_name="config_simulation")
def main(cfg):
    nonnegPCA_coordinator = Coordinator(cfg)
    problem = nonnegPCA_coordinator.run()

if __name__=='__main__':
    main()
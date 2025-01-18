import hydra
import numpy as np
import pymanopt
from dataclasses import dataclass, field
from typing import Any

import sys
sys.path.append('./src/base')
import problem_coordinator

sys.path.append('./src/solver')
import utils

# from coordinator import Coordinator

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
        problem = utils.EuclideanNonlinearProblem(
            cost = costfun,
            ineqconstraints = ineqconstraints,
            eqconstraints = eqconstraints,
            initialpoint = initialpoint,
            maniconstraints = maniconstraints
        )
        return problem

    # Set an inner product of x and the vectorized C as a cost function
    def set_costfun(self):
        # dataset_path = self.dataset_path
        # path = f'{dataset_path}/N.csv'
        # N = np.loadtxt(path)

        def costfun(point):
            r = point[0]
            return -r
        return costfun


    # Set nonnegativity of each element as an inequality function
    def set_ineqconstraints(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/N.csv'
        N = int(np.loadtxt(path))
        path = f'{dataset_path}/a.csv'
        a = int(np.loadtxt(path))
        path = f'{dataset_path}/b.csv'
        b = int(np.loadtxt(path))

        ba2 = (b/a)**2
        ba4 = (b/a)**4

        def build_interiorconstfun(n, ba4):
            def interiorconstfun(point):
                r = point[0]
                u = point[n*3+1]
                v = point[n*3+2]
                s = point[n*3+3]
                val = (r**2) - ((1-s)**2)*(ba4 * u**2 + v**2)
                return val
            return interiorconstfun
        constraint = []

        for idx in range(N):
            cstrfun = build_interiorconstfun(idx, ba4)
            constraint.append(cstrfun)

        def build_nonoverlapfun(i, j, ba2):
            def nonoverlapfun(point):
                r = point[0]
                ui = point[i*3+1]
                vi = point[i*3+2]
                si = point[i*3+3]
                uj = point[j*3+1]
                vj = point[j*3+2]
                sj = point[j*3+3]
                val = 4*(r**2) - ((1+(si-1)*ba2)*ui - (1+(sj-1)*ba2)*uj)**2 - (si*vi - sj*vj)**2
                return val
            return nonoverlapfun
        for i in range(N):
            for j in range(i+1,N):
                cstrfun = build_nonoverlapfun(i, j, ba2)
                constraint.append(cstrfun)


        def build_s_lowerfun(i):
            def s_lowerfun(point):
                s = point[i*3+3]
                return -s
            return s_lowerfun
        for idx in range(N):
            cstrfun = build_s_lowerfun(idx)
            constraint.append(cstrfun)


        def build_s_upperfun(i):
            def s_upperfun(point):
                s = point[i*3+3]
                return s-1
            return s_upperfun
        for idx in range(N):
            cstrfun = build_s_upperfun(idx)
            constraint.append(cstrfun)

        def rnonnegfun(point):
            r = point[0]
            return -r
        constraint.append(rnonnegfun)

        return constraint

    # Set the unit norm constraint on the product of X and V as an equality function
    def set_eqconstraints(self):
        return []

    # Set nonlinear constraints that originally form a manifold
    def set_maniconstraints(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/N.csv'
        N = int(np.loadtxt(path))

        def build_spheremanifun(i):
            def spheremanifun(point):
                u = point[i*3+1]
                v = point[i*3+2]
                return u**2 + v**2 - 1
            return spheremanifun

        constraint = []
        for i in range(N):
            fun = build_spheremanifun(i)
            constraint.append(fun)
        type = ['eq'] * N

        manifoldconstraints = utils.ManifoldConstraints(constraints = constraint, type = type)
        return manifoldconstraints

    # Set initial points with initial Lagrange multipliers
    def set_initialpoint(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/initUV_{self.cfg.problem_initialpoint}.csv'
        initUV = np.loadtxt(path)
        path = f'{dataset_path}/r_scale.csv'
        r_scale = np.loadtxt(path)
        path = f'{dataset_path}/inits_{self.cfg.problem_initialpoint}.csv'
        s = np.loadtxt(path)

        path = f'{dataset_path}/N.csv'
        N = int(np.loadtxt(path))
        path = f'{dataset_path}/a.csv'
        a = int(np.loadtxt(path))
        path = f'{dataset_path}/b.csv'
        b = int(np.loadtxt(path))

        ba2 = (b/a)**2
        ba4 = (b/a)**4

        minval = np.inf
        for i in range(N):
            uvi = initUV[:,i]
            ui = uvi[0]
            vi = uvi[1]
            si = s[i]
            val = ((1-si)**2)*(ba4 * ui**2 + vi**2)
            minval = min(minval, val)
        for i in range(N):
            for j in range(i+1,N):
                uvi = initUV[:,i]
                ui = uvi[0]
                vi = uvi[1]
                si = s[i]
                uvj = initUV[:,j]
                uj = uvj[0]
                vj = uvj[1]
                sj = s[j]
                val = ((1+(si-1)*ba2)*ui - (1+(sj-1)*ba2)*uj)**2 + (si*vi - sj*vj)**2
                minval = min(minval, val)
        assert minval > 0, f"error: minval = {minval}"
        initUVs = np.vstack([initUV, s]).T
        vecinitUVs = initUVs.flatten()
        r = np.array([minval * r_scale])
        initialpoint = np.concatenate([r, vecinitUVs])
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
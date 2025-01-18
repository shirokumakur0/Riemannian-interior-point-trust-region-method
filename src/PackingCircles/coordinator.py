import hydra
import numpy as np
from dataclasses import dataclass, field
import pymanopt
from typing import Any

import sys
sys.path.append('./src/base')
import problem_coordinator

sys.path.append('./src/solver')
import utils

# @dataclass
# class Constraints:
#     has_constraint: bool = False  # whether having a type of constraints or not
#     num_constraint: int = 0  # the number of constraints. must be compatible with the length of 'constraint' list
#     constraint: list = field(default_factory=list)

# @dataclass
# class ManifoldConstraints(Constraints):
#     type: list = field(default_factory=list)

# @dataclass
# class Problem(problem_coordinator.BaseProblem):
#     # costfun: Any
#     # initialpoint: Any
#     initialineqLagmult: field(default_factory=list)
#     initialeqLagmult: field(default_factory=list)
#     searchspace: Any
#     eqconstraints: Constraints
#     ineqconstraints: Constraints
#     # maniconstraints: ManifoldConstraints

class Coordinator(problem_coordinator.Coordinator):
    def run(self):
        manifold = self.set_manifold()
        costfun = self.set_costfun()
        ineqconstraints = self.set_ineqconstraints()
        eqconstraints = self.set_eqconstraints()
        initialpoint = self.set_initialpoint()
        initialineqLagmult = self.set_initialineqLagmult()
        initialeqLagmult = self.set_initialeqLagmult()
        # maniconstraints = self.set_maniconstraints()
        problem = utils.NonlinearProblem(
            manifold=manifold,
            cost=costfun,
            ineqconstraints=ineqconstraints,
            eqconstraints=eqconstraints,
            initialpoint=initialpoint,
            initialineqLagmult=initialineqLagmult,
            initialeqLagmult=initialeqLagmult,
        )
        # searchspace = self.set_searchspace()
        # costfun = self.set_costfun()
        # ineqconstraints = self.set_ineqconstraints()
        # eqconstraints = self.set_eqconstraints()
        # initialpoint = self.set_initialpoint()
        # initialineqLagmult = self.set_initialineqLagmult()
        # initialeqLagmult = self.set_initialeqLagmult()
        # # maniconstraints = self.set_maniconstraints()
        # problem = Problem(searchspace=searchspace,
        #                   costfun=costfun,
        #                   ineqconstraints=ineqconstraints,
        #                   eqconstraints=eqconstraints,
        #                   initialpoint=initialpoint,
        #                   initialineqLagmult=initialineqLagmult,
        #                   initialeqLagmult=initialeqLagmult,
        #                 #   maniconstraints=maniconstraints
        #                   )
        # # problem.costfun = problem._wrap_function(problem.costfun)
        
        return problem

    # Set sphere manifold as a search space
    def set_manifold(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/N.csv'
        N = int(np.loadtxt(path))
        r_euclidean = pymanopt.manifolds.Euclidean(1)
        oblique_mani = pymanopt.manifolds.Oblique(2,N)
        s_euclidean = pymanopt.manifolds.Euclidean(N)
        
        mani = pymanopt.manifolds.product.Product([r_euclidean,
                                                   oblique_mani,
                                                   s_euclidean])
        self.mani = mani
        return mani


    # Set a cost function
    def set_costfun(self):
        mani = self.mani
        pymanopt_problem = pymanopt.Problem(mani, None)

        @pymanopt.function.autograd(mani)
        def costfun(r, UV, s):
            # print("here", args, len(args))
            # print("point:", r)
            # r, UV, s = args
            return -r[0]
        return costfun
        # return pymanopt_problem._wrap_function(costfun)

    # Set nonnegativity of each element as an inequality function
    def set_ineqconstraints(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/N.csv'
        N = int(np.loadtxt(path))
        path = f'{dataset_path}/a.csv'
        a = int(np.loadtxt(path))
        path = f'{dataset_path}/b.csv'
        b = int(np.loadtxt(path))

        mani = self.mani
        ba2 = (b/a)**2
        ba4 = (b/a)**4
        # pymanopt_problem = pymanopt.Problem(mani, None)
        
        def indexdecorated_interiorconstfun(idx, ba4):
            @pymanopt.function.autograd(mani)
            def interiorconstfun(r, UV, s):
                # r, UV, s = args
                uvi = UV[:,idx]
                ui = uvi[0]
                vi = uvi[1]
                si = s[idx]
                val = (r[0]**2) - ((1-si)**2)*(ba4 * ui**2 + vi**2)
                return val
            return interiorconstfun
            # return pymanopt_problem._wrap_function(interiorconstfun)
        constraint = []
        for idx in range(N):
            cstrfun = indexdecorated_interiorconstfun(idx, ba4)
            constraint.append(cstrfun)
            
        def indexdecorated_nonoverlapfun(i, j, ba2):
            @pymanopt.function.autograd(mani)
            def nonoverlapfun(r, UV, s):
                # r, UV, s = args
                uvi = UV[:,i]
                ui = uvi[0]
                vi = uvi[1]
                si = s[i]
                uvj= UV[:,j]
                uj = uvj[0]
                vj = uvj[1]
                sj = s[j]
                val = 4*(r[0]**2) - ((1+(si-1)*ba2)*ui - (1+(sj-1)*ba2)*uj)**2 - (si*vi - sj*vj)**2
                return val
            return nonoverlapfun
            # return pymanopt_problem._wrap_function(nonoverlapfun)
        for i in range(N):
            for j in range(i+1,N):
                cstrfun = indexdecorated_nonoverlapfun(i, j, ba2)
                constraint.append(cstrfun)
        
        def indexdecorated_s_lowerfun(idx):
            @pymanopt.function.autograd(mani)
            def s_lowerfun(r, UV, s):
                # r, UV, s = args
                return -s[idx]
            return s_lowerfun
            # return pymanopt_problem._wrap_function(s_lowerfun)
        for idx in range(N):
            cstrfun = indexdecorated_s_lowerfun(idx)
            constraint.append(cstrfun)


        def indexdecorated_s_upperfun(idx):
            @pymanopt.function.autograd(mani)
            def s_upperfun(r, UV, s):
                # r, UV, s = args
                return s[idx]-1
            return s_upperfun
            # return pymanopt_problem._wrap_function(s_upperfun)
        for idx in range(N):
            cstrfun = indexdecorated_s_upperfun(idx)
            constraint.append(cstrfun)

        @pymanopt.function.autograd(mani)
        def rnonnegfun(r, UV, s):
            # r, UV, s = args
            return -r[0]
        constraint.append(rnonnegfun)
        # constraint.append(pymanopt_problem._wrap_function(rnonnegfun))

        # ineqconstraints = Constraints(has_constraint = True if N > 0 else False,
        #                               num_constraint = N+N*(N-1)//2+2*N+1,
        #                               constraint = constraint)
        # return ineqconstraints
        return constraint

    # Set equality constraints, which are empty in this problem
    def set_eqconstraints(self):
        # return Constraints()
        return []

    # Set Lagrange multipliers for inequality constraints
    def set_initialineqLagmult(self):
        dataset_path = self.dataset_path
        path = f'{dataset_path}/initineqLagmult.csv'
        initialineqLagmult = np.loadtxt(path)
        return initialineqLagmult

    # Set Lagrange multipliers for equality constraints
    def set_initialeqLagmult(self):
        return np.array([])

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
        r = np.array([minval * r_scale])
        initialpoint = [r, initUV, s]
        return initialpoint

@hydra.main(version_base=None, config_path=".", config_name="config_simulation")
def main(cfg):
    nonnegPCA_coordinator = Coordinator(cfg)
    problem = nonnegPCA_coordinator.run()
    print(problem)
if __name__=='__main__':
    main()
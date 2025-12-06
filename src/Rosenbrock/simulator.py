import hydra

import sys
sys.path.append('./src/base')
import base_simulator

import numpy as np

sys.path.append('./src/solver')
import utils

import os
import copy

def identify_active_inequality_constraints(problem, x, threshold=1e-5):
    # Set functions
    ineqconstraints = problem.ineqconstraints_all
    num_ineqconstraints = problem.num_ineqconstraints

    active_ineq_indices = []
    for i in range(num_ineqconstraints):
        if abs(ineqconstraints[i](x)) < threshold:
            active_ineq_indices.append(i)
    return active_ineq_indices

def compute_basis(problem, x, tanvecs, linindtol):
    manifold = problem.manifold
    if len(tanvecs) == 0:
        basis = []
        dim = 0
    else:
        Q, _ = utils.orthogonalize(manifold, x, tanvecs)
        absdiagR = np.abs(np.diag(_))
        independent_indices = np.where(absdiagR > linindtol)[0]
        if independent_indices.size == 0:
            basis = []
        else:
            basis = []
            for idx in independent_indices:
                basis.append(Q[idx])
        dim = len(basis)
        if dim != len(tanvecs):
            print(f"Found {len(tanvecs) - dim} linearly dependent tangent vectors.")
    return basis

def compute_null_basis(problem, x, original_basis):
    manifold = problem.manifold
    manidim = manifold.dim
    originaldim = len(original_basis)
    nulldim = manidim - originaldim
    nullbasis = []
    for _ in range(nulldim):
        v = manifold.random_tangent_vector(x)
        for j in range(originaldim):
            v = v - manifold.inner_product(x, original_basis[j], v) * original_basis[j]
        nullbasis.append(v)
    nullbasis, _ = utils.orthogonalize(manifold, x, nullbasis)
    return nullbasis

def compute_second_order_residual(problem, x, y, z, linindtol=1e-12):
    # Compute the active constraints
    active_ineqconsts_indices = identify_active_inequality_constraints(problem, x)
    active_ineqconsts_gradvec = [problem.ineqconstraints_riemannian_gradient(i)(x) for i in active_ineqconsts_indices]
    eqconsts_gradvecfun = problem.eqconstraints_riemannian_gradient_all
    eqconsts_gradvec = [grad(x) for grad in eqconsts_gradvecfun]
    active_allconsts_gradvec = active_ineqconsts_gradvec + eqconsts_gradvec
    
    # Compute the basis of the subspace spanned by the gradients of active constraints
    Qind = compute_basis(problem, x, active_allconsts_gradvec, linindtol)

    # Compute the null basis of the subspace spanned by the gradients of active constraints
    nullbasis = compute_null_basis(problem, x, Qind)

    def hessLagrangefun(x, dx):
        hesscost = problem.riemannian_hessian
        hessineqconstraints = problem.ineqconstraints_riemannian_hessian_all
        hesseqconstraints = problem.eqconstraints_riemannian_hessian_all
        num_ineqconstraints = problem.num_ineqconstraints
        num_eqconstraints = problem.num_eqconstraints
        vec = hesscost(x, dx)
        for i in range(num_ineqconstraints):
            vec = vec + y[i] * hessineqconstraints[i](x, dx)
        for j in range(num_eqconstraints):
            vec = vec + z[j] * hesseqconstraints[j](x, dx)
        return vec

    manifold = problem.manifold
    hessLagxfun = lambda dx: hessLagrangefun(x, dx)
    hessLagnullspc_mat = utils.selfadj_operator2matrix(manifold, x, hessLagxfun, nullbasis)
    eigvals, _ = np.linalg.eigh(hessLagnullspc_mat)
    if len(eigvals) == 0:
        mineigval = 0
        condnum = None
    else:
        mineigval = np.min(eigvals)
        condnum = np.max(eigvals) / mineigval

    return mineigval, condnum

def callbackfun(problem, x, ineqLagmult, eqLagmult, eval):
    # Compute the 2nd order residual
    second_order_residual, condnum = compute_second_order_residual(problem, x, ineqLagmult, eqLagmult)
    eval["second_order_residual"] = second_order_residual
    eval["condition_number"] = condnum
    return eval

def manviofun(problem, x):
    manvio = 0
    rankp = np.linalg.matrix_rank(x)
    p = problem.manifold._p
    if rankp != p:
        print("Rank deficient")
        manvio = np.inf
    return manvio

class Simulator(base_simulator.Simulator):
    def add_solver_option(self, option):
        option["manviofun"] = manviofun
        option["callbackfun"] = callbackfun
        return option

    def run(self):
        # Make directories if not exist
        os.makedirs(f'intermediate/{self.cfg.problem_name}', exist_ok=True)  # Create {problem_name} folder under 'data' folder
        os.makedirs(f'intermediate/{self.cfg.problem_name}/{self.cfg.problem_instance}/{self.cfg.problem_initialpoint}', exist_ok=True)  # Create {inst} folder under {problem_name} folder
        self.logger.info(f"Running a simulator of class {self.__class__} -- instance: {self.cfg.problem_instance}, initial point: {self.cfg.problem_initialpoint}")

        # Coordinate problem
        problem_coordinator = self.set_coordinator()
        self.logger.info(f"Running a problem coordinator of class {problem_coordinator.__class__}")
        problem = problem_coordinator.run()
        self.logger.info(f"Finished running a problem coordinator of class {problem_coordinator.__class__}")

        # Loop with respect to solver
        solver_name = self.cfg.solver_name
        for name in solver_name:
            solver = self.set_solver(name)  # set solver
            self.logger.info(f"Running a solver of class {solver.__class__}")
            output = solver.run(copy.deepcopy(problem))  # run the experiments

            self.save_output(output.name, output)
            self.logger.info(f"Finished running a solver of class {solver.__class__}")
        self.logger.info(f"Finished running a simulator of class {self.__class__} -- instance: {self.cfg.problem_instance}, initial point: {self.cfg.problem_initialpoint}")

@hydra.main(version_base=None, config_path=".", config_name="config_simulation")
def main(cfg):

    director = Simulator(cfg)
    director.run()

if __name__=='__main__':
    main()
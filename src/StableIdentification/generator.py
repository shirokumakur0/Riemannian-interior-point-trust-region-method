import hydra, math
import numpy as np
import sys
import pymanopt
sys.path.append('./src/base')
import dataset_generator
from scipy.stats import norm
import copy
import sys
sys.path.append('./src/base')

sys.path.append('./src/solver')
import utils
import RALM

class DataGenerator(dataset_generator.Generator):
    def generate(self, data):
        while True:
            try:
                # Set hyperparameters
                N = self.cfg.N
                Xset = self.cfg.Xset
                oneboxratio = self.cfg.oneboxratio
                twoboxratio = self.cfg.twoboxratio
                snr = self.cfg.snr
                dim = self.cfg.dim
                h = self.cfg.h
                scaling = self.cfg.scaling

                data.dim = [[dim]]
                true_J, true_R, true_Q, true_A = self.generate_trueJRQA(dim, scaling)
                constset = self.generate_constraints(dim, true_A, oneboxratio, twoboxratio)

                data.constset = constset
                data.true_J = true_J
                data.true_R = true_R
                data.true_Q = true_Q
                data.true_A = true_A

                for Xindex in Xset:
                    X, noisyX = self.generate_XnoisyX(dim, true_A, h, N, snr)
                    setattr(data, f'X_{Xindex}', X)
                    setattr(data, f'noisyX_{Xindex}', noisyX)

                initialpoints = self.cfg.initialpoints
                for initpt in initialpoints:
                    initJ, initR, initQ, initA = self.generate_initprimaldualpoint(data, scaling)
                    setattr(data, f'initJ_{initpt}', initJ)
                    setattr(data, f'initR_{initpt}', initR)
                    setattr(data, f'initQ_{initpt}', initQ)
                    setattr(data, f'initA_{initpt}', initA)
                break
            except Exception as e:
                print(e)
        return data

    def generate_trueJRQA(self, dim, scaling):
        sqrt_scaling = math.sqrt(scaling)
        J_manifold = pymanopt.manifolds.SkewSymmetric(dim)
        R_manifold = pymanopt.manifolds.SymmetricPositiveDefinite(dim)
        Q_manifold = pymanopt.manifolds.SymmetricPositiveDefinite(dim)
        true_J = sqrt_scaling * J_manifold.random_point()
        true_R = sqrt_scaling * R_manifold.random_point()
        true_Q = sqrt_scaling * Q_manifold.random_point()
        true_A = (true_J - true_R) @ true_Q
        return true_J, true_R, true_Q, true_A

    def generate_constraints(self, dim, true_A, oneboxratio, twoboxratio):
        num_element = true_A.size
        num_onebox = int(num_element * oneboxratio)
        num_twobox = int(num_element * twoboxratio)
        num_const = num_onebox + num_twobox
        rowcolset = np.zeros((num_const, 2), dtype=int)
        constset = []  # 0: 1-box, 1: 2-box-lr, 2: 2-box-ck
        constindices = np.random.permutation(num_element)[:num_const]  # Randomly select indices
        for i in range(num_const):
            cind = constindices[i]
            quo = cind // dim
            rem = cind % dim
            rowcolset[i, 0] = rem
            rowcolset[i, 1] = quo

        const_onebox = rowcolset[:num_onebox]
        const_twobox = rowcolset[num_onebox:]
        for i in range(num_onebox):
            row = const_onebox[i, 0]
            col = const_onebox[i, 1]
            Aval = true_A[row, col]
            absAval = abs(Aval)
            lscoeff = np.random.uniform(low=0.2, high=0.8)  # 0.2
            rscoeff = np.random.uniform(low=0.2, high=0.8)  # 0.2
            ls = Aval - lscoeff * absAval
            rs = Aval + rscoeff * absAval
            constset.append([0, row, col, ls, rs, Aval])

        for i in range(num_twobox):
            row = const_twobox[i, 0]
            col = const_twobox[i, 1]
            Aval = true_A[row, col]
            absAval = abs(Aval)
            ccoeff = np.random.uniform(low=0.2, high=0.8)
            kcoeff = np.random.uniform(low=0.2, high=0.8)
            c = ccoeff * Aval  # 0
            k = c + kcoeff * (Aval - c)  # 0.8 * Aval
            lscoeff = np.random.uniform(low=0.2, high=0.8)  # 0.3
            rscoeff = np.random.uniform(low=0.2, high=0.8)  # 0.3
            ls = - absAval - lscoeff * absAval
            rs = absAval + rscoeff * absAval
            constset.append([1, row, col, ls, rs, Aval])
            constset.append([2, row, col, c, k, Aval])
        constset = np.array(constset)
        self.constset = constset
        return constset

    def awgn(self, signal, snr_dB):
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_dB / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power) * norm.rvs(size=signal.shape)
        return signal + noise

    def generate_XnoisyX(self, dim, true_A, h, N, snr):
        x0 = -1000 + (1000 - (-1000)) * np.random.rand(dim)
        # generate X
        X = np.zeros((dim, N))
        noisyX = np.zeros((dim, N))
        X[:, 0] = x0
        noisyX[:, 0] = self.awgn(x0, snr)
        for i in range(1, N):
            expAh = np.exp(i * h * true_A)
            X[:, i] = expAh @ x0
            noisyX[:, i] = self.awgn(X[:, i], snr)
        X = X / np.linalg.norm(x0)
        noisyX = noisyX / np.linalg.norm(noisyX[:, 0])
        return X, noisyX

    def generate_initprimaldualpoint(self, data, scaling):
        # Set hyperparameters
        interior_scaling = self.cfg.interior_scaling
        manifold = self.set_manifold()
        costfun = self.set_costfun()
        ineqconstraints = self.set_ineqconstraints(interior_scaling)
        eqconstraints = self.set_eqconstraints()
        initialineqLagmult = self.set_initialineqLagmult()
        initialeqLagmult = self.set_initialeqLagmult()

        data.initineqLagmult = initialineqLagmult
        data.initeqLagmult = initialeqLagmult

        original_ineqconstraints = self.set_ineqconstraints(1)


        init_type = self.cfg.init_type
        sqrt_scaling = math.sqrt(scaling)
        if init_type == "interior":  # feasible_ill_conditioned
            loop=0
            while True:
                loop+=1
                if loop > 10:
                    raise ValueError("Cannot find a feasible and interior initial point.")
                random_initialpoint = manifold.random_point()
                random_initialpoint[0] = sqrt_scaling * random_initialpoint[0]
                random_initialpoint[1] = sqrt_scaling * random_initialpoint[1]
                random_initialpoint[2] = sqrt_scaling * random_initialpoint[2]
                problem = utils.NonlinearProblem(
                    manifold=manifold,
                    cost=costfun,
                    ineqconstraints=ineqconstraints,
                    eqconstraints=eqconstraints,
                    initialpoint=random_initialpoint,
                    initialineqLagmult=initialineqLagmult,
                    initialeqLagmult=initialeqLagmult,
                )

                solver_option = self.cfg.solver_option
                option = copy.deepcopy(dict(solver_option["common"]))
                solver = RALM.RALM(option)
                output = solver.run(problem)
                x = output.x
                initJ = x[0]
                initR = x[1]
                initQ = x[2]
                initA = (initJ - initR) @ initQ
                def is_stable(A):
                    """
                    Check if a matrix is stable (all eigenvalues have negative real parts).

                    Parameters:
                        A (numpy.ndarray): Input matrix.

                    Returns:
                        bool: True if A is stable, False otherwise.
                    """
                    eigenvalues = np.linalg.eigvals(A)
                    return np.all(np.real(eigenvalues) < 0)

                def is_interior(J, R, Q, original_ineqconstraints):
                    for constfun in original_ineqconstraints:
                        if constfun(J, R, Q) > 0:
                            return False
                    return True

                if not is_stable(initA):
                    print("Not stable")
                    continue
                if not is_interior(initJ, initR, initQ, original_ineqconstraints):
                    print("Not interior")
                    continue
                break
            print("A stable and interior initial point")
        elif init_type == "random":
            print("initial point randomly generated.")
            random_initialpoint = manifold.random_point()
            random_initialpoint[0] = sqrt_scaling * random_initialpoint[0]
            random_initialpoint[1] = sqrt_scaling * random_initialpoint[1]
            random_initialpoint[2] = sqrt_scaling * random_initialpoint[2]
            initJ = random_initialpoint[0]
            initR = random_initialpoint[1]
            initQ = random_initialpoint[2]
            initA = (initJ - initR) @ initQ
        else:
            raise ValueError("Invalid init_type")
        return initJ, initR, initQ, initA

    # Set sphere manifold as a search space
    def set_manifold(self):
        dim = self.cfg.dim
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
        @pymanopt.function.autograd(mani)
        def costfun(J, R, Q):
            return 0.0
        return costfun

    def set_ineqconstraints(self, interior_scaling):
        constset = self.constset
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
                ls = constset[idx, 3] * interior_scaling
                rs = constset[idx, 4] * interior_scaling
                onebox_lsconstfun, onebox_rsconstfun = build_onebox_constfuns(row, col, ls, rs)
                constraint.append(onebox_lsconstfun)
                constraint.append(onebox_rsconstfun)
            elif type == 2:
                row = constset[idx, 1]
                col = constset[idx, 2]
                c = constset[idx, 3]
                k = constset[idx, 4] * (1 + (1 - interior_scaling))
                twobox_constfun = build_twobox_constfun(row, col, c, k)
                constraint.append(twobox_constfun)
            else:
                raise ValueError("Invalid constraint type")
        self.ineqconstraints = constraint
        return constraint

    # Set equality constraints, which are empty in this problem
    def set_eqconstraints(self):
        return []

    # Set Lagrange multipliers for inequality constraints
    def set_initialineqLagmult(self):
        numineqconst = len(self.ineqconstraints)
        initineqLagmult = np.ones(numineqconst)
        return initineqLagmult

    # Set Lagrange multipliers for equality constraints
    def set_initialeqLagmult(self):
        return np.array([])

@hydra.main(version_base=None, config_path=".", config_name="config_dataset")
def main(cfg):
    generator = DataGenerator(cfg)
    generator.run()

if __name__=='__main__':
    main()
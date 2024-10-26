import hydra, math
import numpy as np
import sys
import pymanopt
import scipy
sys.path.append('./src/base')
import dataset_generator

class XstarCGenerator(dataset_generator.Generator):
    def generate(self, data):
        # Set hyperparameters
        rdim = self.cfg.rdim
        cdim = self.cfg.cdim
        
        t = np.floor(rdim/cdim).astype(int)
        y = rdim - t * cdim
        
        mani = pymanopt.manifolds.sphere.Sphere(t)
        B = np.abs(mani.random_point())
        for _ in range(cdim-2):
            B = scipy.linalg.block_diag(B, np.abs(mani.random_point()))
        mani = pymanopt.manifolds.sphere.Sphere(t+y)
        B = scipy.linalg.block_diag(B, np.abs(mani.random_point()))
        
        B= B.T
        np.random.shuffle(B)
        # print("B", B)
        
        X1 = (B>0) * (1 + np.random.rand(rdim,cdim))
        # print("X1",X1[0])
        # print("X1*X1", (X1*X1)[0])
        # print("sum(X1*X1)", 1 / sum(X1*X1))
        coeff = 1 / np.sqrt(sum(X1 * X1))
        Xstar = coeff * X1

        L = np.random.rand(cdim,cdim)
        L += cdim * np.eye(cdim)
        C = Xstar @ L.T
        # Set dim
        data.rdim = [[rdim]]  # to be compatible with 'save' function
        data.cdim = [[cdim]]  # to be compatible with 'save' function
        data.Xstar =Xstar
        data.C = C
        
        return data
    
class VGenerator(dataset_generator.Generator):
    def generate(self, data):
        # Set hyperparameters
        cdim = self.cfg.cdim
        V = np.ones((cdim, 1))
        V = V / np.linalg.norm(V)
        data.V = V

        return data

# Generator for initial points
class InitialPointGenerator(dataset_generator.Generator):
    def generate(self, data):
        # Set hyperparameters
        initialpoints = self.cfg.initialpoints
        rdim = self.cfg.rdim
        cdim = self.cfg.cdim

        mani = pymanopt.manifolds.oblique.Oblique(rdim, cdim)
        # Generating initial points
        for initpt in initialpoints:
            x0 = mani.random_point()
            setattr(data, f'initx_{initpt}', x0)
        return data

# Generator for initial Lagrange multipliers
class InitialIneqLagMultGenerator(dataset_generator.Generator):
    def generate(self, data):
        # Set hyperparameters
        rdim = self.cfg.rdim
        cdim = self.cfg.cdim

        # Generating :agrange multipliers for inequality constraints
        data.initineqLagmult = np.ones(rdim * cdim)

        return data
    
# Generator for initial Lagrange multipliers
class InitialEqLagMultGenerator(dataset_generator.Generator):
    def generate(self, data):

        # Generating :agrange multipliers for inequality constraints
        data.initeqLagmult = np.ones(1)

        return data

@hydra.main(version_base=None, config_path=".", config_name="config_dataset")
def main(cfg):
    xstarcgenerator = XstarCGenerator(cfg)
    xstarcgenerator.run()
    vgenerator = VGenerator(cfg)
    vgenerator.run()
    initialpointgenerator = InitialPointGenerator(cfg)
    initialpointgenerator.run()
    initialineqlagmultgenerator = InitialIneqLagMultGenerator(cfg)
    initialineqlagmultgenerator.run()
    initialeqlagmultgenerator = InitialEqLagMultGenerator(cfg)
    initialeqlagmultgenerator.run()

if __name__=='__main__':
    main()
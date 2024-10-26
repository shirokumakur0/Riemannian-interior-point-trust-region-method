import hydra, math
import numpy as np
import sys
sys.path.append('./src/base')
import dataset_generator

# Generator for Z in the cost function
class ZGenerator(dataset_generator.Generator):
    def generate(self, data):
        # Set hyperparameters
        dim = self.cfg.dim
        snr = self.cfg.snr
        delta = self.cfg.delta
        samplesize = np.floor(delta*dim)

        # Set dim
        data.dim = [[dim]]  # to be compatible with 'save' function

        # Generate Z
        S = np.random.choice(dim, samplesize.astype(int), replace=False)
        v = np.zeros(dim)
        v[S] = 1 / np.sqrt(samplesize)
        v = np.asmatrix(v)
        Z = np.sqrt(snr) * np.dot(v.T, v)
        Noise = np.random.randn(dim,dim) / np.sqrt(dim)
        for ii in range(dim):
            Noise[ii,ii] = np.random.randn() * 2 / np.sqrt(dim)
        Z = Z + Noise
        data.Z = Z

        return data

# Generator for initial points
class InitialPointGenerator(dataset_generator.Generator):
    def generate(self, data):
        # Set hyperparameters
        initialpoints = self.cfg.initialpoints
        dim = self.cfg.dim

        # Generating initial points
        for initpt in initialpoints:
            x0 = np.random.rand(dim)
            x0 = x0 / np.linalg.norm(x0)
            setattr(data, f'initx_{initpt}', x0)
        return data

# Generator for initial Lagrange multipliers
class InitialIneqLagMultGenerator(dataset_generator.Generator):
    def generate(self, data):
        # Set hyperparameters
        dim = self.cfg.dim

        # Generating :agrange multipliers for inequality constraints
        data.initineqLagmult = np.ones(dim)

        return data

@hydra.main(version_base=None, config_path=".", config_name="config_dataset")
def main(cfg):
    zgenerator = ZGenerator(cfg)
    zgenerator.run()
    initialpointgenerator = InitialPointGenerator(cfg)
    initialpointgenerator.run()
    initialineqLagmultgenerator = InitialIneqLagMultGenerator(cfg)
    initialineqLagmultgenerator.run()

if __name__=='__main__':
    main()
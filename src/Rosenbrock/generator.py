import hydra, math
import numpy as np
import sys
sys.path.append('./src/base')
import dataset_generator

# Tutorial for your project:
# 1. Add classes that will be subclasses of 'dataset_generator.Generator'.
#    Each subclass will represent a specific type of data generator.
# 2. For each class, implement the 'generate(self, data)' method.
#    In this method, generate the required parameters and store them as attributes of 'data' (data.{attribute}).
# 3. At least, you will implement 'InitialPointGenerator', where we generate
#    initial points (x) for the optimization problem.
#
#    Note: The provided code contains example implementations for 'InitialPointGenerator'.
#          You need to customize it based on your specific dataset generation requirements.
#
# 4. Add their instances to the 'def main' of the file and run this file.
#    The data will be generated and stored as expected.
#
# Next, please proceed to the 'coordinator.py' file

# class ZGenerator(dataset_generator.Generator):
#     def generate(self, data):
#         # Set hyperparameters
#         dim = self.cfg.dim
#         snr = self.cfg.snr
#         delta = self.cfg.delta
#         samplesize = np.floor(delta*dim)

#         # Set dim
#         data.dim = [[dim]]  # to be compatible with 'save' function

#         # Generate Z
#         S = np.random.choice(dim, samplesize.astype(int), replace=False)
#         v = np.zeros(dim)
#         v[S] = 1 / np.sqrt(samplesize)
#         v = np.asmatrix(v)
#         Z = np.sqrt(snr) * np.dot(v.T, v)
#         Noise = np.random.randn(dim,dim) / np.sqrt(dim)
#         for ii in range(dim):
#             Noise[ii,ii] = np.random.randn() * 2 / np.sqrt(dim)
#         Z = Z + Noise
#         data.Z = Z

#         return data

# Generator for initial points
class InitialPointGenerator(dataset_generator.Generator):
    def generate(self, data):
        # Set hyperparameters
        initialpoints = self.cfg.initialpoints
        dim = self.cfg.dim
        data.dim = [[dim]]
        x0 = np.eye(dim)
        data.initx = x0
        return data

        # # Generating initial points
        # if self.cfg.initialpoints_type == 'random':
        #     for initpt in initialpoints:
        #         x0 = np.random.rand(dim)
        #         x0 = x0 / np.linalg.norm(x0)
        #         setattr(data, f'initx_{initpt}', x0)
        # elif self.cfg.initialpoints_type == 'feasible':
        #     for initpt in initialpoints:
        #         x0 = np.random.rand(dim)
        #         x0 = x0 / np.linalg.norm(x0)
        #         x0 = np.abs(x0)
        #         setattr(data, f'initx_{initpt}', x0)
        # else:
        #     raise ValueError(f'Initial point type {self.cfg.initialpoints_type} is not supported.')
        # return data

# Generator for initial Lagrange multipliers
class InitialIneqLagMultGenerator(dataset_generator.Generator):
    def generate(self, data):
        # Set hyperparameters
        dim = self.cfg.dim

        # Generating :agrange multipliers for inequality constraints
        data.initineqLagmult = np.ones(dim * dim)

        return data

@hydra.main(version_base=None, config_path=".", config_name="config_dataset")
def main(cfg):
    # zgenerator = ZGenerator(cfg)
    # zgenerator.run()
    initialpointgenerator = InitialPointGenerator(cfg)
    initialpointgenerator.run()
    initialineqLagmultgenerator = InitialIneqLagMultGenerator(cfg)
    initialineqLagmultgenerator.run()

if __name__=='__main__':
    main()


# Generator for initial points
"""
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

@hydra.main(version_base=None, config_path=".", config_name="config_dataset")
def main(cfg):
    initialpointgenerator = InitialPointGenerator(cfg)
    initialpointgenerator.run()

if __name__=='__main__':
    main()
"""
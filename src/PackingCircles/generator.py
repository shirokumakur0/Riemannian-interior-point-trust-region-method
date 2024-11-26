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

# Generator for Z in the cost function
class NabGenerator(dataset_generator.Generator):
    def generate(self, data):
        # Set hyperparameters
        N = self.cfg.N
        a = self.cfg.a
        b = self.cfg.b

        # Set dim
        data.N = [[N]]  # to be compatible with 'save' function
        data.a = [[a]]  # to be compatible with 'save' function
        data.b = [[b]]  # to be compatible with 'save' function

        return data

# Generator for initial points
class InitialPointGenerator(dataset_generator.Generator):
    def generate(self, data):
        # Set hyperparameters
        initialpoints = self.cfg.initialpoints
        N = self.cfg.N

        # Generating initial points
        for initpt in initialpoints:
            UV = np.random.rand(2,N)
            UV = UV / np.linalg.norm(UV, axis=0,keepdims=True)
            setattr(data, f'initUV_{initpt}', UV)
            s = 0.5 * np.ones(N)
            setattr(data, f'inits_{initpt}', s)
        data.r_scale = [[0.9]]

        return data

# Generator for initial Lagrange multipliers
class InitialIneqLagMultGenerator(dataset_generator.Generator):
    def generate(self, data):
        # Set hyperparameters
        N = self.cfg.N

        # Generating :agrange multipliers for inequality constraints
        data.initineqLagmult = np.ones(N+N*(N-1)//2+2*N+1)
        return data

@hydra.main(version_base=None, config_path=".", config_name="config_dataset")
def main(cfg):
    nabgenerator = NabGenerator(cfg)
    nabgenerator.run()
    initialpointgenerator = InitialPointGenerator(cfg)
    initialpointgenerator.run()
    initialineqLagmultgenerator = InitialIneqLagMultGenerator(cfg)
    initialineqLagmultgenerator.run()

if __name__=='__main__':
    main()
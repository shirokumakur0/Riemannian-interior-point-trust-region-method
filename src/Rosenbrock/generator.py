import hydra, math
import numpy as np
import sys
sys.path.append('./src/base')
import dataset_generator

# Generator for initial points
class InitialPointGenerator(dataset_generator.Generator):
    def generate(self, data):
        dim = self.cfg.dim
        data.dim = [[dim]]
        x0 = np.eye(dim)
        data.initx = x0
        return data

# Generator for initial Lagrange multipliers
class InitialIneqLagMultGenerator(dataset_generator.Generator):
    def generate(self, data):
        dim = self.cfg.dim
        data.initineqLagmult = np.ones(dim * dim)
        return data

@hydra.main(version_base=None, config_path=".", config_name="config_dataset")
def main(cfg):
    initialpointgenerator = InitialPointGenerator(cfg)
    initialpointgenerator.run()
    initialineqLagmultgenerator = InitialIneqLagMultGenerator(cfg)
    initialineqLagmultgenerator.run()

if __name__=='__main__':
    main()
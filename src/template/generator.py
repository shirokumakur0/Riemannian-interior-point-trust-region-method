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

@hydra.main(version_base=None, config_path=".", config_name="config_dataset")
def main(cfg):
    initialpointgenerator = InitialPointGenerator(cfg)
    initialpointgenerator.run()

if __name__=='__main__':
    main()
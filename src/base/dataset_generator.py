import hydra, logging, os, csv
import numpy as np
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod

@dataclass  # used as a structure
class Data():
    # Collection of components and initial points
    # that will be generated and saved in Generator()
    pass

class Generator(object, metaclass=ABCMeta):
    def __init__(self, cfg):
        # Assertion
        assert hasattr(cfg, 'problem_name')
        assert hasattr(cfg, 'instance_name')
        assert hasattr(cfg, 'output_path')

        # Set the configuration file
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

    def run(self):
        os.makedirs(f'dataset/{self.cfg.problem_name}', exist_ok=True)  # Create {problem_name} folder under 'data' folder
        os.makedirs(f'dataset/{self.cfg.problem_name}/{self.cfg.instance_name}', exist_ok=True)  # Create {instance_name} folder under {problem_name} folder
        self.logger.info(f"Running a dataset generator of class {self.__class__} -- instance: {self.cfg.instance_name}")
        data = Data()
        data = self.generate(data)  # 'data' has attributes that will be saved in the next line
        self.save(data)

    # By using the hyperparameters from 'self.cfg', we generate components that will become attributes of the 'data' object.
    # It is essential to ensure that the contents of these components are compatible with the CSV format.
    # We can add these components to the 'data' object in two ways: 'data.{name} = content' or 'setattr(data, initialpoint_name, content)'.
    @abstractmethod
    def generate(self, data):
        pass

    # Save all of the attributes in 'data'
    def save(self, data):
        for attr, content in vars(data).items():
            csvpath = f'{self.cfg.output_path}/{attr}.csv'

            # Store content based on its data type.
            # If you require saving data in a different format, override this function to accommodate your specific needs.
            if isinstance(content, np.matrix) or isinstance(content, np.ndarray):
                np.savetxt(csvpath, content)
            else:
                with open(csvpath, 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(content)

# Note that this is just an example of an instance of the Generator class.
# In your project, the class should be edited to suit your project's needs.
class InitialPointGenerator(Generator):
    def generate(self, data):
        # Should be written in your project.
        return data

@hydra.main(version_base=None, config_path=".", config_name="config_dataset")
def main(cfg):
    initialpointgenerator = InitialPointGenerator(cfg)
    initialpointgenerator.run()

if __name__=='__main__':
    main()
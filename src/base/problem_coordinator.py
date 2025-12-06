import hydra, logging
from dataclasses import dataclass
from typing import Any

@dataclass
class BaseProblem:
    costfun: Any
    initialpoint: Any

class Coordinator():
    def __init__(self, cfg):
        # Assertion
        assert hasattr(cfg, 'problem_name')
        assert hasattr(cfg, 'problem_instance')
        assert hasattr(cfg, 'problem_initialpoint')
        assert hasattr(cfg, 'problem_coordinator_name')

        # Set the configuration file
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.dataset_path = f'dataset/{cfg.problem_name}/{cfg.problem_instance}'

    def run(self):
        costfun = self.set_costfun()
        initialpoint = self.set_initialpoint()
        problem = BaseProblem(costfun=costfun,
                              initialpoint=initialpoint)
        return problem

    def set_costfun(self):
        pass

    def set_initialpoint(self):
        pass

@hydra.main(version_base=None, config_path=".", config_name="config_simulation")
def main(cfg):
    problem_coordinator = Coordinator(cfg)
    problem = problem_coordinator.run()

if __name__=='__main__':
    main()
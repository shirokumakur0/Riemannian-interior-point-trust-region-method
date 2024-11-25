import hydra

import sys
sys.path.append('./src/base')
import base_simulator

# Tutorial for your project:
# 1. If your solver employs some callback function or similar ones (e.g., manviofun in RALM), you can integrate them in 'add_solver_option(self, option)',
#    which is assumed to receive, edit and return option. Otherwise, in most cases, you can directly run this file to conduct your experiments,
#    as everything is already implemented in 'src/base/base_simulator.py'.
# 2. If you encounter any errors or need to customize the simulator's behavior for your specific project, you can overwrite the 'simulator' in this file.
#
# Next, please proceed to the 'analyzer.ipynb'.

@hydra.main(version_base=None, config_path=".", config_name="config_simulation")
def main(cfg):

    director = base_simulator.Simulator(cfg)
    director.run()

if __name__=='__main__':
    main()
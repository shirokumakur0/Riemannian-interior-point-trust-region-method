import hydra

import sys
sys.path.append('./src/base')
import solver_tuner

# Tutorial for your project:
# 1. Typically, you can run this file to conduct hyperparameter tuning experiments
#    as everything is already implemented in 'src/base/solver_tuner.py'.
#    If your solver employs callback functions or similar ones (e.g., manviofun in RALM), you can integrate them in 'add_solver_option(self, option)',
#    which is assumed to receive, edit and return option.
# 2. If any errors occur during the tuning process, you have the option to customize or overwrite the functions in this file.
#
# Now, you complete your simulation project. Cheers!

@hydra.main(version_base=None, config_path=".", config_name="config_tuning")
def main(cfg):

    tuner = solver_tuner.Tuner(cfg)
    tuner.run()

if __name__=='__main__':
    main()
import hydra
import numpy as np
import pymanopt

import sys
sys.path.append('./src/base')
import problem_coordinator

# Tutorial for your project:
# 1. Add the class named 'Coordinator', which will be a subclass of 'problem_coordinator.Coordinator'.
#    The name 'Coordinator' must not be changed, as it will be used in other files.
# 2. Implement the following methods in the 'Coordinator' class:
#    -- 'set_costfun(self)': This method should return the cost function (the objective function) for the optimization problem.
#    -- 'set_initialpoint(self)': This method should return the initial x and Lagrange multipliers for the optimization problem.
# 3. Implement other methods as needed such as 'set_searchspace(self)', 'set_ineqconstraints(self)', and 'set_eqconstraints(self)'.
#
# Next, please proceed to the 'config_simulation.yaml' file.
# If you implement your own solvers, you can go through 'src/solver/solver_template.py'.
# Then, proceed to the 'config_simulation.yaml' file.

# Problem coordinator for nonnegative principal component analysis
class Coordinator(problem_coordinator.Coordinator):

    # Set a cost function
    def set_costfun(self):

        def costfun(point):  # edit here
            return - point

        return costfun

    # Set initial points with initial Lagrange multipliers
    def set_initialpoint(self):
        initialpoint = []  # Write here
        return initialpoint

    # Add other set_*** methods as needed
    # def set_***(self):
    # *** = []  # Write here
    # return ***

@hydra.main(version_base=None, config_path=".", config_name="config_simulation")
def main(cfg):
    nonnegPCA_coordinator = Coordinator(cfg)
    problem = nonnegPCA_coordinator.run()

if __name__=='__main__':
    main()
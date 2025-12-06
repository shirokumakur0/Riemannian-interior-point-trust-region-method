# Riemannian interior point trust region method

Code for the paper "M. Obara, T. Okuno, and A. Takeda. A primal-dual interior point trust region method for second-order stationary points of Riemannian inequality-constrained optimization problems, arXiv, 2025" [[link]](https://arxiv.org/abs/2501.15419)

Our implementation includes

- Riemannian interior point trust region method (RIPTRM)
- Riemannian interior point method (RIPM) <!-- in "Z. Lai and A. Yoshise. Riemannian interior point methods for constrained optimization on manifolds, Journal of Optimization Theory and Applications, 201 (2024), pp.433–469."--> [[link]](https://link.springer.com/article/10.1007/s10957-024-02403-8)
- Riemannian sequential quadratic optimization (RSQO) <!-- in "M. Obara, T. Okuno, and A. Takeda. Sequential quadratic optimization for nonlinear optimization problems on Riemannian manifolds, SIAM Journal on Optimization, 32 (2022), pp.822–853."--> [[link]](https://epubs.siam.org/doi/abs/10.1137/20M1370173?download=true&journalCode=sjope8)
- Riemannian augmented Lagrangian method (RALM) <!-- in "C. Liu and N. Boumal. Simple algorithms for optimization on Riemannian manifolds with constraints, Applied Mathematics and Optimization, 82 (2020), pp.949–981."--> [[link]](https://link.springer.com/article/10.1007/s00245-019-09564-3)

## Requirements
This repository requires [Pymanopt](https://pymanopt.org/docs/stable/index.html), [Hydra](https://hydra.cc/), and [Wandb](https://wandb.ai/site) (registration required).
Make sure to install these libraries before running the experiments.

## Directory Structure

The directory structure is as follows:

<pre>
+-- dataset
+-- intermediate
+-- result
+-- src
    +-- solver
            +-- ...
    +-- base
        +-- ...
    +-- NonnegPCA
    +-- Rosenbrock
    +-- StableIdenticaition
</pre>

1. Running the 'generator' module generates input data and saves it in the 'dataset' folder based on the configurations specified in 'config_dataset.yaml'. The generated data includes various parameters and initial conditions essential for the optimization process.

2. Executing the 'simulator' module initiates numerical experiments with the settings provided in 'config_simulation.yaml'.
The outputs of these experiments are stored in the 'intermediate' folder.
During this process, the 'simulator' uses information from the 'dataset' folder to configure the experimental environment.
It also calls the 'problem_coordinator' and 'solver' modules to create and solve the optimization problem, respectively.

1. Running the 'analyzer' module generates final outputs, which are saved in the 'result' folder.
We assume that analyzer.ipynb is run on Google Colab.
The 'analyzer' reads and analyzes the raw data from the 'intermediate' folder to produce these outputs. The outputs may include visualizations, graphs, or comparative tables, depending on the nature of the experiment.

For a visual representation of the relationships and workflow among the modules, you can refer to the following:

![simulation_overview](https://github.com/Mathematical-Informatics-5th-Lab/Simulator-Template/assets/38340809/998edf62-0c45-4446-92cf-ac71af04ff4c)

The following optimization problems are available:

- Nonnegative principal component analysis (NonnegPCA)
- Rosenbrock function minimization (Rosenbrock)
- Stable linear system identification (StableIdentification)

## Note
**1. Wandb visualization**

In the 'config_simulation.yaml' file, there is a setting for wandb visualization as shown below:
```
solver_option:
    # Common settings: stopping criteria and wandb setting
    common:
      ...
      wandb_logging: True
      wandb_project: ${problem_name}-${problem_instance}-${problem_initialpoint}
```
When the 'wandb_logging' option is turned on (set to True), the experimental logs are sent to the wandb platform.
This enables us to visualize graphs and charts of the optimization processes, facilitating better insights into the performance and behavior of the solvers.
Turn off 'wandb_logging' if not necessary.


**2. Hybra multi-run**

Hydra supports [multi-run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/) functionality, allowing us to efficiently handle multiple problem instances and conduct simulations in parallel.
To create problem instances at once, follow the sweeper setting in 'config_dataset.yaml':
```
...
hydra:
  run:
    dir: dataset/${problem_name}
  sweeper:
    params:
      instance_name: 1,2,3
```
By running the following command, problem instances with names '1', '2', and '3' will be created simultaneously:
```
python src/NonnegPCA/generator.py -m
```

Similarly, for conducting simulations for instances all together, use the sweeper setting in 'config_simulation.yaml':
```
...
hydra:
  run:
    dir: intermediate/${problem_name}
  sweeper:
    params:
      problem_instance: 1,2,3
      problem_initialpoint: a,b,c
      solver_name: ["RALM"]
```
Run the following command to solve problem instances '1', '2', and '3' with initial points 'a', 'b', and 'c' using the 'RALM' solver:
```
python src/NonnegPCA/simulator.py -m
```

By leveraging the multi-run feature, we can efficiently manage and process multiple problem instances and obtain results more effectively.
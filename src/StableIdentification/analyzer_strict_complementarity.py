import hydra
import importlib
import numpy as np
import csv

def load_block_file(path):
    mats = []
    with open(path, newline='') as f:
        reader = csv.reader(f)

        for line_idx, row in enumerate(reader):
            vecs = []

            for col_idx, cell in enumerate(row):
                cell = cell.strip()
                if not cell:
                    continue

                s = cell.lstrip('[').rstrip(']')
                vec = np.fromstring(s, sep=' ')

                if vec.size == 0:
                    print(f"  -> WARNING: empty vector at line {line_idx}, col {col_idx}")
                    continue

                vecs.append(vec)

            if not vecs:
                continue

            mat = np.vstack(vecs)
            mats.append(mat)

    return np.stack(mats, axis=0)

@hydra.main(version_base=None, config_path=".", config_name="config_simulation")
def main(cfg):

    problem_coordinator_name = cfg.problem_coordinator_name
    module_coordinator = importlib.import_module(problem_coordinator_name)  # dynamic importation
    coordinator = module_coordinator.Coordinator(cfg)
    problem = coordinator.run()
    ineqconstraints = problem.ineqconstraints_all

    solver_set = ["RIPTRM_tCG", "RIPTRM_Exact_RepMat", "RSQO_reghess_corr1e-04", "RSQO_reghess_corr1e-02","RIPM_RepMat_gamma0.9_beta0.0001_theta0.5"]
    problem_initialpoint_set = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t"]
    problem_name = cfg.problem_name
    problem_instance = cfg.problem_instance
    threshold = 1e-8

    for solver in solver_set:
        for problem_initialpoint in problem_initialpoint_set:
            path = f'intermediate/{problem_name}/{problem_instance}/{problem_initialpoint}'

            path_x = f'{path}/{solver}_x.csv'
            x = load_block_file(path_x)
            path_y = f'{path}/{solver}_ineqLagmult.csv'
            y = np.loadtxt(path_y, delimiter=',')
            ineqval = np.array([ineqfun(x) for ineqfun in ineqconstraints])

            mask = (np.abs(y) <= threshold) & (np.abs(ineqval) <= threshold)

            flag = np.any(mask)
            if flag:
                print(f'Strict complementarity condition does NOT hold for solver {solver}.')
            else:
                pass
                # print(f'Strict complementarity condition holds for solver {solver}.')

if __name__=='__main__':
    main()
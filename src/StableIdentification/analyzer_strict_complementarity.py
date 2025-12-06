import hydra
import importlib

# import sys
# sys.path.append('./src/base')
# import base_simulator
import numpy as np
import csv

# import os
import copy

import csv
import numpy as np

import csv
import numpy as np

def load_block_file(path):
    mats = []
    with open(path, newline='') as f:
        reader = csv.reader(f)  # 必要なら delimiter='\t' など指定

        for line_idx, row in enumerate(reader):
            vecs = []  # この行で集めるベクトルのリスト

            for col_idx, cell in enumerate(row):
                # print("RAW CELL:", cell)

                cell = cell.strip()
                if not cell:
                    continue

                # 角カッコを落として数値だけに
                s = cell.lstrip('[').rstrip(']')
                vec = np.fromstring(s, sep=' ')
                # print("PARSED VECTOR:", vec)

                if vec.size == 0:
                    print(f"  -> WARNING: empty vector at line {line_idx}, col {col_idx}")
                    continue

                vecs.append(vec)

            if not vecs:
                # この行に有効なベクトルが一つもなければスキップ
                continue

            # 1行分のベクトルを縦に積んで行列へ
            mat = np.vstack(vecs)  # ここで初めて 2D ndarray になる
            mats.append(mat)
            # print("MATRIX SHAPE:", mat.shape)

    # mats = [ (5,5), (5,5), ... ] という想定なら stack で (N,5,5)
    return np.stack(mats, axis=0)

@hydra.main(version_base=None, config_path=".", config_name="config_simulation")
def main(cfg):

    problem_coordinator_name = cfg.problem_coordinator_name
    module_coordinator = importlib.import_module(problem_coordinator_name)  # dynamic importation
    coordinator = module_coordinator.Coordinator(cfg)
    problem = coordinator.run()
    # costfun = problem.cost
    ineqconstraints = problem.ineqconstraints_all

    # solver_set = ["RIPTRM_tCG", "RIPTRM_Exact_RepMat"]
    solver_set = ["RIPTRM_tCG", "RIPTRM_Exact_RepMat", "RSQO_reghess_corr1e-04", "RSQO_reghess_corr1e-02","RIPM_RepMat_gamma0.9_beta0.0001_theta0.5"]
    problem_initialpoint_set = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t"]    # problem_initialpoint_set = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t"]
    problem_name = cfg.problem_name
    problem_instance = cfg.problem_instance
    threshold = 1e-8

    for solver in solver_set:
        for problem_initialpoint in problem_initialpoint_set:
            # print(f'Analyzing solver: {solver}, initial point: {problem_initialpoint}')
            path = f'intermediate/{problem_name}/{problem_instance}/{problem_initialpoint}'

            path_x = f'{path}/{solver}_x.csv'
            x = load_block_file(path_x)
            path_y = f'{path}/{solver}_ineqLagmult.csv'
            y = np.loadtxt(path_y, delimiter=',')
            ineqval = np.array([ineqfun(x) for ineqfun in ineqconstraints])
            # print("J", x[0])
            # print("R", x[1])
            # print("Q", x[2])
            # print(f'Cost function value: {costfun(x)}')
            # print('y:', y <= threshold)
            # print('Inequality function values:', ineqval <= threshold)
            # print('Complementarity check (y / ineqval):', y / ineqval)
            # 要素ごとに「両方とも threshold 以下か？」をチェック
            mask = (np.abs(y) <= threshold) & (np.abs(ineqval) <= threshold)
            # print('Complementarity mask:', mask)
            # すべての要素で条件を満たすか？
            flag = np.any(mask)
            # print("flag:", flag)
            if flag:
                print(f'Strict complementarity condition does NOT hold for solver {solver}.')
            else:
                pass
                # print(f'Strict complementarity condition holds for solver {solver}.')

if __name__=='__main__':
    main()
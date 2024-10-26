import cvxpy as cp
import numpy as np

# 変数の定義
x = cp.Variable(2)  # 2次元の変数

# 目的関数の定義: Minimize (1/2) x^T P x + q^T x
P = np.array([[2, 0], [0, 2]])
q = np.array([1, 1])
objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x)

# 制約条件の定義
constraints = [x >= 0, cp.sum(x) == 1]

# 問題の定義
problem = cp.Problem(objective, constraints)

# 問題を解く
problem.solve()

# 結果の出力
print("Optimal solution:", x.value)
print("Optimal value:", problem.value)
import cvxpy as cp
import sccf
import numpy as np

np.random.seed(3) # fishy?
a = np.random.randint(-50, 50, size=5)
idx = np.random.choice(np.arange(5), size=3, replace=False)
a = np.append(
    a,
    -int(np.sum(a[idx]))
)
n = a.size

print("a =", a)

x = cp.Variable(n)
objective = 0.0
for i in range(n):
    objective += sccf.minimum(cp.square(x[i]), 0.25)
    objective += sccf.minimum(cp.square(x[i] - 1.0), 0.25)
objective += cp.square(a*x) - n/4
constraints = [cp.sum(x) >= 1.0]

prob = sccf.Problem(objective, constraints)
print("Alternating minimization:")
result = prob.solve(maxiter=50, verbose=True, warm_start_lam=np.random.uniform(0.0, 1.0, size=2*n+2), solver=cp.MOSEK)

print("objective value = %.3f" % result["final_objective_value"])
print("x =", np.around(x.value))
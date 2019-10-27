import argparse

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

import sccf
from utils import latexify

def main(show):
    # generate data
    np.random.seed(243)
    m, n = 5, 1
    n_outliers = 1
    eta = 0.1
    alpha = 0.5
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    b = A @ x_true + 1e-1 * np.random.randn(m)
    b[np.random.choice(np.arange(m), replace=False, size=n_outliers)] *= -1.

    # alternating
    x_alternating = cp.Variable(n)
    objective = 0.0
    for i in range(m):
        objective += sccf.minimum(cp.square(A[i]@x_alternating-b[i]), alpha)
    objective += eta * cp.sum_squares(x_alternating)
    prob = sccf.Problem(objective)
    prob.solve()

    # solve relaxed problem
    x_relaxed = cp.Variable(n)
    z = [cp.Variable(n) for _ in range(m)]
    s = cp.Variable(m)
    objective = 0.0
    constraints = [0 <= s, s <= 1]
    for i in range(m):
        objective += cp.quad_over_lin(A[i, :] @ z[i] - b[i] * s[i], s[i]) + (1.0 - s[i]) * alpha  + \
                    eta / m * (cp.quad_over_lin(x_relaxed - z[i], 1.0 - s[i]) + eta / m * cp.quad_over_lin(z[i], s[i]))
    prob = cp.Problem(cp.Minimize(objective), constraints)
    result = prob.solve(solver=cp.MOSEK)

    # alternating w/ relaxed initialization
    x_alternating_perspective = cp.Variable(n)
    x_alternating_perspective.value = x_relaxed.value
    objective = 0.0
    for i in range(m):
        objective += sccf.minimum(cp.square(A[i]@x_alternating_perspective-b[i]), alpha)
    objective += eta * cp.sum_squares(x_alternating_perspective)
    prob = sccf.Problem(objective)
    prob.solve(warm_start=True)

    # brute force evaluate function and perspective
    xs = np.linspace(-5, 5, 100)
    f = np.sum(np.minimum(np.square(A * xs - b[:, None]), alpha), axis=0) + eta*xs**2
    f_persp = []
    for x in xs:
        z = [cp.Variable(n) for _ in range(m)]
        s = cp.Variable(m)

        objective = 0.0
        constraints = [0 <= s, s <= 1]
        for i in range(m):
            objective += cp.quad_over_lin(A[i, :] @ z[i] - b[i] * s[i], s[i]) + (1.0 - s[i]) * alpha + \
                        eta / m * (cp.quad_over_lin(x - z[i], 1.0 - s[i]) + eta / m * cp.quad_over_lin(z[i], s[i]))
        prob = cp.Problem(cp.Minimize(objective), constraints)
        result = prob.solve(solver=cp.MOSEK)
        f_persp.append(result)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx 

    # plot
    latexify(fig_width=6, fig_height=4)
    plt.plot(xs, f, '-', label="$L(x)$", c='k')
    plt.plot(xs, f_persp, '--', label="perspective", c='k')
    plt.plot(x_alternating.value[0], )
    plt.scatter(x_alternating.value[0], f[find_nearest(xs, x_alternating.value[0])], marker='o', label="sccf (no init)", c='k')
    plt.scatter(x_alternating_perspective.value[0], f[find_nearest(xs, x_alternating_perspective.value[0])], marker='*', label="sccf (persp init)", c='k')
    plt.legend()
    plt.xlabel("$x$")
    plt.savefig("figs/perspective.pdf")
    if show:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perspective example.')
    parser.add_argument('--noshow', action='store_const', const=True, default=False)

    args = parser.parse_args()
    main(not args.noshow)
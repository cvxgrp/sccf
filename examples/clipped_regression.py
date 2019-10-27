import argparse

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

import sccf
from utils import latexify

import time

def main(show):
    # generate data
    np.random.seed(4)
    m, n = 10, 1
    n_outliers = 2
    alpha = .5
    eta = .1
    A = np.random.randn(m, n)
    x_true = np.array([1.])
    b = A @ x_true + 1e-1 * np.random.randn(m)
    b[np.random.choice(np.arange(m), replace=False, size=n_outliers)] *= -1.0

    # sccf
    x = cp.Variable(n)
    objective = 0.0
    for i in range(m):
        objective += sccf.minimum(cp.square(A[i]@x-b[i]), alpha)
    objective += eta * cp.sum_squares(x)
    prob = sccf.Problem(objective)
    tic = time.time()
    result = prob.solve()
    toc = time.time()
    print ("time (s):", toc-tic)
    print ("iters:", result["iters"])
    outliers = np.square(A@x.value-b) > alpha
    x_alternating = x.value[0]
    print (objective.value)

    # lstsq
    x_lstsq = cp.Variable(n)
    objective = cp.sum_squares(A@x_lstsq-b) + eta*cp.sum_squares(x_lstsq)
    prob = cp.Problem(cp.Minimize(objective))
    prob.solve()

    # huber
    x_huber = cp.Variable(n)
    objective = cp.sum(cp.huber(A@x_huber-b, alpha)) + eta*cp.sum_squares(x_huber)
    prob = cp.Problem(cp.Minimize(objective))
    prob.solve()

    # plot
    latexify(6, 4)
    plt.scatter(A[outliers], b[outliers], marker='x', c='black', label='outlier')
    plt.scatter(A[~outliers], b[~outliers], marker='o', c='black', label='inlier')
    plt.plot([-3,3], [-3*x.value[0], 3*x.value[0]], '-', c='black', label='clipped')
    plt.plot([-3,3], [-3*x_lstsq.value[0], 3*x_lstsq.value[0]], '--', c='black', label='lstsq')
    plt.plot([-3,3], [-3*x_huber.value[0], 3*x_huber.value[0]], ':', c='black', label='huber')
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.savefig("figs/clipped_regression.pdf")
    if show:
        plt.show()
    
    # solve relaxed perspective formulation
    x_relaxed = cp.Variable(n)
    z = [cp.Variable(n) for _ in range(m)]
    s = cp.Variable(m)
    objective = 0.0
    constraints = [0 <= s, s <= 1]
    for i in range(m):
        objective += cp.quad_over_lin(A[i, :] @ z[i] - b[i] * s[i], s[i]) + (1.0 - s[i]) * alpha  + \
                    eta / m * (cp.quad_over_lin(x_relaxed - z[i], 1.0 - s[i]) + eta / m * cp.quad_over_lin(z[i], s[i]))
    prob = cp.Problem(cp.Minimize(objective), constraints)
    tic = time.time()
    result = prob.solve(solver=cp.MOSEK)
    toc = time.time()
    print("relaxed:", toc-tic)

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

    print ("lower bound:", np.min(f_persp))

    # plot
    latexify(fig_width=5)
    plt.plot(xs, f, '-', label="$L(\\theta)$", c='k')
    plt.plot(xs, f_persp, '--', label="perspective", c='k')
    plt.scatter(x_alternating, f[find_nearest(xs, x_alternating)], marker='o', label="$\\theta^\\mathrm{clip}$", c='k')
    plt.axhline(np.min(f_persp), linestyle='-.', label="lower bound", c='k')
    plt.legend()
    plt.ylim(0)
    plt.xlabel("$\\theta$")
    plt.savefig("figs/perspective.pdf")
    if show:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clipped regression example.')
    parser.add_argument('--noshow', action='store_const', const=True, default=False)

    args = parser.parse_args()
    main(not args.noshow)
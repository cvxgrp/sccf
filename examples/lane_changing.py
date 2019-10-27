import argparse

import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import sccf
from utils import latexify

import time
import warnings

def main(show=False):
    margin = .05 # margin for drawing box

    initial_pos = 1
    final_pos = 1

    n = 100

    all_pos = np.linspace(0, 1, n)

    # Complicated lane example

    box_size = .18
    box_pos = [.2, .5, .8]
    box_orientation = [-1, 1, -1]

    x = cp.Variable(n)

    cons = [
        x[0] == initial_pos,
        -2 <= x,
        x <= 2,
        x[-1] == -1
    ]

    obj = 0.0

    for i, pos in enumerate(all_pos):
        obj += sccf.minimum(cp.square(x[i] + 1), 1)
        obj += sccf.minimum(cp.square(x[i] - 1), 1)

        for b_pos, b_or in zip(box_pos, box_orientation):
            if b_pos <= pos and pos <= b_pos + box_size:
                cons.append(
                    x[i] >= 0 if b_or > 0 else x[i] <= 0
                )

    for idx, weight in enumerate([10, 1, .1]):
        obj += weight * cp.sum_squares(cp.diff(x, k=idx+1))

    prob = sccf.Problem(obj, cons)
    tic = time.time()
    result = prob.solve()
    toc = time.time()

    print ("lane change 2:", obj.value)
    print ("time:", toc-tic)
    print ("iters:", result["iters"])

    latexify(fig_width=7, fig_height=2)
    plt.plot(all_pos*100, x.value, c='black')
    plt.ylim(-2, 2)
    for pos, orientation in zip(box_pos, box_orientation):
        plt.gca().add_patch(
        Rectangle((pos*100, .25 if orientation < 0 else -1.75), (box_size - margin)*100,
                1.5, facecolor='none', edgecolor='k')
        )
    plt.axhline(0, ls='--', c='k')
    plt.savefig("figs/lane_changing.pdf")
    if show:
        plt.show()

    # Lower bound
    obj = 0

    z_top = [
        cp.Variable(n)
        for _ in range(n)
    ]
    z_bottom = [
        cp.Variable(n)
        for _ in range(n)
    ]

    x = cp.Variable(n)
    cons = [
        x[0] == initial_pos,
        -2 <= x,
        x <= 2,
        x[-1] == -1
    ]

    lam_top = cp.Variable(n)
    lam_bottom = cp.Variable(n)
    cons.append(0 <= lam_top)
    cons.append(0 <= lam_bottom)
    cons.append(lam_top <= 1)
    cons.append(lam_bottom <= 1)

    for z, lam in zip(z_top + z_bottom, lam_top + lam_bottom):
        cons.append(z[0] == initial_pos * lam)
        cons.append(-2 * lam <= z)
        cons.append(z <= 2 * lam)
        cons.append(z[-1] == -1 * lam)
    
    for i, pos in enumerate(all_pos):
        obj += cp.quad_over_lin(z_top[i][i] + lam_top[i], lam_top[i]) + (1-lam_top[i])
        obj += cp.quad_over_lin(z_bottom[i][i] - lam_bottom[i], lam_bottom[i]) + (1-lam_bottom[i])

        for b_pos, b_or in zip(box_pos, box_orientation):
            if b_pos <= pos and pos <= b_pos + box_size:
                for z in z_top + z_bottom + [x]:
                    cons.append(z[i] >= 0 if b_or > 0 else z[i] <= 0)
    
    for idx, weight in enumerate([10, 1, .1]):
        for z, lam in zip(z_top + z_bottom, lam_top + lam_bottom):
            obj += weight * cp.quad_over_lin(cp.diff(z, k=idx+1), lam) / (2*n)
            obj += weight * cp.quad_over_lin(cp.diff(x - z, k=idx+1), 1 - lam) / (2*n)
    
    prob = cp.Problem(cp.Minimize(obj), cons)
    obj_value = prob.solve(solver=cp.MOSEK)

    print("lane change lower bound:", obj_value)

    # MICP
    obj = 0

    z_top = [
        cp.Variable(n)
        for _ in range(n)
    ]
    z_bottom = [
        cp.Variable(n)
        for _ in range(n)
    ]

    x = cp.Variable(n)
    cons = [
        x[0] == initial_pos,
        -2 <= x,
        x <= 2,
        x[-1] == -1
    ]

    lam_top = cp.Variable(n, boolean=True)
    lam_bottom = cp.Variable(n, boolean=True)

    for z, lam in zip(z_top + z_bottom, lam_top + lam_bottom):
        cons.append(z[0] == initial_pos * lam)
        cons.append(-2 * lam <= z)
        cons.append(z <= 2 * lam)
        cons.append(z[-1] == -1 * lam)
    
    for i, pos in enumerate(all_pos):
        obj += cp.quad_over_lin(z_top[i][i] + lam_top[i], lam_top[i]) + (1-lam_top[i])
        obj += cp.quad_over_lin(z_bottom[i][i] - lam_bottom[i], lam_bottom[i]) + (1-lam_bottom[i])

        for b_pos, b_or in zip(box_pos, box_orientation):
            if b_pos <= pos and pos <= b_pos + box_size:
                for z in z_top + z_bottom + [x]:
                    cons.append(z[i] >= 0 if b_or > 0 else z[i] <= 0)
    
    for idx, weight in enumerate([10, 1, .1]):
        for z, lam in zip(z_top + z_bottom, lam_top + lam_bottom):
            obj += weight * cp.quad_over_lin(cp.diff(z, k=idx+1), lam) / (2*n)
            obj += weight * cp.quad_over_lin(cp.diff(x - z, k=idx+1), 1 - lam) / (2*n)
    
    prob = cp.Problem(cp.Minimize(obj), cons)
    import sys
    while True:
        answer = input("Are you sure you would like to solve the MICP (y/n) ").lower()

        if answer == "y":
            break
        elif answer == "n":
            return
        else:
            print("Invalid answer.") 
            continue
        
    obj_value = prob.solve(solver=cp.MOSEK, verbose=True)

    print("global optimum:", obj_value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lane changing example.')
    parser.add_argument('--noshow', action='store_const', const=True, default=False)

    args = parser.parse_args()
    main(not args.noshow)
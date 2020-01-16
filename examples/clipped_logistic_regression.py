from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import numpy as np

import cvxpy as cp
import sccf
import time

from utils import latexify

import IPython as ipy

np.random.seed(243)
N, n = 1000, 5
X, y = make_blobs(N, n, centers=2, cluster_std=10.0)
Xtr, Xte, ytr, yte = train_test_split(X, y, shuffle=True, test_size=0.9)
idx = np.random.choice(np.arange(ytr.size),
                       size=int(ytr.size * .2), replace=False)
ytr[idx] = 1 - ytr[idx]

theta = cp.Variable(n)
b = cp.Variable(1)
objective = 0.0
logistic_losses = []
for i in range(Xtr.shape[0]):
    logistic_loss = 1./Xtr.shape[0] * (-ytr[i] * (Xtr[i, :] @ theta + b) + cp.logistic(Xtr[i, :] @ theta + b))
    logistic_losses.append(logistic_loss)
    objective += logistic_loss
objective += .1 * cp.sum_squares(theta)
prob = cp.Problem(cp.Minimize(objective))
prob.solve()

losses = np.array([l.value[0] * Xtr.shape[0] for l in logistic_losses])
latexify(3.5)
plt.xlim(-6, 1.5)
plt.hist(np.log10(losses), range=(-6, 1.5), color='black', bins=50)
plt.xlabel("log logistic loss")
plt.ylabel("count")
plt.savefig("figs/logreg_density.pdf")
plt.close()

standard_logreg = np.mean(((Xte @ theta.value + b.value) >= 0) == yte)

print (standard_logreg)

scores, outliers = [], []
alphas = np.logspace(-1,1,50)
avg_time = 0.0
iters = 0.0
for iter, alpha in enumerate(alphas):
    alpha = alphas[iter]
    np.random.seed(0)
    theta = cp.Variable(n)
    b = cp.Variable(1)
    objective = 0.0
    logistic_losses = []
    for i in range(Xtr.shape[0]):
        logistic_loss = 1./Xtr.shape[0] * (-ytr[i] * (Xtr[i, :] @ theta + b) + cp.logistic(Xtr[i, :] @ theta + b))
        logistic_losses.append(logistic_loss)
        objective += sccf.minimum(logistic_loss, 1./Xtr.shape[0]*alpha)
    objective += .1 * cp.sum_squares(theta)
    prob = sccf.Problem(objective)
    tic = time.time()
    result = prob.solve(step_size=.2, maxiter=50)
    print (result["final_objective_value"])
    toc = time.time()
    avg_time += (toc - tic) / 50
    iters += result["iters"] / 50
    
    score = np.mean(((Xte @ theta.value + b.value) >= 0) == yte)
    num_outliers = np.mean([l.value >= 1./Xtr.shape[0]*alpha for l in logistic_losses])

    print (alpha, score, num_outliers, toc-tic, result["iters"])

    scores += [score]
    outliers += [num_outliers]

    if iter == 26:
        lams = np.array(result["lams"])
        latexify(5)
        for i in range(lams.shape[1]):
            plt.plot(lams[:, i], c='black')
        plt.xlabel("$k$")
        plt.ylabel("$\\lambda_i$")
        plt.savefig("figs/logistic_regression_lambda.pdf") 
        plt.close()

        losses = np.array([l.value[0] * Xtr.shape[0] for l in logistic_losses])
        latexify(3.5)
        plt.xlim(-6, 1.5)
        plt.hist(np.log10(losses), range=(-6, 1.5), color='black', bins=50)
        plt.xlabel("log logistic loss")
        plt.ylabel("count")
        plt.savefig("figs/clipped_density.pdf")
        plt.close()

print("average time:", avg_time)
print("average iters:", iters)
latexify(6)
fig, ax1 = plt.subplots()
lns1 = ax1.axhline(standard_logreg, linestyle='-', label='standard', alpha=.5, c='gray')
lns2 = ax1.semilogx(alphas, scores, '-', label='clipped', c='black')
ax1.set_ylabel("test accuracy")
ax1.set_xlabel("$\\alpha$")

ax2 = ax1.twinx()
lns3 = ax2.semilogx(alphas, outliers, '-.', label='outliers', c='black')
ax2.set_ylabel("fraction of outliers")

lns = [lns1] + lns2 + lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="upper right")

plt.savefig("figs/logistic_regression.pdf")
plt.show()

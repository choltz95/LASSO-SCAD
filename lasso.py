# Chester Holtz - chholtz@ucsd.edu
# LASSO w/ smoothly clipped absolute deviation (SCAD) penalty
# solved via local linear approximation (LLA) plus iterative shrinkage-thresholding algorithm (ISTA)

import numpy as np
from numpy import linalg as la
from sklearn.linear_model import Lasso

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

MAXIT = 10000
TOL = 10**(-5)

def l2_error(th_hat, th_star):
    return la.norm(th_hat - th_star, 2)

def fp(th_hat, th_star):
    fp = 0
    for i in range(th_star.shape[0]):
        if th_star[i] == 0:
            if th_hat[i] != 0:
                fp += 1
    return fp

def fn(th_hat, th_star):
    fn = 0
    for i in range(th_star.shape[0]):
        if th_star[i] != 0:
            if th_hat[i] == 0:
                fn += 1
    return fn

def evaluate(th_hat, th_star):
    return {'l2_error' : l2_error(th_hat, th_star),
            'false_pos': fp(th_hat, th_star),
            'false_neg': fn(th_hat, th_star)}

def avg_evaluate(evaluates):
    avg_l2 = sum(d['l2_error'] for d in evaluates) / len(evaluates)
    avg_fp = sum(d['false_pos'] for d in evaluates) / len(evaluates)
    avg_fn = sum(d['false_neg'] for d in evaluates) / len(evaluates)

    return {'l2_error' : avg_l2,
            'false_pos': avg_fp,
            'false_neg': avg_fn}

def update_lambda(l, th, a):
    """
    for i in range(th.shape[0]):
        if th[i] <= l[i]:
            th[i] = l[i]
        elif l[i] <= th[i]  and th[i] <= a*l[i]:
            th[i] = (a*l[i] - th[i])/(a-1)
        elif a*l[i] <= th[i]:
            th[i] = 0
    return th
    """
    if not th.any():
        return np.zeros(th.shape[0])
    return l * np.less_equal(th,l) + np.clip(np.divide((a*l - th),(a-1)*l),0,np.inf) * np.greater(th,l)

def scad(l,th,a):
    for i in range(th.shape[0]):
        if th[i] <= 2*l[i]:
            np.sign(th[i])*(th[i] - l[i])
        elif 2*l[i] <= th[i]  and th[i] <= a*l[i]:
            th[i] = np.sign(th[i])*((a-1)*th[i] - a*l)/(a-2)
        elif a*l[i] <= th[i]:
            th[i] = th[i]
    return th

def obj(X, y, th, l):
    return 1/(2*X.shape[1]) * (la.norm(y - X.dot(th), 2))**2 + scad(l,th)

def backtrack(s_prev):
    pass

def lla(X, y, th_0, s_0, l, n):
    i = 0
    a = 3
    th_prev = th_0
    for i in range(MAXIT):
        th_k = ISTA(X, y, th_prev, s_0, l, n)
        l = update_lambda(l,np.abs(th_k),a)
        if la.norm(th_k - th_prev,np.inf) <= TOL:
            break
        th_prev = th_k
    return th_k, l

def ISTA(X, y, th_0, s_0, l, n):
    i = 0
    th_prev = th_0
    s_k = s_0
    for i in range(MAXIT):
        th_k = th_prev + s_k/n * X.T.dot(y - X.dot(th_prev))
        th_k = np.sign(th_k) * np.maximum(np.abs(th_k) - s_k*l, np.zeros(d))
        if la.norm(th_k - th_prev,np.inf) <= TOL:
            break
        th_prev = th_k

    return th_k

def gen_5folds(X, y):
    X_folds = np.split(X, 5)
    y_folds = np.split(y, 5)
    return X_folds, y_folds

def cv_lambda(X, y, th_0, s_0, lmbda_0, n):
    a = 3
    X_folds, y_folds = gen_5folds(X, y)
    errs = []
    for i, fold in enumerate(tqdm(X_folds, desc='cv')):
        X = np.concatenate(X_folds[:i] + X_folds[(i + 1):])
        y = np.concatenate(y_folds[:i] + y_folds[(i + 1):])
        th_hat, l = lla(X, y, th_0, s_0, lmbda_0, n)
        errs.append(1/(2*y.shape[0])*la.norm(y - X.dot(th_hat),2)**2 + np.sum(scad(l, np.abs(th_hat), a)))
    return np.sum(errs)/len(X_folds)

N = [100, 200]
D = [256, 512, 1024]
s = 10
n = N[0]
d = D[0]
lmbda = 0.85

for n in N:
    for d in D:
        evals_lasso = []
        evals_ista = []
        for _ in tqdm(range(200)):
            #cov =  numpy.fromfunction(lambda i, j: 0.5**(np.abs(i-j)), (n,d)) # (P2)
            cov = np.eye(d)
            X = np.random.multivariate_normal(np.zeros(d),cov,n)
            e = np.random.normal(0,1.5,n)
            th_star = np.array([2,2,2,-1.5,-1.5,-1.5,2,2,2,2] + [0]*(d-10))
            y = X.dot(th_star) + e

            evs = la.eig(X.T.dot(X))[0]
            L = np.amax(np.real(evs))
            s_0 = 1/L;
            th_0 = np.zeros(d, dtype=float)
            lmbda_0 = np.array([lmbda]*d)

            th_hat,l_hat = lla(X, y, th_0, s_0, lmbda_0, n)
            clf = Lasso(alpha = lmbda)
            clf.fit(X, y)
            th_lasso = clf.coef_
            evals_lasso.append(evaluate(th_lasso, th_star))
            evals_ista.append(evaluate(th_hat, th_star))
        print('lasso',n,d,avg_evaluate(evals_lasso))
        print('ista',n,d,avg_evaluate(evals_ista))

"""
X = np.random.normal(0,1,(n,d))
e = np.random.normal(0,1.5,n)
th_star = np.array([2,2,2,-1.5,-1.5,-1.5,2,2,2,2] + [0]*(d-10))
y = X.dot(th_star) + e

evs = la.eig(X.T.dot(X))[0]
L = np.amax(np.real(evs))
s_0 = 1/L;
th_0 = np.zeros(d, dtype=float)
lmbda_0 = np.array([lmbda]*d)

# Cross validation for lambda_0 - best found to be 0.9
#for lmbda in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
#    lmbda_0 = np.array([lmbda]*d)
#    print(lmbda, cv_lambda(X, y, th_0, s_0, lmbda_0, n))

th_hat,l_hat = lla(X, y, th_0, s_0, lmbda_0, n)
clf = Lasso(alpha = lmbda)
clf.fit(X, y)
th_lasso = clf.coef_
print('ista',evaluate(th_hat, th_star))
print('sklearn',evaluate(th_lasso, th_star))
"""

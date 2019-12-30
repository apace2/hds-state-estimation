import copy
import numpy as np


def lineSearch(x, d, f, step=1, delta=.8, maxitr=10):
    it = 0
    obj = f(x)
    xnew = copy.copy(x)
    while it < maxitr and f(xnew) >= obj:
        xnew = x - step*d
        step *= delta
        it += 1
    if f(xnew) < f(x):
        return xnew
    return x


def FISTA(x, g, prox,f , L, iter=5):
    y = copy.copy(x)
    #xp = copy.copy(x)
    t = 1
    loss = []
    for k in range(iter):
        gy = g(y)
        xp = prox(y-gy/L)
        tp = (1+np.sqrt(1+4*t**2))/2.0
        y = xp + (t-1)/tp*(xp-x)
        x = copy.copy(xp)
        t = tp
        loss.append(f(x))
    return x, loss


def simplexConstr(y, h=1.0):
    n = len(y)
    x = np.zeros(n)
    # if on simplex
    s = 0.0
    box = True
    for i in range(n):
        if box and (0 <= y[i] <= 1):
            box = True
        else:
            box = False
        s += y[i]

    if box and s == h:
        x = copy.copy(y)
        return x

    p = np.argsort(y)
    for i in range(n):
        alpha = (s-h)/(n-i)
        if alpha <= y[p[i]]:
            for j in range(i, n):
                x[p[j]] = y[p[j]]-alpha
            break
        else:
            x[p[i]] = 0.0
            s -= y[p[i]]
    return x


def solveTridiag(b, dM, sM):
    N = len(dM)
    iM = [np.linalg.inv(dM[0])]

    for i in range(1, N):
        iM.append(np.linalg.inv(dM[i]-sM[i-1].dot(iM[i-1]).dot(np.transpose(sM[i-1]))))
        b[i] = b[i]-sM[i-1].dot(iM[i-1].dot(b[i-1]))
    sol = [iM[-1].dot(b[-1])]
    for i in range(N-2, -1, -1):
        b[i] = -np.transpose(sM[i]).dot(sol[0]) + b[i]
        sol.insert(0, iM[i].dot(b[i]))
    return sol


def multiplyMV(dM, sM, x):
    N = len(dM)
    y = [np.zeros(dM[0].shape[0]) for _ in range(N)]
    y[0] = dM[0].dot(x[0, :])
    y[0] += np.transpose(sM[0]).dot(x[1, :])

    y[-1] = sM[-1].dot(x[-2, :])
    y[-1] += dM[-1].dot(x[-1, :])

    for i in range(1, N-1):
        y[i] = sM[i-1].dot(x[i-1, :])
        y[i] += dM[i].dot(x[i, :])
        y[i] += np.transpose(sM[i]).dot(x[i+1, :])

    return y


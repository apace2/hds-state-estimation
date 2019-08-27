import numpy as np
import copy

def simplexConstr(y,h = 1.0):
    n = len(y);
    x = np.zeros(n)
    # if on simplex
    s = 0.0;
    box = True;
    for i in range(n):
        if box and (y[i]>=0 and y[i]<=1):
            box = True
        else:
            box = False
        s  += y[i];

    if box and s == h:
        x = copy.copy(y)
        return x

    p = np.argsort(y)
    for i in range(n):
        alpha = (s-h)/(n-i);
        if alpha <= y[p[i]]:
            for j in range(i,n):
                x[p[j]]=y[p[j]]-alpha
            break
        else:
            x[p[i]] = 0.0
            s -= y[p[i]]
    return x

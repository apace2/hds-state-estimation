import numpy as np
import copy

def lineSearch(x,d,f,step = 1,delta = .8,maxitr = 10):
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

import numpy as np
import copy

def FISTA(x,g,prox,f,L,iter = 5):
    y = copy.copy(x)
    xp = copy.copy(x)
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
    return x,loss

def FISTA_faster(x,g,prox,f,L,iter = 5):
    y = copy.copy(x)
    #xp = copy.copy(x)
    t = 1
    loss = []
    for k in range(iter):
        gy = g(y)
        xp = prox(y-gy/L)
        tp = (1+np.sqrt(1+4*t**2))/2.0
        y = xp + (t-1)/tp*(xp-x)
        #x = copy.copy(xp) # in profiling, this line is 13.6 micros
        x = xp             # this is 1.2 micros
        t = tp
        loss.append(f(x))
    return x,loss

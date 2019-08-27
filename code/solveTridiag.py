import numpy as np

def solveTridiag(b,dM,sM):
    N = len(dM)
    iM = [np.linalg.inv(dM[0])]

    for i in range(1,N):
        iM.append(np.linalg.inv(dM[i]-sM[i-1].dot(iM[i-1]).dot(np.transpose(sM[i-1]))))
        b[i] = b[i]-sM[i-1].dot(iM[i-1].dot(b[i-1]))
    sol = [iM[-1].dot(b[-1])]
    for i in range(N-2,-1,-1):
        b[i] = -np.transpose(sM[i]).dot(sol[0]) + b[i]
        sol.insert(0,iM[i].dot(b[i]))
    return sol

def multiplyMV(dM,sM,x):
    N = len(dM)
    y = [np.zeros(dM[0].shape[0]) for _ in range(N)]
    y[0] = dM[0].dot(x[0,:])
    y[0] += np.transpose(sM[0]).dot(x[1,:])

    y[-1] = sM[-1].dot(x[-2,:])
    y[-1] += dM[-1].dot(x[-1,:])

    for i in range(1,N-1):
        y[i] = sM[i-1].dot(x[i-1,:])
        y[i] += dM[i].dot(x[i,:])
        y[i] += np.transpose(sM[i]).dot(x[i+1,:])

    return y

import numpy as np
import copy
from . import utils
from . import gradientTest


class SwitchedKalman:
    def __init__(self, y, G, dG, C, Qinv, Rinv, x0, r):
        """
        Hybrid domain system class
        Args:
            y: measurements
            G: list of process models
            dG: list of derivative of process models
            C: observation model
            Qinv: inverse of process noise covariance matrix
            Rinv: inverse of measurement noise covariance matrix
            x0: initial state
            r: degrees of freedom for student's t distribution
        """
        self.y = y
        self.G = G
        self.dG = dG
        self.C = C
        self.Qinv = Qinv
        self.Rinv = Rinv
        self.x0 = x0
        self.r = r

        self.T = y.shape[0]
        self.d = x0.shape[0]
        self.m = len(G[0])

    def processTerms(self, x):
        process = np.zeros((self.T, self.m, self.d))
        for i in range(self.m):
            for t in range(self.T):
                # as self.G is a function don't think this can be vectorized
                if t == 0:
                    process[t, i, :] = x[0, :] - self.G[t][i](self.x0)
                else:
                    process[t, i, :] = x[t, :] - self.G[t][i](x[t - 1, :])
        return process

    def measurementTerms(self, x):
        meas = np.zeros((self.T,self.y.shape[1]))
        for t in range(self.T):
           meas[t,:] = self.y[t,:] - self.C.dot(x[t,:])
        #meas = self.y - (self.C @ x.T).T
        return meas

    def processErrors(self, w, terms):
        errors = np.zeros((self.T, self.m))
        for m in range(self.m):
            norm = np.einsum('ij,ij->i', terms[:, m, :] @ self.Qinv, terms[:, m, :])  # the norm
            # np.einsum('ij,ij->i',x,x) is equivalent to np.linalg.norm(x,axis=1)**2
            errors[:, m] = w[:, m] * self.r / 2.0 * np.log(1 + norm / self.r)
        return errors

    def measurementErrors(self, terms):
        # errors = np.zeros(self.T)
        # for t in range(self.T):
        #    errors[t] = np.dot(self.Rinv.dot(terms[t,:]),terms[t,:])/2.0
        errors = np.einsum('ij,ij->i', terms @ self.Rinv, terms) / 2.
        return errors

    def computeObj(self, x, w, nu, beta, verbose=False):
        pt = self.processTerms(x)
        pe = np.sum(self.processErrors(w, pt))
        mt = self.measurementTerms(x)
        me = np.sum(self.measurementErrors(mt))

        se = nu / 2.0 * (np.linalg.norm(w[1:, :] - w[:-1, :], ord='fro') ** 2 + np.linalg.norm(w[0, :]) ** 2)

        if verbose:
            print("process error = ", pe, " measurement error = ", me, " smoothing term = ", se)

        return pe + me + se + beta / 2.0 * np.linalg.norm(w) ** 2

    def buildGNMatrices(self, x, w):
        pt = self.processTerms(x)  # p[t] = x[t] - G[t]x[t-1]
        mt = self.measurementTerms(x)

        coefs = np.zeros((self.T, self.m))
        for i in range(self.m):
            norm = np.einsum('ij,ij->i', pt[:, i, :] @ self.Qinv, pt[:, i, :])
            coefs[:, i] = w[:, i] * self.r / (self.r + norm)
        dGs = [[self.dG[t][i](x[t - 1, :]) for i in range(self.m)] for t in range(1, self.T)]
        dGs.insert(0, [self.dG[0][i](self.x0) for i in range(self.m)])

        gx = []
        Ss = []
        Ds = []

        for t in range(self.T):
            gxt = - np.transpose(self.C).dot(self.Rinv.dot(mt[t, :]))
            S = np.zeros((self.d, self.d))
            D = np.transpose(self.C).dot(self.Rinv).dot(self.C)
            for i in range(self.m):
                if t < self.T - 1:
                    gxt += coefs[t, i] * self.Qinv.dot(pt[t, i, :]) - coefs[t + 1, i] * np.dot(
                        np.transpose(dGs[t + 1][i]), self.Qinv.dot(pt[t + 1, i, :]))
                    D += coefs[t, i] * self.Qinv + coefs[t + 1, i] * np.transpose(dGs[t + 1][i]).dot(self.Qinv).dot(
                        dGs[t + 1][i])
                    # D += coefs[t,i]*self.Qinv + coefs[t+1,i]*np.einsum('ji,jk,kl->il',dGs[t+1][i],self.Qinv,dGs[t+1][i])
                    S -= coefs[t + 1, i] * np.dot(np.transpose(dGs[t + 1][i]), self.Qinv)
                else:
                    gxt += coefs[t, i] * self.Qinv.dot(pt[t, i, :])
                    D += coefs[t, i] * self.Qinv

            gx.append(gxt)
            Ss.append(np.transpose(S))
            Ds.append(D)
        return gx, Ds, Ss

    def solveTridiag(self, gx, dM, sM):
        N = len(dM)
        iM = [np.linalg.inv(dM[0])]
        for i in range(1, N):
            iM.append(np.linalg.inv(dM[i] - sM[i - 1].dot(iM[i - 1]).dot(np.transpose(sM[i - 1]))))
            gx[i] = gx[i] - sM[i - 1].dot(iM[i - 1].dot(gx[i - 1]))
        dx = [iM[-1].dot(gx[-1])]
        for i in range(N - 2, -1, -1):
            gx[i] = -np.transpose(sM[i]).dot(dx[0]) + gx[i]
            dx.insert(0, iM[i].dot(gx[i]))
        return dx

    def solveGNDirection(self, x, w, verbose=False):
        gx, Ds, Ss = self.buildGNMatrices(x, w)
        gx_copy = copy.copy(gx)
        dx = np.asarray(utils.solveTridiag(gx, Ds, Ss))
        if verbose:
            print(np.linalg.norm(np.asarray(utils.multiplyMV(Ds, Ss, dx)) - np.asarray(gx_copy)), " ",
                  np.linalg.norm(np.asarray(gx_copy)))
        return dx

    def updateX(self, x, w, iter=1):
        f = lambda x: self.computeObj(x, w, 0.0, 0.0)
        loss = []
        for k in range(iter):
            dx = self.solveGNDirection(x, w)
            x = utils.lineSearch(x, dx, f)
            loss.append(f(x))
        return x, loss

    def computeGradw(self, w, x, nu, beta):
        pt = self.processTerms(x)
        gw = self.processErrors(np.ones((self.T, self.m)), pt) + beta * w
        gw[0, :] += nu * (2 * w[0, :] - w[0 + 1, :])
        gw[self.T - 1, :] += nu * (w[self.T - 1, :] - w[self.T - 2, :])
        gw[1:-1, :] += nu * (2 * w[1:-1] - w[0:-2, :] - w[2:, :])
        return gw

    def updateW(self, w, x, nu, beta, iter=5):
        L = beta + 4 * nu
        g = lambda w: self.computeGradw(w, x, nu, beta)
        f = lambda w: self.computeObj(x, w, nu, beta)

        def prox(z):
            for t in range(self.T):
                z[t, :] = utils.simplexConstr(z[t, :])
            return z

        return utils.FISTA(w, g, prox, f, L, iter=iter)

    def gradTest(self, x, w, nu, beta, pert=1e-1):

        def func(z):
            zz = np.reshape(z, (self.T, self.d))
            fx = self.computeObj(zz, w, 0.0, 0.0)
            gx, _, _ = self.buildGNMatrices(zz, w)
            return fx, np.concatenate(gx)

        gradientTest.gradientTest(func, np.random.randn(self.T * self.d), pert)

        def func2(z):
            zz = np.reshape(z, (self.T, self.m))
            fw = self.computeObj(x, zz, nu, beta)
            gw = self.computeGradw(zz, x, nu, beta)
            return fw, gw.flatten()

        gradientTest.gradientTest(func2, np.random.rand(self.T * self.m), pert)

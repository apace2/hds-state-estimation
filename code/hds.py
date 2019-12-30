import numpy as np
import copy
import simplex
import solveTridiag
import lineSearch
import FISTA
import gradientTest

class HDS:

    def __init__(self, y, G, dG, C, Qinv, Rinv, x0, r):
        """

        Args:
            y:
            G:
            dG:
            C:
            Qinv:
            Rinv:
            x0:
            r:
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
                #as self.G is a function don't think this can be vectorized
                if t == 0:
                    process[t, i, :] = x[0, :] - self.G[t][i](self.x0)
                else:
                    process[t, i, :] = x[t, :] - self.G[t][i](x[t-1, :])
        return process

    def measurementTerms(self,x):
        #meas = np.zeros((self.T,self.y.shape[1]))
        #for t in range(self.T):
        #    meas[t,:] = self.y[t,:] - self.C.dot(x[t,:])
        meas = self.y - (self.C@x.T).T
        return meas

    def processErrors(self,w,terms):
        errors = np.zeros((self.T,self.m))
        for m in range(self.m):
            norm = np.einsum('ij,ij->i', terms[:,m,:]@self.Qinv, terms[:,m,:]) # the norm
            # np.einsum('ij,ij->i',x,x) is equivalent to np.linalg.norm(x,axis=1)**2
            errors[:,m] = w[:,m]*self.r/2.0*np.log(1+ norm/self.r)
        #for t in range(self.T):
            #for i in range(self.m):
        #        errors[t,i] = w[t,i]*self.r/2.0*np.log(1 + np.dot(self.Qinv.dot(terms[t,i,:]),terms[t,i,:])/self.r)
        return errors

    def measurementErrors(self,terms):
        #errors = np.zeros(self.T)
        #for t in range(self.T):
        #    errors[t] = np.dot(self.Rinv.dot(terms[t,:]),terms[t,:])/2.0
        errors = np.einsum('ij,ij->i', terms@self.Rinv, terms)/2.
        return errors

    def computeObj(self,x,w,nu,beta,verbose = False):
        pt = self.processTerms(x)
        pe = np.sum(self.processErrors(w,pt))
        mt = self.measurementTerms(x)
        me = np.sum(self.measurementErrors(mt))

        se = nu/2.0*(np.linalg.norm(w[1:,:] - w[:-1,:],ord='fro')**2 + np.linalg.norm(w[0,:])**2)

        if verbose:
            print("process error = ",pe," measurement error = ",me, " smoothing term = ",se)

        return pe + me + se + beta/2.0*np.linalg.norm(w)**2

    def buildGNMatrices(self,x,w):
        '''
        construct Gauss-Newton matrices
        '''
        pt = self.processTerms(x) # p[t] = x[t] - G[t]x[t-1]
        mt = self.measurementTerms(x)

        coefs = np.zeros((self.T,self.m))
        #for t in range(self.T):
        #    for i in range(self.m):
        #        coefs[t,i] = w[t,i]*self.r/(self.r + np.dot(self.Qinv.dot(pt[t,i,:]),pt[t,i,:]))
        for i in range(self.m):
            norm = np.einsum('ij,ij->i', pt[:,i,:]@self.Qinv, pt[:,i,:])
            coefs[:,i] = w[:,i]*self.r/(self.r + norm)
        dGs = [[self.dG[t][i](x[t-1,:]) for i in range(self.m)] for t in range(1,self.T)]
        dGs.insert(0,[self.dG[0][i](self.x0) for i in range(self.m)])

        gx = []
        Ss = []
        Ds = []

        for t in range(self.T):
            gxt = - np.transpose(self.C).dot(self.Rinv.dot(mt[t,:]))
            S = np.zeros((self.d,self.d))
            D = np.transpose(self.C).dot(self.Rinv).dot(self.C)
            for i in range(self.m):
                if t < self.T - 1:
                    gxt += coefs[t,i]*self.Qinv.dot(pt[t,i,:]) - coefs[t+1,i]*np.dot(np.transpose(dGs[t+1][i]),self.Qinv.dot(pt[t+1,i,:]))
                    D += coefs[t,i]*self.Qinv + coefs[t+1,i]*np.transpose(dGs[t+1][i]).dot(self.Qinv).dot(dGs[t+1][i])
                    #D += coefs[t,i]*self.Qinv + coefs[t+1,i]*np.einsum('ji,jk,kl->il',dGs[t+1][i],self.Qinv,dGs[t+1][i])
                    S -= coefs[t+1,i]*np.dot(np.transpose(dGs[t+1][i]),self.Qinv)
                else:
                    gxt += coefs[t,i]*self.Qinv.dot(pt[t,i,:])
                    D += coefs[t,i]*self.Qinv
                #if t > 0:
                #    S -= coefs[t,i]*self.Qinv.dot(dGs[t][i])

            gx.append(gxt)
            Ss.append(np.transpose(S))
            Ds.append(D)
        return gx,Ds,Ss


    # def solveTridiag(self,gx,dM,sM):
    #     N = len(dM)
    #     iM = [np.linalg.inv(dM[0])]
    #     for i in range(1,N):
    #         iM.append(np.linalg.inv(dM[i]-sM[i-1].dot(iM[i-1]).dot(np.transpose(sM[i-1]))))
    #         gx[i] = gx[i]-sM[i-1].dot(iM[i-1].dot(gx[i-1]))
    #     dx = [iM[-1].dot(gx[-1])]
    #     for i in range(N-2,-1,-1):
    #         gx[i] = -np.transpose(sM[i]).dot(dx[0]) + gx[i]
    #         dx.insert(0,iM[i].dot(gx[i]))
    #     return dx

    def solveGNDirection(self,x,w,verbose = False):
        gx,Ds,Ss = self.buildGNMatrices(x,w)
        gx_copy = copy.copy(gx)
        dx = np.asarray(solveTridiag.solveTridiag(gx,Ds,Ss))
        if verbose:
            print(np.linalg.norm(np.asarray(solveTridiag.multiplyMV(Ds,Ss,dx)) - np.asarray(gx_copy))," ",np.linalg.norm(np.asarray(gx_copy)))
        return dx

    def updateX(self,x,w,iter = 1):
        f = lambda x:self.computeObj(x,w,0.0,0.0)
        loss = []
        for k in range(iter):
            dx = self.solveGNDirection(x,w)
            x = lineSearch.lineSearch(x,dx,f)
            loss.append(f(x))
        return x,loss

    def computeGradw(self,w,x,nu,beta):
        pt = self.processTerms(x)
        #gw = self.processErrors(np.ones((self.T,self.m)),pt) + beta*w
        #for t in range(self.T):
        #    if t == 0:
        #        gw[t,:] += nu*(2*w[t,:]-w[t+1,:])
        #    elif t == self.T - 1:
        #        gw[t,:] += nu*(w[t,:] - w[t-1,:])
        #    else:
        #        gw[t,:] += nu*(2*w[t,:]-w[t-1,:]-w[t+1,:])
        gw = self.processErrors(np.ones((self.T,self.m)),pt) + beta*w
        gw[0,:] += nu*(2*w[0,:]-w[0+1,:])
        gw[self.T-1,:] += nu*(w[self.T-1,:] - w[self.T-2,:])
        gw[1:-1,:] += nu*(2*w[1:-1]-w[0:-2,:]-w[2:,:])
        return gw

    def updateW(self,w,x,nu,beta,iter = 5):
        L = beta + 4*nu
        g = lambda w: self.computeGradw(w,x,nu,beta)
        f = lambda w: self.computeObj(x,w,nu,beta)

        def prox(z):
            for t in range(self.T):
                z[t,:] = simplex.simplexConstr(z[t,:])
            return z

        return FISTA.FISTA(w,g,prox,f,L,iter = iter)

    # def updateW(self,w,x,nu,beta,iter = 5):
    #     loss = []
    #     L = beta + 4*nu
    #     y = copy.copy(w)
    #     wp = copy.copy(w)
    #     t = 1
    #     for k in range(iter):
    #         gy = self.computeGradw(y,x,nu,beta)
    #         for t in range(self.T):
    #             wp[t,:] = simplex.simplexConstr(y[t,:] - gy[t,:]/L)
    #         tp = (1+np.sqrt(1+4*t**2))/2.0
    #         y = wp + (t-1)/tp*(wp-w)
    #         w = copy.copy(wp)
    #         t = tp
    #         loss.append(self.computeObj(x,w,nu,beta))
    #     return w,loss

    def gradTest(self,x,w,nu,beta,pert = 1e-1):

        def func(z):
            zz = np.reshape(z,(self.T,self.d))
            fx = self.computeObj(zz,w,0.0,0.0)
            gx,_,_ = self.buildGNMatrices(zz,w)
            return fx,np.concatenate(gx)
        gradientTest.gradientTest(func,np.random.randn(self.T*self.d),pert)

        def func2(z):
            zz = np.reshape(z,(self.T,self.m))
            fw = self.computeObj(x,zz,nu,beta)
            gw = self.computeGradw(zz,x,nu,beta)
            return fw,gw.flatten()

        gradientTest.gradientTest(func2,np.random.rand(self.T*self.m),pert)

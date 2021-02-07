import math
import numpy as np
from scipy.stats import norm
import scipy.optimize as so
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from UtilityFunctions import *

class MultiStepModel:
    def __init__(self, u_lmb, nX, S0, logvols, Ts, ref_vol, ref_T):
        vol_cutoff= 1e-4
        prob_cutoff = 1e-3

        if nX % 2 == 0:
            nX = nX + 1
        self.nX = nX

        self.set_risk_aversion(u_lmb)

        self.S0 = S0
        self.Ts = Ts
        self.ref_vol = ref_vol
        self.ref_T = ref_T

        self.ref_std = max(self.ref_vol * math.sqrt(self.ref_T),vol_cutoff) * self.S0
        
        s_min = norm.ppf(prob_cutoff, loc=0,scale = self.ref_std)
        s_max = - s_min
        self.xgrid, _ = np.linspace(s_min, s_max,nX,retstep=True)

        self.set_vols(logvols, vol_cutoff)

    def set_risk_aversion(self, u_lmb):
        self.lmb = u_lmb

    def set_vols(self, logvols, vol_cutoff = 1e-4):
        nT = logvols.shape[0]
        self.logvols = logvols
        self.stds = np.maximum(logvols * np.sqrt(self.Ts),np.ones(nT)*vol_cutoff) * self.S0

        # A-D prices/risk neutral probs
        self.qs = []
        for n in np.arange(nT):
            q = norm.pdf(self.xgrid, loc=0,scale=self.stds[n])
            q = q/np.sum(q)
            self.qs.append(q)

        # transition prob priors -- populate with a Gaussian-ish prior
        self.qqs_prior = []
        for n in np.arange(nT-1):
            q12_prior = np.zeros((self.nX,self.nX))
            std12 = math.sqrt(self.stds[n+1]*self.stds[n+1] - self.stds[n]*self.stds[n])
            for i in np.arange(self.nX):
                for j in np.arange(self.nX):
                    q12_prior[i,j] = norm.pdf(self.xgrid[j], loc =self.xgrid[i], scale = std12)
                q12_prior[i,:] = q12_prior[i,:] / np.sum(q12_prior[i,:])

            self.qqs_prior.append(q12_prior)

    def fit_prior_one_step(self, q12_prior, n):

        nX = self.nX
        q12_prior_v = np.reshape(q12_prior, (self.nX*self.nX,))

        # objective function
        def obj_f(q12_v):

            w_entr = 1e-4

            q12_m = np.reshape(q12_v,(self.nX,self.nX))
            q2_t = np.dot(self.qs[n], q12_m)
            q2_diff = q2_t - self.qs[n+1]

            resid = np.zeros(nX*nX + nX + nX)
            resid[:nX*nX] = w_entr*(q12_prior_v - q12_v)
            resid[nX*nX:nX*nX+nX] = q2_diff
            resid[nX*nX+nX:] = np.dot(q12_m,np.ones(nX)) - 1

            return resid

        lo_b = np.ones(nX*nX) * 0.0
        hi_b = np.ones(nX*nX) * (1.0)

        res = so.least_squares(obj_f,q12_prior_v,bounds = (lo_b,hi_b))
        return np.reshape(res.x,(nX,nX))

        showResid = False
        if showResid:
            print('resid\n')
            print(res.fun)
        
            plt.plot(res.fun*100)
            plt.show()

    def fit_priors(self, qqs_prior):
        self.qqs = []
        for n in np.arange(len(qqs_prior)):
            self.qqs.append(self.fit_prior_one_step(qqs_prior[n], n))

import math
import numpy as np
from scipy.stats import norm
import scipy.optimize as so
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from UtilityFunctions import *


class TwoStepModel:
    def __init__(self, u_lmb, nX, S0, logvol1, T1, logvol2, T2, ref_vol):
        vol_cutoff = 1e-4
        prob_cutoff = 1e-3

        if nX % 2 == 0:
            nX = nX + 1
        self.nX = nX

        self.lmb = u_lmb

        self.S0 = S0
        self.logvol1 = logvol1
        self.T1 = T1
        self.logvol2 = logvol2
        self.T2 = T2
        self.ref_vol = ref_vol

        self.ref_std = max(self.ref_vol * math.sqrt(T2), vol_cutoff) * S0

        self.std1 = max(logvol1 * math.sqrt(T1), vol_cutoff) * S0
        self.std2 = max(logvol2 * math.sqrt(T2), vol_cutoff) * S0

        s_min = norm.ppf(prob_cutoff, loc=0, scale=self.ref_std)
        s_max = - s_min

        # [-1,1] to [-1,1] function for grid spacing
        def gmpd(x): return x*abs(x)

#        gmpi = math.sqrt(x) if x >= 0 else -sqrt(-x)  # do not need?
        unigrid = np.linspace(-1, 1, nX)
        self.xgrid = np.vectorize(gmpd)(unigrid)*s_max
#        self.xgrid, _ = np.linspace(s_min, s_max, nX, retstep=True)

        # A-D prices/risk neutral probs
        self.q1 = norm.pdf(self.xgrid, loc=0, scale=self.std1)
        self.q1 = self.q1/np.sum(self.q1)

        self.q2 = norm.pdf(self.xgrid, loc=0, scale=self.std2)
        self.q2 = self.q2/np.sum(self.q2)

# self.util = lambda x:utility(x,self.lmb,scale = self.S0) # rescale utility by S0 so results in % of S0
# self.util_inv = lambda u:utility_inverse(u,self.lmb,scale = self.S0) # rescale inverse utility by S0 so results in % of S0
#        self.util = lambda x:utility(x,self.lmb)
#        self.util_inv = lambda u:utility_inverse(u,self.lmb)

        # transition prob priors -- populate with a Gaussian-ish prior
        self.q12_prior = np.zeros((nX, nX))
        std12 = math.sqrt(self.std2*self.std2 - self.std1*self.std1)
        for i in np.arange(nX):
            for j in np.arange(nX):
                self.q12_prior[i, j] = norm.pdf(
                    self.xgrid[j], loc=self.xgrid[i], scale=std12)
            self.q12_prior[i, :] = self.q12_prior[i, :] / \
                np.sum(self.q12_prior[i, :])

        self.q12 = self.q12_prior.copy()

    def set_risk_aversion(self, u_lmb):
        self.lmb = u_lmb

    def fit_prior(self, q12_prior):

        nX = self.nX
        q12_prior_v = np.reshape(q12_prior, (self.nX*self.nX,))

        # objective function
        def obj_f(q12_v):

            w_entr = 1e-4

            q12_m = np.reshape(q12_v, (self.nX, self.nX))
            q2_t = np.dot(self.q1, q12_m)
            q2_diff = q2_t - self.q2

            resid = np.zeros(nX*nX + nX + nX)
            resid[:nX*nX] = w_entr*(q12_prior_v - q12_v)
            resid[nX*nX:nX*nX+nX] = q2_diff
            resid[nX*nX+nX:] = np.dot(q12_m, np.ones(nX)) - 1

            return resid

        lo_b = np.ones(nX*nX) * 0.0
        hi_b = np.ones(nX*nX) * (1.0)

        res = so.least_squares(obj_f, q12_prior_v, bounds=(lo_b, hi_b))
        self.q12 = np.reshape(res.x, (nX, nX))

        showResid = False
        if showResid:
            print('resid\n')
            print(res.fun)

            plt.plot(res.fun*100)
            plt.show()

# fit the transition probs to minimize whatever is returned by the pricing_func
# with the prior being a regularizer
    def fit_to_min_value(self, pricing_func, q12_prior,
                         prior_w=1e-6, q2_fit_w=1e4, prob_w=1e3, mtg_w=1e2, mtg_scale=1.0, with_plots=False, sparce_initval=True):

        nX = self.nX
        q12_prior_v = np.reshape(q12_prior, (self.nX*self.nX,))
        q12_zero_prior = np.zeros(self.nX*self.nX)
        q12_eye_prior = np.reshape(
            0.99*np.eye(self.nX)+0.005, (self.nX*self.nX,))

        q12_prior_to_use = q12_eye_prior if sparce_initval else q12_prior_v

        # objective function
        def obj_f(q12_v):

            q12_m = np.reshape(q12_v, (self.nX, self.nX))
            q2_t = np.dot(self.q1, q12_m)
            q2_diff = q2_t - self.q2

            self.q12 = q12_m.copy()

            resid = np.zeros(1 + nX*nX + nX + nX + nX)
            resid[0] = pricing_func(self)
#            resid[1:1+nX*nX] = prior_w*(q12_prior_v - q12_v)
            # simulating L1 objective
            resid[1:1+nX*nX] = prior_w*np.abs(q12_zero_prior - q12_v)
            resid[1+nX*nX:1+nX*nX+nX] = q2_fit_w*q2_diff
            resid[1+nX*nX+nX:1+nX*nX+nX+nX] = prob_w * \
                (np.dot(q12_m, np.ones(nX)) - 1)
            tgt_xgrid = np.minimum(np.maximum(
                mtg_scale * self.xgrid, self.xgrid[0]), self.xgrid[-1])
            resid[1+nX*nX+nX+nX:] = mtg_w * \
                (np.dot(q12_m, (self.xgrid)) - tgt_xgrid)

#            print(f'v={resid[0]}')

            return resid

        lo_b = np.ones(nX*nX) * 0.0
        hi_b = np.ones(nX*nX) * (1.0)

        res = so.least_squares(obj_f, q12_prior_to_use, bounds=(lo_b, hi_b))
        self.q12 = np.reshape(res.x, (nX, nX))

        if with_plots:
            print(f'value={res.fun[0]}')
#            print(f'q12 resid={res.fun[1:1+nX*nX]}')
#            plt.plot(res.fun[1+nX*nX:1+nX*nX+nX], label='q2 resid')
            plt.plot(self.xgrid, np.dot(self.q1, self.q12),
                     '.-', label='q2 model')
            plt.plot(self.xgrid, self.q2, '.-', label='q2 mkt')
            plt.legend(loc="upper left")
            plt.show()

            plt.plot(self.xgrid, np.dot(self.q12, np.ones(nX)), '.-',
                     label='T2 prob model')
            plt.legend(loc="upper left")
            plt.show()

            plt.plot(self.xgrid, res.fun[1+nX*nX +
                                         nX+nX:], '.-', label='mtg resid with weight')
            plt.legend(loc="upper left")
            plt.show()

            for i in np.arange(0, self.q12.shape[0], 2):
                plt.plot(self.xgrid, self.q12[i, :], '.-', label=f'i={i}')
            plt.legend(loc="upper left")
            plt.show()

            meanX = np.dot(self.q12, self.xgrid)
            meanX2 = np.dot(self.q12, self.xgrid*self.xgrid)
            meanX3 = np.dot(self.q12, self.xgrid*self.xgrid*self.xgrid)
            stdX = np.sqrt(meanX2-meanX*meanX)
            mu3 = meanX3 - 3*meanX*meanX2+2*meanX*meanX*meanX
            skewX = mu3/(stdX*stdX*stdX)
            plt.plot(self.xgrid, meanX, '.-', label='cond mean')
            plt.plot(self.xgrid, stdX, '.-', label='cond std')
            plt.plot(self.xgrid, skewX*100, '.-', label='cond skew x 100')
            plt.legend(loc="upper left")
            plt.show()

#            plt.plot(res.fun[1:]*100)
#            plt.show()

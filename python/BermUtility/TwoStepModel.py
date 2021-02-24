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

    def fit_prior(self, q12_prior,
                  prior_w=1e-4, mtg_scale=None, mtg_w=1e-2, min_prob=0.0, with_plots=False):

        nX = self.nX
        q12_prior_v = np.reshape(q12_prior, (self.nX*self.nX,))

        # objective function
        def obj_f(q12_v):

            q12_m = np.reshape(q12_v, (self.nX, self.nX))
            q2_t = np.dot(self.q1, q12_m)
            q2_diff = q2_t - self.q2

            if mtg_scale is None:
                resid = np.zeros(nX*nX + nX + nX)
            else:
                resid = np.zeros(nX*nX + nX + nX + nX)
            resid[:nX*nX] = prior_w*(q12_prior_v - q12_v)
            resid[nX*nX:nX*nX+nX] = q2_diff
            resid[nX*nX+nX:nX*nX+nX+nX] = np.dot(q12_m, np.ones(nX)) - 1

            if mtg_scale is not None:
                tgt_xgrid = np.minimum(np.maximum(
                    mtg_scale * self.xgrid, self.xgrid[0]), self.xgrid[-1])
                resid[nX*nX+nX+nX:] = mtg_w * \
                    (np.dot(q12_m, (self.xgrid)) - tgt_xgrid)

            return resid

        lo_b = np.ones(nX*nX) * min_prob
        hi_b = np.ones(nX*nX) * (1.0)

        res = so.least_squares(obj_f, q12_prior_v, bounds=(lo_b, hi_b))
        self.q12 = np.reshape(res.x, (nX, nX))

        showResid = False
        if showResid:
            print('resid\n')
            print(res.fun)

            plt.plot(res.fun*100)
            plt.show()

        if with_plots:
            self.standard_plots(mtg_scale)

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
            resid[1:1+nX*nX] = prior_w*np.abs(q12_prior_to_use - q12_v)
            resid[1+nX*nX:1+nX*nX+nX] = q2_fit_w*q2_diff
            resid[1+nX*nX+nX:1+nX*nX+nX+nX] = prob_w * \
                (np.dot(q12_m, np.ones(nX)) - 1)
            tgt_xgrid = np.minimum(np.maximum(
                mtg_scale * self.xgrid, self.xgrid[0]), self.xgrid[-1])
            resid[1+nX*nX+nX+nX:] = mtg_w * \
                (np.dot(q12_m, (self.xgrid)) - tgt_xgrid)

            return resid

        lo_b = np.ones(nX*nX) * 0.0
        hi_b = np.ones(nX*nX) * (1.0)

        res = so.least_squares(obj_f, q12_prior_to_use, bounds=(lo_b, hi_b))
        self.q12 = np.reshape(res.x, (nX, nX))

        print(f'value={res.fun[0]}')
        if with_plots:
            self.standard_plots(mtg_scale=mtg_scale)

            plt.plot(self.xgrid, res.fun[1+nX*nX +
                                         nX+nX:], '.-', label='mtg resid with weight')
            plt.legend(loc="upper left")
            plt.show()

    def fit_to_min_value_2(self, pricing_func,
                           mtg_scale=1.0, with_plots=False):
        '''
        same as fit_to_min_value but using constrained minimization
        with equality constraints. looks faster than fit_to_min_value
        '''

        nX = self.nX

        # basic ocnstraints
        mtr_rhs = self.standard_constraints(mtg_scale=mtg_scale)
#        A1, rhs1 = mtr_rhs[0]
#        A2, rhs2 = mtr_rhs[1]
#        A3, rhs3 = mtr_rhs[2]
#        constraints = [
#            {'type': 'eq', 'fun': lambda x: A1 @ x - rhs1},
#            {'type': 'eq', 'fun': lambda x: A2 @ x - rhs2},
#            {'type': 'eq', 'fun': lambda x: A3 @ x - rhs3},
#        ]

        # we need to play  some funny games to make sure
        # the iterator variables stay in scope, see
        # https://stackoverflow.com/questions/28014953/capturing-value-instead-of-reference-in-lambdas
        constraints = [
            {'type': 'eq', 'fun': (lambda a, r: lambda x: a @ x - r)(A, rhs)}
            for A, rhs in mtr_rhs
        ]

        # these support constraints in the ory, but it seems on SLSQP is doing something useful
        method = 'SLSQP'
        # method = 'COBYLA'
        # method = 'trust-constr'

        def obj_f(q12_v):

            q12_m = np.reshape(q12_v, (self.nX, self.nX))
            self.q12 = q12_m.copy()
            return pricing_func(self)

        lo_b = np.ones(nX*nX) * 0.0
        hi_b = np.ones(nX*nX) * 1.0

        res = so.minimize(obj_f, self.q12.reshape(-1), method=method,
                          bounds=list(zip(lo_b, hi_b)), constraints=constraints)
        self.q12 = np.reshape(res.x, (nX, nX))

        print(f'value={res.fun}')
        if with_plots:
            self.standard_plots(mtg_scale=mtg_scale)

    def fit_to_min_acceptable_value(self, pricing_func,
                                    mtg_scale=1.0, gamma=0.0, strikes=None,  with_plots=False):
        '''
        Per Madan's paper "Acceptability bounds for forward starting 
        options using disciplined convex programming"

        formulated in terms of z12 which is the density with respect to 
        the initial q12
        '''

        nX = self.nX

        q12_init_v = self.q12.reshape(-1).copy()
        q12_init_d = np.diag(q12_init_v)
        z12_v = np.ones_like(q12_init_v)

        # basic ocnstraints
        mtr_rhs = self.standard_constraints(mtg_scale=mtg_scale)

        # reformula for zs from qs
        mtr_rhs_z = [(A @ q12_init_d, rhs) for A, rhs in mtr_rhs]

        # we need to play  some funny games to make sure
        # the iterator variables stay in scope, see
        # https://stackoverflow.com/questions/28014953/capturing-value-instead-of-reference-in-lambdas
        constraints = [
            {'type': 'eq', 'fun': lambda x, a=A, r=rhs: a @ x - r}
            for A, rhs in mtr_rhs_z
        ]

        #  constraints for acceptability
        if strikes is not None:
            # \Phi in the paper
            if gamma <= 1e-2:
                gamma = 1e-2  # to aviod 0 which is an edge case

            def distortion_conj_f(a):
                eps = 1e-2  # to account for some discretization issues
                return 1.0 + eps - a*math.exp(-gamma*math.log(1+math.pow(a, 1.0/gamma)))

            p12_init_v = np.zeros_like(q12_init_v)
            for n in range(nX):
                p12_init_v[n*nX:(n+1)*nX] = self.q1[n] * \
                    q12_init_v[n*nX:(n+1)*nX]
            for strike in strikes:
                target = distortion_conj_f(strike)
                def acc_bound_f(z, s=strike, t=target): return t - \
                    np.sum(p12_init_v * np.maximum(z-s, 0))
                constraints.append({'type': 'ineq', 'fun': acc_bound_f})

            if with_plots:
                strikes_plt = np.linspace(
                    strikes[0], strikes[-1], 10*(len(strikes)-1)+1)
                acc_val_plt = [np.sum(p12_init_v * np.maximum(z12_v-s, 0))
                               for s in strikes_plt]
                acc_bds_plt = [distortion_conj_f(s) for s in strikes_plt]
                plt.plot(strikes_plt, acc_val_plt, '.-', label='achieved')
                plt.plot(strikes_plt, acc_bds_plt, '.-', label='bound')
                plt.title('Accepability bounds before optimization')
                plt.legend(loc='best')
                plt.show()

        # these support constraints in theory, but it seems on SLSQP is doing something useful
        method = 'SLSQP'
        # method = 'COBYLA'
        # method = 'trust-constr'

        def obj_f(z12_v):

            q12_v = q12_init_v * z12_v
            q12_m = np.reshape(q12_v, (self.nX, self.nX))
            self.q12 = q12_m.copy()
            return pricing_func(self)

        lo_b = [0.0] * (nX*nX)
        hi_b = [None] * (nX*nX)

        res = so.minimize(obj_f, z12_v, method=method,
                          bounds=list(zip(lo_b, hi_b)), constraints=constraints)
        self.q12 = np.reshape(q12_init_v * res.x, (nX, nX))

        print(f'value={res.fun}')
        if with_plots:
            self.standard_plots(mtg_scale=mtg_scale)

            z12_m = res.x.reshape(nX, nX)
            for i in np.arange(0, z12_m.shape[0], 2):
                plt.plot(self.xgrid, z12_m[i, :], '.-', label=f'i={i}')
            plt.title('density Z')
            plt.legend(loc="best")
            plt.show()

            if strikes is not None:
                strikes_plt = np.linspace(
                    strikes[0], strikes[-1], 10*(len(strikes)-1)+1)
                acc_val_plt = [np.sum(p12_init_v * np.maximum(res.x-s, 0))
                               for s in strikes_plt]
                acc_bds_plt = [distortion_conj_f(s) for s in strikes_plt]
                plt.plot(strikes_plt, acc_val_plt, '.-', label='achieved')
                plt.plot(strikes_plt, acc_bds_plt, '.-', label='bound')
                plt.title('Accepability bounds after optimization')
                plt.legend(loc='best')
                plt.show()

    def standard_constraints(self, mtg_scale=1.0):
        nX = self.nX

        # constraints are of the form
        # lb <= A.dot(x) <= ub
        # x is q12 reshaped into a vector
        # reshaping is row-major ie the vector is [row1,...,rowN]

        # sums along rows are 1
        A1 = np.zeros((nX, nX*nX))
        for n in range(nX):
            A1[n, n*nX:(n+1)*nX] = 1.0
        rhs1 = np.ones(nX)

        # recovery of probabilities at T2
        # np.dot(q1, q12) = q2
        A2 = np.zeros((nX, nX*nX))
        for n in range(nX):
            A2[n, n::nX] = self.q1
        # can do smth like np.hstack(np.diag(q1[0]),...,np.diag(q1[nX-1])) or smth similar
        rhs2 = self.q2

        # check if the initial point satisfies this constraint
        # plt.plot(self.xgrid, A2 @ self.q12.reshape(-1), '.-', label='expected')
        # plt.plot(self.xgrid, self.q2, '.-', label='actual')
        # plt.title('q2 constraint')
        # plt.legend(loc='best')
        # plt.show()

        # Now martingale condition
        # in matrix form
        # tgt_xgrid = np.minimum(np.maximum(
        #       mtg_scale * self.xgrid, self.xgrid[0]), self.xgrid[-1])
        # resid[nX*nX+nX+nX:] = mtg_w * \
        #       (np.dot(q12_m, (self.xgrid)) - tgt_xgrid)

        A3 = np.zeros((nX, nX*nX))
        for n in range(nX):
            A3[n, n*nX:(n+1)*nX] = self.xgrid

        tgt_xgrid = np.minimum(np.maximum(
            mtg_scale * self.xgrid, self.xgrid[0]), self.xgrid[-1])

        # use scaled eps as xgrid is of scale 100s whereas others are of scale 1s
        rhs3 = tgt_xgrid

        return [(A1, rhs1), (A2, rhs2), (A3, rhs3)]
        # return [(A1, rhs1), (A3, rhs3)]

    def standard_plots(self, mtg_scale):
        nX = self.nX

        plt.plot(self.xgrid, np.dot(self.q1, self.q12),
                 '.-', label='q2 model')
        plt.plot(self.xgrid, self.q2, '.-', label='q2 mkt')
        plt.legend(loc="upper left")
        plt.title('q2 prob model vs mkt')
        plt.show()

        plt.plot(self.xgrid, np.dot(self.q12, np.ones(nX)), '.-',
                 label='T2 prob model')
        plt.legend(loc="upper left")
        plt.show()

        if mtg_scale is not None:
            plt.plot(self.xgrid, np.dot(self.q12, (self.xgrid)), label='EX2')
            plt.plot(self.xgrid, mtg_scale * self.xgrid, label='target')
            plt.legend(loc='best')
            plt.title(f'Martingale constraint with ms = {mtg_scale}')
            plt.show()

        for i in np.arange(0, self.q12.shape[0], 2):
            plt.plot(self.xgrid, self.q12[i, :], '.-', label=f'i={i}')
        plt.legend(loc="upper left")
        plt.title(f'transition matrix Q12')
        plt.show()

        plot_cond_moments = False
        if plot_cond_moments:
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

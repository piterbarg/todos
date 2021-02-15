
import math
import copy
import numpy as np
import scipy.optimize as so
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from TwoStepModel import *
from MultiStepModel import *


class Bermudan():

    def __init__(self, strikes, scales, isExercise, isPayers=True, notional=1):
        self.strikes = strikes
        self.scales = scales
        self.isExercise = isExercise
        self.isPayers = isPayers
        self.notional = notional

    @classmethod
    def create(cls, strikes, scales, isExercise, isPayers=True, notional=1):
        return cls(strikes, scales, isExercise, isPayers, notional)

    @classmethod
    def create_canary(cls, strike1, strike2, scale1=1.0, scale2=1.0, isPayers=True, notional=1, ex1=True, ex2=True, ):
        return cls(np.array([strike1, strike2]), np.array([scale1, scale2]), np.array([ex1, ex2]), isPayers, notional)

    def exercise_value(self, sgrid, time_index, floor_at_zero=False):
        strike = self.strikes[time_index]
        scale = self.scales[time_index]

        sign = 1 if self.isPayers else -1

        ret = sign*scale*(sgrid - strike)
        if floor_at_zero:
            ret = np.maximum(ret, np.zeros(ret.shape))
        ret = ret*self.notional

        return ret

    # ad_pos is a (long) position in arrow-debrew securities over the two time horizon, with the dimension ngrid x 2

    def price_2(self, model: TwoStepModel, ad_pos=None, with_graphs=False):
        sgrid = model.S0 + model.xgrid

        if ad_pos is None:
            ad_pos = np.zeros((model.nX, 2))

        util_v_2 = np.vectorize(lambda x: utility(x, model.lmb, model.T2))
        hold2 = np.zeros(model.nX)
        exercise2 = util_v_2(self.exercise_value(
            sgrid, 1) + ad_pos[:, 1]) - util_v_2(ad_pos[:, 1])

        if with_graphs:
            plt.plot(sgrid, exercise2, '.-')
            plt.plot(sgrid, hold2, '.-')
            plt.show()

        util_v_1 = np.vectorize(lambda x: utility(x, model.lmb, model.T1))
        hold1 = np.dot(model.q12, np.maximum(hold2, exercise2)
                       if self.isExercise[1] else hold2)
        exercise1 = util_v_1(self.exercise_value(
            sgrid, 0) + ad_pos[:, 0]) - util_v_1(ad_pos[:, 0])

        if with_graphs:
            plt.plot(sgrid, exercise1, '.-')
            plt.plot(sgrid, hold1, '.-')
            plt.show()

        val = np.dot(model.q1, np.maximum(hold1, exercise1)
                     if self.isExercise[0] else hold1)
        val = val \
            + np.dot(model.q1, util_v_1(ad_pos[:, 0]) + np.dot(model.q12, util_v_2(ad_pos[:, 1])))\
            - np.dot(model.q1, ad_pos[:, 0]) - np.dot(model.q2, ad_pos[:, 1])
        return val

    # ad_pos is a (long) position in arrow-debrew securities over exercise dates, with the dimension ngrid x nExercise
    def price_3(self, model: MultiStepModel, ad_pos=None, with_graphs=False):
        sgrid = model.S0 + model.xgrid

        exercise_dates = model.Ts  # argh this is really bad; should be the other way around
        nE = len(exercise_dates)
        if ad_pos is None:
            ad_pos = np.zeros((model.nX, nE))

        val = 0
        eur_acct = 0
        hold = np.zeros(model.nX)
        for n in np.arange(nE-1, -1, -1):
            util_v = np.vectorize(lambda x: utility(x, model.lmb, model.Ts[n]))
            exercise = util_v(self.exercise_value(sgrid, n) +
                              ad_pos[:, n]) - util_v(ad_pos[:, n])
            eur_acct = eur_acct + \
                np.dot(model.qs[n], util_v(ad_pos[:, n])) - \
                np.dot(model.qs[n], ad_pos[:, n])

            if n > 0:
                hold = np.dot(
                    model.qqs[n-1], np.maximum(hold, exercise) if self.isExercise[n] else hold)
            else:
                val = np.dot(model.qs[0], np.maximum(
                    hold, exercise) if self.isExercise[0] else hold)

            if with_graphs:
                plt.plot(sgrid, exercise, '.-')
                plt.plot(sgrid, hold, '.-')
                plt.show()

        val = val + eur_acct
        return val

    # this is an implementation of the method from Sec 6 of my paper, eq (9)
    # ad_pos is ignored here

    def price_4(self, model: MultiStepModel, ad_pos=None, with_graphs=False):
        sgrid = model.S0 + model.xgrid

        exercise_dates = model.Ts  # argh this is really bad; should be the other way around
        nE = len(exercise_dates)
        if ad_pos is None:
            ad_pos = np.zeros((model.nX, nE))

        val = 0
        eur_acct = 0
        hold = np.zeros(model.nX)
        euros = np.zeros((nE, model.nX))
        exercise = np.zeros(model.nX)
        for n in np.arange(nE-1, -1, -1):
            util_v = np.vectorize(lambda x: utility(x, model.lmb, model.Ts[n]))

            euros[n, :] = np.maximum(
                self.exercise_value(sgrid, n), np.zeros(model.nX))
            for k in np.arange(n+1, nE):
                euros[k, :] = np.dot(model.qqs[n], euros[k, :])

            max_euro = np.amax(euros[np.array(self.isExercise), :], 0)
#            max_euro = euros[n,:] if self.isExercise[n] else np.zeros(model.nX)
            hold_u = util_v(hold - max_euro)
#            hold_u= np.zeros(hold.shape)

            if n > 0:
                hold = np.dot(model.qqs[n-1], np.maximum(hold_u, exercise) +
                              max_euro if self.isExercise[n] else hold_u + max_euro)
            else:
                val = np.dot(model.qs[0], np.maximum(
                    hold_u, exercise) + max_euro if self.isExercise[0] else hold_u + max_euro)

            if with_graphs:
                plt.plot(sgrid, exercise, '.-')
                plt.plot(sgrid, hold, '.-')
                plt.show()

        euro_vals = [np.dot(model.qs[0], euros[k, :]) for k in np.arange(nE)]
        val = val + eur_acct
        return val, euro_vals

    # ad_pos is a (long) position in arrow-debrew securities over the two time horizon, with the dimension ngrid x 2
    def price(self, model: MultiStepModel, ad_pos=None, with_graphs=False):
        return self.price_3(model, ad_pos, with_graphs)
        # price_4 is different from others as it does not depen on ad_pos, it uses eq (9) of my paper
#        return self.price_4(model, ad_pos, with_graphs)

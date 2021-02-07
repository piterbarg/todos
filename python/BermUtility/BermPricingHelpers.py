import math
import copy
import numpy as np
import scipy.optimize as so
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from UtilityFunctions import *
from MultiStepModel import *
from Bermudan import *


def price_bermudan_optimal(model, berm: Bermudan, ad_pos_0=None):

    reg_w = 1e-6

    def target(ad_pos_flat: np.ndarray):
        #        ngrid = ad_pos_flat.shape[0]//2
        ngrid = model.nX
        ad_pos = np.reshape(ad_pos_flat, (ngrid, 2))
        test_val = berm.price(model, ad_pos)
        # want a maximum not minimum, and add a bit of regularization
        return -test_val + reg_w*np.linalg.norm(ad_pos_flat)

    if ad_pos_0 is None:
        ad_pos_0 = np.zeros((model.nX, 2))

    res = so.minimize(target, ad_pos_0)
    opt_ad_pos = np.reshape(res.x, (model.nX, 2))
    val = -res.fun  # actually should take out the regularization term; but it is small at the moment

    return (val, opt_ad_pos)


def price_berm_and_euros(model, berm: Bermudan):
    base_model = copy.deepcopy(model)
    base_model.set_risk_aversion(0.0)

    nT = model.Ts.shape[0]
    nT2 = 2 * nT
    res = [None]*(nT2+2)

    res[0], eur_base = berm.price(base_model)
    res[nT + 1], eur_std = berm.price(model)
    for n in np.arange(nT):
        eur_n = copy.deepcopy(berm)
        eur_n.isExercise = [False] * len(eur_n.isExercise)
        eur_n.isExercise[n] = True

        res[n + 1], _ = eur_n.price(base_model)
        res[nT + 1 + n + 1], _ = eur_n.price(model)

    res = res + eur_base + eur_std
    return res


def price_bermudan_opt_eur_hedge(model, berm: Bermudan, eur_pos_0=None):

    reg_w = 1e-6
    nT = model.Ts.shape[0]
    ex = np.zeros((model.nX, nT))
    for n in np.arange(nT):
        ex[:, n] = -np.maximum(berm.exercise_value(model.S0 +
                                                   model.xgrid, n), np.zeros(model.nX))

    calcEur = True

    no_exercise = [False] * nT
    eur_val_b = []
    if calcEur:
        base_model = copy.deepcopy(model)
        base_model.set_risk_aversion(0.0)

        for n in np.arange(nT):
            eur_exercise = copy.deepcopy(no_exercise)
            eur_exercise[n] = True

            eur = copy.deepcopy(berm)
            eur.isExercise = eur_exercise

            eur_v = eur.price(base_model)
            if hasattr(eur_v, '__len__'):
                eur_v = eur_v[0]
            eur_val_b.append(eur_v)

    def target(eur_pos: np.ndarray):
        #        ad_pos = np.vstack((-ex1*eur_pos[0],-ex2*eur_pos[1])).T
        ad_pos = np.matmul(ex, np.diag(eur_pos))
        test_val = berm.price(model, ad_pos)
        if hasattr(test_val, '__len__'):
            test_val = test_val[0]
        # want a maximum not minimum, and add a bit of regularization
        return -test_val + reg_w*np.linalg.norm(eur_pos)

    if eur_pos_0 is None:
        eur_pos_0 = (1./nT) * np.ones(nT)

    res = so.minimize(target, eur_pos_0,
                      bounds=so.Bounds(-1*np.ones(nT), 2*np.ones(nT)))
    return [-res.fun] + list(res.x) + eur_val_b


# Hedge max european and price
def price_bermudan_max_euro(model, berm: Bermudan, with_graphs=False):

    S0 = model.S0
    nX = model.nX
    base_model = copy.deepcopy(model)
    base_model.set_risk_aversion(0.0)

    eur1 = copy.deepcopy(berm)
    eur1.isExercise = [True, False]
    eur2 = copy.deepcopy(berm)
    eur2.isExercise = [False, True]

#    eur1 = Bermudan(berm.strike1, berm.strike2, berm.scale1, berm.scale2, berm.isPayers, berm.notional, True, False)
#    eur2 = Bermudan(berm.strike1, berm.strike2, berm.scale1, berm.scale2, berm.isPayers, berm.notional, False, True)

    berm_val_b, *_ = berm.price(base_model, with_graphs=with_graphs)
    eur1_val_b, *_ = eur1.price(base_model)
    eur2_val_b, *_ = eur2.price(base_model)

    ex1 = np.maximum(berm.exercise_value(
        S0 + model.xgrid, 0), np.zeros(model.nX))
    ex2 = np.maximum(berm.exercise_value(
        S0 + model.xgrid, 1), np.zeros(model.nX))

    #ex1 = ex1 - np.dot(model.q1,ex1)
    #ex2 = ex2 - np.dot(model.q2,ex2)

    ad_pos_1 = np.vstack((-ex1, np.zeros(nX))).T
    ad_pos_2 = np.vstack((np.zeros(nX), -ex2)).T

    berm_val_1, *_ = berm.price(model, ad_pos_1, with_graphs=with_graphs)
    berm_val_2, *_ = berm.price(model, ad_pos_2, with_graphs=with_graphs)

    return (berm_val_1, berm_val_2, max(berm_val_1, berm_val_2) - max(eur1_val_b, eur2_val_b), eur1_val_b, eur2_val_b)


def bermudan_implied_vol(model, berm: Bermudan, berm_price: float, ad_pos=None, vol_guess=0.1):

    if ad_pos is None:
        ad_pos = np.zeros((model.nX, 2))

    def target(logvol: float):
        test_model = copy.deepcopy(model)
        test_model.set_vols(logvol * np.ones(model.logvols.shape[0]))
        test_model.fit_priors(test_model.qqs_prior)

        test_val = berm.price(test_model, ad_pos)
        return berm_price - test_val

    use_brent = True
    if use_brent:
        implied_vol = so.brentq(target, vol_guess*0.01, vol_guess*10)
        return implied_vol
    else:
        implied_vol = so.root(target, vol_guess).x
        return implied_vol[0]

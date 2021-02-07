import math
import numpy as np
from scipy.stats import norm
import scipy.optimize as so
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import copy

from UtilityFunctions import *
from TwoStepModel import *
from MultiStepModel import *
from Bermudan import *
from BermPricingHelpers import *
from VolHelpers import *
from DualLinProg import *


def setup_model():
    u_lmb = 0.0  # 0.1
    nX = 21
    S0 = 100
    logvol1 = 0.1
    logvol2 = 0.2
    ref_vol = 0.2
    T1 = 10
    T2 = 20

    model = TwoStepModel(u_lmb, nX, S0, logvol1, T1, logvol2, T2, ref_vol)
    model.fit_prior(model.q12_prior)

    return model


def setup_berm():

    strike1 = 100
    strike2 = 100
    scale1 = 1
    scale2 = 0.25
    #strike1 = 100
    #strike2 = 100
    #scale1 = 1
    #scale2 = 0.4
    berm = Bermudan.create_canary(
        strike1, strike2, scale1=scale1, scale2=scale2)

    return berm


def min_measure_01(mtg_scale=1, mtg_w=1e2, with_plots=False):

    model = setup_model()

    base_model = copy.deepcopy(model)
    base_model.set_risk_aversion(0.0)

    berm = setup_berm()

    def price_berm_f(mdl):
        return berm.price_2(mdl)

    prior_w = 0
    q2_fit_w = 1e4
    prob_w = 1e3
    model.fit_to_min_value(price_berm_f, model.q12_prior,
                           prior_w=prior_w, q2_fit_w=q2_fit_w, prob_w=prob_w, mtg_w=mtg_w, mtg_scale=mtg_scale, with_plots=with_plots)

    np.savetxt("./res/min_measure_res_01.csv",
               model.q12, delimiter=",", fmt='%.4f')
    np.savetxt("./res/min_measure_xgrid_01.csv",
               model.xgrid, delimiter=",", fmt='%.4f')

    eur1 = copy.deepcopy(berm)
    eur1.isExercise[1] = False
    eur2 = copy.deepcopy(berm)
    eur2.isExercise[0] = False

    with_berm_graphs = False
    berm_value = berm.price_1(model, with_graphs=with_berm_graphs)
    eur1_value = eur1.price_1(model)
    eur2_value = eur2.price_1(model)

    berm_val_0 = berm.price_1(base_model, with_graphs=with_berm_graphs)
    eur1_val_0 = eur1.price_1(base_model)
    eur2_val_0 = eur2.price_1(base_model)

    print(f'{mtg_scale},{berm_value},{eur1_value},{eur2_value},{berm_val_0},{eur1_val_0},{eur2_val_0}')


def min_measure_02():
    lo_s = 0.5  # 2.75  # 0.5
    hi_s = 2.5  # 4.75  # 2.5
    ns = 9  # 18  # 9

    ss = np.linspace(lo_s, hi_s, ns)
    for mtg_scale in ss:
        min_measure_01(mtg_scale, with_plots=False)


def dual_01(eb=110, fit_min_model=False, with_plots=False):
    model = setup_model()

    base_model = copy.deepcopy(model)
    base_model.set_risk_aversion(0.0)

    berm = setup_berm()

    prior_w = 0
    q2_fit_w = 1e4
    prob_w = 1e3
    mtg_w = 1e-2
    mtg_scale = 3

    if fit_min_model:
        def price_berm_f(mdl):
            return berm.price_2(mdl)
        model.fit_to_min_value(price_berm_f, model.q12_prior,
                               prior_w=prior_w, q2_fit_w=q2_fit_w, prob_w=prob_w, mtg_w=mtg_w, mtg_scale=mtg_scale, with_plots=with_plots)

    dp_res = solve_dual_linprog(berm, base_model, model, eb)
    print(f'{eb}, {dp_res.lp_val}, {dp_res.base_val}, {dp_res.other_val}, {dp_res.e1_val}, {dp_res.e2_val}')

    if with_plots:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(dp_res.sg1, dp_res.sg2, dp_res.bpayoff)
        ax.set_xlabel('t1')
        ax.set_ylabel('t2')
        ax.set_zlabel('pf')
        plt.show()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(dp_res.sg1, dp_res.sg2, dp_res.bpayoff-dp_res.gap)
        ax.set_xlabel('t1')
        ax.set_ylabel('t2')
        ax.set_zlabel('fit to payoff')
        plt.show()

        plt.plot(dp_res.xgrid, dp_res.u1, '.-', label='u1')
        plt.plot(dp_res.xgrid, dp_res.u2, '.-', label='u2')
        plt.legend(loc="upper left")
        plt.show()

    return dp_res


def dual_02():
    lo_eb = 100
    hi_eb = 130
    n_eb = 31  # 18  # 9

    ebs = np.linspace(lo_eb, hi_eb, n_eb)
    for eb in ebs:
        dual_01(eb)


if __name__ == "__main__":
    #    min_measure_01(1.0, with_plots=True)
    # min_measure_02()
    dual_01(105, fit_min_model=True, with_plots=True)
    # dual_02()

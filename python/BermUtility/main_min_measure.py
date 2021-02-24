import math
import numpy as np
from scipy.stats import norm
import scipy.optimize as so
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import copy

from setup_min_measure import *
from UtilityFunctions import *
from TwoStepModel import *
from MultiStepModel import *
from Bermudan import *
from BermPricingHelpers import *
from VolHelpers import *
from DualLinProg import *


def min_measure_01(mtg_scale=1, mtg_w=1e2, with_plots=False):

    model = setup_model(mtg_scale=mtg_scale)

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
    berm_value = berm.price_2(model, with_graphs=with_berm_graphs)
    eur1_value = eur1.price_2(model)
    eur2_value = eur2.price_2(model)

    berm_val_0 = berm.price_2(base_model, with_graphs=with_berm_graphs)
    eur1_val_0 = eur1.price_2(base_model)
    eur2_val_0 = eur2.price_2(base_model)

    print(f'{mtg_scale},{berm_value},{eur1_value},{eur2_value},{berm_val_0},{eur1_val_0},{eur2_val_0}')


def min_measure_02():
    lo_s = 0.5  # 2.75  # 0.5
    hi_s = 2.5  # 4.75  # 2.5
    ns = 9  # 18  # 9

    ss = np.linspace(lo_s, hi_s, ns)
    for mtg_scale in ss:
        min_measure_01(mtg_scale, with_plots=False)


def min_measure_01_b(mtg_scale=1, min_prob=0.0, with_plots=False):

    model = setup_model(mtg_scale=mtg_scale,
                        min_prob=min_prob, with_plots=with_plots)

    base_model = copy.deepcopy(model)
    base_model.set_risk_aversion(0.0)

    berm = setup_berm()

    def price_berm_f(mdl):
        v = berm.price_2(mdl)
        # print(v)
        return v

   # model.fit_to_min_value_2(
    model.fit_to_min_acceptable_value(
        price_berm_f, mtg_scale=mtg_scale, with_plots=with_plots)

    np.savetxt("./res/min_measure_1b_res_01.csv",
               model.q12, delimiter=",", fmt='%.4f')
    np.savetxt("./res/min_measure_1b_xgrid_01.csv",
               model.xgrid, delimiter=",", fmt='%.4f')

    eur1 = copy.deepcopy(berm)
    eur1.isExercise[1] = False
    eur2 = copy.deepcopy(berm)
    eur2.isExercise[0] = False

    with_berm_graphs = False
    berm_value = berm.price_2(model, with_graphs=with_berm_graphs)
    eur1_value = eur1.price_2(model)
    eur2_value = eur2.price_2(model)

    berm_val_0 = berm.price_2(base_model, with_graphs=with_berm_graphs)
    eur1_val_0 = eur1.price_2(base_model)
    eur2_val_0 = eur2.price_2(base_model)

    print(f'{mtg_scale},{berm_value},{eur1_value},{eur2_value},{berm_val_0},{eur1_val_0},{eur2_val_0}')


def min_acc_measure_01(mtg_scale=1, min_prob=0.0, gamma=0.0,
                       min_strike=0.1, max_strike=2.1, n_strikes=5, with_plots=False):

    model = setup_model(mtg_scale=mtg_scale,
                        min_prob=min_prob, with_plots=with_plots)

    base_model = copy.deepcopy(model)
    base_model.set_risk_aversion(0.0)

    berm = setup_berm()

    def price_berm_f(mdl):
        v = berm.price_2(mdl)
        # print(v)
        return v

    strikes = None
    if n_strikes > 0:
        strikes = np.linspace(min_strike, max_strike, n_strikes)

    model.fit_to_min_acceptable_value(
        price_berm_f, mtg_scale=mtg_scale,
        gamma=gamma, strikes=strikes, with_plots=with_plots)

    np.savetxt("./res/min_acc_measure_res_01.csv",
               model.q12, delimiter=",", fmt='%.4f')
    np.savetxt("./res/min_acc_measure_xgrid_01.csv",
               model.xgrid, delimiter=",", fmt='%.4f')

    eur1 = copy.deepcopy(berm)
    eur1.isExercise[1] = False
    eur2 = copy.deepcopy(berm)
    eur2.isExercise[0] = False

    with_berm_graphs = False
    berm_value = berm.price_2(model, with_graphs=with_berm_graphs)
    eur1_value = eur1.price_2(model)
    eur2_value = eur2.price_2(model)

    berm_val_0 = berm.price_2(base_model, with_graphs=with_berm_graphs)
    eur1_val_0 = eur1.price_2(base_model)
    eur2_val_0 = eur2.price_2(base_model)

    print(f'{mtg_scale},{berm_value},{eur1_value},{eur2_value},{berm_val_0},{eur1_val_0},{eur2_val_0}')


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


def test_fit_prior_with_mtg_constraint(mtg_scale=1.0):

    model = setup_model(mtg_scale=mtg_scale)
    for i in np.arange(0, model.q12.shape[0], 2):
        plt.plot(model.xgrid, model.q12[i, :], '.-', label=f'i={i}')
    plt.legend(loc="best")
    plt.show()

    plt.plot(model.xgrid, np.dot(model.q12, (model.xgrid)), label='EX2')
    plt.plot(model.xgrid, mtg_scale * model.xgrid, label='target')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    # min_measure_01(1.0, with_plots=True)
    # min_measure_02()
    # dual_01(105, fit_min_model=True, with_plots=True)
    # dual_02()

    # test_fit_prior_with_mtg_constraint(0.5)
    # test_fit_prior_with_mtg_constraint(1.0)
    # test_fit_prior_with_mtg_constraint(2.0)

    # min_measure_01_b(1.0, min_prob=1e-2, with_plots=True)
    min_acc_measure_01(1.0, min_prob=1e-2, gamma=0.05,
                       min_strike=0.1, max_strike=2.1, n_strikes=11, with_plots=True)

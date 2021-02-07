import copy
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.optimize as so
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from UtilityFunctions import *
from MultiStepModel import *
from Bermudan import *
from BermPricingHelpers import *
from VolHelpers import *


def explore_1():
    u_lmb = 0.1  # 0.01  # 0.1
    nX = 21
    S0 = 100
    logvol1 = 0.1
    logvol2 = 0.2
    ref_vol = 0.2
    T1 = 10
    T2 = 20

    model = MultiStepModel(u_lmb, nX, S0, np.array(
        [logvol1, logvol2]), np.array([T1, T2]), ref_vol, T2)
    model.fit_priors(model.qqs_prior)
    # model.fit_prior(np.ones((nX,nX))/(nX*nX))

    base_model = copy.deepcopy(model)
    base_model.set_risk_aversion(0.0)

    print('output\n')

    plt.figure()

    show1d = False
    if show1d:
        plt.plot(model.xgrid, model.q1)
        plt.show()

        plt.plot(model.xgrid, model.q2)
        plt.show()

    showPrior = False

    if showPrior:
        for i in np.arange(nX, step=nX//5):
            plt.plot(model.xgrid, model.q12_prior[i, :])
        plt.show()

    show2d = False
    if show2d:
        for i in np.arange(nX, step=nX//5):
            plt.plot(model.xgrid, model.q12[i, :])
        plt.show()

        X1, X2 = np.meshgrid(model.xgrid, model.xgrid)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # ax.plot_surface(X1,X2,model.q12,cmap='viridis')
        ax.plot_wireframe(X1, X2, model.q12, cmap='viridis')
        ax.set_xlabel('t1')
        ax.set_ylabel('t2')
        ax.set_zlabel('q12')
        plt.show()

    strike1 = 100
    strike2 = 100
    scale1 = 1
    scale2 = 0.4
    berm = Bermudan.create_canary(
        strike1, strike2, scale1=scale1, scale2=scale2)
    eur1 = Bermudan.create_canary(
        strike1, 0, scale1=scale1, scale2=scale2, ex1=True, ex2=False)
    eur2 = Bermudan.create_canary(
        0, strike2, scale1=scale1, scale2=scale2, ex1=False, ex2=True)

    ex1 = np.maximum(berm.exercise_value(
        S0 + model.xgrid, 0), np.zeros(model.nX))
    ex2 = np.maximum(berm.exercise_value(
        S0 + model.xgrid, 1), np.zeros(model.nX))

    #ex1 = ex1 - np.dot(model.q1,ex1)
    #ex2 = ex2 - np.dot(model.q2,ex2)

    use_zero_pos = True
    if use_zero_pos:
        ad_pos = np.zeros((nX, 2))
    else:
        ad_pos = np.vstack((-ex1, -ex2)).T

    berm_val_b = berm.price(base_model, with_graphs=False)
    eur1_val_b = eur1.price(base_model)
    eur2_val_b = eur2.price(base_model)

    print(
        f'berm val_base: {berm_val_b}\neur1 val base: {eur1_val_b}\neur2 val base: {eur2_val_b}\n')

    berm_value = berm.price(model, ad_pos, with_graphs=False)
    berm_val_0 = berm.price(model)
    eur1_value = eur1.price(model)
    eur2_value = eur2.price(model)

    print(
        f'berm value: {berm_value}\nberm val 0: {berm_val_0}\neur1 value: {eur1_value}\neur2 value: {eur2_value}\n')

    calc_impl_vols = False
    if calc_impl_vols:
        berm_impl_vol_b = bermudan_implied_vol(base_model, berm, berm_val_b)
        berm_impl_vol = bermudan_implied_vol(base_model, berm, berm_value)
        berm_impl_vol_0 = bermudan_implied_vol(base_model, berm, berm_val_0)
        eur1_impl_vol = bermudan_implied_vol(base_model, eur1, eur1_value)
        eur2_impl_vol = bermudan_implied_vol(base_model, eur2, eur2_value)

        print(f'berm implied vol base : {berm_impl_vol_b}\nberm implied vol : {berm_impl_vol}\nberm implied vol 0: {berm_impl_vol_0}\neur1 implied vol: {eur1_impl_vol}\neur2 implied vol: {eur2_impl_vol}\n')

    #eur1_val_opt,eur1_ad_pos_opt = price_bermudan_optimal(model, eur1)
    eur1_ad_pos_opt = np.vstack((-ex1, np.zeros(model.nX))).T
    eur1_val_opt = eur1.price(model, eur1_ad_pos_opt)
    eur1_impl_vol_opt = bermudan_implied_vol(base_model, eur1, eur1_val_opt)
    print(
        f'eur1 optimal val: {eur1_val_opt}\neur1 opt val impl vol: {eur1_impl_vol_opt}\n')

    show_opt_ad = True
    if show_opt_ad:
        plt.plot(model.xgrid, ad_pos[:, 0])
        plt.plot(model.xgrid, eur1_ad_pos_opt[:, 0])
        plt.show()

        plt.plot(model.xgrid, ad_pos[:, 1])
        plt.plot(model.xgrid, eur1_ad_pos_opt[:, 1])
        plt.show()

    #eur2_val_opt,eur2_ad_pos_opt = price_bermudan_optimal(model, eur2)
    eur2_ad_pos_opt = np.vstack((np.zeros(model.nX), -ex2)).T
    eur2_val_opt = eur2.price(model, eur2_ad_pos_opt)
    eur2_impl_vol_opt = bermudan_implied_vol(base_model, eur2, eur2_val_opt)
    print(
        f'eur2 optimal val: {eur2_val_opt}\neur2 opt val impl vol: {eur2_impl_vol_opt}\n')

    show_opt_ad = True
    if show_opt_ad:
        plt.plot(model.xgrid, ad_pos[:, 0])
        plt.plot(model.xgrid, eur2_ad_pos_opt[:, 0])
        plt.show()

        plt.plot(model.xgrid, ad_pos[:, 1])
        plt.plot(model.xgrid, eur2_ad_pos_opt[:, 1])
        plt.show()

    berm_val_opt_e1 = berm.price(model, eur1_ad_pos_opt)
    berm_impl_vol_opt_e1 = bermudan_implied_vol(
        base_model, berm, berm_val_opt_e1)
    print(
        f'Values of Berm under the Euro 1 optimal strategy\nberm optimal val: {berm_val_opt_e1}\nberm opt val impl vol: {berm_impl_vol_opt_e1}\n')

    berm_val_opt_e2 = berm.price(model, eur2_ad_pos_opt)
    berm_impl_vol_opt_e2 = bermudan_implied_vol(
        base_model, berm, berm_val_opt_e2)
    print(
        f'Values of Berm under the Euro 2 optimal strategy\nberm optimal val: {berm_val_opt_e2}\nberm opt val impl vol: {berm_impl_vol_opt_e2}\n')

    #berm_val_opt,berm_ad_pos_opt = price_bermudan_optimal(model, berm, eur2_ad_pos_opt if eur2_val_opt > eur1_val_opt else eur1_ad_pos_opt)
    berm_val_opt, berm_ad_pos_opt = price_bermudan_optimal(model, berm)
    berm_val_opt_again = berm.price(model, berm_ad_pos_opt, with_graphs=True)
    berm_impl_vol_opt = bermudan_implied_vol(base_model, berm, berm_val_opt)
    print(
        f'berm optimal val: {berm_val_opt}\nberm optimal 2: {berm_val_opt_again}\nberm opt val impl vol: {berm_impl_vol_opt}\n')

    show_opt_ad = True
    if show_opt_ad:
        plt.plot(model.xgrid, ad_pos[:, 0])
        plt.plot(model.xgrid, berm_ad_pos_opt[:, 0])
        plt.show()

        plt.plot(model.xgrid, ad_pos[:, 1])
        plt.plot(model.xgrid, berm_ad_pos_opt[:, 1])
        plt.show()

    # print(model.q1)
    # print(model.q2)
    # print(model.q12_prior)
    # print(model.q12)


# here we hedge a Berm with every European we have and see how the Bermudan price
# behaves as we crank up risk aversion
def explore_1b():

    lmbs = np.arange(0.0, 2.0, 0.1)

    nX = 21
    S0 = 100
    logvol1 = 0.1
    logvol2 = 0.2
    ref_vol = 0.2
    T1 = 10
    T2 = 20

    model = MultiStepModel(0.0, nX, S0, np.array(
        [logvol1, logvol2]), np.array([T1, T2]), ref_vol, T2)
    model.fit_priors(model.qqs_prior)

    base_model = copy.deepcopy(model)
    base_model.set_risk_aversion(0.0)

    strike1 = 100
    strike2 = 100
    scale1 = 1
    scale2 = 0.4

    berm = Bermudan.create_canary(
        strike1, strike2, scale1=scale1, scale2=scale2)

    ex1 = np.maximum(berm.exercise_value(
        S0 + model.xgrid, 0), np.zeros(model.nX))
    ex2 = np.maximum(berm.exercise_value(
        S0 + model.xgrid, 1), np.zeros(model.nX))

    ad_pos = np.vstack((-ex1, -ex2)).T

    all_res = []

    for u_lmb in tqdm(lmbs):

        model.set_risk_aversion(u_lmb)
        berm_value = berm.price(model, ad_pos, with_graphs=False)
        res0 = list(price_bermudan_opt_eur_hedge(
            base_model, berm, eur_pos_0=np.array([1.0, 1.0])))

        res = [u_lmb] + [berm_value] + res0
#        print(res)
        all_res.append(res)

    df = pd.DataFrame(columns=['risk_av', 'berm_val', 'base_berm_val',
                               'ignore1', 'ignore2', 'eur1', 'eur2'], data=all_res)
    print(df)


def explore_2():
    u_lmb = 1  # 0.1  # 0.01 # 0.1
    nX = 21
    S0 = 100
    logvol1 = 0.1
    logvol2 = 0.2
    ref_vol = 0.2
    T1 = 10
    T2 = 20

    model = MultiStepModel(u_lmb, nX, S0, np.array(
        [logvol1, logvol2]), np.array([T1, T2]), ref_vol, T2)
    model.fit_priors(model.qqs_prior)

    base_model = copy.deepcopy(model)
    base_model.set_risk_aversion(0.0)

    strike1 = 100
    strike2 = 100
    scale1 = 1
    scale2l = 0.0
    scale2u = 1.0
    dscale2 = 0.02
    scales2 = np.arange(scale2l, scale2u, dscale2)

    all_res = []

    for scale2 in tqdm(scales2):
        berm = Bermudan.create_canary(
            strike1, strike2, scale1=scale1, scale2=scale2)
#        res = list(price_bermudan_max_euro(model, berm))
        res = list(price_bermudan_opt_eur_hedge(
            model, berm, eur_pos_0=np.array([0.5, 0.5])))

        res0 = list(price_bermudan_opt_eur_hedge(
            base_model, berm, eur_pos_0=np.array([0.0, 0.0])))

        res = [scale2] + [res0[0]] + res
        all_res.append(res)
#        print(res)

    df = pd.DataFrame(columns=['scale', 'base_val', 'berm_val',
                               'w1', 'w2', 'eur1', 'eur2'], data=all_res)
    print(df)


def explore_4():
    u_lmb = 0.1  # 0.01 # 0.1
    nX = 21
    S0 = 100
    logvol1 = 0.2
    logvol2 = 0.2
    ref_vol = 0.2
    ref_T = 2
    T1s = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    tau = 1

    strike1 = 95
    strike2 = 100
    scale1 = 1
    scale2 = 0.6

    berm = Bermudan.create_canary(
        strike1, strike2, scale1=scale1, scale2=scale2)

    all_res = None
    for n, T1 in enumerate(T1s):

        T2 = T1 + tau
        model = MultiStepModel(u_lmb, nX, S0, np.array(
            [logvol1, logvol2]), np.array([T1, T2]), ref_vol, ref_T)
        model.fit_priors(model.qqs_prior)
#        res = list(price_bermudan_max_euro(model, berm))
        res = list(price_bermudan_opt_eur_hedge(
            model, berm, eur_pos_0=np.array([0.5, 0.5])))
        res = [T1] + res

        if all_res is None:
            all_res = np.zeros((len(T1s), len(res)))

        all_res[n, :] = np.array(res)
        print(res)


def explore_3():
    u_lmb = 0.01  # 0.01 # 0.1
    nX = 21
    S0 = 100
    logvol1 = 0.1
    logvol2l = 0.075
    logvol2u = 0.4
    dlogvol2 = 0.005
    logvols2 = np.arange(logvol2l, logvol2u, dlogvol2)
    ref_vol = 0.2
    T1 = 10
    T2 = 20

    strike1 = 100
    strike2 = 100
    scale1 = 1
    scale2 = 0.4
    berm = Bermudan.create_canary(
        strike1, strike2, scale1=scale1, scale2=scale2)

    for logvol2 in logvols2:
        model = MultiStepModel(u_lmb, nX, S0, np.array(
            [logvol1, logvol2]), np.array([T1, T2]), ref_vol, T2)
        model.fit_priors(model.qqs_prior)

        res = list(price_bermudan_max_euro(model, berm))
        res = [logvol2] + res
        print(res)


def explore_5():
    u_lmb = 0.0  # 0.01 # 0.1
    nX = 21
    S0 = 100
    logvols = np.array([0.2, 0.2, 0.2])
    Ts = [1, 2, 3]
    ref_vol = 0.2
    ref_T = 3

    model = MultiStepModel(u_lmb, nX, S0, logvols, Ts, ref_vol, Ts[-1])
    model.fit_priors(model.qqs_prior)

#    strikes = np.array([100,98,106])
#    ini_scales = np.array([1.0,0.3,0.6])
#    fin_scales = np.array([1.0,1.0,1.0])
    strikes = np.array([100, 96, 98])
    ini_scales = np.array([1.0, 0.5, 0.2])
    fin_scales = np.array([1.0, 1.0, 1.0])
    isExercise = [True, True, True]

    pos_initial_guess = np.array([0.3, 0.3, 0.3])

    ws = np.linspace(0, 1, 101, endpoint=True)
    all_res = None
    for n, w in enumerate(ws):
        scales = ini_scales + w*(fin_scales - ini_scales)
        berm = Bermudan.create(strikes, scales, isExercise)
        res = list(price_bermudan_opt_eur_hedge(
            model, berm, eur_pos_0=pos_initial_guess))
        res = [w] + res

        if all_res is None:
            all_res = np.zeros((len(ws.tolist()), len(res)))

        all_res[n, :] = np.array(res)
        print(res)

    np.savetxt("./res/berm_util_explore5.csv", all_res, delimiter=",",
               fmt='%.4f', header="scale2, berm, w0, w1, w2, e0, e1, e2")


def explore_6():
    u_lmb = 10.0  # 0.01 # 0.1
    nX = 21
    S0 = 100
    logvols_base = np.array([0.2, 0.2, 0.2])
    ref_vol = 0.2
    ref_T = 3
    eur_pos_0_base = np.array([1.0, 0.1, 0.0])

    strikes_base = np.array([90, 100, 100])
    scales_base = np.array([1.0, 0.6, 0.4])
    isExercise_base = [True, True, True]

    T1s = [0.5, 0.4, 0.3, 0.2, 0.19, 0.18, 0.17, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09,
           0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.001, -0.001, -0.01, -0.02, -0.03]
    tau1 = 1
    tau2 = 2

    all_res = None
    for n, T1 in enumerate(T1s):

        Ts_base = np.array([T1, T1+tau1, T1 + tau2])

        if(T1 > 0):
            logvols = logvols_base
            Ts = Ts_base
            eur_pos_0 = eur_pos_0_base
            strikes = strikes_base
            scales = scales_base
            isExercise = isExercise_base

        else:
            logvols = logvols_base[1:]
            Ts = Ts_base[1:]
            eur_pos_0 = eur_pos_0_base[1:]
            strikes = strikes_base[1:]
            scales = scales_base[1:]
            isExercise = isExercise_base[1:]

        berm = Bermudan.create(strikes, scales, isExercise)
        model = MultiStepModel(u_lmb, nX, S0, logvols, Ts, ref_vol, ref_T)
        model.fit_priors(model.qqs_prior)
#        res = list(price_bermudan_opt_eur_hedge(model, berm, eur_pos_0=eur_pos_0))
#        res = list(berm.price(model))
        res = price_berm_and_euros(model, berm)
        res = [T1] + res

        if all_res is None:
            all_res = np.zeros((len(T1s), len(res)))
            res_length = len(res)

#        all_res[n,:] = np.array(res).resize((res_length,)) # length may change as we drop exericse dates
        npres = np.array(res)
        all_res[n, :] = np.resize(npres, (res_length,))
        print(res)

    np.savetxt("./res/berm_util_explore6.csv", all_res, delimiter=",",
               fmt='%.4f', header="T, berm, w0, w1, w2, e0, e1, e2")


def explore_7():
    u_lmb = 20.0  # 0.01 # 0.1
    nX = 21
    S0 = 100
    logvols_base = np.array([0.2, 0.2, 0.2])
    ref_vol = 0.2
    ref_T = 10
    eur_pos_0_base = np.array([1.0, 0.1, 0.0])

    strikes_base = np.array([95, 100, 105])
    scales_base = np.array([1.0, 0.5, 0.3])
    isExercise_base = [True, True, True]

    T1s = [0.5, 0.4, 0.3, 0.2, 0.19, 0.18, 0.17, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09,
           0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.001, -0.001, -0.01, -0.02, -0.03]
    tau1 = 5
    tau2 = 10

    all_res = None
    for n, T1 in enumerate(T1s):

        Ts_base = np.array([T1, T1+tau1, T1 + tau2])

        if(T1 > 0):
            logvols = logvols_base
            Ts = Ts_base
            eur_pos_0 = eur_pos_0_base
            strikes = strikes_base
            scales = scales_base
            isExercise = isExercise_base

        else:
            logvols = logvols_base[1:]
            Ts = Ts_base[1:]
            eur_pos_0 = eur_pos_0_base[1:]
            strikes = strikes_base[1:]
            scales = scales_base[1:]
            isExercise = isExercise_base[1:]

        berm = Bermudan.create(strikes, scales, isExercise)
        model = MultiStepModel(u_lmb, nX, S0, logvols, Ts, ref_vol, ref_T)
        model.fit_priors(model.qqs_prior)
#        res = list(price_bermudan_opt_eur_hedge(model, berm, eur_pos_0=eur_pos_0))
#        res = list(berm.price(model))
        res = price_berm_and_euros(model, berm)
        res = [T1] + res

        if all_res is None:
            all_res = np.zeros((len(T1s), len(res)))
            res_length = len(res)

#        all_res[n,:] = np.array(res).resize((res_length,)) # length may change as we drop exericse dates
        npres = np.array(res)
        all_res[n, :] = np.resize(npres, (res_length,))
        print(res)

    np.savetxt("./res/berm_util_explore7.csv", all_res, delimiter=",",
               fmt='%.4f', header="T, berm, w0, w1, w2, e0, e1, e2")


def explore_8():
    u_lmb = 5.0  # 0.01 # 0.1
    nX = 21
    S0 = 100
    ref_vol = 0.2
    nE = 9
    tau_base = 1
    logvols_base = ref_vol * np.ones(nE)
    ref_T = nE * tau_base + 1

    strikes_base = 100*np.ones(nE)
    strikes_base[0] = 100-13

    scales_base = 1.0 * np.linspace(1, 0.5, nE, endpoint=True)
    isExercise_base = nE * [True]

    T1s = [0.5, 0.4, 0.3, 0.2, 0.19, 0.18, 0.17, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09,
           0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.001, -0.001, -0.01, -0.02, -0.03]
    taus = tau_base * np.arange(nE)

    all_res = None
    for n, T1 in enumerate(T1s):

        Ts_base = T1 + taus

        if(T1 > 0):
            logvols = logvols_base
            Ts = Ts_base
            strikes = strikes_base
            scales = scales_base
            isExercise = isExercise_base

        else:
            logvols = logvols_base[1:]
            Ts = Ts_base[1:]
            strikes = strikes_base[1:]
            scales = scales_base[1:]
            isExercise = isExercise_base[1:]

        berm = Bermudan.create(strikes, scales, isExercise)
        model = MultiStepModel(u_lmb, nX, S0, logvols, Ts, ref_vol, ref_T)
        model.fit_priors(model.qqs_prior)
        res = price_berm_and_euros(model, berm)
        res = [T1] + res

        if all_res is None:
            all_res = np.zeros((len(T1s), len(res)))
            res_length = len(res)

#        all_res[n,:] = np.array(res).resize((res_length,)) # length may change as we drop exericse dates
        npres = np.array(res)
        all_res[n, :] = np.resize(npres, (res_length,))
        print(res)

    np.savetxt("./res/berm_util_explore8.csv", all_res, delimiter=",",
               fmt='%.4f', header="T, berm, w0, w1, w2, e0, e1, e2")


def explore_9():
    u_lmb = 0.01  # 0.01 # 0.1
    nX = 21
    S0 = 100
    ref_vol = 0.2
    nE = 9
    tau_base = 1
    logvols_base = ref_vol * np.ones(nE)
    ref_T = nE * tau_base + 1

    strikes_base = 100*np.ones(nE)
    scales_0 = 1.0 * np.linspace(1.0, 0.1, nE, endpoint=True)
    scales_1 = 1.0 * np.linspace(0.1, 1.0, nE, endpoint=True)
    scales_lin = np.linspace(0, 1, 11, endpoint=True)

    isExercise_base = nE * [True]
    taus = tau_base * np.arange(nE)
    T1 = 0.1
    Ts_base = T1 + taus

    all_res = None
    for n, s in enumerate(scales_lin):
        scales_base = scales_0 * (1.0 - s) + scales_1 * (s)

        if(T1 > 0):
            logvols = logvols_base
            Ts = Ts_base
            strikes = strikes_base
            scales = scales_base
            isExercise = isExercise_base

        else:
            logvols = logvols_base[1:]
            Ts = Ts_base[1:]
            strikes = strikes_base[1:]
            scales = scales_base[1:]
            isExercise = isExercise_base[1:]

        berm = Bermudan.create(strikes, scales, isExercise)
        model = MultiStepModel(u_lmb, nX, S0, logvols, Ts, ref_vol, ref_T)
        model.fit_priors(model.qqs_prior)
        res = price_berm_and_euros(model, berm)
        res = [s] + res

        if all_res is None:
            all_res = np.zeros((len(scales_lin), len(res)))
            res_length = len(res)

#        all_res[n,:] = np.array(res).resize((res_length,)) # length may change as we drop exericse dates
        npres = np.array(res)
        all_res[n, :] = np.resize(npres, (res_length,))
        print(res)

    np.savetxt("./res/berm_util_explore9.csv", all_res, delimiter=",",
               fmt='%.4f', header="s, berm, w0, w1, w2, e0, e1, e2")


def explore_model_mix_1():
    u_lmb = 0.0  # 0.01 # 0.1
    nX = 21
    S0 = 100
    logvols = np.array([0.2, 0.2, 0.2])
    Ts = np.array([1, 2, 3])
    ref_vol = 0.5
    ref_T = 3

    j_in = 1
    scale_vol = np.exp(Ts*j_in*0.5)  # _vol_ scale hence 0.5
    fwd_vols = term_vols_to_fwd_vols(logvols, Ts)
    scaled_fwd_vols = fwd_vols*scale_vol

    small_vol = 0.01

    scaled_fwd_vols0 = scaled_fwd_vols.copy()
    scaled_fwd_vols0[0:] = small_vol
    logvols0 = fwd_vols_to_term_vols(scaled_fwd_vols0, Ts)

    scaled_fwd_vols1 = scaled_fwd_vols.copy()
    scaled_fwd_vols1[1:] = small_vol
    logvols1 = fwd_vols_to_term_vols(scaled_fwd_vols1, Ts)

    scaled_fwd_vols2 = scaled_fwd_vols.copy()
    scaled_fwd_vols2[2:] = small_vol
    logvols2 = fwd_vols_to_term_vols(scaled_fwd_vols2, Ts)

    scaled_fwd_vols3 = scaled_fwd_vols.copy()
    #scaled_fwd_vols3[3:] = small_vol
    logvols3 = fwd_vols_to_term_vols(scaled_fwd_vols3, Ts)

#    probs = np.zeros(4)
#    probs[3] = 1/scale_vol[2]
#    probs[2] = 1/scale_vol[1] - probs[3]
#    probs[1] = 1/scale_vol[0] - probs[3] - probs[2]
#    probs[0] = 1              - probs[3] - probs[2] - probs[1]

    probs = np.zeros(4)
    probs[3] = 1/scale_vol[2]**2
    probs[2] = 1/scale_vol[1]**2 - probs[3]
    probs[1] = 1/scale_vol[0]**2 - probs[3] - probs[2]
    probs[0] = 1 - probs[3] - probs[2] - probs[1]

    print(scale_vol)
    print(probs)
    print(logvols)
    print(logvols0)
    print(logvols1)
    print(logvols2)
    print(logvols3)

    models = [MultiStepModel(u_lmb, nX, S0, logvols, Ts, ref_vol, Ts[-1]),
              MultiStepModel(u_lmb, nX, S0, logvols0, Ts, ref_vol, Ts[-1]),
              MultiStepModel(u_lmb, nX, S0, logvols1, Ts, ref_vol, Ts[-1]),
              MultiStepModel(u_lmb, nX, S0, logvols2, Ts, ref_vol, Ts[-1]),
              MultiStepModel(u_lmb, nX, S0, logvols3, Ts, ref_vol, Ts[-1]), ]

    for model in models:
        model.fit_priors(model.qqs_prior)

#    strikes = np.array([100,96,98])
    strikes = np.array([100, 100, 100])
    scales = np.array([1.0, 0.5, 0.2])
    isExercise = [True, True, True]
    berm = Bermudan.create(strikes, scales, isExercise)

    all_res = None
    for n, model in enumerate(models):
        res = list(price_berm_and_euros(model, berm))
#        res = [w] + res

        if all_res is None:
            all_res = np.zeros((len(models), len(res)))

        all_res[n, :] = np.array(res)
        print(res)

    mix_val = probs @ all_res[1:, 0:4]
    print(mix_val)

    # , header = "scale2, berm, w0, w1, w2, e0, e1, e2")
    np.savetxt("./res/berm_model_mix_1.csv",
               all_res, delimiter=",", fmt='%.4f')


def explore_model_mix_2():
    u_lmb = 0.0  # 0.01 # 0.1
    nX = 21
    S0 = 100
    logvols = np.array([0.2, 0.2, 0.2])
    Ts = np.array([1, 2, 3])
    ref_vol = 0.5
    ref_T = 3

    j_in = 0.3
#    scale_vol = np.exp(Ts*j_in*0.5) # _vol_ scale hence 0.5
    scale_vol = np.exp((ref_T - Ts)*j_in*0.5)  # _vol_ scale hence 0.5
    fwd_vols = term_vols_to_fwd_vols(logvols, Ts)
    scaled_fwd_vols = fwd_vols*scale_vol

    small_vol = 0.01

    scaled_fwd_vols0 = scaled_fwd_vols.copy()
    scaled_fwd_vols0[0:] = small_vol
    logvols0 = fwd_vols_to_term_vols(scaled_fwd_vols0, Ts)

    scaled_fwd_vols1 = scaled_fwd_vols.copy()
    scaled_fwd_vols1[1:] = small_vol
    logvols1 = fwd_vols_to_term_vols(scaled_fwd_vols1, Ts)

    scaled_fwd_vols2 = scaled_fwd_vols.copy()
    scaled_fwd_vols2[2:] = small_vol
    logvols2 = fwd_vols_to_term_vols(scaled_fwd_vols2, Ts)

    scaled_fwd_vols3 = scaled_fwd_vols.copy()
    #scaled_fwd_vols3[3:] = small_vol
    logvols3 = fwd_vols_to_term_vols(scaled_fwd_vols3, Ts)


#    print(scale_vol)
    print(logvols)
    print(logvols0)
    print(logvols1)
    print(logvols2)
    print(logvols3)

    models = [MultiStepModel(u_lmb, nX, S0, logvols, Ts, ref_vol, Ts[-1]),
              MultiStepModel(u_lmb, nX, S0, logvols0, Ts, ref_vol, Ts[-1]),
              MultiStepModel(u_lmb, nX, S0, logvols1, Ts, ref_vol, Ts[-1]),
              MultiStepModel(u_lmb, nX, S0, logvols2, Ts, ref_vol, Ts[-1]),
              MultiStepModel(u_lmb, nX, S0, logvols3, Ts, ref_vol, Ts[-1]), ]

    for model in models:
        model.fit_priors(model.qqs_prior)

#    strikes = np.array([100,96,98])
    strikes = np.array([100, 100, 100])
    scales = np.array([1.0, 0.5, 0.2])
    isExercise = [True, True, True]
    berm = Bermudan.create(strikes, scales, isExercise)

    all_res = None
    for n, model in enumerate(models):
        res = list(price_berm_and_euros(model, berm))
#        res = [w] + res

        if all_res is None:
            all_res = np.zeros((len(models), len(res)))

        all_res[n, :] = np.array(res)
        print(res)

    base_eur = all_res[0, 1:4]
    scen_eur = all_res[1:, 1:4]

    scen_mtr = np.hstack((scen_eur, np.ones((4, 1))))
    base_rhs = np.append(base_eur, 1)
    probs = np.linalg.solve(scen_mtr.T, base_rhs)

#    probs = np.zeros(4)
#    probs[3] = 1/scale_vol[2]**2
#    probs[2] = 1/scale_vol[1]**2 - probs[3]
#    probs[1] = 1/scale_vol[0]**2 - probs[3] - probs[2]
#    probs[0] = 1                 - probs[3] - probs[2] - probs[1]

    mix_val = probs @ all_res[1:, 0:4]
    print(probs)
    print(mix_val)

    # , header = "scale2, berm, w0, w1, w2, e0, e1, e2")
    np.savetxt("./res/berm_model_mix_2.csv",
               all_res, delimiter=",", fmt='%.4f')


if __name__ == "__main__":
    # explore_1()
    explore_1b()
    explore_2()
    # explore_3()
    # explore_4()
    #    explore_5()
    #    explore_6()
    #    explore_7()
    #    explore_8()
    #    explore_9()
    #    explore_model_mix_1()
    # explore_model_mix_2()

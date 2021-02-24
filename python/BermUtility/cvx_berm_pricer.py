import math
import copy
import numpy as np
import cvxopt as co
import scipy.linalg as la
import cvxopt.modeling as com

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Bermudan import *
from setup_min_measure import *


def cvx_price_1(berm: Bermudan, model: TwoStepModel, z: com.variable):
    '''
    Bermudan price as a function of the density matrix z
    that is represented as a vector in row-major (numpy style) order
    as a cvsopt.modeling.variable type
    The function returns a cvxopt.modeling style 'function'

    we only support plain valuation ie no ad_pos, no utility
    '''
    if model.lmb != 0.0:
        raise Exception(f'this method only supports zero risk aversion')
    sgrid = model.S0 + model.xgrid
    nX = model.nX

    hold2 = np.zeros(model.nX)
    exercise2 = berm.exercise_value(sgrid, 1)
    payoff2 = np.maximum(hold2, exercise2)

    hold1 = co.matrix(0.0, (nX, 1))
    nX = len(model.xgrid)
    if berm.isExercise[1]:
        rollback_m = np.zeros((nX, nX*nX))
        for n in range(nX):
            rollback_m[n, n*nX:(n+1)*nX] = payoff2 * model.q12[n, :]

        hold1 = co.matrix(rollback_m) * z

    exercise1 = co.matrix(berm.exercise_value(sgrid, 0))
    payoff1 = hold1

    # this does not work because payoff is not an affine function
    # if berm.isExercise[0]:
    #    payoff1 = com.max(exercise1, hold1)
    #
    # val = com.dot(co.matrix(model.q1), payoff1)

    # so instead move q1 inide max since q1 is always non-negative
    if berm.isExercise[0]:
        q1m = co.matrix(np.diag(model.q1))
        payoff1 = com.max(q1m*exercise1, q1m*hold1)

    val = com.sum(payoff1)

    return val, hold1, exercise1


def cvx_fit_to_min_acceptable_value(model: TwoStepModel, berm: Bermudan,
                                    mtg_scale=1.0, gamma=0.0, strikes=None,  with_plots=False):
    '''
    Per Madan's paper "Acceptability bounds for forward starting
    options using disciplined convex programming"

    formulated in terms of z12 which is the density with respect to
    the initial q12

    using cvxopt package -- basically solving an LP problem
    '''

    nX = model.nX

    q12_init_v = model.q12.reshape(-1).copy()
    q12_init_d = np.diag(q12_init_v)

    z = com.variable(nX*nX, 'z')

    # objective function
    obj_f, *_ = cvx_price_1(berm, model, z)

    # basic ocnstraints
    mtr_rhs = model.standard_constraints(mtg_scale=mtg_scale)

    # reformulate for zs from qs
    # try to have the coefs of size closer to each other
    scale = [1e2, 1e4, 1]
    mtr_rhs_z = []
    for n, (A, rhs) in enumerate(mtr_rhs):
        mtr_rhs_z.append(((A*scale[n]) @ q12_init_d, rhs*scale[n]))

    # constraints in the format understood by cvxopt
#    constraints = [co.sparse(co.matrix(A))*z == co.matrix(rhs)
#                   for A, rhs in mtr_rhs_z[2:3]]

    As = [A for A, r in mtr_rhs_z]
    rs = [r.reshape(-1, 1) for A, r in mtr_rhs_z]
    big_A = np.vstack(As)
    big_r = np.vstack(rs)
    np.savetxt("./res/big_A.csv",
               big_A, delimiter=",")  # , fmt='%.4f')

    # reduce num rows of big_A to be equal to rank
    # otherwise the solver complains. Use svd
    U, S, Vh = np.linalg.svd(big_A)
    rk = sum(np.abs(S) > 1e-16*np.abs(S.max() * max(S.shape)))
    red_r = (U.T @ big_r)[:rk]
    red_A = la.diagsvd(S[:rk], rk, big_A.shape[1]) @ Vh

    # try to zero out as many as possible
    red_A[np.abs(red_A) < 1e-8*np.max(red_A)] = 0.0

    constraints = [co.sparse(co.matrix(red_A))*z == co.matrix(red_r)]

    constraints.append(z >= co.matrix(0.0))
    constraints.append(z <= co.matrix(100.0))

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
            p12_init_v[n*nX:(n+1)*nX] = model.q1[n] * \
                q12_init_v[n*nX:(n+1)*nX]
        p12_com_m = co.spdiag(co.matrix(p12_init_v))

        for strike in strikes:

            target = distortion_conj_f(strike)
            opt_on_z = com.sum(com.max(p12_com_m * (z-co.matrix(strike)), 0))

            constraints.append(opt_on_z <= co.matrix(target))

    op_problem = com.op(obj_f, constraints)
    op_problem.solve()
    print(f'LP status = {op_problem.status}')

    z12_v = np.array(z.value).reshape(-1)
    model.q12 = np.reshape(q12_init_v * z12_v, (nX, nX))

    print(f'value={op_problem.objective.value()}')
    if with_plots:
        model.standard_plots(mtg_scale=mtg_scale)

        z12_m = z12_v.reshape(nX, nX)
        for i in np.arange(0, z12_m.shape[0], 2):
            plt.plot(model.xgrid, z12_m[i, :], '.-', label=f'i={i}')
        plt.title('density Z')
        plt.legend(loc="best")
        plt.show()

        if strikes is not None:
            strikes_plt = np.linspace(
                strikes[0], strikes[-1], 10*(len(strikes)-1)+1)
            acc_val_plt = [np.sum(p12_init_v * np.maximum(z12_v-s, 0))
                           for s in strikes_plt]
            acc_bds_plt = [distortion_conj_f(s) for s in strikes_plt]
            plt.plot(strikes_plt, acc_val_plt, '.-', label='achieved')
            plt.plot(strikes_plt, acc_bds_plt, '.-', label='bound')
            plt.title('Accepability bounds after optimization')
            plt.legend(loc='best')
            plt.show()


def test_cvx_pricer_01():
    berm = setup_berm()
    model = setup_model()
    nX = model.nX

    z = com.variable(nX*nX, 'z')
    val, hold1, exercise1 = cvx_price_1(berm, model, z)

    z.value = co.matrix(1.0, (nX*nX, 1))
    print(val.value()[0])
    print(berm.price_2(model))

    plt.plot(model.xgrid, hold1.value(), '.-', label='hold')
    plt.plot(model.xgrid, exercise1, '.-', label='exercise')
    plt.title('Time T1')
    plt.legend(loc='best')
    plt.show()


def test_cvx_fit_to_min_acceptable_value_01(
        mtg_scale=1, min_prob=0.0, gamma=0.0,
        min_strike=0.1, max_strike=2.1, n_strikes=5, with_plots=False):
    model = setup_model(mtg_scale=mtg_scale,
                        min_prob=min_prob, with_plots=with_plots)

    base_model = copy.deepcopy(model)
    base_model.set_risk_aversion(0.0)

    berm = setup_berm()

    strikes = None
    if n_strikes > 0:
        strikes = np.linspace(min_strike, max_strike, n_strikes)

    cvx_fit_to_min_acceptable_value(
        model, berm, mtg_scale=mtg_scale,
        gamma=gamma, strikes=strikes, with_plots=with_plots)

    np.savetxt("./res/cvx_min_acc_measure_res_01.csv",
               model.q12, delimiter=",", fmt='%.4f')
    np.savetxt("./res/cvx_min_acc_measure_xgrid_01.csv",
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


if __name__ == '__main__':
    # test_cvx_pricer_01()
    test_cvx_fit_to_min_acceptable_value_01(mtg_scale=0.5,
                                            min_prob=0.5e-4, gamma=0.0,
                                            min_strike=0.5, max_strike=1.5,
                                            n_strikes=11, with_plots=True)

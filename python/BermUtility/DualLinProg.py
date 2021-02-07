import numpy as np
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linprog
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class berm_linprog_res():
    def __init__(self):
        self.lp_val = 0.0
        self.base_val = 0.0
        self.other_val = 0.0
        self.e1_val = 0.0
        self.e2_val = 0.0
        self.u1 = None
        self.u2 = None
        self.gap = None
        self.lp_res = None
        self.base_prob2d = None
        self.other_prob2d = None
        self.bpayoff = None
        self.ex1m = None
        self.ex2m = None
        self.c = None
        self.A_ub = None
        self.b_ub = None
        self.sg1 = None
        self.sg2 = None
        self.xgrid = None
        self.sgrid = None


def solve_dual_linprog(berm, base_model, other_model, eb):

    allres = berm_linprog_res()

    S0 = base_model.S0
    xgrid = base_model.xgrid
    sgrid = S0 + xgrid
    sg1, sg2 = np.meshgrid(sgrid, sgrid)
    # sg1 increases across ie -->
    # sg2 increases down ie V

    nx = xgrid.shape[0]

    ex1 = berm.exercise_value(sgrid, 0, True)
    ex2 = berm.exercise_value(sgrid, 1, True)

    # sg1 increses across in the column dimension -->
    ex1m = np.tile(ex1, (nx, 1))
    ex2m = np.tile(ex2, (nx, 1)).T  # sg2 increases in the row dimension, ie V

    payoff = ex1m*(sg1 > eb) + ex2m*(sg1 <= eb)

    allres.ex1m = ex1m
    allres.ex2m = ex2m
    allres.bpayoff = payoff
    allres.sg1 = sg1
    allres.sg2 = sg2

    # sg1 increases in the
    base_prob2d = np.dot(np.diag(base_model.q1), base_model.q12).T
    other_prob2d = np.dot(np.diag(other_model.q1),
                          other_model.q12).T  # sg1 increases in the

    allres.base_prob2d = base_prob2d
    allres.other_prob2d = other_prob2d

    allres.base_val = np.sum(payoff*base_prob2d)
    allres.other_val = np.sum(payoff*other_prob2d)
    allres.e1_val = np.sum(ex1m*base_prob2d)
    allres.e2_val = np.sum(ex2m*base_prob2d)

    c = np.concatenate((-base_model.q1, -base_model.q2))

    A_ub_l = np.zeros((0, nx))
    for n in np.arange(nx):
        z = np.zeros((nx, nx))
        z[:, n] = 1
        A_ub_l = np.concatenate((A_ub_l, z), axis=0)

    eye = np.eye(nx)
    A_ub_r = np.tile(eye, (nx, 1))
    A_ub = np.concatenate((A_ub_l, A_ub_r), axis=1)

    # the order F to match the fact that A_ub first column correspond to sg1[0], second to sg1[1] etc
    b_ub = np.reshape(payoff, (nx*nx, 1), order='F')

    allres.c = c
    allres.A_ub = A_ub
    allres.b_ub = b_ub

    lp_res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, method='revised simplex')
    u1 = lp_res.x[0:nx]
    u2 = lp_res.x[nx:]
    gap = np.reshape(lp_res.slack, (nx, nx), order='F')

    allres.lp_val = -lp_res.fun
    allres.u1 = u1
    allres.u2 = u2
    allres.gap = gap
    allres.lp_res = lp_res
    allres.xgrid = xgrid
    allres.sgrid = sgrid

    return allres


def optimal_call_spread_01(berm, base_model, use_sort=True, with_plots=False):

    S0 = base_model.S0
    xgrid = base_model.xgrid
    sgrid = S0 + xgrid

    nx = xgrid.shape[0]

    ex1 = berm.exercise_value(sgrid, 0, True)
    ex2 = berm.exercise_value(sgrid, 1, True)

    def call_spread(c):
        u1 = np.maximum(ex1-c, 0.0)
        u2 = np.minimum(ex2, c)

        u1v = np.dot(base_model.q1, u1)
        u2v = np.dot(base_model.q2, u2)

        return u1v + u2v

    if with_plots:
        cs = np.arange(np.max(ex2))
        plt.plot(cs, [call_spread(c) for c in cs], '-', label=f'callspr val')
        plt.legend(loc="upper right")
        plt.show()

    bounds = (0, np.max(ex2))
    if use_sort:
        cs = np.linspace(bounds[0], bounds[1], 1001)
        vs = [call_spread(s) for s in cs]
        opt_idx = np.argmax(np.array(vs))
        return (cs[opt_idx], vs[opt_idx])

    else:
        x0 = np.average(ex1)
        res = minimize(lambda c: -call_spread(c),
                       method='Nelder-Mead', x0=[x0], bounds=[bounds])
        return (res.x[0], -res.fun)

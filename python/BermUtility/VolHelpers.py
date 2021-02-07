import numpy as np
import math

def term_vols_to_fwd_vols(vols, Ts):
    
    n_vols = vols.shape[0]
    fwd_vols = np.zeros(n_vols)

    prev_var = 0.0
    prev_T = 0.0
    for n in np.arange(n_vols):
        T = Ts[n]
        var_to_T = T*vols[n]*vols[n]
        fwd_vols[n] = math.sqrt((var_to_T - prev_var)/(T-prev_T))
        prev_T = T
        prev_var = var_to_T

    return fwd_vols


def fwd_vols_to_term_vols(fwd_vols, Ts):

    n_vols = fwd_vols.shape[0]
    vols = np.zeros(n_vols)

    prev_var = 0.0
    prev_T = 0.0
    for n in np.arange(n_vols):
        T = Ts[n]
        fwd_var = (T-prev_T)*fwd_vols[n]**2
        var_to_T = prev_var + fwd_var
        vols[n] = math.sqrt(var_to_T/T)
        prev_T = T
        prev_var = var_to_T

    return vols


def scale_term_vols_by_fwd_vol_scales(term_vols, vol_scale, Ts):
    fwd_vols = term_vols_to_fwd_vols(term_vols,Ts)
    scaled_fwd_vols = fwd_vols * vol_scale
    scaled_term_vols = fwd_vols_to_term_vols(scaled_fwd_vols,Ts)
    return scaled_term_vols


import math

def utility_exp(x,lmb, t = 0, x0=0,scale = 1):
    # ignore time t
    cutoff = 1e-4
    if(math.fabs(lmb) < cutoff):
        return (x - x0)/scale

#    return (1-math.exp(-lmb*(x - x0)/scale))/lmb if x > x0 else (x-x0)/scale
    return (1-math.exp(-lmb*(x - x0)/scale))/lmb 

def utility_disc_1(x,lmb, t = 0, x0=0,scale = 1):
    if x>0:
        return math.exp(-lmb*t) * x
    else:
        return x

def utility_disc_2(x,lmb, t=0, x0=0,scale = 1):
    if x<0:
        return math.exp(lmb*t) * x
    else:
        return x

def utility_disc_3(x,lmb, t=0, x0=0,scale = 1):
    if x<0:
        return math.exp(lmb*t) * x
    else:
        return math.exp(-lmb*t) * x
    
def utility_disc_4(x,lmb, t=0, x0=0,scale = 1):
    return math.exp(-lmb*t) * x

# note no time dimension
def utility_disc_time_constant(x,lmb, t=0, x0=0,scale = 1):
    if x<0:
        return math.exp(lmb) * x
    else:
        return math.exp(-lmb) * x

# not quite softplus as we ensure that f(0)=0
def safe_softplus(x):
    return math.log(0.5+0.5*math.exp(-math.fabs(x))) + max(x,0)

def utility_softplus(x,lmb, t=0, x0=0,scale = 1):

    b = math.exp(lmb*t)
    a = 1/b-b

    return (a*safe_softplus(x*scale) + b*x*scale)/scale
    
def utility(x,lmb, t=0,x0=0,scale = 10):
    return utility_disc_time_constant(x,lmb,t,x0,scale)
#    return utility_disc_3(x,lmb,t,x0,scale)
#    return utility_softplus(x,lmb,t,x0,scale)
#    return utility_exp(x,lmb,t,x0,scale)

# do not seem to need atm
def utility_exp_inverse(u,lmb, x0=0,scale = 1):
    cutoff = 1e-4
    if(math.fabs(lmb) < cutoff):
        return scale * u + x0
    return (-math.log( 1- u*lmb)/lmb)*scale + x0

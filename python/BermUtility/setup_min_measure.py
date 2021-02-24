from TwoStepModel import *
from Bermudan import *


def setup_model(mtg_scale=None, min_prob=0.0, with_plots=False):
    u_lmb = 0.0  # 0.1
    nX = 21
    S0 = 100
    logvol1 = 0.1
    logvol2 = 0.2
    ref_vol = 0.2
    T1 = 10
    T2 = 20

    model = TwoStepModel(u_lmb, nX, S0, logvol1, T1, logvol2, T2, ref_vol)

    uniform_prior = np.ones((nX, nX))/nX
#    model.fit_prior(model.q12_prior, mtg_scale=mtg_scale)
    model.fit_prior(uniform_prior, mtg_scale=mtg_scale,
                    min_prob=min_prob, with_plots=with_plots)

    # if with_plots:
    #    for i in np.arange(0, model.q12.shape[0], 2):
    #        plt.plot(model.xgrid, model.q12[i, :], '.-', label=f'i={i}')
    #    plt.title('initial Q12')
    #    plt.legend(loc="best")
    #    plt.show()

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


def test_01():
    model = setup_model(with_plots=True)
    berm = setup_berm()
    print(berm)


if __name__ == "__main__":
    test_01()

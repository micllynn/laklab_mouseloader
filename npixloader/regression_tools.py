import numpy as np
import scipy.signal as sp_sig

import statsmodels as sm
from statsmodels.nonparametric.kernel_regression import KernelReg as kr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns

from types import SimpleNamespace

# test functions (continuous)
# ---------------


def calc_y_sqrt(x, noise_scale=0.1):
    y_clean = x ** 0.5
    y = y_clean + np.random.normal(scale=noise_scale,
                                   size=x.shape[0])
    return y


def calc_y_sin(x, noise_scale=0.1):
    y_clean = np.sin(x)
    y = y_clean + np.random.normal(scale=noise_scale,
                                   size=x.shape[0])
    return y


def make_data(y_noise=0, x_noise=0.2):
    sig = SimpleNamespace()

    x_unispace = np.linspace(0, 10, 100)
    sig.x = x_unispace + np.random.normal(scale=x_noise,
                                          size=x_unispace.shape[0])

    sig.y_sqrt = calc_y_sqrt(sig.x, noise_scale=y_noise)
    sig.y_sqrt_and_sin = calc_y_sqrt(sig.x, noise_scale=y_noise) \
        + calc_y_sin(sig.x, noise_scale=y_noise)

    return sig


def kernel_reg_sample(sig):
    kmodel = kr(endog=sig.y_sqrt, exog=sig.x, var_type='c',
                bw='cv_ls', reg_type='ll')

    fit_out = SimpleNamespace()
    fit_out_new = SimpleNamespace()

    fit_out.pred_y, fit_out.marginal_effects = kmodel.fit()
    fit_out_new.pred_y, fit_out_new.marginal_effects = kmodel.fit(
        sig.y_sqrt_and_sin)

    return fit_out_new

# test functions (discrete)
# -------------------

def convolve_kern_matrix(t, ):


class KernGen(object):
    def __init__(self, rate=5, t_end=10, t_int=0.01,
                 rate_scaling_factor=10,
                 kern_std_ind=4, noise_std=0.1):
        # generate target rate signals for inhom poisson
        self.t = np.arange(0, 10, t_int)
        self.y1_rate = self.t ** 0.5
        self.y2_rate = np.sin(self.t)

        # normalize
        self.y1_rate /= (np.max(self.y1_rate)*rate_scaling_factor)
        self.y2_rate /= (np.max(self.y2_rate)*rate_scaling_factor)

        # generate pointprocs
        self.y1_pointproc = self.y1_rate > np.random.rand(self.t.shape[0])
        self.y2_pointproc = self.y2_rate > np.random.rand(self.t.shape[0])

        self.y1_pointproc = self.y1_pointproc.astype(int)
        self.y2_pointproc = self.y2_pointproc.astype(int)

        # convolve with kernel
        self.kern = sp_sig.windows.gaussian(self.t.shape[0],
                                            std=kern_std_ind)
        self.y1 = sp_sig.convolve(self.y1_pointproc,
                                  self.kern, mode='same')
        self.y2 = sp_sig.convolve(self.y2_pointproc,
                                  self.kern, mode='same')

        # add some noise
        self.y1 += np.random.normal(size=self.y1.shape[0], scale=noise_std)
        self.y2 += np.random.normal(size=self.y2.shape[0], scale=noise_std)
        self.y1_y2_sum = self.y1 + self.y2

        return

    def plt(self, markersize=5):
        fig = plt.figure()
        spec = gs.GridSpec(nrows=2, ncols=1, height_ratios=[1, 0.2])
        ax_sig = fig.add_subplot(spec[0, 0])
        ax_pp = fig.add_subplot(spec[1, 0])

        ax_sig.plot(self.t, self.y1,
                    color=sns.xkcd_rgb['coral'])
        ax_sig.plot(self.t, self.y2,
                    color=sns.xkcd_rgb['steel blue'])

        ax_pp.scatter(self.t[self.y1_pointproc == 1],
                      np.ones_like(self.t[self.y1_pointproc == 1]),
                      color=sns.xkcd_rgb['coral'], s=markersize)
        ax_pp.scatter(self.t[self.y2_pointproc == 1],
                      np.ones_like(self.t[self.y2_pointproc == 1]),
                      color=sns.xkcd_rgb['steel blue'], s=markersize)

        plt.show()

        return


class KernelRegPPTest(object):
    def __init__(self):
        self.data_test = KernGen()


# real functions for kernel regression on movement and npix data
# ----------------------

class KernelRegMovement(object):
    def __init__(self,
                 movement_train,
                 npix_data_train,
                 kernreg_opts={'var_type': 'c',
                               'bw': 'cv_ls',
                               'reg_type': 'll'}):
        self.kmodel = kr(
            endog=preprocess_movement(movement_train),
            exog=preprocess_npix(npix_data_train),
            **kernreg_opts)

    def apply_model(self, movement, npix_data):
        npix_data = SimpleNamespace()
        npix_data.pred, _marg_eff = self.kmodel.fit(movement)
        npix_data.resid = self.calc_resid(npix_data, npix_data.pred)
        return npix_data

    def calc_resid(self, y_real, y_pred):
        return y_real - y_pred


def preprocess_npix(spk_matrix):
    return spk_matrix_preprocessed


def preprocess_movement(mvmt_times):
    return mvmt_times_preprocessed

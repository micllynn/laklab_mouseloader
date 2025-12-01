import numpy as np
import scipy.signal as sp_signal

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns

from .beh import BehDataSimpleLoad
from .utils import find_event_onsets_plateau, find_event_onsets_autothresh, \
    remove_lick_artefact_after_rew

from types import SimpleNamespace
import os
import pathlib


def licktimes_to_licktrain(licktimes, t,
                           method='argmin',
                           smooth=True,
                           smooth_std_ms=50):
    licktrain = np.zeros_like(t, dtype=np.int8)
    _samp_rate_hz = 1/(abs(t[1]-t[0]))

    for _t_spk in licktimes:
        if method == 'calc':
            _ind_spk = int(_t_spk * _samp_rate_hz)
        elif method == 'argmin':
            _ind_spk = np.argmin(np.abs(t-_t_spk))
        licktrain[_ind_spk] = 1

    if smooth is True:
        licktrain = smooth_licktrain(licktrain, t,
                                     gauss_stdev_ms=smooth_std_ms)
    return licktrain


def smooth_licktrain(licktrain, t,
                     gauss_stdev_ms):
    """
    Parameters
    ----------
    pointproc : np.ndarray
        Point process of a single neuron's spiking outputs
    t : np.ndarray
        Time vector associated with pointproc, in units of seconds
    gauss_stdev : np.ndarray
        Standard devation of the gaussian kernel used for convolution,
        in units of ms
    """
    # setup parameters
    sampling_rate_hz = 1/(abs(t[1]-t[0]))
    gauss_stdev_inds = (gauss_stdev_ms/1000) * sampling_rate_hz
    gauss_npoints = int(gauss_stdev_inds*8)

    # define gauss kernel and normalize auc to 1
    gauss_kern = sp_signal.windows.gaussian(
        gauss_npoints, gauss_stdev_inds)
    gauss_kern_integral = np.trapezoid(
        gauss_kern,
        x=t[0:gauss_kern.shape[0]])
    gauss_kern /= gauss_kern_integral

    # convolve point process with gaussian
    spktrain_smoothed = sp_signal.convolve(
        licktrain,
        gauss_kern, mode='same')

    return spktrain_smoothed


class VisualPavlovAnalysis(object):
    def __init__(self, path,
                 parse_by='stimulusOrientation',
                 lick_type='normal',
                 lick_sin_v=5,
                 lick_sin_tol=0.02,
                 lick_sin_t_plateau=0.05):
        """
        - path is a string to a behavior folder (eg MBL001/2025-03-19/1)
        - lick_type can either be 'normal' (lick data recorded correctly)
        or 'noise' (grounding issue on Dual2P, closely resembles a sin wave
        with brief periods of signal=5 that correspond to licks).
        """
        self.params = SimpleNamespace()
        self.params.parse_by = parse_by
        self.params.lick_type = lick_type

        self.path = SimpleNamespace()
        self.path.raw = pathlib.Path(path)
        self.path.beh_folder = self.path.raw.parts[-1]
        self.path.date = self.path.raw.parts[-2]
        self.path.animal = self.path.raw.parts[-3]

        self.beh = BehDataSimpleLoad(path, parse_by=parse_by)
        self.beh.rew = SimpleNamespace()
        self.beh.stim = SimpleNamespace()

        self.beh.stim.t = self.beh._data.get_event_var('stimulusOnTimes')
        self.beh.stim.parsed_param = self.beh._stimparser._all_parsed_param
        self.beh.stim.ori = self.beh._stimparser._all_stimoris
        self.beh.stim.prob = self.beh._stimparser._all_stimprobs
        self.beh.stim.rew = self.beh._stimparser._all_stimsizes

        self.beh.rew.t = self.beh._data.get_event_var('totalRewardTimes')
        self.beh.rew.prob = self.beh.stim.prob
        self.lick = SimpleNamespace()

        t_licksig = self.beh._daq_data.t
        if lick_type == 'normal':
            lick_onset_inds = find_event_onsets_autothresh(
                self.beh._daq_data.sig['lickDetector'], n_stdevs=4)
        elif lick_type == 'noise':
            lick_onset_inds = find_event_onsets_plateau(
                self.beh._daq_data.sig['lickDetector'],
                self.beh._daq_data.t,
                v=lick_sin_v,
                tol=lick_sin_tol,
                t_thresh=lick_sin_t_plateau)
        self.lick.t_raw = t_licksig[lick_onset_inds]

        return

    def plt(self, t_prestim=3, t_postrew=3,
            n_trials='all', kern_sd=50):
        fig = plt.figure(figsize=(4, 8))
        spec = gs.GridSpec(nrows=4, ncols=1,
                           figure=fig)
        ax_stim0 = fig.add_subplot(spec[0, 0])
        ax_stim0p5 = fig.add_subplot(spec[1, 0],
                                     sharex=ax_stim0)
        ax_stim1 = fig.add_subplot(spec[2, 0],
                                   sharex=ax_stim0)
        ax_summary = fig.add_subplot(spec[3, 0],
                                     sharex=ax_stim0)

        ax_summary.set_xlabel('time (s)')
        ax_summary.set_ylabel('trial')

        trialcount_stim0 = 0
        trialcount_stim0p5 = 0
        trialcount_stim1 = 0

        t = np.arange(-1*t_prestim, 2+t_postrew, 0.001)
        licktrain_stim0 = np.zeros_like(t, dtype=float)
        licktrain_stim0p5 = np.zeros_like(t, dtype=float)
        licktrain_stim1 = np.zeros_like(t, dtype=float)

        if n_trials == 'all':
            n_trials = len(self.beh.rew.t)

        for trial in range(n_trials):
            # get the data
            _t_stim = self.beh.stim.t[trial]
            _t_rew = self.beh.rew.t[trial]
            _lick_inds = np.logical_and(
                self.lick.t_raw > (_t_stim - t_prestim),
                self.lick.t_raw < (_t_rew + t_postrew))
            _licks = self.lick.t_raw[_lick_inds]
            _licks -= _t_stim

            # correct licks
            _t_rew_rel_to_stim = _t_rew - _t_stim
            _licks = remove_lick_artefact_after_rew(
                _licks, [_t_rew_rel_to_stim+0.03,
                         _t_rew_rel_to_stim+0.06])

            # store smoothed data
            _licktrain = licktimes_to_licktrain(_licks, t,
                                                smooth_std_ms=kern_sd)
            self._licks = _licks
            self._licktrain = _licktrain

            # plot in appropriate place
            if self.beh.rew.prob[trial] == 0:
                licktrain_stim0 += _licktrain
                ax_stim0.scatter(
                    _licks, np.ones_like(_licks) * trialcount_stim0,
                    s=4, color='k')
                trialcount_stim0 += 1
            elif self.beh.rew.prob[trial] == 0.5:
                licktrain_stim0p5 += _licktrain
                ax_stim0p5.scatter(
                    _licks, np.ones_like(_licks) * trialcount_stim0p5,
                    s=4, color='k')
                trialcount_stim0p5 += 1
            elif self.beh.rew.prob[trial] == 1:
                licktrain_stim1 += _licktrain
                ax_stim1.scatter(
                    _licks, np.ones_like(_licks) * trialcount_stim1,
                    s=4, color='k')
                trialcount_stim1 += 1

        licktrain_stim0 /= trialcount_stim0
        licktrain_stim0p5 /= trialcount_stim0p5
        licktrain_stim1 /= trialcount_stim1

        for _ax in [ax_stim0, ax_stim0p5, ax_stim1, ax_summary]:
            _ax.axvline(0, color=sns.xkcd_rgb['forest green'],
                        alpha=0.5, linewidth=2)
        ax_stim0.axvline(_t_rew-_t_stim, color=sns.xkcd_rgb['bright blue'],
                         alpha=0.1, linewidth=2)
        ax_stim0p5.axvline(_t_rew-_t_stim, color=sns.xkcd_rgb['bright blue'],
                           alpha=0.4, linewidth=2)
        ax_stim1.axvline(_t_rew-_t_stim, color=sns.xkcd_rgb['bright blue'],
                         alpha=1.0, linewidth=2)

        # config and plot summary licktrain
        ax_summary.plot(t, licktrain_stim0,
                        color=sns.xkcd_rgb['black'], alpha=0.1)
        ax_summary.plot(t, licktrain_stim0p5,
                        color=sns.xkcd_rgb['black'], alpha=0.4)
        ax_summary.plot(t, licktrain_stim1,
                        color=sns.xkcd_rgb['black'], alpha=1.0)
        ax_summary.axvline(_t_rew-_t_stim, color=sns.xkcd_rgb['bright blue'],
                           alpha=0.8, linewidth=2)
        ax_summary.set_xlabel('time (s)')
        ax_summary.set_ylabel('licks (Hz)')

        fig.savefig(os.path.join(pathlib.Path(*self.path.raw.parts[0:-1]),
                                 f'{self.path.animal}'
                                 + f'_{self.path.date}_{self.path.beh_folder}'
                                 + '_behavior_summary.pdf'))

        plt.show()

    def plt_new(self,
                t_prestim=3, t_postrew=3,
                n_trials='all', kern_sd=50):

        # Numbers of trials and relative plot sizes
        # ------------
        n_conds = len(self.beh._stimparser.parsed_param)

        trial_n_conds = np.zeros(n_conds)
        # for n_cond in range(n_conds):
        #     trial_n_conds[n_cond] = len(np.where(
        #         self.beh.stim.ori == self.beh._stimparser.ori[n_cond])[0])
        for n_cond in range(n_conds):
            trial_n_conds[n_cond] = len(np.where(
                self.beh._stimparser.parsed_param[n_cond]
                == self.beh._stimparser._all_parsed_param)[0])
        plt_size = (trial_n_conds / np.max(trial_n_conds)) * 2
        height_ratios_axs = np.append(trial_n_conds, np.max(trial_n_conds))

        # setup plots
        # ----------
        fig = plt.figure(figsize=(4, (np.sum(plt_size)+2)))
        spec = gs.GridSpec(nrows=n_conds+1, ncols=2,
                           height_ratios=height_ratios_axs,
                           figure=fig)
        ax_conds = []
        ev_conds = []
        for n_cond in range(n_conds):
            ev_conds.append(self.beh._stimparser.prob[n_cond]
                            * self.beh._stimparser.size[n_cond])

            if n_cond == 0:
                ax_conds.append(
                    fig.add_subplot(spec[n_cond, :]))
            else:
                ax_conds.append(
                    fig.add_subplot(spec[n_cond, :],
                                    sharex=ax_conds[0]))

        alpha_conds = ev_conds / (np.max(ev_conds)) + 0.3
        alpha_conds /= np.max(alpha_conds)

        ax_summary = fig.add_subplot(spec[-1, 0])
        ax_hist = fig.add_subplot(spec[-1, 1])

        ax_summary.set_xlabel('time (s)')
        ax_summary.set_ylabel('trial')

        # iterate through trials, processing data
        # -------------------
        if n_trials == 'all':
            n_trials = len(self.beh.rew.t)

        t = np.arange(-1*t_prestim, 2+t_postrew, 0.001)
        trialcounts = np.zeros(n_conds)
        licktrains = np.zeros((len(t), n_conds), dtype=float)
        antic_licks_hz = np.zeros(n_trials)

        for trial in range(n_trials):
            # Process data
            # ----------
            _t_stim = self.beh.stim.t[trial]
            _t_rew = self.beh.rew.t[trial]
            _lick_inds = np.logical_and(
                self.lick.t_raw > (_t_stim - t_prestim),
                self.lick.t_raw < (_t_rew + t_postrew))
            _licks = self.lick.t_raw[_lick_inds]
            _licks -= _t_stim

            # correct licks
            _t_rew_rel_to_stim = _t_rew - _t_stim
            _licks = remove_lick_artefact_after_rew(
                _licks, [_t_rew_rel_to_stim+0.03,
                         _t_rew_rel_to_stim+0.06])

            # calculate anticipatory lickrate and store
            _antic = np.sum(np.logical_and(_licks > 0,
                                           _licks < _t_rew_rel_to_stim))
            antic_licks_hz[trial] = _antic / _t_rew_rel_to_stim

            # store smoothed data
            _licktrain = licktimes_to_licktrain(_licks, t,
                                                smooth_std_ms=kern_sd)
            self._licks = _licks
            self._licktrain = _licktrain

            # Plot data in appropriate subplot
            # ----------
            for ind, param in enumerate(self.beh._stimparser.parsed_param):
                if self.beh.stim.parsed_param[trial] == param:
                    # update licktrain and trialcount
                    licktrains[:, ind] += _licktrain
                    trialcounts[ind] += 1

                    # plot
                    ax_conds[ind].scatter(
                        _licks, np.ones_like(_licks) * trialcounts[ind],
                        s=4, color='k')

        for ind, _ax in enumerate(ax_conds):
            licktrains[:, ind] /= trialcounts[ind]

            _ax.axvline(0, color=sns.xkcd_rgb['forest green'],
                        alpha=0.5, linewidth=2)
            _ax.axvline(_t_rew-_t_stim, color=sns.xkcd_rgb['bright blue'],
                        alpha=alpha_conds[ind], linewidth=2)

            if self.params.parse_by == 'stimulusOrientation':
                _text = f'ori={self.beh._stimparser.ori[ind]}deg \n' + \
                    f'size={self.beh._stimparser.size[ind]}ul \n' + \
                    f'p(rew)={self.beh._stimparser.prob[ind]}'
            elif self.params.parse_by == 'stimulusTypeValues':
                _text = f'type={self.beh._stimparser.stimtype[ind]} \n' + \
                    f'size={self.beh._stimparser.size[ind]}ul \n' + \
                    f'p(rew)={self.beh._stimparser.prob[ind]}'
            _ax.text(0.01, 0.95, _text,
                     transform=_ax.transAxes,
                     fontsize=6, va='top', ha='left',
                     fontfamily='monospace',
                     fontweight=1000, color=[0.561, 0.078, 0.008],
                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.5))

        # config and plot summary licktrain and histogram
        # ---------------
        for ind, _ax in enumerate(ax_conds):
            ax_summary.plot(t, licktrains[:, ind],
                            color=sns.xkcd_rgb['black'],
                            alpha=alpha_conds[ind])

        ax_summary.axvline(_t_rew-_t_stim, color=sns.xkcd_rgb['bright blue'],
                           alpha=0.8, linewidth=2)
        ax_summary.axvline(0, color=sns.xkcd_rgb['forest green'],
                           alpha=0.8, linewidth=2)
        ax_summary.set_xlabel('time (s)')
        ax_summary.set_ylabel('licks (Hz)')

        # hist
        # z_score_
        bins = np.arange(0, 8, 0.5)
        for ind, param in enumerate(self.beh._stimparser.parsed_param):
            _filt = np.where(self.beh.stim.parsed_param[0:n_trials] == param)[0]
            _antic_licks_filt = antic_licks_hz[_filt]
            ax_hist.hist(_antic_licks_filt, bins=bins,
                         histtype='step', alpha=alpha_conds[ind],
                         edgecolor=sns.xkcd_rgb['forest green'],
                         density=True)
        ax_hist.set_xlabel('antic. licks (Hz)')
        ax_hist.set_ylabel('pdf')

        fig.savefig(os.path.join(pathlib.Path(*self.path.raw.parts[0:-1]),
                                 f'{self.path.animal}'
                                 + f'_{self.path.date}_{self.path.beh_folder}'
                                 + '_behavior_summary.pdf'))

        plt.show()

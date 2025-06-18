import numpy as np
import scipy as sp
import scipy.stats as sp_stats
from sklearn import decomposition
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import pandas as pd

from types import SimpleNamespace
import gc
import os
import copy
import pickle

from .load_exp_twop import TwoPRec_New
from .utils import rescale_to_frac
from .dset_twop import DSetObj_5HTCtx

try:
    plt.style.use('publication_ml')
except:
    pass


def plot_antic_licking(dset_obj, figsize=(2, 2),
                       marker_size=60,
                       plt_xlim=[-0.5, 2.5]):
    n_recs = dset_obj._dset_raw['expref'].shape[0]
    antic_licks = {'0': [], '0.5': [], '1': []}

    colors = sns.cubehelix_palette(
         n_colors=3,
         start=2, rot=0,
         dark=0.2, light=0.8)

    # store anticipatory lickrate for each rec
    for rec in range(n_recs):
        print(f'---------\n{rec=}\n----------')
        try:
            exp = TwoPRec_New(dset_obj=dset_obj, dset_ind=rec)
            exp.add_lickrates()
            _antic_licks = exp.get_antic_licks_by_trialtype()

            antic_licks['0'].append(_antic_licks['0'])
            antic_licks['0.5'].append(_antic_licks['0.5'])
            antic_licks['1'].append(_antic_licks['1'])
        except:
            pass

    # make plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot([antic_licks['0'],
             antic_licks['0.5'],
             antic_licks['1']],
            color=sns.xkcd_rgb['light grey'])

    ax.scatter(np.ones_like(antic_licks['0'])*0,
               antic_licks['0'], s=marker_size,
               facecolors='none',
               edgecolors=colors[0])
    ax.scatter(np.ones_like(antic_licks['0.5'])*1,
               antic_licks['0.5'], s=marker_size,
               facecolors='none',
               edgecolors=colors[1])
    ax.scatter(np.ones_like(antic_licks['1'])*2,
               antic_licks['1'], s=marker_size,
               facecolors='none',
               edgecolors=colors[2])

    ax.set_xticks([0, 1, 2], ['0%', '50%', '100%'])
    ax.set_ylabel('antic. licking (Hz)')
    ax.set_xlabel('rew. prob.')
    ax.set_xlim(plt_xlim)

    fig.savefig('summary_licks.pdf')

    _res_integ = sp.stats.ttest_rel(
        antic_licks['0'], antic_licks['1'])
    print(f'integ pval={_res_integ.pvalue}')
    print(f'integ stat={_res_integ.statistic}')

    return


def plt_fig1_grab_summary(dset_obj,
                          t_pre=1,
                          t_post=5):
    """
    Plots all stimulus-evoked GRAB5-HT responses across the whole
    dataset.
    """
    # load and store
    # -----------
    n_recs = dset_obj._dset_raw['exprefs'].shape[0]
    tr_conds = ['0', '0.5_rew', '0.5_norew', '1']

    for dset_ind in range(n_recs):
        _exp = TwoPRec_New(dset_obj=dset_obj,
                           dset_ind=dset_ind,
                           ch_img=1,
                           trial_end=dset_obj._dset_raw['trial_end'][dset_ind])
        _exp.plt_stim_aligned_avg(t_pre=t_pre, t_post=t_post, plot=False)
        _ind_t_stim = np.argmin(np.abs(_exp.frame_avg.t - 0))
        _ind_t_rew = np.argmin(np.abs(_exp.frame_avg.t - 2))

        # On first recording, setup data structs
        if dset_ind == 0:
            len_dff = _exp.frame_avg.dff['0'].shape[1]

            data = SimpleNamespace
            data.stats = SimpleNamespace(
                stim_resp={},
                rew_resp={},
                rew_omission_resp={})

            for tr_cond in tr_conds:
                data.dff_stim[tr_cond] = np.zeros(n_recs, len_dff)
                data.stats.stim_resp[tr_cond] = np.zeros(n_recs)
                data.stats.rew_resp[tr_cond] = np.zeros(n_recs)
                data.stats.between_sector_variance_stim[tr_cond] \
                    = np.zeros(n_recs)
                data.stats.within_sector_variance_stim[tr_cond] \
                    = np.zeros(n_recs)

        # for this rec, iterate through all trials and store
        for tr_cond in tr_conds:
            data.dff_stim[tr_cond][dset_ind, :] = np.mean(
                _exp.frame_avg.dff[tr_cond], axis=0)
            data.t_dff_stim = _exp.frame_avg.t
            data.stats.stim_resp[tr_cond][dset_ind] = np.mean(np.mean(
                _exp.frame_avg.dff[tr_cond][:, _ind_t_stim:_ind_t_rew],
                axis=0), axis=0)

    # plot data
    # -----------
    fig = plt.figure(figsize=(6.86, 4))
    spec = gs.GridSpec(nrows=1, ncols=4,
                       width_ratios=[1, 0.3, 0.3, 0.3],
                       figure=fig)
    ax_dff = fig.add_subplot(spec[0, 0])
    ax_stats_stim = fig.add_subplot(spec[0, 1])
    ax_stats_rew = fig.add_subplot(spec[0, 2])
    ax_stats_licking = fig.add_subplot(spec[0, 3])

    ax_dff.plot()

    return


def plt_grab_responses_aligned(dset_obj, dset_ind_start=0,
                               t_pre=1, t_post=6,
                               trial_end=60):
    n_recs = dset_obj._dset_raw['expref'].shape[0]

    plot_conds_chs = SimpleNamespace()
    plot_conds_chs.ch = [1, 2]
    plot_conds_chs.plt_dff_y_gain = [10, 5]

    plot_conds_spec = SimpleNamespace()
    plot_conds_spec.plot_special = ['rew', 'rew_norew']
    for rec in range(dset_ind_start, n_recs):
        for ch in plot_conds_chs.ch:
            print(f'-------\n{ch=}')
            try:
                print('--------------')
                print(f'expref: {dset_obj._dset_raw.iloc[rec]["expref"]}')
                exp = TwoPRec_New(dset_obj=dset_obj, dset_ind=rec,
                                  ch_img=ch,
                                  trial_end=trial_end)
            except Exception as e:
                print('failed to load or analyze recording')
                print(e)
                pass

            for _ind_sp, _plot_special in enumerate(plot_conds_spec.plot_special):
                _y_gain = plot_conds_chs.plt_dff_y_gain[_ind_sp]
                exp.plt_stim_aligned_sectors(
                    n_sectors=10, img_ds_factor=50,
                    plt_dff={'x': {'gain': 0.6, 'offset': 0.2},
                             'y': {'gain': _y_gain, 'offset': 0.2}},
                    t_pre=t_pre, t_post=t_post,
                    plot_special=_plot_special,
                    plt_show=False)

    return


def plt_rew_value_from_dset(dset_obj, dset_ind_start=0,
                            ch_img=1, trial_end=60):
    n_recs = dset_obj._dset_raw['expref'].shape[0]
    for rec in range(dset_ind_start, n_recs):
        try:
            print('--------------')
            print(f'expref: {dset_obj._dset_raw.iloc[rec]["expref"]}')
            exp = TwoPRec_New(dset_obj=dset_obj, dset_ind=rec,
                              ch_img=ch_img,
                              trial_end=trial_end)
            exp.plt_stim_aligned_sectors(
                n_sectors=10, img_ds_factor=50,
                plt_dff={'x': {'gain': 0.6, 'offset': 0.2},
                         'y': {'gain': 10, 'offset': 0.2}},
                t_pre=2, t_post=5, plot_special=None,
                plt_show=False)
            exp.plt_psychometric_stimscaling(plt_show=False)
        except Exception as e:
            print('failed to load or analyze recording')
            print(e)
            pass

    return

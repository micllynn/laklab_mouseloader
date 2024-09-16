import numpy as np
import scipy as sp
import scipy.stats as sp_stats
from sklearn import decomposition
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns

from types import SimpleNamespace
import gc
import os
import copy

from .load_exp import ExpObj
from .utils import rescale_to_frac

try:
    plt.style.use('publication_ml')
except:
    pass


def get_neur_count(dset, region='supplemental somatosensory',
                   start_at_ind=0):
    region_inds = dset.get_region_inds(region)

    sess_count = 0
    neur_count = 0
    for ind in region_inds:
        if ind >= start_at_ind:
            try:
                exp = ExpObj(dset_obj=dset, dset_ind=ind)
                _count = len(np.where(
                    np.array(exp.ephys.spk.info.region) == region)[0])
                neur_count += _count
                sess_count += 1
                print(f'** neurs in this rec: {_count} **')
                print(f'** total neurs: {neur_count} **')
                print(f'** total sess: {sess_count} **')
                print('------------------')
                del exp
                gc.collect()
            except:
                print('!! could not load expref ',
                      f'{dset.npix.iloc[ind]["Exp_Ref"]}!! ')
                print('--------------')
                pass

    return neur_count


def get_ephys_all(dset, region='supplemental somatosensory',
                  start_at_ind=0,
                  ref='stim',
                  t_pre=1, t_post=1,
                  kern_fr_sd=25, sampling_rate=100,
                  get_zscore=True,
                  filter_high_perf=None,
                  wheel_turn_thresh_opto_incorr=None,
                  wheel_turn_thresh_dir_opto_incorr='neg',
                  n_neur_max=2000):

    trial_conds = ['opto', 'no_opto']
    choice_conds = ['corr', 'incorr']

    # load ephys from all datasets
    # -----------------
    region_inds = dset.get_region_inds(region)

    ephys = SimpleNamespace()
    ephys.tr = SimpleNamespace(
        opto=SimpleNamespace(),
        no_opto=SimpleNamespace())
    ephys.avg = SimpleNamespace(
        opto=SimpleNamespace(),
        no_opto=SimpleNamespace())
    ephys.n_sess = SimpleNamespace(
        opto=SimpleNamespace(),
        no_opto=SimpleNamespace())
    ephys.n_sess_abs = SimpleNamespace(
        opto=SimpleNamespace(),
        no_opto=SimpleNamespace())
    ephys.n_tr = SimpleNamespace(
        opto=SimpleNamespace(),
        no_opto=SimpleNamespace())

    ephys.params = SimpleNamespace(
        region=region,
        ref=ref,
        wheel_turn_thresh_opto_incorr=wheel_turn_thresh_opto_incorr,
        wheel_turn_thresh_dir_opto_incorr=wheel_turn_thresh_dir_opto_incorr,
        filter_high_perf=filter_high_perf)

    first_rec = SimpleNamespace(
        opto=SimpleNamespace(corr=True, incorr=True),
        no_opto=SimpleNamespace(corr=True, incorr=True))

    neur_count = SimpleNamespace(
        opto=SimpleNamespace(corr=0, incorr=0),
        no_opto=SimpleNamespace(corr=0, incorr=0))
    sess_count = 0

    for ind in region_inds:
        if ind >= start_at_ind:
            try:
                print('--------------')
                _exp = ExpObj(dset_obj=dset, dset_ind=ind)
                _ephys_rec = _exp.get_aligned_ephys(
                    region=[region], ref=ref,
                    t_pre=t_pre, t_post=t_post,
                    kern_fr_sd=kern_fr_sd, sampling_rate=sampling_rate,
                    get_zscore=get_zscore,
                    filter_high_perf=filter_high_perf,
                    wheel_turn_thresh_opto_incorr=wheel_turn_thresh_opto_incorr,
                    wheel_turn_thresh_dir_opto_incorr=wheel_turn_thresh_dir_opto_incorr)

                print('appending to all ephys...')

                # only include if there are incorrect trials
                if _ephys_rec.opto.incorr[region] is not None:
                    print('\tincluding session...')

                    for trial_cond in trial_conds:
                        ephys.t = getattr(_ephys_rec, trial_cond).t

                        for choice_cond in choice_conds:
                            print(f'\t{trial_cond=}, {choice_cond=}')
                            _first_rec = getattr(getattr(
                                first_rec, trial_cond), choice_cond)
                            _neur_count = getattr(getattr(
                                neur_count, trial_cond), choice_cond)
                            _ephys_rec_tr_ch = getattr(getattr(
                                _ephys_rec, trial_cond), choice_cond)

                            if _ephys_rec_tr_ch[region] is not None:
                                # first rec special tasks
                                # -------------
                                if _first_rec is True:
                                    print('\t\tfirst rec')
                                    # trial-averaged response in ephys.avg
                                    setattr(getattr(
                                        ephys.avg, trial_cond),
                                            choice_cond,
                                            np.mean(_ephys_rec_tr_ch[region],
                                                    axis=0))

                                    # setup attrs for tr, n_tr and n_sess
                                    setattr(getattr(
                                        ephys.tr, trial_cond),
                                            choice_cond,
                                            np.empty(n_neur_max,
                                                     dtype=np.ndarray))
                                    setattr(getattr(
                                        ephys.n_tr, trial_cond),
                                            choice_cond,
                                            np.zeros(n_neur_max, dtype=int))
                                    setattr(getattr(
                                        ephys.n_sess, trial_cond),
                                            choice_cond,
                                            np.zeros(n_neur_max, dtype=int))
                                    setattr(getattr(
                                        ephys.n_sess_abs, trial_cond),
                                            choice_cond,
                                            np.zeros(n_neur_max, dtype=int))

                                    # record if this is first rec and
                                    # update session count
                                    setattr(getattr(
                                        first_rec, trial_cond),
                                            choice_cond, False)

                                # non-first rec special tasks
                                # --------------
                                elif _first_rec is False:
                                    print('\t\tnot first rec')
                                    # append trial-averaged response
                                    # in ephys.avg and setattr back
                                    _ephys_avg = getattr(getattr(
                                        ephys.avg, trial_cond), choice_cond)
                                    _ephys_avg = np.append(
                                        _ephys_avg,
                                        np.mean(_ephys_rec_tr_ch[region],
                                                axis=0),
                                        axis=0)
                                    setattr(getattr(
                                        ephys.avg, trial_cond), choice_cond,
                                            _ephys_avg)

                                # common tasks regardless of first rec or not
                                # ---------------
                                # tr, n_tr and n_sess
                                for neur in range(_ephys_rec_tr_ch[region].shape[1]):
                                    # all trials in ephys.tr
                                    getattr(getattr(
                                        ephys.tr, trial_cond),
                                            choice_cond)[_neur_count] \
                                            = _ephys_rec_tr_ch[region][
                                                :, neur, :]
                                    # num trials in ephys.n_tr
                                    getattr(getattr(
                                        ephys.n_tr, trial_cond),
                                            choice_cond)[_neur_count] \
                                            = _ephys_rec_tr_ch[region][
                                                :, neur, :].shape[0]
                                    # session number
                                    getattr(getattr(
                                        ephys.n_sess, trial_cond),
                                            choice_cond)[_neur_count] \
                                            = sess_count
                                    getattr(getattr(
                                        ephys.n_sess_abs, trial_cond),
                                            choice_cond)[_neur_count] \
                                            = ind
                                    _neur_count += 1
                                    # print(f'\t\t\t{_neur_count=}')

                                setattr(getattr(neur_count, trial_cond),
                                        choice_cond, _neur_count)

                            elif _ephys_rec_tr_ch[region] is None:
                                print('\t\tno trials, skipping...')

                    sess_count += 1
                    del _exp
                    gc.collect()

                elif _ephys_rec.opto.incorr[region] is None:
                    print('\tskipping session due to no incorr trials...')

            except Exception as error:
                print('!! could not load expref ',
                      f'{dset.npix.iloc[ind]["Exp_Ref"]}!! ')
                print(f'error: {error}')
                pass

    # # cleanup (remove extra elements from ephys.tr)
    # # --------------
    for trial_cond in trial_conds:
        for choice_cond in choice_conds:
            _ephys_tr = getattr(getattr(
                ephys.tr, trial_cond), choice_cond)
            _ephys_maxneur_filt = [_ephys_tr[ind] is None
                                   for ind in range(_ephys_tr.shape[0])]
            _ephys_tr = np.delete(_ephys_tr, _ephys_maxneur_filt)
            setattr(getattr(ephys.tr, trial_cond), choice_cond,
                    _ephys_tr)

            _ephys_n_sess = getattr(getattr(
                ephys.n_sess, trial_cond),
                                    choice_cond)
            _ephys_n_sess = np.delete(_ephys_n_sess,
                                      _ephys_maxneur_filt)
            setattr(getattr(ephys.n_sess, trial_cond), choice_cond,
                    _ephys_n_sess)

            _ephys_n_sess_abs = getattr(getattr(
                ephys.n_sess_abs, trial_cond),
                                    choice_cond)
            _ephys_n_sess_abs = np.delete(_ephys_n_sess_abs,
                                          _ephys_maxneur_filt)
            setattr(getattr(ephys.n_sess_abs, trial_cond), choice_cond,
                    _ephys_n_sess_abs)

            _ephys_n_tr = getattr(getattr(
                ephys.n_tr, trial_cond),
                                    choice_cond)
            _ephys_n_tr = np.delete(_ephys_n_tr,
                                    _ephys_maxneur_filt)
            setattr(getattr(ephys.n_tr, trial_cond), choice_cond,
                    _ephys_n_tr)

    return ephys


def get_ephys_all_dict(dset,
                       region=['supplemental somatosensory',
                               'caudoputamen',
                               'thalamus'],
                       **kwargs):
    ephys_all_dict = {}
    for reg in region:
        ephys_all_dict[reg] = get_ephys_all(
            dset, region=reg, **kwargs)

    return ephys_all_dict


def plt_ephys_all_ordered(ephys_all,
                          figsize=(3.43, 3.43),
                          time=None,
                          ref='stim', sorting='time',
                          max_threshold_factor=2,
                          cmap=sns.diverging_palette(220, 20, as_cmap=True),
                          cmap_equal_pos_neg=True,
                          resample_hits_equal_n_tr=False,
                          len_metrics=20):
    os.chdir('/Users/michaellynn/Desktop/postdoc/proj/blake_perception_popanalysis')
    ephys_all_cp = copy.deepcopy(ephys_all)

    # if resample_hits_equal_n_tr is True:
    #     for neur in range(ephys_all_cp.avg.opto.corr.shape[0]):
    #         _n_tr_miss = ephys_all_cp.n_tr.opto.incorr[neur]
    #         _n_tr_hit = ephys_all_cp.n_tr.opto.corr[neur]
    #         _inds_random_tr_hit = np.random.choice(
    #             _n_tr_hit, _n_tr_miss)
    #         _new_travg_hit = np.mean(
    #             ephys_all_cp.tr.opto.corr[neur][_inds_random_tr_hit, :], axis=0)
    #         ephys_all_cp.avg.opto.corr[neur] = _new_travg_hit

    if time is not None:
        _ind_t_start = np.argmin(np.abs(ephys_all_cp.t-time[0]))
        _ind_t_end = np.argmin(np.abs(ephys_all_cp.t-time[1]))
        ephys_all_cp.t = ephys_all_cp.t[_ind_t_start:_ind_t_end]

        ephys_all_cp.avg.opto.corr = ephys_all_cp.avg.opto.corr[
            :, _ind_t_start:_ind_t_end]
        ephys_all_cp.avg.opto.incorr = ephys_all_cp.avg.opto.incorr[
            :, _ind_t_start:_ind_t_end]
        ephys_all_cp.avg.no_opto.corr = ephys_all_cp.avg.no_opto.corr[
            :, _ind_t_start:_ind_t_end]
        ephys_all_cp.avg.no_opto.incorr = ephys_all_cp.avg.no_opto.incorr[
            :, _ind_t_start:_ind_t_end]

    if sorting == 'time':
        sort_inds_corr = np.argsort(np.argmax(
            ephys_all_cp.avg.opto.corr, axis=1))
    elif sorting == 'diff':
        diff_corr_incorr = np.max(np.abs(
            ephys_all_cp.avg.opto.corr), axis=1) \
            - np.max(np.abs(
                ephys_all_cp.avg.opto.incorr),
                axis=1)
        # diff_corr_incorr = ephys_all_cp.avg.opto.corr - ephys_all_cp.avg.opto.incorr

        sort_inds_corr = np.flip(np.argsort(diff_corr_incorr))

    mean_act_corr = ephys_all_cp.avg.opto.corr[sort_inds_corr]
    mean_act_incorr = ephys_all_cp.avg.opto.incorr[sort_inds_corr]

    # plot
    # -----------
    fig = plt.figure(figsize=figsize, dpi=800)
    spec = gs.GridSpec(nrows=1, ncols=7,
                       width_ratios=[1, 1, 0.2, 0.2,
                                     0.05, 0.05, 0.05],
                       figure=fig)
    ax_corr = fig.add_subplot(spec[0, 0])
    ax_incorr = fig.add_subplot(spec[0, 1], sharey=ax_corr)
    ax_n_tr = fig.add_subplot(spec[0, 2], sharey=ax_corr)
    ax_n_sess = fig.add_subplot(spec[0, 3], sharey=ax_corr)

    ax_cbar_zscore = fig.add_subplot(spec[0, 4])
    ax_cbar_n_tr = fig.add_subplot(spec[0, 5])
    ax_cbar_n_sess = fig.add_subplot(spec[0, 6])

    vmin = np.min(mean_act_corr)
    vmax = np.max(mean_act_corr)/max_threshold_factor
    if cmap_equal_pos_neg is True:
        vmin = -1 * vmax

    if resample_hits_equal_n_tr is True:
        for neur in range(ephys_all_cp.avg.opto.corr.shape[0]):
            _n_tr_miss = ephys_all_cp.n_tr.opto.incorr[neur]
            _n_tr_hit = ephys_all_cp.n_tr.opto.corr[neur]
            _inds_random_tr_hit = np.random.choice(
                _n_tr_hit, _n_tr_miss)
            _new_travg_hit = np.mean(
                ephys_all_cp.tr.opto.corr[neur][_inds_random_tr_hit, :], axis=0)
            ephys_all_cp.avg.opto.corr[neur] = _new_travg_hit

            mean_act_corr = ephys_all_cp.avg.opto.corr[sort_inds_corr]
            mean_act_incorr = ephys_all_cp.avg.opto.incorr[sort_inds_corr]

    im_corr = ax_corr.imshow(mean_act_corr,
                             cmap=cmap,
                             vmin=vmin,
                             vmax=vmax)
    im_incorr = ax_incorr.imshow(mean_act_incorr,
                                 cmap=cmap,
                                 vmin=vmin,
                                 vmax=vmax)

    im_n_sess = ax_n_sess.imshow(np.tile(np.expand_dims(
        ephys_all_cp.n_sess.opto.corr[sort_inds_corr], axis=1),
                                         len_metrics))
    im_n_tr = ax_n_tr.imshow(np.tile(np.expand_dims(
        ephys_all_cp.n_tr.opto.incorr[sort_inds_corr], axis=1),
                                     len_metrics))

    _ind_zero = np.argmin(np.abs(ephys_all_cp.t))

    for _ax in [ax_corr, ax_incorr]:
        _ax.axvline(_ind_zero)
        _ax.set_xticks(
            ticks=[0, _ind_zero, ephys_all_cp.t.shape[-1]],
            labels=[f'{ephys_all_cp.t[0]:.2f}', 0,
                    f'{ephys_all_cp.t[-1]:.2f}'])

        if ref == 'stim':
            _ax.set_xlabel('time from stim (s)')
        if ref == 'choice':
            _ax.set_xlabel('time from choice (s)')

    ax_corr.set_ylabel('neur')
    ax_corr.set_title('hit trials')
    ax_incorr.set_title('miss trials')

    fig.colorbar(im_incorr, cax=ax_cbar_zscore, fraction=0.05)
    fig.colorbar(im_n_sess, cax=ax_cbar_n_sess, fraction=0.05)
    fig.colorbar(im_n_tr, cax=ax_cbar_n_tr, fraction=0.05)

    fig.savefig(f'neur_activ_{ephys_all.params.region}_{sorting=}_{ref=}_'
                + f'{ephys_all.params.wheel_turn_thresh_opto_incorr=}_'
                + f'{ephys_all.params.filter_high_perf=}_'
                + f'{resample_hits_equal_n_tr=}.pdf')

    plt.show()
    return


def plt_ephys_all_grandavg(dict_activ, figsize=(6.86, 1.5),
                           ref='stim', sort_by_diff=False,
                           sort_frac=[0, 0.25],
                           xlim=[-0.2, 0.5],
                           title_fontsize=8):
    dict_activ_cp = copy.deepcopy(dict_activ)
    region = dict_activ_cp.keys()

    if sort_by_diff is True:
        for reg in region:
            _n_neurs = dict_activ_cp[reg].avg.opto.corr.shape[0]
            _ind_start = int(sort_frac[0]*_n_neurs)
            _ind_end = int(sort_frac[1]*_n_neurs)

            diff_corr_incorr = np.max(np.abs(
                dict_activ_cp[reg].avg.opto.corr), axis=1) \
                - np.max(np.abs(
                    dict_activ_cp[reg].avg.opto.incorr),
                    axis=1)
            sort_inds_corr = np.flip(np.argsort(diff_corr_incorr))

            dict_activ_cp[reg].avg.opto.corr \
                = dict_activ_cp[reg].avg.opto.corr[sort_inds_corr][
                    _ind_start:_ind_end]
            dict_activ_cp[reg].avg.opto.incorr \
                = dict_activ_cp[reg].avg.opto.incorr[sort_inds_corr][
                    _ind_start:_ind_end]
            dict_activ_cp[reg].avg.no_opto.corr \
                = dict_activ_cp[reg].avg.no_opto.corr[sort_inds_corr][
                    _ind_start:_ind_end]

    # plot data
    # -------------
    fig = plt.figure(figsize=figsize)
    spec = gs.GridSpec(1, 4, figure=fig)
    axs = SimpleNamespace(opto=SimpleNamespace(),
                          no_opto=SimpleNamespace())

    axs.opto.corr = fig.add_subplot(spec[0, 0])
    axs.opto.incorr = fig.add_subplot(spec[0, 1], sharey=axs.opto.corr)
    axs.no_opto.corr = fig.add_subplot(spec[0, 2], sharey=axs.opto.corr)
    axs.no_opto.incorr = fig.add_subplot(spec[0, 3], sharey=axs.opto.corr)

    colors = {}
    colors['primary somatosensory'] = sns.xkcd_rgb['seafoam green']
    colors['supplemental somatosensory'] = sns.xkcd_rgb['blue green']
    colors['field CA'] = sns.xkcd_rgb['apple green']

    colors['caudoputamen'] = sns.xkcd_rgb['powder blue']
    colors['striatum'] = sns.xkcd_rgb['powder blue']
    colors['pallidum'] = sns.xkcd_rgb['faded blue']

    colors['lateral geniculate'] = sns.xkcd_rgb['salmon']
    colors['thalamus'] = sns.xkcd_rgb['pinkish']
    colors['midbrain'] = sns.xkcd_rgb['fuchsia']

    colors['hypothalamus'] = sns.xkcd_rgb['scarlet']

    colors['none'] = sns.xkcd_rgb['grey']

    for reg in region:
        for trial_cond in ['opto', 'no_opto']:
            for choice_cond in ['corr', 'incorr']:
                _ax = getattr(getattr(axs, trial_cond), choice_cond)
                _ephys = getattr(getattr(
                    dict_activ_cp[reg].avg, trial_cond), choice_cond)
                _t = dict_activ_cp[reg].t

                if _ephys is not None:
                    _ax.plot(
                        _t, np.mean(_ephys, axis=0),
                        color=colors[reg], label=reg)
                    _ax.fill_between(
                        _t,
                        np.mean(_ephys, axis=0)
                        + sp_stats.sem(_ephys, axis=0),
                        np.mean(_ephys, axis=0)
                        - sp_stats.sem(_ephys, axis=0),
                        facecolor=colors[reg],
                        alpha=0.2)

                # configure each plot
                if ref == 'stim':
                    _ax.axvline(0, color='k')
                    _ax.set_xlabel('time from stim (s)')
                if ref == 'choice':
                    _ax.axvline(0, color=sns.xkcd_rgb['grass green'])
                    _ax.set_xlabel('time from choice (s)')

                if xlim is not None:
                    _ax.set_xlim(xlim)

                _ax.set_title(trial_cond + ' ' + choice_cond,
                              fontsize=title_fontsize)

    axs.opto.corr.set_ylabel('pop. activity (z-score)')
    # axs.no_opto.incorr.legend(markerscale=0.5)

    figtitle = f'pop_fr_grandavg_{region=}.pdf'
    fig.savefig(figtitle)
    plt.show()

    return


def plt_ephys_all_corr_between_regions(dict_activ,
                                       dset,
                                       n_sess=0,
                                       t_start=0.15, t_end=0.25,
                                       t_delay=0.2,
                                       plt_keys=['supplemental somatosensory',
                                                 'thalamus'],
                                       plt_type='max',
                                       return_arr=False):
    # get common recs
    # ---------------
    sess_id_abs = np.intersect1d(
        dset.get_region_inds('supplemental somatosensory'),
        dset.get_region_inds('thalamus'))
    n_sess_in_id = sess_id_abs[n_sess]

    # analyze activity metrics
    # ----------------
    neur_dict = {}
    ephys_activ = SimpleNamespace(corr={}, incorr={})

    for key in dict_activ.keys():
        neur_dict[key] = np.where(
            dict_activ[key].n_sess_abs.opto.corr == n_sess_in_id)[0]

        n_trials_corr = dict_activ[key].tr.opto.corr[
            neur_dict[key][0]].shape[0]
        n_trials_incorr = dict_activ[key].tr.opto.incorr[
            neur_dict[key][0]].shape[0]
        t = dict_activ[key].t

        ephys_activ.corr[key] = np.zeros(n_trials_corr, dtype=np.ndarray)
        ephys_activ.incorr[key] = np.zeros(n_trials_incorr, dtype=np.ndarray)

        for tr in range(n_trials_corr):
            _neur_count = 0
            for ind_neur, neur in enumerate(neur_dict[key]):
                if _neur_count == 0:
                    ephys_activ.corr[key][tr] \
                        = dict_activ[key].tr.opto.corr[neur][tr, :]
                else:
                    ephys_activ.corr[key][tr] \
                        += dict_activ[key].tr.opto.corr[neur][tr, :]
                _neur_count += 1
            ephys_activ.corr[key][tr] /= neur_dict[key].shape[0]

        for tr in range(n_trials_incorr):
            _neur_count = 0
            for ind_neur, neur in enumerate(neur_dict[key]):
                if _neur_count == 0:
                    ephys_activ.incorr[key][tr] \
                        = dict_activ[key].tr.opto.incorr[neur][tr, :]
                else:
                    ephys_activ.incorr[key][tr] \
                        += dict_activ[key].tr.opto.incorr[neur][tr, :]
                _neur_count += 1
            ephys_activ.incorr[key][tr] /= neur_dict[key].shape[0]

    if return_arr is True:
        return ephys_activ

    # plot
    # ---------------
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot()

    _ind_t_start = np.argmin(np.abs(t-t_start))
    _ind_t_end = np.argmin(np.abs(t-t_end))
    _ind_t_delay = np.argmin(np.abs((t-t[0])-t_delay))

    # print(_ind_t_start, _ind_t_end, _ind_t_delay)

    for tr in range(n_trials_corr):
        if plt_type == 'max':
            _corr_xax = np.max(ephys_activ.corr[plt_keys[0]][tr][
                _ind_t_start+_ind_t_delay:_ind_t_end+_ind_t_delay])
            _corr_yax = np.max(ephys_activ.corr[plt_keys[1]][tr][
                _ind_t_start+_ind_t_delay:_ind_t_end+_ind_t_delay])
        if plt_type == 'mean':
            _corr_xax = np.mean(ephys_activ.corr[plt_keys[0]][tr][
                _ind_t_start+_ind_t_delay:_ind_t_end+_ind_t_delay])
            _corr_yax = np.mean(ephys_activ.corr[plt_keys[1]][tr][
                _ind_t_start+_ind_t_delay:_ind_t_end+_ind_t_delay])
        ax.scatter(_corr_xax, _corr_yax,
                   c=sns.xkcd_rgb['slate blue'])

    for tr in range(n_trials_incorr):
        if plt_type == 'max':
            _incorr_xax = np.max(ephys_activ.incorr[plt_keys[0]][tr][
                _ind_t_start+_ind_t_delay:_ind_t_end+_ind_t_delay])
            _incorr_yax = np.max(ephys_activ.incorr[plt_keys[1]][tr][
                _ind_t_start+_ind_t_delay:_ind_t_end+_ind_t_delay])
        if plt_type == 'mean':
            _incorr_xax = np.mean(ephys_activ.incorr[plt_keys[0]][tr][
                _ind_t_start+_ind_t_delay:_ind_t_end+_ind_t_delay])
            _incorr_yax = np.mean(ephys_activ.incorr[plt_keys[1]][tr][
                _ind_t_start+_ind_t_delay:_ind_t_end+_ind_t_delay])

        ax.scatter(_incorr_xax, _incorr_yax,
                   c=sns.xkcd_rgb['coral'])

    ax.set_xlabel(plt_keys[0])
    ax.set_ylabel(plt_keys[1])
    fig.savefig('/Users/michaellynn/Desktop/postdoc/proj/'
                + 'blake_perception_popanalysis/'
                + f'pop_activ_corr_sess={n_sess_in_id}_'
                + f'{plt_keys=}'
                + f'{t_start=}_{t_end=}_{t_delay=}.pdf')
    plt.show()
    return


def plt_wheel_data(exp, figsize=(3, 3), plt_thresh=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()

    n_trials = exp.beh.wheel.sig.shape[0]
    for trial in range(n_trials):
        if exp.beh.stim.opto[trial] == True:
            if exp.beh.choice.correct[trial] == True:
                _alpha = 0.8
                _color = sns.xkcd_rgb['bright blue']
            elif exp.beh.choice.correct[trial] == False:
                _alpha = 0.4
                _color = sns.xkcd_rgb['orange red']                

            ax.plot(exp.beh.wheel.t[trial],
                    np.abs(exp.beh.wheel.sig[trial]),
                    color=_color, alpha=_alpha)
    ax.axvline(0, color=sns.xkcd_rgb['grey'], linestyle='dashed')
    if plt_thresh is not None:
        ax.axhline(plt_thresh, color=sns.xkcd_rgb['black'])
    ax.set_xlabel('time from stim (s)')
    ax.set_ylabel('wheel pos. (A.U.)')
    fig.savefig(f'wheel_data_{exp.folder.enclosing.split("/")[-2]}'
                + f'_{exp.folder.enclosing.split("/")[-1]}_{plt_thresh=}.pdf')
    plt.show()


def extract_beh_metrics_dset(figsize=(6.86, 6.86),
                             n_beh_bins=51,
                             ind_range=None, show_plts=False,
                             kern_fr_sd=500, kern_perf_fast=5,
                             kern_rew_rate=5, kern_t_react=5):
    dset = DSetObj()
    len_dset = dset.npix.shape[0]
    metrics = SimpleNamespace(pop=SimpleNamespace(),
                              ind=SimpleNamespace())

    if ind_range is None:
        ind_range = [0, len_dset]
    n_inds = int(ind_range[1]-ind_range[0])

    metrics.frac = np.linspace(0, 1, n_beh_bins)

    metrics.pop.perf = np.zeros(n_beh_bins)
    metrics.pop.rew_rate = np.zeros(n_beh_bins)
    metrics.pop.reaction_time = np.zeros(n_beh_bins)

    metrics.ind.perf = np.zeros((n_inds, n_beh_bins))
    metrics.ind.rew_rate = np.zeros((n_inds, n_beh_bins))
    metrics.ind.reaction_time = np.zeros((n_inds, n_beh_bins))

    metrics._inds_bad = []
    count_inds_good = 0
    for ind_rel_dset, ind_abs_dset in enumerate(
            range(ind_range[0], ind_range[1])):
        print(f'\n{ind_abs_dset}/{len_dset} recs...')
        print('------------------')
        try:
            exp = ExpObj(dset_obj=dset, dset_ind=ind_abs_dset)
            exp.beh.add_metrics(kern_perf_fast=kern_perf_fast,
                                kern_rew_rate=kern_rew_rate,
                                kern_t_react=kern_t_react)

            _perf_fast = rescale_to_frac(exp.beh.metrics.t_trial,
                                         exp.beh.metrics.perf.fast,
                                         n_bins=n_beh_bins)
            _rew_rate = rescale_to_frac(exp.beh.metrics.t_trial,
                                        exp.beh.metrics.rew_rate,
                                        n_bins=n_beh_bins)
            _reaction_time = rescale_to_frac(exp.beh.metrics.t_trial,
                                             exp.beh.metrics.reaction_time,
                                             n_bins=n_beh_bins)

            metrics.pop.perf += _perf_fast.beh
            metrics.ind.perf[ind_rel_dset, :] = _perf_fast.beh

            metrics.pop.rew_rate += _rew_rate.beh
            metrics.ind.rew_rate[ind_rel_dset, :] = _rew_rate.beh

            metrics.pop.reaction_time += _reaction_time.beh
            metrics.ind.reaction_time[ind_rel_dset, :] = _reaction_time.beh

            count_inds_good += 1

        except:
            metrics._inds_bad.append(ind_rel_dset)
            _expref = dset._parse_expref(ind_abs_dset)
            print(f'\n*** could not load'
                  f'{_expref[0]} || {_expref[1]} || {_expref[2]}.')

    # calc mean for pop data
    for _metric in [metrics.pop.perf, metrics.pop.rew_rate,
                    metrics.pop.reaction_time]:
        _metric /= count_inds_good

    metrics.ind.perf = np.delete(metrics.ind.perf, metrics._inds_bad, axis=0)
    metrics.ind.rew_rate = np.delete(
        metrics.ind.rew_rate, metrics._inds_bad, axis=0)
    metrics.ind.reaction_time = np.delete(
        metrics.ind.reaction_time, metrics._inds_bad, axis=0)

    return metrics


def plt_beh_metrics(metrics, figsize=(6.86, 3.43),
                    inds_to_highlight=[],
                    save_dir='/Users/michaellynn/Desktop/postdoc/'
                    + 'proj/antara_caitlin_mousegambling_popanalysis/figs'):
    fig = plt.figure(figsize=figsize)
    spec = gs.GridSpec(2, 3, figure=fig)
    ax_perf = fig.add_subplot(spec[0, 0])
    ax_rew = fig.add_subplot(spec[0, 1])
    ax_rxn = fig.add_subplot(spec[0, 2])

    ax_resid_perf = fig.add_subplot(spec[1, 0])
    ax_resid_rew = fig.add_subplot(spec[1, 1])
    ax_resid_rxn = fig.add_subplot(spec[1, 2])

    ax_perf.plot(metrics.frac, metrics.pop.perf,
                 color=sns.xkcd_rgb['grass green'],
                 linewidth=2)
    ax_rew.plot(metrics.frac, metrics.pop.rew_rate,
                color=sns.xkcd_rgb['cerulean'],
                linewidth=2)
    ax_rxn.plot(metrics.frac, metrics.pop.reaction_time,
                color=sns.xkcd_rgb['dull orange'],
                linewidth=2)

    for rec in range(metrics.pop.perf.shape[0]):
        # plot indiv beh metrics

        ax_perf.plot(metrics.frac, metrics.ind.perf[rec, :],
                     color=sns.xkcd_rgb['grass green'],
                     linewidth=0.5, alpha=0.3)
        ax_rew.plot(metrics.frac, metrics.ind.rew_rate[rec, :],
                    color=sns.xkcd_rgb['cerulean'],
                    linewidth=0.5, alpha=0.3)
        ax_rxn.plot(metrics.frac, metrics.ind.reaction_time[rec, :],
                    color=sns.xkcd_rgb['dull orange'],
                    linewidth=0.5, alpha=0.3)

        # plot residuals of individ beh metrics
        if rec in inds_to_highlight:
            _lw_resid = 0.8
            _alpha = 1
            _color_perf = sns.xkcd_rgb['black']
            _color_rew = sns.xkcd_rgb['black']
            _color_rxn = sns.xkcd_rgb['black']
        else:
            _lw_resid = 0.5
            _alpha = 0.3
            _color_perf = sns.xkcd_rgb['grass green']
            _color_rew = sns.xkcd_rgb['cerulean']
            _color_rxn = sns.xkcd_rgb['dull orange']

        _resid_perf = metrics.ind.perf[rec, :] - metrics.pop.perf
        _resid_rew_rate = metrics.ind.rew_rate[rec, :] - metrics.pop.rew_rate
        _resid_reaction_time = metrics.ind.reaction_time[rec, :] \
            - metrics.pop.reaction_time
        ax_resid_perf.plot(metrics.frac, _resid_perf,
                           color=_color_perf,
                           linewidth=_lw_resid, alpha=_alpha)
        ax_resid_rew.plot(metrics.frac, _resid_rew_rate,
                          color=_color_rew,
                          linewidth=_lw_resid, alpha=_alpha)
        ax_resid_rxn.plot(metrics.frac, _resid_reaction_time,
                          color=_color_rxn,
                          linewidth=_lw_resid, alpha=_alpha)

    for _ax in [ax_resid_perf, ax_resid_rew, ax_resid_rxn]:
        _ax.plot(metrics.frac, np.zeros_like(metrics.frac),
                 '--', color='k')
        _ax.set_xlabel('frac. session')

    ax_perf.set_ylabel('performance')
    ax_rew.set_ylabel('rew rate (ul/s)')
    ax_rxn.set_ylabel('react. time (s)')

    ax_resid_perf.set_ylabel('resid. perf.')
    ax_resid_rew.set_ylabel('resid. rew rate')
    ax_resid_rxn.set_ylabel('resid. react t')

    ax_rxn.set_yscale('log')

    plt.show()
    fig.savefig(os.path.join(save_dir, 'beh_metrics.pdf'))
    fig.clear()
    gc.collect()


def pca_on_beh_metrics(metrics, metric_key='perf',
                       n_components=5,
                       save_dir='/Users/michaellynn/Desktop/postdoc/'
                       + 'proj/antara_caitlin_mousegambling_popanalysis/figs'):
    metric = getattr(metrics.ind, metric_key)

    pca = decomposition.PCA(n_components=n_components)
    pca.fit(metric.T)
    pca_weights = pca.components_

    # plot
    # -------------
    fig = plt.figure(figsize=(3.43, 6))
    spec = gs.GridSpec(2, 3, figure=fig)
    ax_perf_pc1 = fig.add_subplot(spec[0, 0])
    ax_rew_pc1 = fig.add_subplot(spec[0, 1], sharey=ax_perf_pc1)
    ax_rxn_pc1 = fig.add_subplot(spec[0, 2], sharey=ax_perf_pc1)

    ax_perf_pc2 = fig.add_subplot(spec[1, 0], sharex=ax_perf_pc1)
    ax_rew_pc2 = fig.add_subplot(spec[1, 1], sharex=ax_rew_pc1,
                                 sharey=ax_perf_pc2)
    ax_rxn_pc2 = fig.add_subplot(spec[1, 2], sharex=ax_rxn_pc1,
                                 sharey=ax_perf_pc2)

    ax_perf_pc1.imshow(
        metrics.ind.perf[np.flip(pca_weights[0, :].argsort())],
        cmap='viridis', interpolation=None)
    ax_rew_pc1.imshow(
        metrics.ind.rew_rate[np.flip(pca_weights[0, :].argsort())],
        cmap='plasma', interpolation=None)
    ax_rxn_pc1.imshow(
        metrics.ind.reaction_time[np.flip(pca_weights[0, :].argsort())],
        cmap='cividis', interpolation=None, norm=mpl.colors.LogNorm())

    ax_perf_pc2.imshow(
        metrics.ind.perf[np.flip(pca_weights[1, :].argsort())],
        cmap='viridis', interpolation=None)
    ax_rew_pc2.imshow(
        metrics.ind.rew_rate[np.flip(pca_weights[1, :].argsort())],
        cmap='plasma', interpolation=None)
    ax_rxn_pc2.imshow(
        metrics.ind.reaction_time[np.flip(pca_weights[1, :].argsort())],
        cmap='cividis', interpolation=None, norm=mpl.colors.LogNorm())

    if metric_key == 'perf':
        ax_perf_pc1.set_title('perf, sort by pc1')
        ax_perf_pc2.set_title('perf, sort by pc2')
        ax_rew_pc1.set_title('rew rate')
        ax_rew_pc2.set_title('rew rate')
        ax_rxn_pc1.set_title('rxn t')
        ax_rxn_pc2.set_title('rxn t')
    if metric_key == 'rew_rate':
        ax_perf_pc1.set_title('perf')
        ax_perf_pc2.set_title('perf')
        ax_rew_pc1.set_title('rew rate, sort by pc1')
        ax_rew_pc2.set_title('rew rate, sort by pc2')
        ax_rxn_pc1.set_title('rxn t')
        ax_rxn_pc2.set_title('rxn t')
    if metric_key == 'reaction_time':
        ax_perf_pc1.set_title('perf')
        ax_perf_pc2.set_title('perf')
        ax_rew_pc1.set_title('rew rate')
        ax_rew_pc2.set_title('rew rate')
        ax_rxn_pc1.set_title('rxn t, sort by pc1')
        ax_rxn_pc2.set_title('rxn t, sort by pc2')

    for _ax in [ax_perf_pc1, ax_perf_pc2]:
        _ax.set_ylabel('session num.')
    for _ax in [ax_perf_pc2, ax_rew_pc2, ax_rxn_pc2]:
        _ax.set_xlabel('frac. of session')

    plt.show()
    fig.savefig(os.path.join(save_dir, f'pca_beh_sortby={metric_key}.pdf'),
                dpi=800)
    fig.clear()
    gc.collect()

    return


def plt_corr_beh_metrics(metrics,
                         save_dir='/Users/michaellynn/Desktop/'
                         + 'postdoc/proj/antara_caitlin_'
                         + 'mousegambling_popanalysis/figs',
                         ch_start=-0.6, ch_rot=0.5, ch_hue=0.8,
                         ch_rev=False,
                         reg_logscale=True):
    # setup figs
    # ------------------
    fig = plt.figure(figsize=(3.43, 2))
    fig2_qt = plt.figure(figsize=(3.43, 1.5))

    spec = gs.GridSpec(1, 3, figure=fig)
    spec_qt = gs.GridSpec(1, 4, figure=fig2_qt)

    ax_perf_vs_rewrate = fig.add_subplot(spec[0, 0])
    ax_perf_vs_react_t = fig.add_subplot(spec[0, 1])
    ax_corr_perf_vs_react_t = fig.add_subplot(spec[0, 2])

    ax_q1 = fig2_qt.add_subplot(spec_qt[0, 0])
    ax_q2 = fig2_qt.add_subplot(spec_qt[0, 1], sharey=ax_q1)
    ax_q3 = fig2_qt.add_subplot(spec_qt[0, 2], sharey=ax_q1)
    ax_q4 = fig2_qt.add_subplot(spec_qt[0, 3], sharey=ax_q1)

    colorpal = sns.cubehelix_palette(n_colors=metrics.ind.perf.shape[1],
                                     start=ch_start,
                                     rot=ch_rot,
                                     hue=ch_hue,
                                     reverse=ch_rev)

    # analyze data and plot for fig1
    # --------------
    corr_perf_vs_react_t = np.zeros((metrics.ind.perf.shape[0], metrics.ind.perf.shape[1]-1))
    _x_corr = np.linspace(-0.5, 0.5, metrics.ind.perf.shape[1]-1)

    for rec in range(metrics.ind.perf.shape[0]):
        corr_perf_vs_react_t[rec, :] = sp.signal.correlate(
            sp.stats.zscore(np.diff(metrics.ind.perf[rec, :])),
            sp.stats.zscore(np.diff(metrics.ind.reaction_time[rec, :])),
            mode='same')
        ax_corr_perf_vs_react_t.plot(_x_corr,
                                     corr_perf_vs_react_t[rec, :],
                                     linewidth=0.5, alpha=0.2,
                                     color=sns.xkcd_rgb['sea blue'])

    for frac_ses in range(metrics.ind.perf.shape[1]):
        ax_perf_vs_rewrate.scatter(metrics.ind.perf[:, frac_ses],
                                   metrics.ind.rew_rate[:, frac_ses],
                                   color=colorpal[frac_ses],
                                   alpha=0.3)
        ax_perf_vs_react_t.scatter(metrics.ind.perf[:, frac_ses],
                                   metrics.ind.reaction_time[:, frac_ses],
                                   color=colorpal[frac_ses],
                                   alpha=0.3)

    ax_corr_perf_vs_react_t.plot(_x_corr,
                                 np.mean(corr_perf_vs_react_t, axis=0),
                                 linewidth=2, alpha=1,
                                 color=sns.xkcd_rgb['sea blue'])
    ax_corr_perf_vs_react_t.set_xlabel('$\Delta$frac. session')
    ax_corr_perf_vs_react_t.set_ylabel('xcorr (perf vs $t_{react}$)')

    ax_perf_vs_rewrate.set_xlabel('perf')
    ax_perf_vs_rewrate.set_ylabel('rew rate')

    ax_perf_vs_react_t.set_xlabel('perf')
    ax_perf_vs_react_t.set_ylabel('$t_{react}$')
    ax_perf_vs_react_t.set_yscale('log')

    ax_corr_perf_vs_react_t.plot([0, 0],
                                 [ax_corr_perf_vs_react_t.get_ylim()[0],
                                  ax_corr_perf_vs_react_t.get_ylim()[1]],
                                 '--', color='k')


    # analyze and plot for fig2
    # -------------
    pop = SimpleNamespace()
    pop.perf = SimpleNamespace(q1=[], q2=[], q3=[], q4=[])
    pop.rt = SimpleNamespace(q1=[], q2=[], q3=[], q4=[])

    for frac_ses in range(metrics.ind.perf.shape[1]):
        if metrics.frac[frac_ses] < 0.25:
            ax_q1.scatter(metrics.ind.perf[:, frac_ses],
                          metrics.ind.reaction_time[:, frac_ses],
                          color=colorpal[frac_ses],
                          alpha=0.3)
            pop.perf.q1 = np.append(pop.perf.q1,
                                    metrics.ind.perf[:, frac_ses])
            pop.rt.q1 = np.append(pop.rt.q1,
                                  metrics.ind.reaction_time[:, frac_ses])

        elif np.logical_and(metrics.frac[frac_ses] > 0.25,
                            metrics.frac[frac_ses] < 0.5):
            ax_q2.scatter(metrics.ind.perf[:, frac_ses],
                          metrics.ind.reaction_time[:, frac_ses],
                          color=colorpal[frac_ses],
                          alpha=0.3)
            pop.perf.q2 = np.append(pop.perf.q2,
                                    metrics.ind.perf[:, frac_ses])
            pop.rt.q2 = np.append(pop.rt.q2,
                                  metrics.ind.reaction_time[:, frac_ses])
        elif np.logical_and(metrics.frac[frac_ses] > 0.5,
                            metrics.frac[frac_ses] < 0.75):
            ax_q3.scatter(metrics.ind.perf[:, frac_ses],
                          metrics.ind.reaction_time[:, frac_ses],
                          color=colorpal[frac_ses],
                          alpha=0.3)
            pop.perf.q3 = np.append(pop.perf.q3,
                                    metrics.ind.perf[:, frac_ses])
            pop.rt.q3 = np.append(pop.rt.q3,
                                  metrics.ind.reaction_time[:, frac_ses])

        elif metrics.frac[frac_ses] > 0.75:
            ax_q4.scatter(metrics.ind.perf[:, frac_ses],
                          metrics.ind.reaction_time[:, frac_ses],
                          color=colorpal[frac_ses],
                          alpha=0.3)
            pop.perf.q4 = np.append(pop.perf.q4,
                                    metrics.ind.perf[:, frac_ses])
            pop.rt.q4 = np.append(pop.rt.q4,
                                  metrics.ind.reaction_time[:, frac_ses])

    # fits
    # -----------
    linreg = SimpleNamespace()
    line_x = np.linspace(0, 1, 1000)
    line_y = SimpleNamespace()

    if reg_logscale is True:
        linreg.q1 = sp.stats.linregress(pop.perf.q1, np.log10(pop.rt.q1))
        linreg.q2 = sp.stats.linregress(pop.perf.q2, np.log10(pop.rt.q2))
        linreg.q3 = sp.stats.linregress(pop.perf.q3, np.log10(pop.rt.q3))
        linreg.q4 = sp.stats.linregress(pop.perf.q4, np.log10(pop.rt.q4))

        line_y.q1 = np.exp(linreg.q1.slope*line_x
                           + linreg.q1.intercept)
        line_y.q2 = np.exp(linreg.q2.slope*line_x
                           + linreg.q2.intercept)
        line_y.q3 = np.exp(linreg.q3.slope*line_x
                           + linreg.q3.intercept)
        line_y.q4 = np.exp(linreg.q4.slope*line_x
                           + linreg.q4.intercept)
    elif reg_logscale is False:
        linreg.q1 = sp.stats.linregress(pop.perf.q1, pop.rt.q1)
        linreg.q2 = sp.stats.linregress(pop.perf.q2, pop.rt.q2)
        linreg.q3 = sp.stats.linregress(pop.perf.q3, pop.rt.q3)
        linreg.q4 = sp.stats.linregress(pop.perf.q4, pop.rt.q4)

        line_y.q1 = linreg.q1.slope*line_x + linreg.q1.intercept
        line_y.q2 = linreg.q2.slope*line_x + linreg.q2.intercept
        line_y.q3 = linreg.q3.slope*line_x + linreg.q3.intercept
        line_y.q4 = linreg.q4.slope*line_x + linreg.q4.intercept

    ax_q1.plot(line_x, line_y.q1, linewidth=1, color='k')
    ax_q2.plot(line_x, line_y.q2, linewidth=1, color='k')
    ax_q3.plot(line_x, line_y.q3, linewidth=1, color='k')
    ax_q4.plot(line_x, line_y.q4, linewidth=1, color='k')

    # configure plots and save
    # -----------
    for _ax in [ax_q1, ax_q2, ax_q3, ax_q4]:
        _ax.set_xlabel('perf')
        _ax.set_ylabel('$t_{react}$')
        _ax.set_yscale('log')

    ax_q1.set_title('Q1')
    ax_q2.set_title('Q2')
    ax_q3.set_title('Q3')
    ax_q4.set_title('Q4')

    plt.show()
    fig.savefig(os.path.join(save_dir, 'beh_metrics_corr.pdf'))
    fig2_qt.savefig(os.path.join(save_dir, 'beh_metrics_corr_quartiles.pdf'))

    fig.clear()
    fig2_qt.clear()

    gc.collect()

    # save text
    with open(os.path.join(save_dir, 'beh_metrics_linreg_data.txt'), 'w') as f:
        f.write(f'{reg_logscale=}\n')
        for _q in ['q1', 'q2', 'q3', 'q4']:
            f.write(_q+'\n')
            f.write('\t'+str(getattr(linreg, _q))+'\n')

    return linreg


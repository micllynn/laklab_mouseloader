"""
This file contains classes to loads the ephys and behavior files
from a full session from Blake Russel's ReportOpto project.

Main usage
-----------
from dset import DSetObj
dset = DSetObj()  # loads dataset .csv
exp = ExpObj(dset_obj=dset, dset_ind=5)  # loads expref 5 into exp
"""


import numpy as np
import scipy.stats as sp_stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns

from types import SimpleNamespace
import os
import gc

from .utils import calc_alpha, smooth_spktrain
from .beh import BehData
from .ephys import EphysData
from .dset import DSetObj
from .align_ephysbeh import Aligner_EphysBeh


class ExpObj(object):
    def __init__(self, dset_obj=None, dset_ind=None,
                 enclosing_folder=None,
                 ephys_folder='ephys_g0', beh_folder='1'):
        """
        This class loads the ephys and behavior files from a full
        session from Blake Russel's ReportOpto project.

        The location of a session can either be specified manually
        (using enclosing_folder, ephys_folder, beh_folder),
        or by specifying a dataset folder (dset_obj and dset_ind)

        Parameters
        -------------
        ** option 1: DSet loading **
        dset_obj : DSetObj (from the module dset)
            Indicates a DSetObj to load information about Blake's dataset
        dset_ind : int
            Index of dataset to load data from (specified as rows from top of
            dataset .csv
        ** option 2: manual folder loading **
        enclosing folder: str
            Location of enclosing folder (eg expref with both behavior
            and ephys folders
            located inside)
        ephys_folder : str
            Name of ephys folder inside of enclosing folder
        beh_folder : str
            Name of beh folder inside of enclosing folder

        Main usage
        -----------
        from dset import DSetObj
        dset = DSetObj()  # loads dataset .csv
        exp = ExpObj(dset_obj=dset, dset_ind=5)  # loads expref 5 into exp
        """
        self.folder = SimpleNamespace()
        if dset_obj is None:
            self.folder.enclosing = enclosing_folder
            self.folder.ephys = ephys_folder
            self.folder.beh = beh_folder
        elif 'DSetObj' in str(type(dset_obj)):
            self.folder.enclosing = dset_obj.get_path_expref(dset_ind)
            self.folder.ephys = dset_obj.get_path_ephys(dset_ind)
            self.folder.beh = dset_obj.get_path_beh(dset_ind)
        else:
            raise TypeError('dset_obj must be type DSetObj')

        print(f'loading {self.folder.enclosing}...')
        os.chdir(self.folder.enclosing)

        # setup aligner objects
        print('\tcreating aligner object...')
        self.aligner_obj = Aligner_EphysBeh()
        self.aligner_obj.parse_ephys_rewechoes(folder=self.folder.ephys)
        self.aligner_obj.parse_beh_rewechoes(folder=self.folder.beh)
        self.aligner_obj.compute_alignment()

        # setup behavior and ephys objects
        self.beh = BehData(self.folder.beh)
        self.ephys = EphysData(self.folder.ephys,
                               aligner_obj=self.aligner_obj)
        self.ephys.add_spktrain_kernconv(kern_sd=5)

        # setup colors
        self.reg_colors = {}
        self.reg_colors['Primary somatosensory'] \
            = sns.xkcd_rgb['seafoam green']
        self.reg_colors['Supplemental somatosensory'] \
            = sns.xkcd_rgb['blue green']
        self.reg_colors['Field CA'] = sns.xkcd_rgb['apple green']

        self.reg_colors['Caudoputamen'] = sns.xkcd_rgb['powder blue']
        self.reg_colors['Striatum'] = sns.xkcd_rgb['powder blue']
        self.reg_colors['Pallidum'] = sns.xkcd_rgb['faded blue']

        self.reg_colors['lateral geniculate'] = sns.xkcd_rgb['salmon']
        self.reg_colors['thalamus'] = sns.xkcd_rgb['pinkish']
        self.reg_colors['Thalamus'] = sns.xkcd_rgb['pinkish']
        self.reg_colors['Midbrain'] = sns.xkcd_rgb['fuchsia']

        self.reg_colors['Hypothalamus'] = sns.xkcd_rgb['scarlet']

        self.reg_colors['none'] = sns.xkcd_rgb['grey']

    def plt_exp(self, scatter_size=2, figsize=(6.86, 6.86),
                fig_save=True, fig_save_duplicate_folder=None,
                t_range=None, region=None, kern_fr_sd=50,
                plt_areas_separately=True,
                plt_wheel_turn_vel=False, t_pre_stim=2,
                t_post_stim=3, plt_dpi=600,
                kern_perf_fast=5, kern_perf_slow=51,
                kern_rew_rate=5, kern_t_react=5,
                psi_spkbin=10, psi_indbin=1000,
                show_plts=True):
        """
        This method generates a summary plot of the loaded experiment,
        including behavior and ephys components.

        brain regions and colors
        -----------
        """
        self.plt = SimpleNamespace()
        self.plt.fig = plt.figure(figsize=figsize)

        self.plt.spec = gs.GridSpec(nrows=7, ncols=1,
                                    height_ratios=[0.2, 0.15, 0.15, 0.15,
                                                   1, 0.1, 0.15],
                                    figure=self.plt.fig)

        self.plt.ax_beh = self.plt.fig.add_subplot(self.plt.spec[0, 0])
        self.plt.ax_beh_metrics_wheel = self.plt.fig.add_subplot(
            self.plt.spec[1, 0], sharex=self.plt.ax_beh)
        self.plt.ax_beh_metrics_perf = self.plt.fig.add_subplot(
            self.plt.spec[2, 0], sharex=self.plt.ax_beh)
        self.plt.ax_beh_metrics_rxn = self.plt.fig.add_subplot(
            self.plt.spec[3, 0], sharex=self.plt.ax_beh)
        self.plt.ax_raster = self.plt.fig.add_subplot(self.plt.spec[4, 0],
                                                      sharex=self.plt.ax_beh)
        self.plt.ax_fr = self.plt.fig.add_subplot(self.plt.spec[5, 0],
                                                  sharex=self.plt.ax_beh)
        self.plt.ax_psi = self.plt.fig.add_subplot(self.plt.spec[6, 0],
                                                   sharex=self.plt.ax_beh)

        # behavior plot (basic)
        # -------------
        print('plotting behavior...')
        print('\ttask')
        n_trials = self.beh.stim.t_start.shape[0]
        for trial in range(n_trials):
            # plot opto stim
            if self.beh.stim.opto[trial] == 1:
                self.plt.ax_beh.plot(np.tile(self.beh.stim.t_start[trial], 2),
                                     [0, -1], color='k', alpha=1, linewidth=1.5)
            if self.beh.stim.opto[trial] == 0:
                self.plt.ax_beh.plot(np.tile(self.beh.stim.t_start[trial], 2),
                                     [0, -1], '--', color='k', alpha=0.4,
                                     linewidth=1.5)

            # plot choice event
            _response = self.beh.choice.response[trial]

            if _response == 1:
                self.plt.ax_beh.plot(np.tile(self.beh.choice.t[trial], 2),
                                     [0, -1],
                                     color=sns.xkcd_rgb['grass green'],
                                     linewidth=1.5)

                # plot reward event
                self.plt.ax_beh.plot(np.tile(self.beh.rew.t[trial], 2),
                                     [0, -1], color=sns.xkcd_rgb['cerulean'],
                                     alpha=1,
                                     linewidth=1.5)

        self.plt.ax_beh.set_yticks([-0.5])
        self.plt.ax_beh.set_yticklabels(['opto'])

        self.beh.add_metrics(kern_perf_fast=kern_perf_fast,
                             kern_perf_slow=kern_perf_slow)

        # behavioral metrics (wheel turning)
        # -----------
        print('\twheel turning')
        for trial in range(n_trials):
            self.plt.ax_beh_metrics_wheel.plot(
                self.beh.wheel.t_abs[trial],
                self.beh.wheel.sig[trial],
                color=sns.xkcd_rgb['cerulean'],
                alpha=0.7)
        self.plt.ax_beh_metrics_wheel.set_ylabel(
            'wheel pos.\n(mm)')

        # behavioral metrics (perf)
        # -----------
        print('\tperformance metrics')
        if self.beh.metrics._params.sort_by_ttype is False:
            self.plt.ax_beh_metrics_perf.plot(
                self.beh.metrics.t_trial, self.beh.metrics.perf.slow,
                color=sns.xkcd_rgb['grass green'],
                linewidth=2)
            self.plt.ax_beh_metrics_perf.plot(
                self.beh.metrics.t_trial, self.beh.metrics.perf.fast,
                color=sns.xkcd_rgb['grass green'],
                linewidth=0.5)
        elif self.beh.metrics._params.sort_by_ttype is True:
            self.plt.ax_beh_metrics_perf.plot(
                self.beh.metrics.perf.t_opto,
                self.beh.metrics.perf.opto,
                color=sns.xkcd_rgb['grass green'],
                linewidth=1)
            self.plt.ax_beh_metrics_perf.plot(
                self.beh.metrics.perf.t_no_opto,
                self.beh.metrics.perf.no_opto,
                '--',
                color=sns.xkcd_rgb['grass green'],
                linewidth=1)

        self.plt.ax_beh_metrics_perf.set_ylabel('perf.')

        # behavioral metrics (rxn)
        # -----------
        print('\treaction metrics')
        self.plt.ax_beh_metrics_rxn.semilogy(
            self.beh.metrics.reaction_time_t_correct,
            self.beh.metrics.reaction_time,
            color=sns.xkcd_rgb['dull orange'],
            alpha=0.7)
        self.plt.ax_beh_metrics_rxn.set_ylabel(
            'reaction\ntime (s)')

        # ephys raster plot
        # -------------
        print('plotting ephys...')
        print('\traster plot')
        for ind_clust, raw_clust_id in enumerate(self.ephys.spk.raw_clust_id):
            print(f'plotting {ind_clust=}...', end='\r')
            _region_key = self.ephys.spk.info.region[ind_clust]

            if region is None:
                _t_spk = self.ephys.spk_raw.times[
                    self.ephys.spk_raw.clusters == raw_clust_id]
                _color = self.ephys.plt.colors[_region_key]
                self.plt.ax_raster.scatter(
                    _t_spk, np.ones_like(_t_spk)*-1*ind_clust,
                    s=scatter_size,
                    color=_color)
            else:
                if _region_key in region:
                    _t_spk = self.ephys.spk_raw.times[
                        self.ephys.spk_raw.clusters == raw_clust_id]
                    _color = self.ephys.plt.colors[_region_key]
                    self.plt.ax_raster.scatter(
                        _t_spk, np.ones_like(_t_spk)*-1*ind_clust,
                        s=scatter_size,
                        color=_color)

        # firing rate plot
        # ------------
        print('\tfiring rate')
        kern_sd_key = str(kern_fr_sd) + 'ms'
        self.ephys.add_spktrain_kernconv(kern_sd=kern_fr_sd)

        if plt_areas_separately is True:
            _list_areas = list(self.ephys.plt.colors.keys())

            # create struct to store region-specific firing rates
            self.ephys.plt.fr = {}
            self.ephys.plt.fr_neur_counter = {}
            for _area in _list_areas:
                self.ephys.plt.fr[_area] = np.zeros_like(
                    self.ephys.spk.train_kernconv.t,
                    dtype='single')
                self.ephys.plt.fr_neur_counter[_area] = 0

            # for clust_id in range(max_n_clusts):
            for ind_clust, raw_clust_id in enumerate(self.ephys.spk.raw_clust_id):
                try:
                    # _key_id = find_key_partialmatch(
                    #     self.ephys.spk_raw.clust_info, 'id')
                    # _clust_info = self.ephys.spk_raw.clust_info[
                    #     self.ephys.spk_raw.clust_info[_key_id] == raw_clust_id]
                    # _ch = np.array(_clust_info['ch'])[0]

                    # _region_id = self.ephys.ch.region[_ch]
                    # _region_key = match_region_partial(
                    #     _region_id,
                    #     list(self.ephys.plt.colors.keys()))

                    _region_key = self.ephys.spk.info.region[ind_clust]
                    self.ephys.plt.fr[_region_key] \
                        += self.ephys.spk.train_kernconv.fr[kern_sd_key] \
                        [ind_clust].astype(float)
                    self.ephys.plt.fr_neur_counter[_region_key] += 1
                except IndexError:
                    pass

            # normalize each region firing rate by number of neurons and plot
            for _area in _list_areas:
                if region is None:
                    if self.ephys.plt.fr_neur_counter[_area] != 0:
                        self.ephys.plt.fr[_area] \
                            /= self.ephys.plt.fr_neur_counter[_area]
                    elif self.ephys.plt.fr_neur_counter[_area] == 0:
                        pass

                    self.plt.ax_fr.plot(self.ephys.spk.train_kernconv.t,
                                        self.ephys.plt.fr[_area],
                                        color=self.ephys.plt.colors[_area],
                                        linewidth=0.6)
                else:
                    if _area in region:
                        self.ephys.plt.fr[_area] \
                            /= self.ephys.plt.fr_neur_counter[_area]
                        self.plt.ax_fr.plot(
                            self.ephys.spk.train_kernconv.t,
                            self.ephys.plt.fr[_area],
                            color=self.ephys.plt.colors[_area],
                            linewidth=0.6)

        self.plt.ax_fr.set_ylabel('fr (Hz)')
        self.plt.ax_raster.set_ylabel('dv (um)')

        if t_range is not None:
            self.plt.ax_raster.set_xlim(t_range[0], t_range[1])

        # # population synchrony plot
        # # ------------
        # print('\tpopulation synchrony')
        # self.ephys.calc_synchrony_index(
        #     bin_size_count_spks=psi_spkbin,
        #     bin_size_synch_index=psi_indbin,
        #     region=region)

        # for _area in _list_areas:
        #     if region is None:
        #         self.plt.ax_psi.plot(self.ephys.psi.t,
        #                              self.ephys.psi.index,
        #                              color='k',
        #                              linewidth=0.6)
        #     else:
        #         if _area in region:
        #             self.plt.ax_psi.plot(self.ephys.psi.t,
        #                                  self.ephys.psi.index,
        #                                  color=self.ephys.plt.colors[_area],
        #                                  linewidth=0.6)
        # self.plt.ax_psi.set_ylabel('synch. index')
        # self.plt.ax_psi.set_xlabel('time (s)')

        # save and show
        # --------------
        _enclosing_fold_splt = self.folder.enclosing.split('/')
        _mouse = _enclosing_fold_splt[3]
        _date = _enclosing_fold_splt[4]

        self.plt.fig.suptitle(f'{_mouse} {_date}', fontsize=10)

        if fig_save is True:
            print('saving fig...')
            fname = f'exp_summary_{_mouse}_{_date}.jpg'
            self.plt.fig.savefig(fname, dpi=800)
            if fig_save_duplicate_folder is not None:
                print('saving fig in duplicate path...')
                dup_path_fname = os.path.join(fig_save_duplicate_folder,
                                              fname)
                self.plt.fig.savefig(dup_path_fname, dpi=plt_dpi)

        if show_plts is True:
            plt.show()

        print('cleanup...')
        self.plt.fig.clear()
        gc.collect()

        return

    def plt_pca(self, n_trials=100, s=80,
                region=None, timeperiod='stim_to_choice',
                plt_perf_state=False):

        self.ephys.calc_pca(region=region)
        self.beh.add_metrics()

        self.plt_pca_attr = SimpleNamespace()
        self.plt_pca_attr.fig = plt.figure()
        self.plt_pca_attr.ax = self.plt_pca_attr.fig.add_subplot(
            1, 1, 1,
            projection='3d')

        if n_trials == None:
            n_trials = self.beh.choice.t.shape[0]

        for ind_trial in range(n_trials):
            if timeperiod == 'stim_to_choice':
                t_start = self.beh.stim.t_start[ind_trial]
                t_end = self.beh.choice.t[ind_trial]
            if timeperiod == 'prestim':
                t_start = self.beh.stim.t_start[ind_trial] - 1
                t_end = self.beh.stim.t_start[ind_trial]
            if timeperiod == 'poststim':
                t_start = self.beh.stim.t_start[ind_trial]
                t_end = self.beh.stim.t_start[ind_trial] + 0.2

            _ind_t_start = np.argmin(np.abs(self.ephys.spk.t-t_start))
            _ind_t_end = np.argmin(np.abs(self.ephys.spk.t-t_end))

            _choice = self.beh.choice.correct[ind_trial]
            _perf = self.beh.metrics.perf.fast[ind_trial]

            # plot the trajectories
            if plt_perf_state is False:
                if _choice > 0.5:
                    self.plt_pca_attr.ax.plot(
                        self.ephys.pointproc_pca[_ind_t_start:_ind_t_end, 0],
                        self.ephys.pointproc_pca[_ind_t_start:_ind_t_end, 1],
                        self.ephys.pointproc_pca[_ind_t_start:_ind_t_end, 2],
                        alpha=calc_alpha(_perf, 1.2, alpha_min=0.1),
                        color=sns.xkcd_rgb['cerulean'])
                    self.plt_pca_attr.ax.scatter(
                        self.ephys.pointproc_pca[_ind_t_start, 0],
                        self.ephys.pointproc_pca[_ind_t_start, 1],
                        self.ephys.pointproc_pca[_ind_t_start, 2],
                        s=s, color=sns.xkcd_rgb['cobalt'])
                    self.plt_pca_attr.ax.scatter(
                        self.ephys.pointproc_pca[_ind_t_end, 0],
                        self.ephys.pointproc_pca[_ind_t_end, 1],
                        self.ephys.pointproc_pca[_ind_t_end, 2],
                        s=s, color=sns.xkcd_rgb['bright blue'])
                elif _choice < 0.5:
                    self.plt_pca_attr.ax.plot(
                        self.ephys.pointproc_pca[_ind_t_start:_ind_t_end, 0],
                        self.ephys.pointproc_pca[_ind_t_start:_ind_t_end, 1],
                        self.ephys.pointproc_pca[_ind_t_start:_ind_t_end, 2],
                        alpha=calc_alpha(_perf, 1.2, alpha_min=0.1),
                        color=sns.xkcd_rgb['light red'])
                    self.plt_pca_attr.ax.scatter(
                        self.ephys.pointproc_pca[_ind_t_start, 0],
                        self.ephys.pointproc_pca[_ind_t_start, 1],
                        self.ephys.pointproc_pca[_ind_t_start, 2],
                        s=s, color=sns.xkcd_rgb['wine'])
                    self.plt_pca_attr.ax.scatter(
                        self.ephys.pointproc_pca[_ind_t_end, 0],
                        self.ephys.pointproc_pca[_ind_t_end, 1],
                        self.ephys.pointproc_pca[_ind_t_end, 2],
                        s=s, color=sns.xkcd_rgb['deep pink'])
            if plt_perf_state == 'low':
                if _perf < 0.5:
                    if _choice > 0.5:
                        self.plt_pca_attr.ax.plot(
                            self.ephys.pointproc_pca[_ind_t_start:_ind_t_end, 0],
                            self.ephys.pointproc_pca[_ind_t_start:_ind_t_end, 1],
                            self.ephys.pointproc_pca[_ind_t_start:_ind_t_end, 2],
                            alpha=calc_alpha(_perf, 1.2, alpha_min=0.1),
                            color=sns.xkcd_rgb['cerulean'])
                        self.plt_pca_attr.ax.scatter(
                            self.ephys.pointproc_pca[_ind_t_start, 0],
                            self.ephys.pointproc_pca[_ind_t_start, 1],
                            self.ephys.pointproc_pca[_ind_t_start, 2],
                            s=s, color=sns.xkcd_rgb['cobalt'])
                        self.plt_pca_attr.ax.scatter(
                            self.ephys.pointproc_pca[_ind_t_end, 0],
                            self.ephys.pointproc_pca[_ind_t_end, 1],
                            self.ephys.pointproc_pca[_ind_t_end, 2],
                            s=s, color=sns.xkcd_rgb['bright blue'])
                    elif _choice < 0.5:
                        self.plt_pca_attr.ax.plot(
                            self.ephys.pointproc_pca[_ind_t_start:_ind_t_end, 0],
                            self.ephys.pointproc_pca[_ind_t_start:_ind_t_end, 1],
                            self.ephys.pointproc_pca[_ind_t_start:_ind_t_end, 2],
                            alpha=calc_alpha(_perf, 1.2, alpha_min=0.1),
                            color=sns.xkcd_rgb['light red'])
                        self.plt_pca_attr.ax.scatter(
                            self.ephys.pointproc_pca[_ind_t_start, 0],
                            self.ephys.pointproc_pca[_ind_t_start, 1],
                            self.ephys.pointproc_pca[_ind_t_start, 2],
                            s=s, color=sns.xkcd_rgb['wine'])
                        self.plt_pca_attr.ax.scatter(
                            self.ephys.pointproc_pca[_ind_t_end, 0],
                            self.ephys.pointproc_pca[_ind_t_end, 1],
                            self.ephys.pointproc_pca[_ind_t_end, 2],
                            s=s, color=sns.xkcd_rgb['deep pink'])
            if plt_perf_state == 'high':
                if _perf > 0.5:
                    if _choice > 0.5:
                        self.plt_pca_attr.ax.plot(
                            self.ephys.pointproc_pca[_ind_t_start:_ind_t_end, 0],
                            self.ephys.pointproc_pca[_ind_t_start:_ind_t_end, 1],
                            self.ephys.pointproc_pca[_ind_t_start:_ind_t_end, 2],
                            alpha=calc_alpha(_perf, 1.2, alpha_min=0.1),
                            color=sns.xkcd_rgb['cerulean'])
                        self.plt_pca_attr.ax.scatter(
                            self.ephys.pointproc_pca[_ind_t_start, 0],
                            self.ephys.pointproc_pca[_ind_t_start, 1],
                            self.ephys.pointproc_pca[_ind_t_start, 2],
                            s=s, color=sns.xkcd_rgb['cobalt'])
                        self.plt_pca_attr.ax.scatter(
                            self.ephys.pointproc_pca[_ind_t_end, 0],
                            self.ephys.pointproc_pca[_ind_t_end, 1],
                            self.ephys.pointproc_pca[_ind_t_end, 2],
                            s=s, color=sns.xkcd_rgb['bright blue'])
                    elif _choice < 0.5:
                        self.plt_pca_attr.ax.plot(
                            self.ephys.pointproc_pca[
                                _ind_t_start:_ind_t_end, 0],
                            self.ephys.pointproc_pca[
                                _ind_t_start:_ind_t_end, 1],
                            self.ephys.pointproc_pca[
                                _ind_t_start:_ind_t_end, 2],
                            alpha=calc_alpha(_perf, 1.2, alpha_min=0.1),
                            color=sns.xkcd_rgb['light red'])
                        self.plt_pca_attr.ax.scatter(
                            self.ephys.pointproc_pca[_ind_t_start, 0],
                            self.ephys.pointproc_pca[_ind_t_start, 1],
                            self.ephys.pointproc_pca[_ind_t_start, 2],
                            s=s, color=sns.xkcd_rgb['wine'])
                        self.plt_pca_attr.ax.scatter(
                            self.ephys.pointproc_pca[_ind_t_end, 0],
                            self.ephys.pointproc_pca[_ind_t_end, 1],
                            self.ephys.pointproc_pca[_ind_t_end, 2],
                            s=s, color=sns.xkcd_rgb['deep pink'])

            # plot start points and choice points

        plt.show()
        self.plt_pca_attr.fig.clear()
        gc.collect()

        return

    def get_aligned_ephys(self, region=None,
                          ref='stim',
                          t_pre=1, t_post=1,
                          kern_fr_sd=50, sampling_rate=100,
                          get_zscore=True,
                          region_snake='supplemental somatosensory',
                          figsize=(3.43, 2),
                          verbose=False, fig_save=True,
                          filter_high_perf=None,
                          wheel_turn_thresh_opto_incorr=None,
                          wheel_turn_thresh_dir_opto_incorr='neg'):

        print('aligning ephys to behavior...')

        self.ephys.add_spktrain_kernconv(
            kern_sd=kern_fr_sd, samp_rate_spktrain_hz=sampling_rate)

        kern_fr_sd_key = str(kern_fr_sd)+'ms'

        # parse beh trials
        # ---------
        if ref == 'stim':
            period = 'poststim'
        elif ref == 'choice':
            period = 'postchoice'

        ind_tr = SimpleNamespace()
        ind_tr.opto = SimpleNamespace()
        ind_tr.no_opto = SimpleNamespace()

        ind_tr.opto.corr = self.beh.get_subset_trials(
            trial_type_opto=True,
            correct=True, period=period,
            filter_high_perf=filter_high_perf)
        ind_tr.opto.incorr = self.beh.get_subset_trials(
            trial_type_opto=True,
            correct=False, period=period,
            filter_high_perf=filter_high_perf,
            wheel_turn_thresh=wheel_turn_thresh_opto_incorr,
            wheel_turn_thresh_dir=wheel_turn_thresh_dir_opto_incorr)
        ind_tr.no_opto.corr = self.beh.get_subset_trials(
            trial_type_opto=False,
            correct=True, period=period,
            filter_high_perf=filter_high_perf)
        ind_tr.no_opto.incorr = self.beh.get_subset_trials(
            trial_type_opto=False,
            correct=False, period=period,
            filter_high_perf=filter_high_perf)

        _templ_neur_data = self.ephys.get_subset_spktrain(
            t_start=ind_tr.opto.corr.t.start[1]-t_pre,
            t_end=ind_tr.opto.corr.t.start[1]+t_post,
            get_kernconv=True, get_zscore=get_zscore,
            kernconv_sd_key=kern_fr_sd_key,
            region=None)

        n_timepoints = _templ_neur_data.train.shape[1]

        # parse ephys
        # ---------------
        self.ephys_trsort = SimpleNamespace()
        self.ephys_trsort.opto = SimpleNamespace(
            corr={}, incorr={})
        self.ephys_trsort.no_opto = SimpleNamespace(
            corr={}, incorr={})

        self.ephys_trsort.opto.t = _templ_neur_data.t \
            - ind_tr.opto.corr.t.start[1]
        self.ephys_trsort.no_opto.t = self.ephys_trsort.opto.t

        _list_areas = list(self.ephys.plt.colors.keys())
        for _area in _list_areas:

            # decide if including this area
            if region is None:
                _correct_region = True
            else:
                if _area in region:
                    _correct_region = True
                else:
                    _correct_region = False

            # if correct region, store attributes
            if _correct_region is True:
                print(f'\textracting area {_area}...')
                _n_neurs = np.where(np.array(
                    self.ephys.spk.info.region) == _area)[0].shape[0]

                for trial_cond in ['opto', 'no_opto']:
                    for choice_cond in ['corr', 'incorr']:
                        _ephys = getattr(getattr(
                            self.ephys_trsort, trial_cond), choice_cond)
                        _ind_tr = getattr(getattr(
                            ind_tr, trial_cond), choice_cond)

                        if _ind_tr.inds.shape[0] > 0:
                            _ephys[_area] = np.zeros(
                                (len(_ind_tr.inds), _n_neurs, n_timepoints))

                            # go through hit and miss trials
                            for _trial in range(len(_ind_tr.inds)):
                                if verbose is True:
                                    print(f'\t\t{trial_cond} {choice_cond}'
                                          + f' trials... {_trial}')

                                _ephys[_area][_trial, :, :] \
                                    = self.ephys.get_subset_spktrain(
                                        t_start=_ind_tr.t.start[_trial]-t_pre,
                                        t_end=_ind_tr.t.start[_trial]+t_post,
                                        get_kernconv=True,
                                        get_zscore=get_zscore,
                                        kernconv_sd_key=kern_fr_sd_key,
                                        region=_area).train

                        elif _ind_tr.inds.shape[0] == 0:
                            _ephys[_area] = None

        return self.ephys_trsort

    def plt_aligned_ephys(self, region=None,
                          ref='stim',
                          t_pre=1, t_post=1,
                          get_zscore=True,
                          figsize=(3.43, 2), kern_fr_sd=25,
                          region_snake='supplemental somatosensory',
                          sampling_rate=100,
                          verbose=False,
                          title_fontsize=8,
                          fig_save=True):

        # process ephys
        # ----------------
        self.get_aligned_ephys(region=region,
                               ref=ref,
                               t_pre=t_pre, t_post=t_post,
                               figsize=figsize, kern_fr_sd=kern_fr_sd,
                               region_snake=region_snake,
                               sampling_rate=sampling_rate,
                               verbose=verbose,
                               get_zscore=get_zscore,
                               fig_save=fig_save)

        # plot
        # ---------------
        fig = plt.figure(figsize=figsize)
        spec = gs.GridSpec(1, 4, figure=fig)
        axs = SimpleNamespace(opto=SimpleNamespace(),
                              no_opto=SimpleNamespace())

        axs.opto.corr = fig.add_subplot(spec[0, 0])
        axs.opto.incorr = fig.add_subplot(spec[0, 1], sharey=axs.opto.corr)
        axs.no_opto.corr = fig.add_subplot(spec[0, 2], sharey=axs.opto.corr)
        axs.no_opto.incorr = fig.add_subplot(spec[0, 3], sharey=axs.opto.corr)

        for reg in region:
            for trial_cond in ['opto', 'no_opto']:
                for choice_cond in ['corr', 'incorr']:
                    _ax = getattr(getattr(axs, trial_cond), choice_cond)
                    _ephys = getattr(getattr(
                        self.ephys_trsort, trial_cond), choice_cond)[reg]
                    _t = getattr(self.ephys_trsort, trial_cond).t

                    if _ephys is not None:
                        _ax.plot(
                            _t, np.mean(np.mean(_ephys, axis=0), axis=0),
                            color=self.ephys.plt.colors[reg])
                        _ax.fill_between(
                            _t,
                            np.mean(np.mean(_ephys, axis=0), axis=0)
                            + sp_stats.sem(np.mean(_ephys, axis=0)),
                            np.mean(np.mean(_ephys, axis=0), axis=0)
                            - sp_stats.sem(np.mean(_ephys, axis=0)),
                            facecolor=self.ephys.plt.colors[reg],
                            alpha=0.2)

                    # configure each plot
                    if get_zscore is False:
                        _ax.set_ylabel('pop. fr (Hz)')
                    elif get_zscore is True:
                        _ax.set_ylabel('Z-score activity')

                    if ref == 'stim':
                        _ax.axvline(0, color='k')
                        _ax.set_xlabel('time from stim (s)')
                    if ref == 'choice':
                        _ax.axvline(0, color=sns.xkcd_rgb['grass green'])
                        _ax.set_xlabel('time from choice (s)')

                    _ax.set_title(trial_cond + ' ' + choice_cond,
                                  fontsize=title_fontsize)

        figtitle = f'pop_fr_summary_{region=}_{kern_fr_sd=}_{get_zscore=}.pdf'
        fig.savefig(figtitle)
        plt.show()

    def plt_summary_fr(self, region=None,
                       ref='stim',
                       t_pre=1, t_post=1,
                       figsize=(3.43, 2), kern_fr_sd=25,
                       region_snake='supplemental somatosensory',
                       sampling_rate=100,
                       verbose=False,
                       fig_save=True):

        self.ephys.add_spktrain(sampling_rate)

        # parse beh trials
        # ---------
        if ref == 'stim':
            _beh_miss = self.beh.get_subset_trials(hit=False,
                                                   period='poststim')
            _beh_hit = self.beh.get_subset_trials(hit=True,
                                                  period='poststim')
        elif ref == 'choice':
            _beh_miss = self.beh.get_subset_trials(hit=False,
                                                   period='postchoice')
            _beh_hit = self.beh.get_subset_trials(hit=True,
                                                  period='postchoice')

        _templ_neur_data = self.ephys.get_subset_spktrain(
            t_start=_beh_hit.t.start[1]-t_pre,
            t_end=_beh_hit.t.start[1]+t_post,
            region=None)

        n_timepoints = _templ_neur_data.train.shape[1]
        n_trials_miss = len(_beh_miss.inds)
        n_trials_hit = len(_beh_hit.inds)

        # parse ephys
        # ---------------
        self.ephys_trsort = SimpleNamespace()

        self.ephys_trsort.miss_mean = {}
        self.ephys_trsort.hit_mean = {}

        self.ephys_trsort.miss_all = {}
        self.ephys_trsort.hit_all = {}

        self.ephys_trsort.t = _templ_neur_data.t \
            - _beh_hit.t.start[1]

        _list_areas = list(self.ephys.plt.colors.keys())
        for _area in _list_areas:
            # decide if plotting this area
            if region is None:
                _plt_this_area = True
            else:
                if _area in region:
                    _plt_this_area = True
                else:
                    _plt_this_area = False

            # if plotting, store attributes
            if _plt_this_area is True:
                print(f'\tplotting area {_area}...')
                self.ephys_trsort.miss_mean[_area] = np.zeros(
                    (n_trials_miss, n_timepoints))
                self.ephys_trsort.hit_mean[_area] = np.zeros(
                    (n_trials_hit, n_timepoints))

                self.ephys_trsort.miss_all[_area] = np.empty(
                    n_trials_miss, dtype=np.ndarray)
                self.ephys_trsort.hit_all[_area] = np.empty(
                    n_trials_hit, dtype=np.ndarray)

                # go through hit and miss trials
                for _trial in range(n_trials_hit):
                    if verbose is True:
                        print(f'\t\thit trials... {_trial}')

                    _hit = self.ephys.get_subset_spktrain(
                        t_start=_beh_hit.t.start[_trial]-t_pre,
                        t_end=_beh_hit.t.start[_trial]+t_post,
                        region=_area)
                    _hit_mean = smooth_spktrain(
                        np.mean(_hit.train, axis=0),
                        _hit.t,
                        kern_fr_sd)

                    # adjust shape to fit
                    if _hit_mean.shape[0] > n_timepoints:
                        _hit_mean = _hit_mean[0:n_timepoints]
                    elif _hit_mean.shape[0] < n_timepoints:
                        _vals_to_append = np.zeros(
                            int(n_timepoints-_hit_mean.shape[0]))
                        _hit_mean = np.append(_hit_mean, _vals_to_append)

                    self.ephys_trsort.hit_mean[_area][_trial, :] = _hit_mean

                    self.ephys_trsort.hit_all[_area][_trial] \
                        = np.zeros_like(_hit.train)
                    for neur in range(_hit.train.shape[0]):
                        self.ephys_trsort.hit_all[_area][_trial][neur, :] \
                            = smooth_spktrain(
                                _hit.train[neur, :], _hit.t, kern_fr_sd)

                for _trial in range(n_trials_miss):
                    if verbose is True:
                        print(f'\t\tmiss trials... {_trial}')

                    _miss = self.ephys.get_subset_spktrain(
                        t_start=_beh_miss.t.start[_trial]-t_pre,
                        t_end=_beh_miss.t.start[_trial]+t_post,
                        region=_area)
                    _miss_mean = smooth_spktrain(
                        np.mean(_miss.train, axis=0),
                        _miss.t,
                        kern_fr_sd)

                    # adjust shape to fit
                    if _miss_mean.shape[0] > n_timepoints:
                        _miss_mean = _miss_mean[0:n_timepoints]
                    elif _miss_mean.shape[0] < n_timepoints:
                        _vals_to_append = np.zeros(
                            int(n_timepoints-_miss_mean.shape[0]))
                        _miss_mean = np.append(_miss_mean, _vals_to_append)

                    self.ephys_trsort.miss_mean[_area][_trial, :] = _miss_mean

                    self.ephys_trsort.miss_all[_area][_trial] \
                        = np.zeros_like(_miss.train)
                    for neur in range(_miss.train.shape[0]):
                        self.ephys_trsort.miss_all[_area][_trial][neur, :] \
                            = smooth_spktrain(
                                _miss.train[neur, :], _miss.t, kern_fr_sd)

        # analyze snake data
        # --------------
        _mean_act_hit = np.zeros_like(
            self.ephys_trsort.hit_all[region_snake][0])
        _n_tr_hit = self.ephys_trsort.hit_all[region_snake].shape[0]
        for tr in range(_n_tr_hit):
            _mean_act_hit += self.ephys_trsort.hit_all[region_snake][tr]
        _mean_act_hit /= _n_tr_hit

        _mean_act_miss = np.zeros_like(
            self.ephys_trsort.miss_all[region_snake][0])
        _n_tr_miss = self.ephys_trsort.miss_all[region_snake].shape[0]
        for tr in range(_n_tr_miss):
            _mean_act_miss += self.ephys_trsort.miss_all[region_snake][tr]
        _mean_act_miss /= _n_tr_miss

        sort_inds_hit = np.argsort(np.argmax(_mean_act_hit, axis=1))
        _mean_act_hit_sorted = np.array(_mean_act_hit)[sort_inds_hit]
        _mean_act_miss_sorted = np.array(_mean_act_miss)[sort_inds_hit]

        # plot main figure
        # -----------------
        fig = plt.figure(figsize=figsize)
        spec = gs.GridSpec(nrows=1, ncols=2, figure=fig)

        ax_miss = fig.add_subplot(spec[0, 0])
        ax_hit = fig.add_subplot(spec[0, 1], sharey=ax_miss,
                                 sharex=ax_miss)


        # main figure
        for _area in _list_areas:
            # decide if plotting this area
            if region is None:
                _plt_this_area = True
            else:
                if _area in region:
                    _plt_this_area = True
                else:
                    _plt_this_area = False

            # plot hit and miss trials for area
            if _plt_this_area is True:
                _hit_mean_travg = np.mean(
                    self.ephys_trsort.hit_mean[_area], axis=0)
                _hit_mean_trstd = np.std(
                    self.ephys_trsort.hit_mean[_area], axis=0)

                ax_hit.plot(self.ephys_trsort.t, _hit_mean_travg,
                            color=self.ephys.plt.colors[_area])
                ax_hit.fill_between(self.ephys_trsort.t,
                                    _hit_mean_travg+_hit_mean_trstd,
                                    _hit_mean_travg-_hit_mean_trstd,
                                    facecolor=self.ephys.plt.colors[_area],
                                    alpha=0.2)

                _miss_mean_travg = np.mean(
                    self.ephys_trsort.miss_mean[_area], axis=0)
                _miss_mean_trstd = np.std(
                    self.ephys_trsort.miss_mean[_area], axis=0)

                ax_miss.plot(self.ephys_trsort.t, _miss_mean_travg,
                             color=self.ephys.plt.colors[_area])
                ax_miss.fill_between(self.ephys_trsort.t,
                                     _miss_mean_travg+_miss_mean_trstd,
                                     _miss_mean_travg-_miss_mean_trstd,
                                     facecolor=self.ephys.plt.colors[_area],
                                     alpha=0.2)

        for _ax in [ax_miss, ax_hit]:
            _ax.set_ylabel('pop. fr (Hz)')
            if ref == 'stim':
                _ax.axvline(0, color='k')
                _ax.set_xlabel('time from stim (s)')
            if ref == 'choice':
                _ax.axvline(0, color=sns.xkcd_rgb['grass green'])
                _ax.set_xlabel('time from choice (s)')

        ax_miss.set_title('miss trials')
        ax_hit.set_title('hit trials')

        # supp figure
        # --------------
        fig_snake = plt.figure(figsize=figsize)
        spec_snake = gs.GridSpec(nrows=1, ncols=2, figure=fig_snake)

        ax_snake_miss = fig_snake.add_subplot(spec_snake[0, 0])
        ax_snake_hit = fig_snake.add_subplot(spec_snake[0, 1],
                                             sharey=ax_snake_miss,
                                             sharex=ax_snake_miss)

        ax_snake_hit.imshow(_mean_act_hit_sorted)
        ax_snake_miss.imshow(_mean_act_miss_sorted)

        _ind_zero = np.argmin(np.abs(self.ephys_trsort.t))

        for _ax in [ax_snake_hit, ax_snake_miss]:
            _ax.axvline(_ind_zero)
            if ref == 'stim':
                _ax.set_xlabel('ind from stim')
            if ref == 'choice':
                _ax.set_xlabel('ind from choice')

        ax_snake_miss.set_title('miss trials (sort by hit)')
        ax_snake_hit.set_title('hit trials')

        # save and cleanup
        if fig_save is True:
            fig.savefig('fig_popfr.pdf')
            fig_snake.savefig(f'fig_snake_popfr_{region_snake=}.pdf')

        plt.show()
        return

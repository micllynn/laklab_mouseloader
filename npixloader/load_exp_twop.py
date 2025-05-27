from types import SimpleNamespace
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import os
import pathlib
import tifffile

from .align_imgbeh import Aligner_ImgBeh
from .utils import find_event_onsets_autothresh, remove_lick_artefact_after_rew, \
    check_folder_exists
from .utils_twop import XMLParser, calc_dff
from .beh import BehDataSimpleLoad, StimParser


class TwoPRec_New(object):
    def __init__(self,
                 enclosing_folder=None,
                 folder_beh=None,
                 folder_img=None,
                 fname_img=None,
                 dset_obj=None,
                 dset_ind=None,
                 ch_img=2,
                 trial_end=None,
                 rec_type='trig_rew',
                 n_px_remove_sides=10):
        """
        Loads a 2p recording (tiff) and associated behavioral folder

        fname : str
            Name of the 2p recording .tiff
        folder_beh : str
            Name of the behavior folder
        folder_img : str
        Name of imaging folder (typical format of 'TwoP/xxxx_t-001')
        dset_obj : None or DSetObj class instance
            Not initialized yet as of 25.3, but will allow passing of a dataset
            object
        rec_type : str | 'trig_rew' or 'paqio'
            Recording type. Can either be 'trig_rew' (Michael-style),
            where imaging is triggered to start upon the first reward
            delivered during the task, or 'paqio' (Marko/Sandra/Jess style)
            where imaging and behavior acquisitions are manually started,
            and synchronized with a simultaneously recorded .paq file
            that has frame times (imaging start) and reward echoes (beh start)
        n_px_remove_sides : int
            Number of pixels to remove on each side of the frame (necessary if
            dealing with suite2p motion corrected tiffs, as these can
            introduce artifacts with high values on the edges
        """

        # setup names of folders and files
        # --------------
        self.folder = SimpleNamespace()
        self.path = SimpleNamespace()
        if dset_obj is None:
            self.folder.enclosing = enclosing_folder
            self.folder.img = folder_img
            self.folder.beh = folder_beh

        elif 'DSetObj' in str(type(dset_obj)):
            self.dset_obj = dset_obj
            self.folder.enclosing = self.dset_obj.get_path_expref(dset_ind)
            self.folder.img = self.dset_obj.get_path_img(dset_ind)
            self.folder.beh = self.dset_obj.get_path_beh(dset_ind)

        self.path.raw = pathlib.Path(self.folder.enclosing)
        self.path.animal = self.path.raw.parts[-2]
        self.path.date = self.path.raw.parts[-1]
        self.path.beh_folder = self.folder.beh
        self.ch_img = ch_img

        os.chdir(self.folder.enclosing)
        check_folder_exists('figs_mbl')  # to store all figs
        self.ops = SimpleNamespace()
        self.ops.n_px_remove_sides = n_px_remove_sides

        # get filename of image of the appropriate channel
        list_img = os.listdir(self.folder.img)
        if fname_img is None:
            for _fname in list_img:
                if f'Ch{ch_img}.tif' in _fname and 'compiled' in _fname:
                    self.fname_img = _fname
        else:
            self.fname_img = fname_img

        # load behavioral data
        # -------------
        self.beh = BehDataSimpleLoad(self.folder.beh)

        self.beh.rew = SimpleNamespace()
        self.beh.stim = SimpleNamespace()

        self.beh.rew.delivered = self.beh._data.get_event_var(
            'isRewardGivenValues')

        self.beh.rew.t = self.beh._data.get_event_var('totalRewardTimes')[
            np.where(self.beh.rew.delivered == 1)[0]]

        _daq_data = self.beh._timeline.get_daq_data()
        self.beh.licks = find_event_onsets_autothresh(
            _daq_data.sig['lickDetector'])
        self.beh.t_licks = _daq_data.t[self.beh.licks]

        self.beh.stim.t_start = self.beh._data.get_event_var(
            'stimulusOnTimes')
        self.beh.stim.stimlist = StimParser(self.beh)

        self.beh.stim.id = self.beh.stim.stimlist._all_stimtypes
        self.beh.stim.prob = self.beh.stim.stimlist._all_stimprobs
        self.beh.stim.size = self.beh.stim.stimlist._all_stimsizes

        self.beh.lick = SimpleNamespace()

        t_licksig = self.beh._daq_data.t
        lick_onset_inds = find_event_onsets_autothresh(
            self.beh._daq_data.sig['lickDetector'], n_stdevs=4)
        self.beh.lick.t_raw = t_licksig[lick_onset_inds]

        # load imaging data
        # ------------
        print('loading imaging...')
        for _file in os.listdir(self.folder.img):
            if _file.endswith('BACKUP.xml'):
                try:
                    xmlobj = XMLParser(os.path.join(self.folder.img, _file))
                    sampling_rate = xmlobj.get_framerate()
                    print(f'\tframerate is {sampling_rate:.2f}Hz')
                except:
                    print('could not parse framerate from BACKUP.xml')
                    pass
        self.samp_rate = sampling_rate

        print('\tloading tiff...')
        self.rec = tifffile.memmap(os.path.join(
            self.folder.img, self.fname_img))[
            :, n_px_remove_sides:-1*n_px_remove_sides,
                n_px_remove_sides:-1*n_px_remove_sides]

        # try to load suite2p output if available
        if ch_img == 2:
            print('\tloading suite2p neurs...')
            try:
                self.neur = SimpleNamespace()
                self.neur.f = np.load(
                    os.path.join(self.folder.img,
                                 'suite2p', 'plane0', 'F.npy'),
                    allow_pickle=True)
                self.neur.ops = np.load(
                    os.path.join(self.folder.img,
                                 'suite2p', 'plane0', 'ops.npy'),
                    allow_pickle=True)
                self.neur.stat = np.load(
                    os.path.join(self.folder.img,
                                 'suite2p', 'plane0', 'stat.npy'),
                    allow_pickle=True)
                self.neur.iscell = np.load(
                    os.path.join(self.folder.img,
                                 'suite2p', 'plane0', 'iscell.npy'),
                    allow_pickle=True)
            except Exception as e:
                print('\tcould not load suite2p output:')
                print(f'\t\t{e}')

        # align behavior and imaging data
        # ----------------------
        print('\tcreating timestamps...')

        if rec_type == 'trig_rew':
            _t_start = self.beh.rew.t[0]
            _t_end = (self.rec.shape[0]/self.samp_rate) + _t_start
            self.rec_t = np.linspace(_t_start,
                                     _t_end,
                                     num=self.rec.shape[0])
        elif rec_type == 'paqio':
            self._aligner = Aligner_ImgBeh()
            self._aligner.parse_img_rewechoes()
            self._aligner.parse_beh_rewechoes()
            self._aligner.compute_alignment()

            _t_start = 0
            _t_end = self.rec.shape[0]/self.samp_rate
            self.rec_t = np.linspace(_t_start, _t_end, num=self.rec.shape[0])
            self.rec_t = self._aligner.correct_img_data(self.rec_t)

        # store a copy of rec_t in neur attribute for convenience
        if hasattr(self, 'neur'):
            self.neur.t = self.rec_t

        # note the first and last stimulus/rew within recording bounds
        # ---------------
        # stims
        self.beh._stimrange = SimpleNamespace()
        n_stims = self.beh.stim.t_start.shape[0]

        _temp_first = 0
        _temp_last = n_stims - 1
        for ind_stim in range(n_stims):
            if self.beh.stim.t_start[ind_stim] + 4 > self.rec_t[-1]:
                _temp_last = ind_stim-1
            if self.beh.stim.t_start[ind_stim] - 2 < self.rec_t[0]:
                _temp_first = ind_stim+1

        self.beh._stimrange.first = _temp_first
        self.beh._stimrange.last = _temp_last

        # rews
        self.beh._rewrange = SimpleNamespace()
        n_rews = self.beh.rew.t.shape[0]

        _temp_first = 0
        _temp_last = n_rews - 1
        for ind_rew in range(n_rews):
            if self.beh.rew.t[ind_rew] + 4 > self.rec_t[-1]:
                _temp_last = ind_stim-1
            if self.beh.rew.t[ind_rew] - 2 < self.rec_t[0]:
                _temp_first = ind_stim+1

        self.beh._rewrange.first = _temp_first
        self.beh._rewrange.last = _temp_last

        # if trial_end is manually specified, replace these attributes
        if trial_end is not None:
            self.beh._stimrange.last = trial_end
            self.beh._rewrange.last = trial_end

        return

    def add_lickrates(self, t_prestim=2,
                      t_poststim=5):
        """
        Adds a lickrate segregated by trial-type
        """
        n_trials = self.beh.stim.t_start.shape[0]

        self.beh.lick.t = np.empty(n_trials,
                                   dtype=np.ndarray)
        self.beh.lick.antic = np.empty(n_trials,
                                       dtype=np.ndarray)
        self.beh.lick.base = np.empty(n_trials,
                                      dtype=np.ndarray)

        for trial in range(n_trials):
            # get the data
            _t_stim = self.beh.stim.t_start[trial]
            _t_rew = _t_stim + 2
            _lick_inds = np.logical_and(
                self.beh.lick.t_raw > (_t_stim - t_prestim),
                self.beh.lick.t_raw < (_t_rew + t_poststim))
            _licks = self.beh.lick.t_raw[_lick_inds]
            _licks -= _t_stim

            # correct licks
            _t_rew_rel_to_stim = _t_rew - _t_stim
            _licks = remove_lick_artefact_after_rew(
                _licks, [_t_rew_rel_to_stim+0.03,
                         _t_rew_rel_to_stim+0.06])
            self.beh.lick.t[trial] = _licks

            # compute anticipatory lickrates
            _licks_antic = np.sum(np.logical_and(
                _licks > 0, _licks < 2))
            self.beh.lick.antic[trial] = (_licks_antic
                                          / (_t_rew - _t_stim))
            _licks_base = np.sum(np.logical_and(
                _licks < 0, _licks > -4))
            self.beh.lick.base[trial] = (_licks_base
                                         / (t_prestim - 0))

    def get_antic_licks_by_trialtype(self, bl_norm=False):
        _tr_counts = {'0': 0, '0.5': 0, '1': 0}
        lickrates_hz = {'0': 0, '0.5': 0, '1': 0}
        for trial in range(self.beh._stimrange.first,
                           self.beh._stimrange.last):
            if bl_norm is True:
                _antic_licks = self.beh.lick.antic[trial] \
                    - self.beh.lick.base[trial]
            elif bl_norm is False:
                _antic_licks = self.beh.lick.antic[trial]

            if self.beh.stim.prob[trial] == 0:
                lickrates_hz['0'] += _antic_licks
                _tr_counts['0'] += 1
            if self.beh.stim.prob[trial] == 0.5:
                lickrates_hz['0.5'] += _antic_licks
                _tr_counts['0.5'] += 1
            if self.beh.stim.prob[trial] == 1:
                lickrates_hz['1'] += _antic_licks
                _tr_counts['1'] += 1

        lickrates_hz['0'] /= _tr_counts['0']
        lickrates_hz['0.5'] /= _tr_counts['0.5']
        lickrates_hz['1'] /= _tr_counts['1']

        return lickrates_hz

    def _plt_rew_aligned_spatial_sectors(self, n_sectors,
                                         figsize=(3.43, 2),
                                         dpi=300,
                                         scaling_factor_trace=2,
                                         scaling_factor_img=10,
                                         t_rew_pre=1, t_rew_post=3,
                                         img_ds_factor=50,
                                         ind_lastrew=None):
        print('plotting spatial sectors...')
        self.neur = SimpleNamespace()
        self.neur.params = SimpleNamespace()

        self.neur.params.n_sectors = n_sectors
        self.neur.params.t_rew_pre = t_rew_pre
        self.neur.params.t_rew_post = t_rew_post

        fig = plt.figure(figsize=figsize, dpi=dpi)
        spec = gs.GridSpec(nrows=2, ncols=2,
                           height_ratios=[0.8, 0.2],
                           figure=fig)
        ax_img = fig.add_subplot(spec[0, 0])
        ax_traces = fig.add_subplot(spec[0, 1])
        ax_rew = fig.add_subplot(spec[1, 1], sharex=ax_traces)

        _rec_max = np.max(self.rec[::img_ds_factor, :, :], axis=0)
        _rec_max[:, ::int(_rec_max.shape[0]/n_sectors)] = 0
        _rec_max[::int(_rec_max.shape[0]/n_sectors), :] = 0

        ax_img.imshow(_rec_max,
                      extent=[0, n_sectors,
                              n_sectors, 0])

        # setup reward-aligned traces
        self.neur.dff_rewaligned = np.empty(n_sectors*n_sectors,
                                            dtype=np.ndarray)
        n_frames_pre = int(t_rew_pre * self.samp_rate)
        n_frames_post = int(t_rew_post * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        self.neur.t = np.linspace(
            -1*t_rew_pre, t_rew_post, n_frames_tot)

        _n_trace = 0
        for n_x in range(n_sectors):
            for n_y in range(n_sectors):
                print(f'\tsector {_n_trace}/{n_sectors**2}...', end='\r')
                # calculate location of sector
                _ind_x_lower = int((n_x / n_sectors) * self.rec.shape[1])
                _ind_x_upper = int(((n_x+1) / n_sectors) * self.rec.shape[1])

                _ind_y_lower = int((n_y / n_sectors) * self.rec.shape[1])
                _ind_y_upper = int(((n_y+1) / n_sectors) * self.rec.shape[1])

                # plot trace
                _trace = np.mean(np.mean(
                    self.rec[:,
                             _ind_x_lower:_ind_x_upper,
                             _ind_y_lower:_ind_y_upper], axis=1), axis=1)

                ax_traces.plot(self.rec_t,
                               sp.stats.zscore(_trace)*scaling_factor_trace
                               + _n_trace,
                               color=sns.xkcd_rgb['ocean green'],
                               linewidth=0.5, alpha=0.8)

                # plot reward-aligned trace
                self.neur.dff_rewaligned[_n_trace] = np.zeros((
                    self.beh.rew.t[self.beh._rewrange.first:
                                   self.beh._rewrange.last].shape[0],
                    n_frames_tot))

                for ind, t_rew in enumerate(
                        self.beh.rew.t[self.beh._rewrange.first:
                                       self.beh._rewrange.last]):
                    ind_rew = np.argmin(np.abs(self.rec_t-t_rew))
                    _rew_trace = _trace[ind_rew-n_frames_pre:
                                        ind_rew+n_frames_post]
                    self.neur.dff_rewaligned[_n_trace][ind, :] = calc_dff(
                        _rew_trace, baseline_frames=n_frames_pre)

                _rec_dt = np.diff(self.rec_t[0:n_frames_tot])[0]
                _rec_t_templ = np.arange(0, (n_frames_tot+5)*_rec_dt, _rec_dt)

                t_rewaligned_norm = ((_rec_t_templ[0:n_frames_tot]
                                      / _rec_t_templ[n_frames_tot])
                                     * 0.6)
                t_rewaligned_norm = t_rewaligned_norm + n_x + 0.2

                # note here that we must invert dff_rewaligned_mean
                # because plt() traces are  plotted in the 'negative' direction
                # on top of imshow() images for some reason
                dff_rewaligned_mean = np.mean(
                    self.neur.dff_rewaligned[_n_trace], axis=0)
                dff_rewaligned_mean_shifted = (-1 * dff_rewaligned_mean
                                               * scaling_factor_img
                                               + n_y + 0.5)

                t_rewonset = ((_rec_t_templ[n_frames_pre]
                               / _rec_t_templ[n_frames_tot]) * 0.6) \
                               + n_x + 0.2
                ax_img.plot(t_rewaligned_norm,
                            dff_rewaligned_mean_shifted,
                            color=sns.xkcd_rgb['orangered'],
                            linewidth=0.5,
                            alpha=0.8)
                ax_img.plot([t_rewonset, t_rewonset], [n_y+0.2, n_y+0.8],
                            color=sns.xkcd_rgb['white'], linestyle='dashed',
                            linewidth=0.3)

                _n_trace += 1
                self._last_trace = _trace

        for ind, t_rew in enumerate(self.beh.rew.t):
            ax_rew.plot([t_rew, t_rew], [0, 1],
                        color=sns.xkcd_rgb['bright blue'])

        if self.fname_img.endswith('Ch1.tif'):
            prefix = 'grab'
        elif self.fname_img.endswith('Ch2.tif'):
            prefix = 'gcamp'

        fig.savefig(os.path.join(
            self.folder.enclosing,
            f'{self.path.animal}'
            + f'_{self.path.date}_{self.path.beh_folder}'
            + f'_{prefix}_sector_fig.pdf'))

        plt.show()

    def plt_stim_aligned_avg(self,
                             figsize=(3.43, 2),
                             t_pre=2, t_post=10,
                             colors=sns.cubehelix_palette(
                                 n_colors=3,
                                 start=2, rot=0,
                                 dark=0.1, light=0.6)):
        """
        Plots the average fluorescence across the whole fov,
        separated by trial-type (eg 0%, 50%, 100% rewarded trials)
        """

        print('plotting stim_aligned traces...')
        self.frame_avg = SimpleNamespace()

        self.frame_avg.params = SimpleNamespace()
        self.frame_avg.params.t_rew_pre = t_pre
        self.frame_avg.params.t_rew_post = t_post

        self.frame_avg.colors = {'0': colors[0],
                                '0.5': colors[1],
                                '1': colors[2]}

        fig_avg = plt.figure(figsize=figsize)
        spec = gs.GridSpec(nrows=1, ncols=1,
                           figure=fig_avg)
        ax_trace = fig_avg.add_subplot(spec[0, 0])

        # setup stim-aligned traces
        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int((2 + t_post) * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        self.frame_avg.tr_inds = {}
        self.frame_avg.tr_inds['0'] = np.where(
            self.beh.stim.prob[
                self.beh._stimrange.first:self.beh._stimrange.last] == 0)[0]
        self.frame_avg.tr_inds['0.5'] = np.where(
            self.beh.stim.prob[
                self.beh._stimrange.first:self.beh._stimrange.last] == 0.5)[0]
        self.frame_avg.tr_inds['1'] = np.where(
            self.beh.stim.prob[
                self.beh._stimrange.first:self.beh._stimrange.last] == 1)[0]

        self.frame_avg.dff = {}
        self.frame_avg.dff['0'] = np.zeros((
            self.frame_avg.tr_inds['0'].shape[0], n_frames_tot))
        self.frame_avg.dff['0.5'] = np.zeros((
            self.frame_avg.tr_inds['0.5'].shape[0], n_frames_tot))
        self.frame_avg.dff['1'] = np.zeros((
            self.frame_avg.tr_inds['1'].shape[0], n_frames_tot))

        self.frame_avg.t = np.linspace(
            -1*t_pre, 2+t_post, n_frames_tot)

        # store stim-aligned trace
        self.frame_avg.tr_counts = {'0': 0, '0.5': 0, '1': 0}
        for trial in range(self.beh._stimrange.first,
                           self.beh._stimrange.last):
            print(f'\t{trial=}', end='\r')
            _t_stim = self.beh.stim.t_start[trial]
            _t_start = _t_stim - t_pre
            _t_end = self.beh._data.get_event_var(
                'totalRewardTimes')[trial] + t_post

            _ind_t_start = np.argmin(np.abs(
                self.rec_t - _t_start))
            _ind_t_end = np.argmin(np.abs(
                self.rec_t - _t_end)) + 1   # add frames to end

            # extract fluorescence
            _f = np.mean(np.mean(
                self.rec[_ind_t_start:_ind_t_end, :, :], axis=1), axis=1)
            _dff = calc_dff(_f, baseline_frames=n_frames_pre)

            if self.beh.stim.prob[trial] == 0:
                self.frame_avg.dff['0'][self.frame_avg.tr_counts['0'], :] \
                    = _dff[0:n_frames_tot]
                self.frame_avg.tr_counts['0'] += 1
            if self.beh.stim.prob[trial] == 0.5:
                self.frame_avg.dff['0.5'][self.frame_avg.tr_counts['0.5'], :] \
                    = _dff[0:n_frames_tot]
                self.frame_avg.tr_counts['0.5'] += 1
            if self.beh.stim.prob[trial] == 1:
                self.frame_avg.dff['1'][self.frame_avg.tr_counts['1'], :] \
                    = _dff[0:n_frames_tot]
                self.frame_avg.tr_counts['1'] += 1

        # plot stim-aligned traces
        for stim_cond in ['0', '0.5', '1']:
            ax_trace.plot(self.frame_avg.t,
                          np.mean(self.frame_avg.dff[stim_cond], axis=0),
                          color=self.frame_avg.colors[stim_cond])
            ax_trace.fill_between(
                self.frame_avg.t,
                np.mean(self.frame_avg.dff[stim_cond], axis=0) -
                np.std(self.frame_avg.dff[stim_cond], axis=0),
                np.mean(self.frame_avg.dff[stim_cond], axis=0) +
                np.std(self.frame_avg.dff[stim_cond], axis=0),
                facecolor=self.frame_avg.colors[stim_cond],
                alpha=0.2)

        # plot rew and stim traces
        ax_trace.axvline(x=0, color=sns.xkcd_rgb['dark grey'],
                         linewidth=1.5, alpha=0.8)
        ax_trace.axvline(x=2, color=sns.xkcd_rgb['bright blue'],
                         linewidth=1.5, alpha=0.8)

        ax_trace.set_xlabel('time (s)')
        ax_trace.set_ylabel('df/f')

        fig_avg.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'_ch={self.ch_img}_mean_trial_activity.pdf'))

        plt.show()

    def plt_stim_aligned_sectors(self,
                                 n_sectors=10,
                                 t_pre=2, t_post=5,
                                 plot_special=None,
                                 figsize=(6, 6),
                                 img_ds_factor=50,
                                 img_alpha=0.5,
                                 plt_dff={'x': {'gain': 0.6,
                                                'offset': 0.2},
                                          'y': {'gain': 10,
                                                'offset': 0.2}},
                                 plt_prefix='',
                                 plt_show=True,
                                 colors=sns.cubehelix_palette(
                                     n_colors=3,
                                     start=2, rot=0,
                                     dark=0.2, light=0.8)):
        """
        Divides the field of view into sectors, and plots a set of trial-types
        separately within each sector.

        plot_special controls the type of plot:
            if plot_special is None:
                stim_list = ['0', '0.5', '1']
            elif plot_special == 'rew_norew':
                stim_list = ['0.5_rew', '0.5_norew']
            elif plot_special == 'prelick_noprelick':
                stim_list = ['0.5_prelick', '0.5_noprelick']
            elif plot_special == 'rew':
                stim_list = ['0', '0.5_rew', '1']
        """
        print('plotting stim_aligned traces...')
        self.add_lickrates(t_prestim=t_pre,
                           t_poststim=t_post)

        self.sector = SimpleNamespace()

        self.sector.params = SimpleNamespace()
        self.sector.n_sectors = n_sectors
        self.sector.params.t_rew_pre = t_pre
        self.sector.params.t_rew_post = t_post
        self.sector.n_sectors = n_sectors

        self.sector.colors = {'0': colors[0],
                                '0.5': colors[1],
                                '1': colors[2],
                                '0.5_rew': colors[1],
                                '0.5_norew': colors[1],
                                '0.5_prelick': colors[1],
                                '0.5_noprelick': colors[1]}
        self.sector.linestyle = {'0': 'solid',
                                 '0.5': 'solid',
                                 '1': 'solid',
                                 '0.5_rew': 'solid',
                                 '0.5_norew': 'dashed',
                                 '0.5_prelick': 'solid',
                                 '0.5_noprelick': 'dashed'}

        fig_avg = plt.figure(figsize=figsize)
        spec = gs.GridSpec(nrows=1, ncols=1,
                           figure=fig_avg)

        ax_img = fig_avg.add_subplot(spec[0, 0])

        print('\tcreating max projection image...')
        _rec_max = np.max(self.rec[::img_ds_factor, :, :], axis=0)
        _rec_max[:, ::int(_rec_max.shape[0]/n_sectors)] = 0
        _rec_max[::int(_rec_max.shape[0]/n_sectors), :] = 0

        ax_img.imshow(_rec_max,
                      extent=[0, n_sectors,
                              n_sectors, 0],
                      alpha=img_alpha)

        # setup stim-aligned traces
        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int((2 + t_post) * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        _stim_prob_trrange = self.beh.stim.prob[
            self.beh._stimrange.first:self.beh._stimrange.last]
        _stim_rewdeliv_trrange = self.beh.rew.delivered[
            self.beh._stimrange.first:self.beh._stimrange.last]
        _licks_trrange = self.beh.lick.antic[
            self.beh._stimrange.first:self.beh._stimrange.last]

        self.sector.tr_inds = {}
        self.sector.tr_inds['0'] = np.where(_stim_prob_trrange == 0)[0]
        self.sector.tr_inds['0.5'] = np.where(_stim_prob_trrange == 0.5)[0]
        self.sector.tr_inds['1'] = np.where(_stim_prob_trrange == 1)[0]

        self.sector.tr_inds['0.5_rew'] = np.where(np.logical_and(
            _stim_prob_trrange == 0.5, _stim_rewdeliv_trrange == True))[0]
        self.sector.tr_inds['0.5_norew'] = np.where(np.logical_and(
            _stim_prob_trrange == 0.5, _stim_rewdeliv_trrange == False))[0]
        self.sector.tr_inds['0.5_prelick'] = np.where(np.logical_and(
            _stim_prob_trrange == 0.5, _licks_trrange > 0))[0]
        self.sector.tr_inds['0.5_noprelick'] = np.where(np.logical_and(
            _stim_prob_trrange == 0.5, _licks_trrange == 0))[0]

        self.sector.dff = np.empty((n_sectors, n_sectors), dtype=dict)

        self.sector.t = np.linspace(
            -1*t_pre, 2+t_post, n_frames_tot)

        # store edges of sectors
        self.sector.x = SimpleNamespace()
        self.sector.x.lower = np.zeros((n_sectors, n_sectors))
        self.sector.x.upper = np.zeros((n_sectors, n_sectors))
        self.sector.y = SimpleNamespace()
        self.sector.y.lower = np.zeros((n_sectors, n_sectors))
        self.sector.y.upper = np.zeros((n_sectors, n_sectors))

        # ---------------------
        print('\textracting aligned fluorescence traces...')
        _n_trace = 1
        for n_x in range(n_sectors):
            for n_y in range(n_sectors):
                print(f'\t\tsector {_n_trace}/{n_sectors**2}...      ')
                # calculate location of sector
                _ind_x_lower = int((n_x / n_sectors) * self.rec.shape[1])
                _ind_x_upper = int(((n_x+1) / n_sectors) * self.rec.shape[1])

                _ind_y_lower = int((n_y / n_sectors) * self.rec.shape[1])
                _ind_y_upper = int(((n_y+1) / n_sectors) * self.rec.shape[1])

                self.sector.x.lower[n_x, n_y] = _ind_x_lower
                self.sector.x.upper[n_x, n_y] = _ind_x_upper
                self.sector.y.lower[n_x, n_y] = _ind_y_lower
                self.sector.y.upper[n_x, n_y] = _ind_y_upper

                # setup structure
                self.sector.dff[n_x, n_y] = {}
                self.sector.dff[n_x, n_y]['0'] = np.zeros((
                    self.sector.tr_inds['0'].shape[0], n_frames_tot))
                self.sector.dff[n_x, n_y]['0.5'] = np.zeros((
                    self.sector.tr_inds['0.5'].shape[0], n_frames_tot))
                self.sector.dff[n_x, n_y]['1'] = np.zeros((
                    self.sector.tr_inds['1'].shape[0], n_frames_tot))

                self.sector.dff[n_x, n_y]['0.5_rew'] = np.zeros((
                    self.sector.tr_inds['0.5_rew'].shape[0], n_frames_tot))
                self.sector.dff[n_x, n_y]['0.5_norew'] = np.zeros((
                    self.sector.tr_inds['0.5_norew'].shape[0], n_frames_tot))

                self.sector.dff[n_x, n_y]['0.5_prelick'] = np.zeros((
                    self.sector.tr_inds['0.5_prelick'].shape[0],
                    n_frames_tot))
                self.sector.dff[n_x, n_y]['0.5_noprelick'] = np.zeros((
                    self.sector.tr_inds['0.5_noprelick'].shape[0],
                    n_frames_tot))

                # store stim-aligned traces
                self.sector.tr_counts = {'0': 0, '0.5': 0, '1': 0,
                                         '0.5_rew': 0, '0.5_norew': 0,
                                         '0.5_prelick': 0,
                                         '0.5_noprelick': 0}
                for trial in range(self.beh._stimrange.first,
                                   self.beh._stimrange.last):
                    print(f'\t\t\t{trial=}', end='\r')
                    _t_stim = self.beh.stim.t_start[trial]
                    _t_start = _t_stim - t_pre
                    _t_end = self.beh._data.get_event_var(
                        'totalRewardTimes')[trial] + t_post

                    _ind_t_start = np.argmin(np.abs(
                        self.rec_t - _t_start))
                    _ind_t_end = np.argmin(np.abs(
                        self.rec_t - _t_end)) + 2   # add frames to end

                    # extract fluorescence
                    _f = np.mean(np.mean(
                        self.rec[_ind_t_start:_ind_t_end,
                                 _ind_x_lower:_ind_x_upper,
                                 _ind_y_lower:_ind_y_upper], axis=1),
                                 axis=1)
                    _dff = calc_dff(_f, baseline_frames=n_frames_pre)
                    # _dff_shifted = (-1 * _dff * plt_dff['y']['gain']) \
                    #     + n_y - plt_dff['y']['offset'] + 1

                    if self.beh.stim.prob[trial] == 0:
                        self.sector.dff[n_x, n_y]['0'][
                            self.sector.tr_counts['0'], :] \
                            = _dff[0:n_frames_tot]
                        self.sector.tr_counts['0'] += 1
                    elif self.beh.stim.prob[trial] == 0.5:
                        self.sector.dff[n_x, n_y]['0.5'][
                            self.sector.tr_counts['0.5'], :] \
                            = _dff[0:n_frames_tot]
                        self.sector.tr_counts['0.5'] += 1
                    elif self.beh.stim.prob[trial] == 1:
                        self.sector.dff[n_x, n_y]['1'][
                            self.sector.tr_counts['1'], :] \
                            = _dff[0:n_frames_tot]
                        self.sector.tr_counts['1'] += 1

                    if self.beh.stim.prob[trial] == 0.5 \
                       and self.beh.rew.delivered[trial] == 1:
                        self.sector.dff[n_x, n_y]['0.5_rew'][
                            self.sector.tr_counts['0.5_rew'], :] \
                            = _dff[0:n_frames_tot]
                        self.sector.tr_counts['0.5_rew'] += 1
                    elif self.beh.stim.prob[trial] == 0.5 \
                         and self.beh.rew.delivered[trial] == 0:
                        self.sector.dff[n_x, n_y]['0.5_norew'][
                            self.sector.tr_counts['0.5_norew'], :] \
                            = _dff[0:n_frames_tot]
                        self.sector.tr_counts['0.5_norew'] += 1

                    if self.beh.stim.prob[trial] == 0.5 \
                       and self.beh.lick.antic[trial] > 0:
                        self.sector.dff[n_x, n_y]['0.5_prelick'][
                            self.sector.tr_counts['0.5_prelick'], :] \
                            = _dff[0:n_frames_tot]
                        self.sector.tr_counts['0.5_prelick'] += 1
                    elif self.beh.stim.prob[trial] == 0.5 \
                         and self.beh.lick.antic[trial] == 0:
                        self.sector.dff[n_x, n_y]['0.5_noprelick'][
                            self.sector.tr_counts['0.5_noprelick'], :] \
                            = _dff[0:n_frames_tot]
                        self.sector.tr_counts['0.5_noprelick'] += 1

                # plot stim-aligned traces
                _t_sector = ((self.sector.t / self.sector.t[-1])
                             * plt_dff['x']['gain'])
                _t_sector = _t_sector + n_x + plt_dff['x']['offset']

                if plot_special is None:
                    stim_list = ['0', '0.5', '1']
                elif plot_special == 'rew_norew':
                    stim_list = ['0.5_rew', '0.5_norew']
                elif plot_special == 'prelick_noprelick':
                    stim_list = ['0.5_prelick', '0.5_noprelick']
                elif plot_special == 'rew':
                    stim_list = ['0', '0.5_rew', '1']

                for stim_cond in stim_list:
                    _dff_mean = np.mean(self.sector.dff[n_x, n_y][stim_cond],
                                        axis=0)
                    _dff_shifted = (-1 * _dff_mean * plt_dff['y']['gain']) \
                        + n_y - plt_dff['y']['offset'] + 1

                    ax_img.plot(
                        _t_sector, _dff_shifted,
                        color=self.sector.colors[stim_cond],
                        linestyle=self.sector.linestyle[stim_cond])

                # plot rew and stim lines
                _t_stim = (0 * plt_dff['x']['gain']) \
                    + n_x + plt_dff['x']['offset']
                _t_rew = ((2 / self.sector.t[-1])
                          * plt_dff['x']['gain']) \
                    + n_x + plt_dff['x']['offset']

                ax_img.plot([_t_stim, _t_stim],
                            [n_y + 0.2, n_y + 0.8],
                            color=sns.xkcd_rgb['dark grey'],
                            linewidth=1,
                            linestyle='dashed')
                ax_img.plot([_t_rew, _t_rew],
                            [n_y + 0.2, n_y + 0.8],
                            color=sns.xkcd_rgb['bright blue'],
                            linewidth=1,
                            linestyle='dashed')

                # update _n_trace
                _n_trace += 1

        fig_avg.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{plt_prefix}_'
            + f'{n_sectors=}_{plot_special=}_{t_pre=}_{t_post=}_'
            + f'ch={self.ch_img}_sector_trial_activity.pdf'))

        if plt_show is True:
            plt.show()

        return

    def plt_psychometric_stimscaling(self, figsize=(2, 2),
                                     marker_size=60,
                                     plt_xlim=[-0.5, 2.5],
                                     plt_show=True):
        """
        Generates a psychometric curve of grab response vs stim
        probability, independently plotting each roi.

        must have ran plt_stim_aligned_sectors() to generate
        the stim-aligned dff traces in self.stim_plt.dff[x, y][stim_cond].
        """
        try:
            self.psychometrics = SimpleNamespace()
            self.psychometrics.p_rew = ['0', '0.5', '1']
            self.psychometrics.stim_resp = {'0': [],
                                            '0.5': [],
                                            '1': []}
            self.psychometrics.rew_resp = {'0': [],
                                           '0.5': [],
                                           '1': []}

            _ind_stim_start = np.argmin(np.abs(self.neur.t - 0))
            _ind_stim_end = np.argmin(np.abs(self.neur.t - 2))

            _ind_rew_bl = np.argmin(np.abs(self.neur.t - 1.5))
            _ind_rew_start = np.argmin(np.abs(self.neur.t - 2))
            _ind_rew_end = np.argmin(np.abs(self.neur.t - 5))

            _n_trace = 0
            for n_x in range(self.neur.n_sectors):
                for n_y in range(self.neur.n_sectors):
                    print(f'\t\tsector {_n_trace}/{self.neur.n_sectors**2}'
                          + '...      ')

                    for trial_cond in ['0', '0.5', '1']:
                        _dff = np.mean(
                            self.neur.dff[n_x, n_y][trial_cond],
                            axis=0)

                        _mean_stim_resp = np.mean(
                            _dff[_ind_stim_start:_ind_stim_end])
                        _mean_rew_resp = np.mean(
                            _dff[_ind_rew_start:_ind_rew_end]) \
                            - np.mean(
                                _dff[_ind_rew_bl:_ind_rew_start])

                        self.psychometrics.stim_resp[trial_cond].append(
                           _mean_stim_resp)
                        self.psychometrics.rew_resp[trial_cond].append(
                            _mean_rew_resp)

                    _n_trace += 1

            # plot psychometric
            # ---------------
            colors = sns.cubehelix_palette(
                 n_colors=3,
                 start=2, rot=0,
                 dark=0.2, light=0.8)

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)

            ax.plot([self.psychometrics.stim_resp['0'],
                     self.psychometrics.stim_resp['0.5'],
                     self.psychometrics.stim_resp['1']],
                    color=sns.xkcd_rgb['light grey'])

            ax.scatter(
                np.ones_like(self.psychometrics.stim_resp['0'])*0,
                self.psychometrics.stim_resp['0'],
                s=marker_size,
                facecolors='none',
                edgecolors=colors[0])
            ax.scatter(
                np.ones_like(self.psychometrics.stim_resp['0.5'])*1,
                self.psychometrics.stim_resp['0.5'],
                s=marker_size,
                facecolors='none',
                edgecolors=colors[1])
            ax.scatter(
                np.ones_like(self.psychometrics.stim_resp['1'])*2,
                self.psychometrics.stim_resp['1'],
                s=marker_size,
                facecolors='none',
                edgecolors=colors[2])

            ax.set_xticks([0, 1, 2], ['0%', '50%', '100%'])
            ax.set_ylabel('GRAB 5-HT stim resp.')
            ax.set_xlabel('rew. prob.')
            ax.set_xlim(plt_xlim)

            # plot correlation
            # ---------------
            fig_corr = plt.figure(figsize=figsize)
            ax_corr = fig_corr.add_subplot(1, 1, 1)

            ax_corr.scatter(self.psychometrics.stim_resp['1'],
                            self.psychometrics.rew_resp['1'],
                            facecolors='none',
                            edgecolors=colors[2], s=10)
            ax_corr.set_xlabel('GRAB 5-HT stim resp. (100%)')
            ax_corr.set_ylabel('GRAB 5-HT rew resp. (100%)')

            fig.savefig(os.path.join(
                os.getcwd(), 'figs_mbl',
                f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
                + f'ch={self.ch_img}_psychometric_curve.pdf'))
            fig_corr.savefig(os.path.join(
                os.getcwd(), 'figs_mbl',
                f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
                + f'ch={self.ch_img}_stim_rew_resp_corr.pdf'))

            if plt_show is True:
                plt.show()

        except Exception as e:
            print('failed to generate psychometric curve')
            print(e)

    def plt_stim_aligned_neurs(self, t_pre=1, t_post=2,
                               figsize=(3.43, 3.43), zscore=False,
                               plt_equal=True,
                               cmap=sns.diverging_palette(
                                   220, 20, as_cmap=True)):
        self.add_lickrates(t_prestim=t_pre,
                           t_poststim=t_post)

        # setup structure
        # ---------------
        neurs = np.where(self.neur.iscell[:, 0] == 1)[0]
        n_neurs = neurs.shape[0]

        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int((2 + t_post) * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        _stim_prob_trrange = self.beh.stim.prob[
            self.beh._stimrange.first:self.beh._stimrange.last]
        _stim_rewdeliv_trrange = self.beh.rew.delivered[
            self.beh._stimrange.first:self.beh._stimrange.last]
        _licks_trrange = self.beh.lick.antic[
            self.beh._stimrange.first:self.beh._stimrange.last]

        self.neur.tr_inds = {}
        self.neur.tr_inds['0'] = np.where(_stim_prob_trrange == 0)[0]
        self.neur.tr_inds['0.5'] = np.where(_stim_prob_trrange == 0.5)[0]
        self.neur.tr_inds['1'] = np.where(_stim_prob_trrange == 1)[0]

        self.neur.tr_inds['0.5_rew'] = np.where(np.logical_and(
            _stim_prob_trrange == 0.5, _stim_rewdeliv_trrange == True))[0]
        self.neur.tr_inds['0.5_norew'] = np.where(np.logical_and(
            _stim_prob_trrange == 0.5, _stim_rewdeliv_trrange == False))[0]
        self.neur.tr_inds['0.5_prelick'] = np.where(np.logical_and(
            _stim_prob_trrange == 0.5, _licks_trrange > 0))[0]
        self.neur.tr_inds['0.5_noprelick'] = np.where(np.logical_and(
            _stim_prob_trrange == 0.5, _licks_trrange == 0))[0]

        self.neur.dff_aln = np.empty(n_neurs, dtype=dict)
        self.neur.dff_aln_mean = {}
        for tr_type in ['0', '0.5', '1', '0.5_rew',
                        '0.5_norew', '0.5_prelick',
                        '0.5_noprelick']:
            self.neur.dff_aln_mean[tr_type] = np.zeros((n_neurs, n_frames_tot))
        self.neur.x = np.zeros(n_neurs)
        self.neur.y = np.zeros(n_neurs)

        self.neur.t_aln = np.linspace(
            -1*t_pre, 2+t_post, n_frames_tot)

        # iterate through trials
        # ----------------
        for ind, neur in enumerate(neurs):
            # extract x/y locations
            self.neur.x[ind] = self.neur.stat[neur]['med'][0]
            self.neur.y[ind] = self.neur.stat[neur]['med'][1]
            # correct for image crop
            self.neur.x[ind] -= self.ops.n_px_remove_sides
            self.neur.y[ind] -= self.ops.n_px_remove_sides

            # setup structure
            # -------------
            self.neur.dff_aln[ind] = {}
            self.neur.dff_aln[ind]['0'] = np.zeros((
                self.neur.tr_inds['0'].shape[0], n_frames_tot))
            self.neur.dff_aln[ind]['0.5'] = np.zeros((
                self.neur.tr_inds['0.5'].shape[0], n_frames_tot))
            self.neur.dff_aln[ind]['1'] = np.zeros((
                self.neur.tr_inds['1'].shape[0], n_frames_tot))

            self.neur.dff_aln[ind]['0.5_rew'] = np.zeros((
                self.neur.tr_inds['0.5_rew'].shape[0], n_frames_tot))
            self.neur.dff_aln[ind]['0.5_norew'] = np.zeros((
                self.neur.tr_inds['0.5_norew'].shape[0], n_frames_tot))

            self.neur.dff_aln[ind]['0.5_prelick'] = np.zeros((
                self.neur.tr_inds['0.5_prelick'].shape[0],
                n_frames_tot))
            self.neur.dff_aln[ind]['0.5_noprelick'] = np.zeros((
                self.neur.tr_inds['0.5_noprelick'].shape[0],
                n_frames_tot))

            self.neur.dff_aln_mean[ind] = {}

            _tr_counts = {'0': 0, '0.5': 0, '1': 0,
                          '0.5_rew': 0, '0.5_norew': 0,
                          '0.5_prelick': 0,
                          '0.5_noprelick': 0}

            # extract fluorescence for each trial
            # -------------
            for trial in range(self.beh._stimrange.first,
                               self.beh._stimrange.last):
                print(f'\t\t\tneur={int(neur)} | {trial=}', end='\r')
                _t_stim = self.beh.stim.t_start[trial]
                _t_start = _t_stim - t_pre
                _t_end = self.beh._data.get_event_var(
                    'totalRewardTimes')[trial] + t_post

                _ind_t_start = np.argmin(np.abs(
                    self.neur.t - _t_start))
                _ind_t_end = np.argmin(np.abs(
                    self.neur.t - _t_end)) + 2   # add frames to end

                if zscore is True:
                    _f = sp.stats.zscore(self.neur.f[neur, :])[
                        _ind_t_start:_ind_t_end]
                elif zscore is False:
                    _f = self.neur.f[neur, _ind_t_start:_ind_t_end]
                _dff = calc_dff(_f, baseline_frames=n_frames_pre)

                if self.beh.stim.prob[trial] == 0:
                    self.neur.dff_aln[ind]['0'][_tr_counts['0'], :] \
                        = _dff[0:n_frames_tot]
                    _tr_counts['0'] += 1
                elif self.beh.stim.prob[trial] == 0.5:
                    self.neur.dff_aln[ind]['0.5'][_tr_counts['0.5'], :] \
                        = _dff[0:n_frames_tot]
                    _tr_counts['0.5'] += 1
                elif self.beh.stim.prob[trial] == 1:
                    self.neur.dff_aln[ind]['1'][_tr_counts['1'], :] \
                        = _dff[0:n_frames_tot]
                    _tr_counts['1'] += 1

                if self.beh.stim.prob[trial] == 0.5 \
                   and self.beh.rew.delivered[trial] == 1:
                    self.neur.dff_aln[ind]['0.5_rew'][
                        _tr_counts['0.5_rew'], :] \
                        = _dff[0:n_frames_tot]
                    _tr_counts['0.5_rew'] += 1
                elif self.beh.stim.prob[trial] == 0.5 \
                     and self.beh.rew.delivered[trial] == 0:
                    self.neur.dff_aln[ind]['0.5_norew'][
                        _tr_counts['0.5_norew'], :] \
                        = _dff[0:n_frames_tot]
                    _tr_counts['0.5_norew'] += 1

                if self.beh.stim.prob[trial] == 0.5 \
                   and self.beh.lick.antic[trial] > 0:
                    self.neur.dff_aln[ind]['0.5_prelick'][
                        _tr_counts['0.5_prelick'], :] \
                        = _dff[0:n_frames_tot]
                    _tr_counts['0.5_prelick'] += 1
                elif self.beh.stim.prob[trial] == 0.5 \
                     and self.beh.lick.antic[trial] == 0:
                    self.neur.dff_aln[ind]['0.5_noprelick'][
                        _tr_counts['0.5_noprelick'], :] \
                        = _dff[0:n_frames_tot]
                    _tr_counts['0.5_noprelick'] += 1

            for tr_type in ['0', '0.5', '1', '0.5_rew',
                            '0.5_norew', '0.5_prelick',
                            '0.5_noprelick']:
                self.neur.dff_aln_mean[tr_type][ind, :] = \
                    np.mean(self.neur.dff_aln[ind][tr_type], axis=0)

        # process data for plotting
        # ------------
        _ind_stim_start = np.argmin(np.abs(self.neur.t_aln))
        _ind_stim_end = np.argmin(np.abs(self.neur.t_aln-2))
        diff_0_100_rew = np.abs(np.max(
            self.neur.dff_aln_mean['1'][:, _ind_stim_start:_ind_stim_end]
            - self.neur.dff_aln_mean['0'][:, _ind_stim_start:_ind_stim_end],
            axis=1))

        sort_inds_corr = np.argsort(diff_0_100_rew)

        plt_vmin = np.min(
            [self.neur.dff_aln_mean['0'],
             self.neur.dff_aln_mean['0.5'],
             self.neur.dff_aln_mean['1']])
        plt_vmax = np.max(
            [self.neur.dff_aln_mean['0'],
             self.neur.dff_aln_mean['0.5'],
             self.neur.dff_aln_mean['1']])

        if plt_equal is True:
            if abs(plt_vmin) < abs(plt_vmax):
                plt_vmin = -1 * plt_vmax
            elif abs(plt_vmin) > abs(plt_vmax):
                plt_vmax = -1 * plt_vmin

        # plot figure
        # -----------------
        fig = plt.figure(figsize=figsize)
        spec = gs.GridSpec(nrows=1, ncols=3,
                           figure=fig)
        ax_0pc = fig.add_subplot(spec[0, 0])
        ax_50pc = fig.add_subplot(spec[0, 1], sharey=ax_0pc)
        ax_100pc = fig.add_subplot(spec[0, 2], sharey=ax_0pc)

        ax_0pc.pcolormesh(
            self.neur.dff_aln_mean['0'][sort_inds_corr],
            vmin=plt_vmin, vmax=plt_vmax,
            cmap=cmap)
        ax_50pc.pcolormesh(
            self.neur.dff_aln_mean['0.5'][sort_inds_corr],
            vmin=plt_vmin, vmax=plt_vmax,
            cmap=cmap)
        ax_100pc.pcolormesh(
            self.neur.dff_aln_mean['1'][sort_inds_corr],
            vmin=plt_vmin, vmax=plt_vmax,
            cmap=cmap)

        _ind_zero = np.argmin(np.abs(self.neur.t_aln))
        _ind_rew = np.argmin(np.abs(self.neur.t_aln-2))

        for _ax in [ax_0pc, ax_50pc, ax_100pc]:
            _ax.axvline(_ind_zero,
                        color=sns.xkcd_rgb['dark grey'],
                        linewidth=1,
                        linestyle='dashed')
            _ax.axvline(_ind_rew,
                        color=sns.xkcd_rgb['bright blue'],
                        linewidth=1,
                        linestyle='dashed')
            _ax.set_xticks(
                ticks=[0, _ind_zero, self.neur.t_aln.shape[-1]],
                labels=[f'{self.neur.t_aln[0]:.2f}', 0,
                        f'{self.neur.t_aln[-1]:.2f}'])
        ax_0pc.set_title('p(rew)=0')
        ax_50pc.set_title('p(rew)=0.5')
        ax_100pc.set_title('p(rew)=1')
        ax_0pc.set_ylabel('neuron')
        ax_50pc.set_xlabel('time from stim (s)')

        fig.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{t_pre=}_{t_post=}_{zscore=}_{plt_equal=}_'
            + 'neur_trial_activity.pdf'))

        plt.show()

        return

    def plt_rew_trace_by_iti(self, ind_sector=10, n_time_divs=5,
                             sns_palette='mako',
                             figsize=(3.43, 3.43), savefig_prefix='grab'):
        palette = sns.color_palette(sns_palette, n_time_divs)

        self.dff_binned_time = np.empty(n_time_divs, dtype=np.ndarray)
        itis = np.diff(self.beh.rew.t)
        _count, t_bin_edges = np.histogram(itis, bins=n_time_divs)

        for ind_timediv in range(n_time_divs):
            _trials_in_timediv = np.logical_and(
                itis > t_bin_edges[ind_timediv],
                itis < t_bin_edges[ind_timediv+1])
            _mean_dff_in_timediv = np.mean(
                self.neur.dff_rewaligned[ind_sector][_trials_in_timediv, :],
                axis=0)
            self.dff_binned_time[ind_timediv] = _mean_dff_in_timediv

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        for ind_timediv in range(n_time_divs):
            label = f'{t_bin_edges[ind_timediv]:.1f}' \
                + f'-{t_bin_edges[ind_timediv+1]:.1f}'
            ax.plot(self.neur.t,
                    self.dff_binned_time[ind_timediv],
                    color=palette[ind_timediv], linewidth=0.8,
                    label=label)
        ax.axvline(x=0, color=sns.xkcd_rgb['bright blue'],
                   linestyle='dashed',
                   linewidth=0.8)
        ax.legend()

        ax.set_xlabel('time from rew (s)')
        ax.set_ylabel('df/f')

        fig.savefig(f'{savefig_prefix}_dff_by_{n_time_divs}'
                    + f'iti_sector{ind_sector}.pdf')

        plt.show()

    def plt_dffs_including_iti(self, ind_sector=10, n_time_divs=5,
                               t_rew_pre=2,
                               sns_palette='mako',
                               figsize=(3.43, 3.43), savefig_prefix='grab'):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        palette = sns.color_palette(sns_palette, n_time_divs)

        self.dff_binned_time = np.empty(n_time_divs, dtype=np.ndarray)
        itis = np.diff(self.beh.rew.t)

        _count, t_bin_edges = np.histogram(itis, bins=n_time_divs)

        for ind_timediv in range(n_time_divs):
            _trials_in_timediv = np.logical_and(
                itis > t_bin_edges[ind_timediv],
                itis < t_bin_edges[ind_timediv+1])

            t_rew_post = t_bin_edges[ind_timediv]

            n_frames_pre = int(t_rew_pre * self.samp_rate)
            n_frames_post = int(t_rew_post * self.samp_rate)
            n_frames_tot = n_frames_pre + n_frames_post

            for ind, t_rew in enumerate(self.beh.rew.t[1:]):
                ind_rew = np.argmin(np.abs(self.rec_t-t_rew))
                _rew_trace = _trace[ind_rew-n_frames_pre:
                                    ind_rew+n_frames_post]
                self.neur.dff_rewaligned[_n_trace][ind, :] = calc_dff(
                    _rew_trace, baseline_frames=n_frames_pre)

            _mean_dff_in_timediv = np.mean(
                self.neur.dff_rewaligned[ind_sector][_trials_in_timediv, :],
                axis=0)
            self.dff_binned_time[ind_timediv] = _mean_dff_in_timediv

        for ind_timediv in range(n_time_divs):
            label=f'{t_bin_edges[ind_timediv]:.1f}-{t_bin_edges[ind_timediv+1]:.1f}'
            ax.plot(self.neur.t,
                    self.dff_binned_time[ind_timediv],
                    color=palette[ind_timediv], linewidth=0.8,
                    label=label)
        ax.axvline(x=0, color=sns.xkcd_rgb['bright blue'],
                   linestyle='dashed',
                   linewidth=0.8)
        ax.legend()

        ax.set_xlabel('time from rew (s)')
        ax.set_ylabel('df/f')

        fig.savefig(f'{savefig_prefix}_dff_by_{n_time_divs}iti_sector{ind_sector}.pdf')

        return

    def plt_dffs_single(self, sector=0,
                        sns_palette='mako',
                        figsize=(3.43, 3.43),
                        savefig_prefix='grab'):

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        for trial in range(self.neur.dff_rewaligned[sector].shape[0]):
            ax.plot(self.neur.t,
                    self.neur.dff_rewaligned[sector][trial, :],
                    color=sns.xkcd_rgb['ocean green'], linewidth=0.8)
        ax.axvline(x=0, color=sns.xkcd_rgb['bright blue'],
                   linestyle='dashed',
                   linewidth=0.8)

        ax.set_xlabel('time from rew (s)')
        ax.set_ylabel('df/f')

        fig.savefig(f'{savefig_prefix=}_dff_{sector=}.pdf')

        plt.show()

    def plt_dffs_all(self,
                     sns_palette='mako',
                     figsize=(3.43, 3.43),
                     savefig_prefix='grab'):

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        for sector in range(self.neur.dff_rewaligned.shape[0]):
            ax.plot(self.neur.t,
                    np.mean(self.neur.dff_rewaligned[sector], axis=0),
                    color=sns.xkcd_rgb['ocean green'], linewidth=0.8)
        ax.axvline(x=0, color=sns.xkcd_rgb['bright blue'],
                   linestyle='dashed',
                   linewidth=0.8)

        ax.set_xlabel('time from rew (s)')
        ax.set_ylabel('df/f')

        fig.savefig(f'{savefig_prefix}_dff_all.pdf')

        plt.show()

    def plt_total_resp(self, figsize=(3.43, 3.43),
                       savefig_prefix='grab'):

        n_rews = self.neur.dff_rewaligned[0].shape[0]
        n_sectors = self.neur.dff_rewaligned.shape[0]

        self.resp_5ht = np.zeros(n_rews)

        for rew in range(self.beh._rewrange.first, self.beh._rewrange.last):
            _resp_5ht = 0
            for sector in range(n_sectors):
                _dff_integ = np.trapz(
                    self.neur.dff_rewaligned[sector][rew, :],
                    dx=1/self.samp_rate)

                _resp_5ht += _dff_integ
            _resp_5ht /= n_sectors

            self.resp_5ht[rew] = _resp_5ht

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(np.arange(n_rews), self.resp_5ht,
                color=sns.xkcd_rgb['ocean green'], linewidth=0.8)
        ax.set_xlabel('reward number')
        ax.set_ylabel('integral df/f')

        fig.savefig(f'{savefig_prefix}_total_response.pdf')

        plt.show()

        return

    def plt_lick_resp(self, t_pre=3, t_post=3,
                      figsize=(3.43, 3.43),
                      savefig_prefix='grab'):
        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int(t_post * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        t_lickaligned = np.linspace(
            -1*t_pre, t_post, n_frames_tot)

        dff_lick = np.zeros(n_frames_tot)
        count_dffs = 0
        for lick in self.beh.t_licks:
            _closest_rew = np.min(np.abs(
                self.beh.rew.t - lick))
            if _closest_rew > np.max([t_pre, t_post]):
                ind_lick = np.argmin(np.abs(
                    self.rec_t - lick))
                f = np.mean(np.mean(
                    self.rec[ind_lick-n_frames_pre:
                             ind_lick+n_frames_post, :, :],
                    axis=1), axis=1)
                dff = calc_dff(f, n_frames_pre)
                dff_lick += dff
                count_dffs += 1

        dff_lick /= count_dffs

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t_lickaligned, dff_lick,
                color=sns.xkcd_rgb['grey'], linewidth=0.8)
        ax.set_xlabel('time from lick (s)')
        ax.set_ylabel('dff')
        ax.axvline(x=0, color=sns.xkcd_rgb['black'],
                   linestyle='dashed',
                   linewidth=0.8)

        fig.savefig(f'{savefig_prefix}_lick_resp_{t_pre=}_{t_post=}.pdf')
        plt.show()


class TwoPRec(object):
    def __init__(self, fname,
                 folder_beh,
                 sampling_rate=29.87,
                 n_px_remove_sides=10):
        self.fname = fname
        self.folder_beh = folder_beh

        # load behavioral data
        # -------------
        self.beh = BehDataSimpleLoad(self.folder_beh)

        self.beh.rew = SimpleNamespace()
        self.beh.rew.t = self.beh._data.get_event_var('isRewardGivenTimes')

        _daq_data = self.beh._timeline.get_daq_data()
        self.beh.licks = find_event_onsets_autothresh(
            _daq_data.sig['lickDetector'])
        self.beh.t_licks = _daq_data.t[self.beh.licks]

        # load imaging data
        # ------------
        print('loading imaging...')
        # parse sampling rate
        _dir_list = os.listdir(os.path.split(fname)[0])
        for _file in _dir_list:
            if _file.endswith('BACKUP.xml'):
                _xml_path = os.path.join(
                    os.path.split(fname)[0], _file)
                try:
                    xmlobj = XMLParser(_xml_path)
                    sampling_rate = xmlobj.get_framerate()
                    print(f'\tframerate is {sampling_rate:.2f}Hz')
                except:
                    pass

        self.samp_rate = sampling_rate

        # goto imaging data folder
        os.chdir(os.path.split(fname)[0])

        # load tiff and generate timestamps for the frames aligned to rew onset
        print('\tloading tiff...')
        self.rec = tifffile.imread(fname)[
            :, n_px_remove_sides:, n_px_remove_sides:]

        print('\tcreating timestamps...')
        _t_start = self.beh.rew.t[0]
        _t_end = (self.rec.shape[0]/self.samp_rate) + _t_start

        self.rec_t = np.linspace(_t_start,
                                 _t_end,
                                 num=self.rec.shape[0])

        return

    def plt_spatial_sectors(self, n_sectors,
                            figsize=(3.43, 2),
                            dpi=300,
                            scaling_factor=2,
                            scaling_factor_rewlocked=10,
                            t_rew_pre=1, t_rew_post=3,
                            ind_lastrew=None):
        print('plotting spatial sectors...')
        self.params = SimpleNamespace()
        self.params.n_sectors = n_sectors
        self.params.t_rew_pre = t_rew_pre
        self.params.t_rew_post = t_rew_post

        fig = plt.figure(figsize=figsize, dpi=dpi)
        spec = gs.GridSpec(nrows=2, ncols=2,
                           height_ratios=[0.8, 0.2],
                           figure=fig)
        ax_img = fig.add_subplot(spec[0, 0])
        ax_traces = fig.add_subplot(spec[0, 1])
        ax_rew = fig.add_subplot(spec[1, 1], sharex=ax_traces)

        _rec_max = np.max(self.rec, axis=0)
        _rec_max[:, ::int(_rec_max.shape[0]/n_sectors)] = 0
        _rec_max[::int(_rec_max.shape[0]/n_sectors), :] = 0

        ax_img.imshow(_rec_max,
                      extent=[0, n_sectors,
                              n_sectors, 0])

        # setup reward-aligned traces
        self.dff_rewaligned = np.empty(n_sectors*n_sectors,
                                       dtype=np.ndarray)
        n_frames_pre = int(t_rew_pre * self.samp_rate)
        n_frames_post = int(t_rew_post * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        self.t_rewaligned = np.linspace(
            -1*t_rew_pre, t_rew_post, n_frames_tot)

        _n_trace = 0
        for n_x in range(n_sectors):
            for n_y in range(n_sectors):
                print(f'\tsector {_n_trace}/{n_sectors**2}...', end='\r')
                # calculate location of sector
                _ind_x_lower = int((n_x / n_sectors) * self.rec.shape[1])
                _ind_x_upper = int(((n_x+1) / n_sectors) * self.rec.shape[1])

                _ind_y_lower = int((n_y / n_sectors) * self.rec.shape[1])
                _ind_y_upper = int(((n_y+1) / n_sectors) * self.rec.shape[1])

                # plot trace
                _trace = np.mean(np.mean(
                    self.rec[:,
                             _ind_x_lower:_ind_x_upper,
                             _ind_y_lower:_ind_y_upper], axis=1), axis=1)

                ax_traces.plot(self.rec_t,
                               sp.stats.zscore(_trace)*scaling_factor
                               + _n_trace,
                               color=sns.xkcd_rgb['ocean green'],
                               linewidth=0.5, alpha=0.8)

                # plot reward-aligned trace
                self.dff_rewaligned[_n_trace] = np.zeros((
                    self.beh.rew.t[1:ind_lastrew].shape[0],
                    n_frames_tot))

                for ind, t_rew in enumerate(self.beh.rew.t[1:ind_lastrew]):
                    ind_rew = np.argmin(np.abs(self.rec_t-t_rew))
                    _rew_trace = _trace[ind_rew-n_frames_pre:
                                        ind_rew+n_frames_post]
                    self.dff_rewaligned[_n_trace][ind, :] = calc_dff(
                        _rew_trace, baseline_frames=n_frames_pre)

                t_rewaligned_norm = ((self.rec_t[0:n_frames_tot]
                                      / self.rec_t[n_frames_tot])
                                     * 0.6)
                t_rewaligned_norm = t_rewaligned_norm + n_x + 0.2

                # note here that we must invert dff_rewaligned_mean
                # because plt() traces are  plotted in the 'negative' direction
                # on top of imshow() images for some reason
                dff_rewaligned_mean = np.mean(
                    self.dff_rewaligned[_n_trace], axis=0)
                dff_rewaligned_mean_shifted = (-1 * dff_rewaligned_mean
                                               * scaling_factor_rewlocked
                                               + n_y + 0.5)

                t_rewonset = ((self.rec_t[n_frames_pre]
                               / self.rec_t[n_frames_tot]) * 0.6) \
                               + n_x + 0.2
                ax_img.plot(t_rewaligned_norm,
                            dff_rewaligned_mean_shifted,
                            color=sns.xkcd_rgb['orangered'],
                            linewidth=0.5,
                            alpha=0.8)
                ax_img.plot([t_rewonset, t_rewonset], [n_y+0.2, n_y+0.8],
                            color=sns.xkcd_rgb['white'], linestyle='dashed',
                            linewidth=0.3)

                _n_trace += 1
                self._last_trace = _trace

        for ind, t_rew in enumerate(self.beh.rew.t):
            ax_rew.plot([t_rew, t_rew], [0, 1],
                        color=sns.xkcd_rgb['bright blue'])

        if self.fname.endswith('Ch1.tif'):
            prefix = 'grab'
        elif self.fname.endswith('Ch2.tif'):
            prefix = 'gcamp'

        fig.savefig(os.path.split(self.fname)[0] + '/'
                    + f'{prefix}_quadrant_fig.pdf')

        plt.show()

    def plt_dff_by_iti(self, ind_sector=10, n_time_divs=5,
                       sns_palette='mako',
                       figsize=(3.43, 3.43), savefig_prefix='grab'):
        palette = sns.color_palette(sns_palette, n_time_divs)

        self.dff_binned_time = np.empty(n_time_divs, dtype=np.ndarray)
        itis = np.diff(self.beh.rew.t)

        _count, t_bin_edges = np.histogram(itis, bins=n_time_divs)

        for ind_timediv in range(n_time_divs):
            _trials_in_timediv = np.logical_and(
                itis > t_bin_edges[ind_timediv],
                itis < t_bin_edges[ind_timediv+1])
            _mean_dff_in_timediv = np.mean(
                self.dff_rewaligned[ind_sector][_trials_in_timediv, :],
                axis=0)
            self.dff_binned_time[ind_timediv] = _mean_dff_in_timediv

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        for ind_timediv in range(n_time_divs):
            label = f'{t_bin_edges[ind_timediv]:.1f}' \
                + f'-{t_bin_edges[ind_timediv+1]:.1f}'
            ax.plot(self.t_rewaligned,
                    self.dff_binned_time[ind_timediv],
                    color=palette[ind_timediv], linewidth=0.8,
                    label=label)
        ax.axvline(x=0, color=sns.xkcd_rgb['bright blue'],
                   linestyle='dashed',
                   linewidth=0.8)
        ax.legend()

        ax.set_xlabel('time from rew (s)')
        ax.set_ylabel('df/f')

        fig.savefig(f'{savefig_prefix}_dff_by_{n_time_divs}'
                    + f'iti_sector{ind_sector}.pdf')

        plt.show()

    def plt_dffs_including_iti(self, ind_sector=10, n_time_divs=5,
                               t_rew_pre=2,
                               sns_palette='mako',
                               figsize=(3.43, 3.43), savefig_prefix='grab'):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        palette = sns.color_palette(sns_palette, n_time_divs)

        self.dff_binned_time = np.empty(n_time_divs, dtype=np.ndarray)
        itis = np.diff(self.beh.rew.t)

        _count, t_bin_edges = np.histogram(itis, bins=n_time_divs)

        for ind_timediv in range(n_time_divs):
            _trials_in_timediv = np.logical_and(
                itis > t_bin_edges[ind_timediv],
                itis < t_bin_edges[ind_timediv+1])

            t_rew_post = t_bin_edges[ind_timediv]

            n_frames_pre = int(t_rew_pre * self.samp_rate)
            n_frames_post = int(t_rew_post * self.samp_rate)
            n_frames_tot = n_frames_pre + n_frames_post

            for ind, t_rew in enumerate(self.beh.rew.t[1:]):
                ind_rew = np.argmin(np.abs(self.rec_t-t_rew))
                _rew_trace = _trace[ind_rew-n_frames_pre:
                                    ind_rew+n_frames_post]
                self.dff_rewaligned[_n_trace][ind, :] = calc_dff(
                    _rew_trace, baseline_frames=n_frames_pre)

            _mean_dff_in_timediv = np.mean(
                self.dff_rewaligned[ind_sector][_trials_in_timediv, :],
                axis=0)
            self.dff_binned_time[ind_timediv] = _mean_dff_in_timediv

        for ind_timediv in range(n_time_divs):
            label=f'{t_bin_edges[ind_timediv]:.1f}-{t_bin_edges[ind_timediv+1]:.1f}'
            ax.plot(self.t_rewaligned,
                    self.dff_binned_time[ind_timediv],
                    color=palette[ind_timediv], linewidth=0.8,
                    label=label)
        ax.axvline(x=0, color=sns.xkcd_rgb['bright blue'],
                   linestyle='dashed',
                   linewidth=0.8)
        ax.legend()

        ax.set_xlabel('time from rew (s)')
        ax.set_ylabel('df/f')

        fig.savefig(f'{savefig_prefix}_dff_by_{n_time_divs}iti_sector{ind_sector}.pdf')

        plt.show()

        return

    def plt_dffs_single(self, sector=0,
                        sns_palette='mako',
                        figsize=(3.43, 3.43),
                        savefig_prefix='grab'):

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        for trial in range(self.dff_rewaligned[sector].shape[0]):
            ax.plot(self.t_rewaligned,
                    self.dff_rewaligned[sector][trial, :],
                    color=sns.xkcd_rgb['ocean green'], linewidth=0.8)
        ax.axvline(x=0, color=sns.xkcd_rgb['bright blue'],
                   linestyle='dashed',
                   linewidth=0.8)

        ax.set_xlabel('time from rew (s)')
        ax.set_ylabel('df/f')

        fig.savefig(f'{savefig_prefix=}_dff_{sector=}.pdf')

        plt.show()

    def plt_dffs_all(self,
                     sns_palette='mako',
                     figsize=(3.43, 3.43),
                     savefig_prefix='grab'):

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        for sector in range(self.dff_rewaligned.shape[0]):
            ax.plot(self.t_rewaligned,
                    np.mean(self.dff_rewaligned[sector], axis=0),
                    color=sns.xkcd_rgb['ocean green'], linewidth=0.8)
        ax.axvline(x=0, color=sns.xkcd_rgb['bright blue'],
                   linestyle='dashed',
                   linewidth=0.8)

        ax.set_xlabel('time from rew (s)')
        ax.set_ylabel('df/f')

        fig.savefig(f'{savefig_prefix}_dff_all.pdf')

        plt.show()

    def plt_total_resp(self, figsize=(3.43, 3.43),
                       savefig_prefix='grab'):

        n_rews = self.dff_rewaligned[0].shape[0]
        n_sectors = self.dff_rewaligned.shape[0]

        self.resp_5ht = np.zeros(n_rews)

        for rew in range(n_rews):
            _resp_5ht = 0
            for sector in range(n_sectors):
                _dff_integ = np.trapz(
                    self.dff_rewaligned[sector][rew, :],
                    dx=1/self.samp_rate)

                _resp_5ht += _dff_integ
            _resp_5ht /= n_sectors

            self.resp_5ht[rew] = _resp_5ht

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(np.arange(n_rews), self.resp_5ht,
                color=sns.xkcd_rgb['ocean green'], linewidth=0.8)
        ax.set_xlabel('reward number')
        ax.set_ylabel('integral df/f')

        fig.savefig(f'{savefig_prefix}_total_response.pdf')

        plt.show()

        return

    def plt_lick_resp(self, t_pre=3, t_post=3,
                      figsize=(3.43, 3.43),
                      savefig_prefix='grab'):
        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int(t_post * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        t_lickaligned = np.linspace(
            -1*t_pre, t_post, n_frames_tot)

        dff_lick = np.zeros(n_frames_tot)
        count_dffs = 0
        for lick in self.beh.t_licks:
            _closest_rew = np.min(np.abs(
                self.beh.rew.t - lick))
            if _closest_rew > np.max([t_pre, t_post]):
                ind_lick = np.argmin(np.abs(
                    self.rec_t - lick))
                f = np.mean(np.mean(
                    self.rec[ind_lick-n_frames_pre:
                             ind_lick+n_frames_post, :, :],
                    axis=1), axis=1)
                dff = calc_dff(f, n_frames_pre)
                dff_lick += dff
                count_dffs += 1

        dff_lick /= count_dffs

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t_lickaligned, dff_lick,
                color=sns.xkcd_rgb['grey'], linewidth=0.8)
        ax.set_xlabel('time from lick (s)')
        ax.set_ylabel('dff')
        ax.axvline(x=0, color=sns.xkcd_rgb['black'],
                   linestyle='dashed',
                   linewidth=0.8)

        fig.savefig(f'{savefig_prefix}_lick_resp_{t_pre=}_{t_post=}.pdf')
        plt.show()


class TwoPRec_DualColor(object):
    def __init__(self, fname_grab, fname_gcamp,
                 folder_beh,
                 sampling_rate=29.87,
                 n_px_remove_sides=10):
        print('loading grab ch...\n-----------')
        self.grab = GRABRec(fname=fname_grab,
                            folder_beh=folder_beh,
                            n_px_remove_sides=n_px_remove_sides)
        print('loading gcamp ch...\n-----------')
        self.gcamp = GRABRec(fname=fname_gcamp,
                             folder_beh=folder_beh,
                             n_px_remove_sides=n_px_remove_sides)

    def setup_compare(self, n_sectors, scaling_factor_rewlocked=10,
                      t_rew_pre=1, t_rew_post=3, ind_lastrew=None):

        self.grab.plt_spatial_sectors(
            n_sectors,
            figsize=(3.43, 2),
            dpi=300,
            scaling_factor=2,
            scaling_factor_rewlocked=scaling_factor_rewlocked,
            t_rew_pre=t_rew_pre, t_rew_post=t_rew_post,
            ind_lastrew=ind_lastrew)
        self.gcamp.plt_spatial_sectors(
            n_sectors,
            figsize=(3.43, 2),
            dpi=300,
            scaling_factor=2,
            scaling_factor_rewlocked=scaling_factor_rewlocked,
            t_rew_pre=t_rew_pre, t_rew_post=t_rew_post,
            ind_lastrew=ind_lastrew)

    def compare(self, ex_sector=0, figsize=(5, 2),
                markersize=5, color_palette='magma'):
        print('comparing grab and gcamp...')
        # compare responses
        n_rews = self.grab.dff_rewaligned[0].shape[0]
        n_sectors = self.grab.dff_rewaligned.shape[0]

        fig = plt.figure(figsize=figsize)
        spec = gs.GridSpec(nrows=1, ncols=3, figure=fig)

        ax_meancorr = fig.add_subplot(spec[0, 0])
        ax_trcorr_ex = fig.add_subplot(spec[0, 1])
        ax_trcorr_all = fig.add_subplot(spec[0, 2])

        colors = sns.color_palette('magma', n_sectors)
        for sector in range(n_sectors):
            print(f'\tsector {sector}/{n_sectors}...', end='\r')
            _dff_integ_grab_mean = np.trapz(np.mean(
                self.grab.dff_rewaligned[sector], axis=0),
                    dx=1/self.grab.samp_rate)
            _dff_integ_gcamp_mean = np.trapz(np.mean(
                self.gcamp.dff_rewaligned[sector], axis=0),
                    dx=1/self.gcamp.samp_rate)

            ax_meancorr.scatter([_dff_integ_grab_mean],
                                [_dff_integ_gcamp_mean],
                                s=markersize, color=colors[sector])

            for rew in range(n_rews):
                _dff_integ_grab = np.trapz(
                    self.grab.dff_rewaligned[sector][rew, :],
                    dx=1/self.grab.samp_rate)
                _dff_integ_gcamp = np.trapz(
                    self.gcamp.dff_rewaligned[sector][rew, :],
                    dx=1/self.gcamp.samp_rate)

                ax_trcorr_all.scatter([_dff_integ_grab], [_dff_integ_gcamp],
                                      s=markersize, color=colors[sector])

                if sector == ex_sector:
                    ax_trcorr_ex.scatter([_dff_integ_grab], [_dff_integ_gcamp],
                                         s=markersize, color=colors[sector])

            for _ax in [ax_trcorr_all, ax_trcorr_ex]:
                _ax.set_xlabel('integ(dff) grab')
                _ax.set_ylabel('integ(dff) gcamp')
                _ax.axhline(0, color=sns.xkcd_rgb['grey'],
                            linestyle='dashed',
                            linewidth=0.8)
                _ax.axvline(0, color=sns.xkcd_rgb['grey'],
                            linestyle='dashed',
                            linewidth=0.8)

            ax_meancorr.set_xlabel('mean integ(dff) grab')
            ax_meancorr.set_ylabel('mean integ(dff) gcamp')
            ax_meancorr.axhline(0, color=sns.xkcd_rgb['grey'],
                                linestyle='dashed',
                                linewidth=0.8)
            ax_meancorr.axvline(0, color=sns.xkcd_rgb['grey'],
                                linestyle='dashed',
                                linewidth=0.8)

        fig.savefig(f'compare_grab_gcamp_{ex_sector=}.pdf')
        plt.show()



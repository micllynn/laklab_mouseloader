from types import SimpleNamespace
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import os
import tifffile

from .align_imgbeh import Aligner_ImgBeh
from .utils import find_event_onsets_autothresh
from .utils_twop import XMLParser, calc_dff
from .beh import BehDataSimpleLoad, StimParser


class TwoPRec_New(object):
    def __init__(self,
                 enclosing_folder,
                 folder_beh,
                 folder_img,
                 ch_img=2,
                 # fname_img,
                 dset_obj=None,
                 dset_ind=None,
                 rec_type='paqio',
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
        if dset_obj is None:
            self.folder.enclosing = enclosing_folder
            self.folder.img = folder_img
            self.folder.beh = folder_beh
        elif 'DSetObj' in str(type(dset_obj)):
            pass

        os.chdir(self.folder.enclosing)

        # get filename of image of the appropriate channel
        list_img = os.listdir(self.folder.img)
        for _fname in list_img:
            if f'Ch{ch_img}.tif' in _fname and 'compiled' in _fname:
                self.fname_img = _fname

        # load behavioral data
        # -------------
        self.beh = BehDataSimpleLoad(self.folder.beh)

        self.beh.rew = SimpleNamespace()
        self.beh.stim = SimpleNamespace()

        self.beh.rew.t = self.beh._data.get_event_var('totalRewardTimes')

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
            :, n_px_remove_sides:, n_px_remove_sides:]

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

        return

    def plt_spatial_sectors(self, n_sectors,
                            figsize=(3.43, 2),
                            dpi=300,
                            scaling_factor_trace=2,
                            scaling_factor_img=10,
                            t_rew_pre=1, t_rew_post=3,
                            img_ds_factor=50,
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

        _rec_max = np.max(self.rec[::img_ds_factor, :, :], axis=0)
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
                               sp.stats.zscore(_trace)*scaling_factor_trace
                               + _n_trace,
                               color=sns.xkcd_rgb['ocean green'],
                               linewidth=0.5, alpha=0.8)

                # plot reward-aligned trace
                self.dff_rewaligned[_n_trace] = np.zeros((
                    self.beh.rew.t[self.beh._rewrange.first:
                                   self.beh._rewrange.last].shape[0],
                    n_frames_tot))

                for ind, t_rew in enumerate(
                        self.beh.rew.t[self.beh._rewrange.first:
                                       self.beh._rewrange.last]):
                    ind_rew = np.argmin(np.abs(self.rec_t-t_rew))
                    _rew_trace = _trace[ind_rew-n_frames_pre:
                                        ind_rew+n_frames_post]
                    self.dff_rewaligned[_n_trace][ind, :] = calc_dff(
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
                    self.dff_rewaligned[_n_trace], axis=0)
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

        fig.savefig(os.path.join(os.getcwd(),
                    f'{prefix}_quadrant_fig.pdf'))

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

        for rew in range(self.beh._rewrange.first, self.beh._rewrange.last):
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



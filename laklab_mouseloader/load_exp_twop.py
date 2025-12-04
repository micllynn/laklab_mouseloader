from types import SimpleNamespace
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import sklearn
import sklearn.linear_model
import sklearn.preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
import scipy.stats as sp_stats

import zetapy

import os
import pathlib
import tifffile

from .align_imgbeh import Aligner_ImgBeh
from .utils import find_event_onsets_autothresh, remove_lick_artefact_after_rew, \
    check_folder_exists
from .utils_twop import XMLParser, calc_dff
from .beh import BehDataSimpleLoad, StimParser


class TwoPRec(object):
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
        Loads a 2p recording (tiff) and associated behavioral folder.

        Two methods of loading are available.
        ---------------
        1. Load using a dataset object built for your experiment.
        (Specify dset_obj and dset_ind, and optionally ch_img.)
            - dset_obj is a DSetObj_5HTCtx object
            which references a .csv containing full ExpRef information for
            each recording.
            - dset_ind is an index within this dataset specifying
            recording number.
            - ch_img specifies which channel (1 or 2) tiff to
            load for the recording.
        2. Load by manually inputting folders related to the recording.
        (Specify an enclosing_folder, a folder_beh, a folder_img,
        and a fname_img.)
            - enclosing_folder is a string specifying the ExpRef folder
            (ie 'Data/MBLXXX/2025-XX-XX/')
            - folder_beh is a string specifying the behavior folder,
            referenced from the enclosing_folder location.
            (ie '1' or '2')
            - folder_img is a string specifying the imaging folder,
            referenced from the enclosing_folder location.
            (ie 'TwoP/2025-06-17_t-001')
            - fname_img is a string specifying the imaging name
            within folder_img
            (ie '2025-06-17_t-003_Cycle00001_Ch2.tif')

        Other parameters
        ----------------
        trial_end : None or int
            Optional parameter specifying the index of the last trial
            to analyze.
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

        Object layout
        -------------
        .rec : np.ndarray
            Raw version of the .tiff file (memory-mapped), with dimensions
            (t, pix_x, pix_y).
        .rec_t : np.ndarray
            Vector with timestamps for the t dimension of .rec, in sec.
        .beh : SimpleNamespace | Behavior data for task
            .stim
            .rew
            .lick
        .ops : SimpleNamespace | stores all relevant options for 
        """

        # setup names of folders and files
        # --------------
        self.folder = SimpleNamespace()
        self.path = SimpleNamespace()
        if dset_obj is None:
            self.folder.enclosing = enclosing_folder
            self.folder.img = folder_img
            self.folder.beh = folder_beh
            print(f'loading {self.folder.enclosing}...\n---------------')

        elif 'DSetObj' in str(type(dset_obj)):
            self.dset_obj = dset_obj
            self.folder.enclosing = self.dset_obj.get_path_expref(dset_ind)
            self.folder.img = self.dset_obj.get_path_img(dset_ind)
            self.folder.beh = self.dset_obj.get_path_beh(dset_ind)
            print(f'loading {self.dset_obj.get_path_expref(dset_ind)}...')
            print('---------------')

        self.path.raw = pathlib.Path(self.folder.enclosing)
        self.path.animal = self.path.raw.parts[-2]
        self.path.date = self.path.raw.parts[-1]
        self.path.beh_folder = self.folder.beh
        self.ch_img = ch_img

        os.chdir(self.folder.enclosing)
        check_folder_exists('figs_mbl')  # to store all figs
        self.ops = SimpleNamespace()
        self.ops.n_px_remove_sides = n_px_remove_sides
        self.ops.rec_type = rec_type

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

        # compute trial indices for each trtype
        # -------------------
        self.beh.tr_inds = {}
        self.beh.tr_inds['0'] = np.where(self.beh.stim.prob == 0)[0]
        self.beh.tr_inds['0.5'] = np.where(self.beh.stim.prob == 0.5)[0]
        self.beh.tr_inds['1'] = np.where(self.beh.stim.prob == 1)[0]
        self.beh.tr_inds['0.5_rew'] = np.where(np.logical_and(
            self.beh.stim.prob == 0.5, self.beh.rew.delivered == True))[0]
        self.beh.tr_inds['0.5_norew'] = np.where(np.logical_and(
            self.beh.stim.prob == 0.5, self.beh.rew.delivered == False))[0]

        # remove trinds outside of correct range
        for tr_cond in ['0', '0.5', '1', '0.5_rew', '0.5_norew']:
            self.beh.tr_inds[tr_cond] = np.delete(
                self.beh.tr_inds[tr_cond],
                ~np.isin(self.beh.tr_inds[tr_cond],
                         np.arange(
                             self.beh._stimrange.first,
                             self.beh._stimrange.last+1)))
        # add lickrates
        # ----------------
        self.add_lickrates()

        self.beh.tr_inds['0.5_prelick'] = np.where(np.logical_and(
            self.beh.stim.prob == 0.5, self.beh.lick.antic_raw > 0))[0]
        self.beh.tr_inds['0.5_noprelick'] = np.where(np.logical_and(
            self.beh.stim.prob == 0.5, self.beh.lick.antic_raw == 0))[0]
        for tr_cond in ['0.5_prelick', '0.5_noprelick']:
            self.beh.tr_inds[tr_cond] = np.delete(
                self.beh.tr_inds[tr_cond],
                ~np.isin(self.beh.tr_inds[tr_cond],
                         np.arange(
                             self.beh._stimrange.first,
                             self.beh._stimrange.last+1)))
        return

    def add_lickrates(self, t_prestim=2,
                      t_poststim=5,
                      bl_norm=False):
        """
        Adds a lickrate segregated by trial-type
        """

        # add base/antic lickrates for all trialtypes
        n_trials = self.beh.stim.t_start.shape[0]

        self.beh.lick.t = np.empty(n_trials,
                                   dtype=np.ndarray)
        self.beh.lick.antic_raw = np.empty(n_trials,
                                           dtype=np.ndarray)
        self.beh.lick.base_raw = np.empty(n_trials,
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
            self.beh.lick.antic_raw[trial] = (_licks_antic
                                              / (_t_rew - _t_stim))
            _licks_base = np.sum(np.logical_and(
                _licks < 0, _licks > -4))
            self.beh.lick.base_raw[trial] = (_licks_base
                                             / (t_prestim - 0))

        # segregate lickrates per trialtype
        self.beh.lick.antic = {}
        self.beh.lick.base = {}
        for trial_type in ['0', '0.5', '1', '0.5_rew', '0.5_norew']:
            self.beh.lick.antic[trial_type] = np.zeros(
                self.beh.tr_inds[trial_type].shape[0])
            self.beh.lick.base[trial_type] = np.zeros(
                self.beh.tr_inds[trial_type].shape[0])

            for ind_trial, trial in enumerate(self.beh.tr_inds[trial_type]):
                if bl_norm is True:
                    _antic_licks = self.beh.lick.antic_raw[trial] \
                        - self.beh.lick.base_raw[trial]
                elif bl_norm is False:
                    _antic_licks = self.beh.lick.antic_raw[trial]

                self.beh.lick.antic[trial_type][ind_trial] \
                    = _antic_licks
                self.beh.lick.base[trial_type][ind_trial] \
                    = self.beh.lick.base_raw[trial]

        return

    def add_frame(self, t_pre=2, t_post=5):

        """
        Plots the average fluorescence across the whole fov,
        separated by trial-type (eg 0%, 50%, 100% rewarded trials)

        plot_type controls the things plotted.
            None: plots 0, 0.5, 1
            rew_norew: plots 0.5_rew, 0.5_norew
            prelick_noprelick: plots 0.5_prelick, 0.5_noprelick

        """
        print('creating trial-averaged signal (whole-frame)...')

        # setup
        self.add_lickrates()

        self.frame = SimpleNamespace()
        self.frame.params = SimpleNamespace()
        self.frame.params.t_rew_pre = t_pre
        self.frame.params.t_rew_post = t_post

        # setup stim-aligned traces
        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int((2 + t_post) * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        self.frame.dff = {}
        self.frame.dff['0'] = np.zeros((
            self.beh.tr_inds['0'].shape[0], n_frames_tot))
        self.frame.dff['0.5'] = np.zeros((
            self.beh.tr_inds['0.5'].shape[0], n_frames_tot))
        self.frame.dff['1'] = np.zeros((
            self.beh.tr_inds['1'].shape[0], n_frames_tot))

        self.frame.dff['0.5_rew'] = np.zeros((
            self.beh.tr_inds['0.5_rew'].shape[0], n_frames_tot))
        self.frame.dff['0.5_norew'] = np.zeros((
            self.beh.tr_inds['0.5_norew'].shape[0], n_frames_tot))
        self.frame.dff['0.5_prelick'] = np.zeros((
            self.beh.tr_inds['0.5_prelick'].shape[0], n_frames_tot))
        self.frame.dff['0.5_noprelick'] = np.zeros((
            self.beh.tr_inds['0.5_noprelick'].shape[0], n_frames_tot))

        self.frame.t = np.linspace(-1*t_pre, 2+t_post,
                                   n_frames_tot)

        # store stim-aligned trace
        self.frame.tr_counts = {'0': 0, '0.5': 0, '1': 0,
                                '0.5_rew': 0, '0.5_norew': 0,
                                '0.5_prelick': 0, '0.5_noprelick': 0}
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
                self.rec_t - _t_end)) + 1  # add frames to end

            # extract fluorescence
            _f = np.mean(np.mean(
                self.rec[_ind_t_start:_ind_t_end, :, :], axis=1), axis=1)
            _dff = calc_dff(_f, baseline_frames=n_frames_pre)

            if self.beh.stim.prob[trial] == 0:
                self.frame.dff['0'][
                    self.frame.tr_counts['0'], :] \
                    = _dff[0:n_frames_tot]
                self.frame.tr_counts['0'] += 1
            elif self.beh.stim.prob[trial] == 0.5:
                self.frame.dff['0.5'][
                    self.frame.tr_counts['0.5'], :] \
                    = _dff[0:n_frames_tot]
                self.frame.tr_counts['0.5'] += 1
            elif self.beh.stim.prob[trial] == 1:
                self.frame.dff['1'][
                    self.frame.tr_counts['1'], :] \
                    = _dff[0:n_frames_tot]
                self.frame.tr_counts['1'] += 1

            if self.beh.stim.prob[trial] == 0.5 \
               and self.beh.rew.delivered[trial] == 1:
                self.frame.dff['0.5_rew'][
                    self.frame.tr_counts['0.5_rew'], :] \
                    = _dff[0:n_frames_tot]
                self.frame.tr_counts['0.5_rew'] += 1
            elif self.beh.stim.prob[trial] == 0.5 \
                 and self.beh.rew.delivered[trial] == 0:
                self.frame.dff['0.5_norew'][
                    self.frame.tr_counts['0.5_norew'], :] \
                    = _dff[0:n_frames_tot]
                self.frame.tr_counts['0.5_norew'] += 1

            if self.beh.stim.prob[trial] == 0.5 \
               and self.beh.lick.antic_raw[trial] > 0:
                self.frame.dff['0.5_prelick'][
                    self.frame.tr_counts['0.5_prelick'], :] \
                    = _dff[0:n_frames_tot]
                self.frame.tr_counts['0.5_prelick'] += 1
            elif self.beh.stim.prob[trial] == 0.5 \
                 and self.beh.lick.antic_raw[trial] == 0:
                self.frame.dff['0.5_noprelick'][
                    self.frame.tr_counts['0.5_noprelick'], :] \
                    = _dff[0:n_frames_tot]
                self.frame.tr_counts['0.5_noprelick'] += 1

    def add_sectors(self,
                    n_sectors=10,
                    t_pre=2, t_post=5,
                    n_null=500,
                    resid_type='sector',
                    resid_corr_tr_cond='1',
                    colors=sns.cubehelix_palette(
                        n_colors=3,
                        start=2, rot=0,
                        dark=0.2, light=0.8)):
        """
        Divides the field of view into sectors, and plots a set of trial-types
        separately within each sector.

        plot_type controls the type of plot:
            if plot_type is None:
                stim_list = ['0', '0.5', '1']
            elif plot_type == 'rew_norew':
                stim_list = ['0.5_rew', '0.5_norew']
            elif plot_type == 'prelick_noprelick':
                stim_list = ['0.5_prelick', '0.5_noprelick']
            elif plot_type == 'rew':
                stim_list = ['0', '0.5_rew', '1']
        """
        # compute whole-frame avg in case needed
        if not hasattr(self, 'frame'):
            self.add_frame(t_pre=t_pre, t_post=t_post)

        print('creating trial-averaged signal (sectors)...')
        self.add_lickrates(t_prestim=t_pre,
                           t_poststim=t_post)

        self.sector = SimpleNamespace()

        self.sector.params = SimpleNamespace()
        self.sector.n_sectors = n_sectors
        self.sector.params.t_rew_pre = t_pre
        self.sector.params.t_rew_post = t_post
        self.sector.n_sectors = n_sectors
        self.sector.n_null = n_null

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

        # setup stim-aligned traces
        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int((2 + t_post) * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        self.sector.dff = np.empty((n_sectors, n_sectors), dtype=dict)
        self.sector.dff_resid = np.empty((n_sectors, n_sectors), dtype=dict)

        self.sector.t = np.linspace(
            -1*t_pre, 2+t_post, n_frames_tot)

        # store edges of sectors
        self.sector.x = SimpleNamespace()
        self.sector.x.lower = np.zeros((n_sectors, n_sectors), dtype=int)
        self.sector.x.upper = np.zeros((n_sectors, n_sectors), dtype=int)
        self.sector.y = SimpleNamespace()
        self.sector.y.lower = np.zeros((n_sectors, n_sectors), dtype=int)
        self.sector.y.upper = np.zeros((n_sectors, n_sectors), dtype=int)

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
                self.sector.dff_resid[n_x, n_y] = {}

                for tr_cond in ['0', '0.5', '1', '0.5_rew',
                                '0.5_norew', '0.5_prelick', '0.5_noprelick']:
                    self.sector.dff[n_x, n_y][tr_cond] = np.zeros((
                        self.beh.tr_inds[tr_cond].shape[0], n_frames_tot))
                    self.sector.dff_resid[n_x, n_y][tr_cond] = np.zeros_like(
                        self.sector.dff[n_x, n_y][tr_cond])

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
                       and self.beh.lick.antic_raw[trial] > 0:
                        self.sector.dff[n_x, n_y]['0.5_prelick'][
                            self.sector.tr_counts['0.5_prelick'], :] \
                            = _dff[0:n_frames_tot]
                        self.sector.tr_counts['0.5_prelick'] += 1
                    elif self.beh.stim.prob[trial] == 0.5 \
                         and self.beh.lick.antic_raw[trial] == 0:
                        self.sector.dff[n_x, n_y]['0.5_noprelick'][
                            self.sector.tr_counts['0.5_noprelick'], :] \
                            = _dff[0:n_frames_tot]
                        self.sector.tr_counts['0.5_noprelick'] += 1

                # store dff residuals
                # ----------
                for tr_cond in ['0', '0.5', '1']:
                    for tr in range(
                            self.sector.dff[n_x, n_y][tr_cond].shape[0]):
                        if resid_type == 'sector':
                            _dff_mean = np.mean(
                                self.sector.dff[n_x, n_y][tr_cond], axis=0)
                        elif resid_type == 'trial':
                            _dff_mean = self.frame.dff[tr_cond][tr, :]

                        self.sector.dff_resid[n_x, n_y][tr_cond][tr, :] \
                            = self.sector.dff[n_x, n_y][tr_cond][tr, :] \
                            - _dff_mean

                # update _n_trace
                _n_trace += 1

        print('\n')
        # compute and plot cross-correlations
        self.sector.resid_corr_stat = np.zeros(
            (self.sector.n_sectors**2, self.sector.n_sectors**2))
        self.sector.resid_corr_pval = np.zeros(
            (self.sector.n_sectors**2, self.sector.n_sectors**2))

        for sec_1 in range(self.sector.n_sectors**2):
            for sec_2 in range(self.sector.n_sectors**2):
                sec_1_x, sec_1_y = np.divmod(sec_1, self.sector.n_sectors)
                sec_2_x, sec_2_y = np.divmod(sec_2, self.sector.n_sectors)

                _sec_1_sig = np.mean(
                    self.sector.dff_resid[sec_1_x, sec_1_y][resid_corr_tr_cond],
                    axis=1)
                _sec_2_sig = np.mean(
                    self.sector.dff_resid[sec_2_x, sec_2_y][resid_corr_tr_cond],
                    axis=1)

                _corr = sp.stats.pearsonr(_sec_1_sig, _sec_2_sig)
                self.sector.resid_corr_stat[sec_1, sec_2] = _corr.statistic
                self.sector.resid_corr_pval[sec_1, sec_2] = _corr.pvalue

        # add null distributions
        # -------------
        self.add_null_dists(n_null=n_null, t_pre=t_pre, t_post=t_post)

        return

    def add_neurs(self, t_pre=1, t_post=2, zscore=True):
        self.add_lickrates(t_prestim=t_pre,
                           t_poststim=t_post)

        print('adding aligned neur avg...')
        # setup structure
        # ---------------
        neurs = np.where(self.neur.iscell[:, 0] == 1)[0]
        n_neurs = neurs.shape[0]

        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int((2 + t_post) * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

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
            print(f'\tneur={int(neur)}', end='\r')
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
                self.beh.tr_inds['0'].shape[0], n_frames_tot))
            self.neur.dff_aln[ind]['0.5'] = np.zeros((
                self.beh.tr_inds['0.5'].shape[0], n_frames_tot))
            self.neur.dff_aln[ind]['1'] = np.zeros((
                self.beh.tr_inds['1'].shape[0], n_frames_tot))

            self.neur.dff_aln[ind]['0.5_rew'] = np.zeros((
                self.beh.tr_inds['0.5_rew'].shape[0], n_frames_tot))
            self.neur.dff_aln[ind]['0.5_norew'] = np.zeros((
                self.beh.tr_inds['0.5_norew'].shape[0], n_frames_tot))

            self.neur.dff_aln[ind]['0.5_prelick'] = np.zeros((
                self.beh.tr_inds['0.5_prelick'].shape[0],
                n_frames_tot))
            self.neur.dff_aln[ind]['0.5_noprelick'] = np.zeros((
                self.beh.tr_inds['0.5_noprelick'].shape[0],
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
                   and self.beh.lick.antic_raw[trial] > 0:
                    self.neur.dff_aln[ind]['0.5_prelick'][
                        _tr_counts['0.5_prelick'], :] \
                        = _dff[0:n_frames_tot]
                    _tr_counts['0.5_prelick'] += 1
                elif self.beh.stim.prob[trial] == 0.5 \
                     and self.beh.lick.antic_raw[trial] == 0:
                    self.neur.dff_aln[ind]['0.5_noprelick'][
                        _tr_counts['0.5_noprelick'], :] \
                        = _dff[0:n_frames_tot]
                    _tr_counts['0.5_noprelick'] += 1

            for tr_type in ['0', '0.5', '1', '0.5_rew',
                            '0.5_norew', '0.5_prelick',
                            '0.5_noprelick']:
                self.neur.dff_aln_mean[tr_type][ind, :] = \
                    np.mean(self.neur.dff_aln[ind][tr_type], axis=0)
        print('')
        return

    def add_psychometrics(self, t_pre=5, t_post=2):
        if not hasattr(self, 'sector'):
            self.add_sectors(t_pre=t_pre,
                             t_post=t_post)
        if not hasattr(self.neur, 'dff_aln'):
            self.add_neurs(t_pre=t_pre,
                           t_post=t_post)

        print('adding psychometrics...')

        # set up psychometrics attribute and variables
        # -------------
        print('\tsetting up attributes...')
        n_sec = self.sector.n_sectors
        self.psychometrics = SimpleNamespace(
            grab=SimpleNamespace(),
            grab_null=SimpleNamespace(),
            grab_frame=SimpleNamespace(),
            neur=SimpleNamespace(),
            lick=SimpleNamespace())
        self.psychometrics.p_rew = ['0', '0.5', '1']

        for item in ['grab', 'grab_frame', 'neur', 'lick']:
            setattr(getattr(self.psychometrics, item),
                    'stim_resp', {'0': [], '0.5': [], '1': []})
            setattr(getattr(self.psychometrics, item),
                    'stim_resp_raw', {
                        '0': np.empty(n_sec**2,
                                      dtype=np.ndarray),
                        '0.5': np.empty(n_sec**2,
                                        dtype=np.ndarray),
                        '1': np.empty(n_sec**2,
                                      dtype=np.ndarray)})
        self.psychometrics.grab_null.stim_resp = {
            '0': np.empty(n_sec**2, dtype=np.ndarray),
            '0.5': np.empty(n_sec**2, dtype=np.ndarray),
            '1': np.empty(n_sec**2, dtype=np.ndarray)}
        self.psychometrics.grab_null.stim_resp_raw = {
            '0': np.empty(n_sec**2, dtype=np.ndarray),
            '0.5': np.empty(n_sec**2, dtype=np.ndarray),
            '1': np.empty(n_sec**2, dtype=np.ndarray)}

        _ind_stim_start = np.argmin(np.abs(self.sector.t - 0))
        _ind_stim_end = np.argmin(np.abs(self.sector.t - 2))
        dt = self.sector.t[1] - self.sector.t[0]

        # extract licking data
        # -------------
        print('\textracting lick data...')
        for trial_cond in ['0', '0.5', '1']:
            self.psychometrics.lick.stim_resp[trial_cond] = np.mean(
                self.beh.lick.antic[trial_cond])

        # extract GRAB full-frame data
        # ---------------
        print('\textracting GRAB full-frame data...')
        for trial_cond in ['0', '0.5', '1']:
            self.psychometrics.grab_frame.stim_resp[trial_cond] = np.mean(
                sp.integrate.trapezoid(
                    self.frame.dff[trial_cond]
                    [:, _ind_stim_start:_ind_stim_end], dx=dt, axis=1), axis=0)
            self.psychometrics.grab_frame.stim_resp_raw[trial_cond] \
                = sp.integrate.trapezoid(
                    self.frame.dff[trial_cond]
                    [:, _ind_stim_start:_ind_stim_end], dx=dt, axis=1)

        # extract GRAB sector data
        # -----------
        print('\textracting GRAB sector data...')
        # setup sector.stim_resp for storing per-trial stim responses
        self.sector.stim_resp = {}
        for tr_cond in ['0', '0.5', '1']:
            self.sector.stim_resp[tr_cond] = np.zeros((
                self.beh.tr_inds[tr_cond].shape[0],
                self.sector.n_sectors, self.sector.n_sectors))

        _n_sec = 0
        for n_x in range(self.sector.n_sectors):
            for n_y in range(self.sector.n_sectors):
                print(f'\t\tsector {_n_sec}/{self.sector.n_sectors**2}...',
                      end='\r')
                for trial_cond in ['0', '0.5', '1']:

                    # regular GRAB
                    # --------
                    _dff_raw = self.sector.dff[n_x, n_y][trial_cond]
                    _stim_resp = sp.integrate.trapezoid(
                           _dff_raw[:, _ind_stim_start:_ind_stim_end], axis=1,
                           dx=dt)

                    self.psychometrics.grab.stim_resp_raw[
                        trial_cond][_n_sec] = _stim_resp
                    self.sector.stim_resp[trial_cond][:, n_x, n_y] \
                        = _stim_resp

                    self.psychometrics.grab.stim_resp[trial_cond].append(
                       np.mean(_stim_resp))

                    # GRAB null
                    # ---------
                    self.psychometrics.grab_null.stim_resp[trial_cond][
                        _n_sec] = []

                    for ind_null_sim in range(self.sector.n_null):
                        _dff = np.mean(
                            self.sector_null.dff[n_x, n_y][trial_cond][
                                ind_null_sim, :, :], axis=0)
                        _mean_stim_resp = sp.integrate.trapezoid(
                            _dff[_ind_stim_start:_ind_stim_end], dx=dt)
                        self.psychometrics.grab_null.stim_resp[
                            trial_cond][_n_sec].append(
                                _mean_stim_resp)

                        # _dff_raw = self.sector_null.dff[n_x, n_y][trial_cond]
                        # self.psychometrics.grab_null.stim_resp[
                        #     trial_cond][_n_sec].append(
                        #         np.mean(np.mean(
                        #             _dff_raw[:, _ind_stim_start:_ind_stim_end]),
                        #                 axis=0), axis=0)

                _n_sec += 1

        # extract neur (GCaMP) data
        # --------------
        print('\textracting neur data...')
        n_neurs = self.neur.dff_aln.shape[0]
        self.neur.stim_resp = {}
        for tr_cond in ['0', '0.5', '1']:
            self.neur.stim_resp[tr_cond] = np.zeros((
                self.beh.tr_inds[tr_cond].shape[0],
                n_neurs))

        for neur in range(n_neurs):
            print(f'\t\tneur {neur}/{n_neurs}...',
                  end='\r')
            for trial_cond in ['0', '0.5', '1']:
                _dff_raw = self.neur.dff_aln_mean[trial_cond][neur, :]
                _mean_stim_resp = np.mean(
                    _dff_raw[_ind_stim_start:_ind_stim_end])
                self.psychometrics.neur.stim_resp[trial_cond].append(
                    _mean_stim_resp)

                self.neur.stim_resp[trial_cond][:, neur] \
                    = sp.integrate.trapezoid(
                        self.neur.dff_aln[neur][trial_cond]
                        [:, _ind_stim_start:_ind_stim_end], dx=dt, axis=1)

        # compute psychometrics
        # -----------
        print('\tcomputing optimism...')

        # **** GRAB *****
        # calculation on whole-frame GRAB (all trials)
        _raw = {
            '0-0.5': np.array(self.psychometrics.grab_frame.stim_resp['0.5'])
            - np.array(self.psychometrics.grab_frame.stim_resp['0']),
            '0.5-1': np.array(self.psychometrics.grab_frame.stim_resp['1'])
            - np.array(self.psychometrics.grab_frame.stim_resp['0.5']),
            '0-1': np.array(self.psychometrics.grab_frame.stim_resp['1'])
            - np.array(self.psychometrics.grab_frame.stim_resp['0'])}
        _optimism = ((_raw['0-0.5']) / (_raw['0-0.5'] + _raw['0.5-1'])) - 0.5
        self.psychometrics.grab_frame.optimism = _optimism

        # calculation on sector-by-sector GRAB (all trials)
        _raw = {
            '0-0.5': np.array(self.psychometrics.grab.stim_resp['0.5'])
            - np.array(self.psychometrics.grab.stim_resp['0']),
            '0.5-1': np.array(self.psychometrics.grab.stim_resp['1'])
            - np.array(self.psychometrics.grab.stim_resp['0.5']),
            '0-1': np.array(self.psychometrics.grab.stim_resp['1'])
            - np.array(self.psychometrics.grab.stim_resp['0'])}
        _optimism = ((_raw['0-0.5']) / (_raw['0-0.5'] + _raw['0.5-1'])) - 0.5
        self.psychometrics.grab.optimism = _optimism

        # calculation on sector-by-sector GRAB (trial split)
        self.psychometrics.grab.optimism_trsplit_a = []
        self.psychometrics.grab.optimism_trsplit_b = []

        for sector in range(self.sector.n_sectors**2):
            _raw_trsplit_a = {
                '0-0.5': np.mean(self.psychometrics.grab.stim_resp_raw['0.5'][
                    sector][::2])
                - np.mean(self.psychometrics.grab.stim_resp_raw['0'][
                    sector][::2]),
                '0.5-1': np.mean(self.psychometrics.grab.stim_resp_raw['1'][
                    sector][::2])
                - np.mean(self.psychometrics.grab.stim_resp_raw['0.5'][
                    sector][::2])}
            _raw_trsplit_b = {
                '0-0.5': np.mean(self.psychometrics.grab.stim_resp_raw['0.5'][
                    sector][1::2])
                - np.mean(self.psychometrics.grab.stim_resp_raw['0'][
                    sector][1::2]),
                '0.5-1': np.mean(self.psychometrics.grab.stim_resp_raw['1'][
                    sector][1::2])
                - np.mean(self.psychometrics.grab.stim_resp_raw['0.5'][
                    sector][1::2])}
            _optimism_trsplit_a = (
                (_raw_trsplit_a['0-0.5']) / (_raw_trsplit_a['0-0.5']
                                             + _raw_trsplit_a['0.5-1'])) - 0.5
            _optimism_trsplit_b = (
                (_raw_trsplit_b['0-0.5']) / (_raw_trsplit_b['0-0.5']
                                             + _raw_trsplit_b['0.5-1'])) - 0.5

            self.psychometrics.grab.optimism_trsplit_a.append(_optimism_trsplit_a)
            self.psychometrics.grab.optimism_trsplit_b.append(_optimism_trsplit_b)

        self.psychometrics.grab.optimism_trsplit_a = np.array(
            self.psychometrics.grab.optimism_trsplit_a)
        self.psychometrics.grab.optimism_trsplit_b = np.array(
            self.psychometrics.grab.optimism_trsplit_b)

        # ******** GRAB null ********
        self.psychometrics.grab_null.optimism = np.empty(
            self.sector.n_sectors**2, dtype=np.ndarray)
        for sector in range(self.sector.n_sectors**2):
            _raw = {
                '0-0.5': np.array(
                    self.psychometrics.grab_null.stim_resp['0.5'][sector])
                - np.array(
                    self.psychometrics.grab_null.stim_resp['0'][sector]),
                '0.5-1': np.array(
                    self.psychometrics.grab.stim_resp['1'][sector])
                - np.array(
                    self.psychometrics.grab.stim_resp['0.5'][sector]),
                '0-1': np.array(
                    self.psychometrics.grab.stim_resp['1'][sector])
                - np.array(
                    self.psychometrics.grab.stim_resp['0'][sector])}
            _optimism = ((_raw['0-0.5']) / (_raw['0-0.5'] + _raw['0.5-1'])) - 0.5
            self.psychometrics.grab_null.optimism[sector] = _optimism

        # ******* GCaMP *******
        _raw = {
            '0-0.5': np.array(self.psychometrics.neur.stim_resp['0.5'])
            - np.array(self.psychometrics.neur.stim_resp['0']),
            '0.5-1': np.array(self.psychometrics.neur.stim_resp['1'])
            - np.array(self.psychometrics.neur.stim_resp['0.5']),
            '0-1': np.array(self.psychometrics.neur.stim_resp['1'])
            - np.array(self.psychometrics.neur.stim_resp['0'])}
        _optimism = ((_raw['0-0.5']) / (_raw['0-0.5'] + _raw['0.5-1'])) - 0.5
        self.psychometrics.neur.optimism = _optimism

        # ***** lick *****
        _raw = {
            '0-0.5': np.array(self.psychometrics.lick.stim_resp['0.5'])
            - np.array(self.psychometrics.lick.stim_resp['0']),
            '0.5-1': np.array(self.psychometrics.lick.stim_resp['1'])
            - np.array(self.psychometrics.lick.stim_resp['0.5']),
            '0-1': np.array(self.psychometrics.lick.stim_resp['1'])
            - np.array(self.psychometrics.lick.stim_resp['0'])}
        _optimism = ((_raw['0-0.5']) / (_raw['0-0.5'] + _raw['0.5-1'])) - 0.5

        self.psychometrics.lick.optimism = _optimism

        return

    def plt_optimism(self,
                     markersize=20,
                     violin_color=sns.xkcd_rgb['moss green'],
                     zetatest_mask=False,
                     sig_thresh=0.01,
                     fontsize=10,
                     ylims=[-2, 2],
                     plt_show=True):

        # run some checks
        # --------
        if not hasattr(self, 'psychometrics'):
            self.add_psychometrics()
        if zetatest_mask is True and not hasattr(self.sector, 'zeta_pval'):
            self.add_zetatest_sectors()
            self.add_zetatest_neurs()

        # setup
        # ---------
        ind_sig_lower = (sig_thresh *
                         self.psychometrics.grab_null.optimism[0].shape[0])
        ind_sig_upper = (self.psychometrics.grab_null.optimism[0].shape[0]
                         - (sig_thresh *
                            self.psychometrics.grab_null.optimism[0].shape[0]))

        print('plotting optimism...')
        # plot optimism/pessimism, null distribution and significance
        # ----------------
        # setup figures
        fig_distrl = plt.figure(figsize=(6, 3))
        spec = gs.GridSpec(nrows=2, ncols=2,
                           height_ratios=[0.05, 0.95],
                           figure=fig_distrl)
        ax_distrl_grab_signif = fig_distrl.add_subplot(spec[0, 0])
        ax_distrl_grab_signif.set_yticks([])
        ax_distrl_grab = fig_distrl.add_subplot(
            spec[1, 0], sharex=ax_distrl_grab_signif)

        ax_distrl_neur_signif = fig_distrl.add_subplot(spec[0, 1])
        ax_distrl_neur_signif.set_yticks([])
        ax_distrl_neur = fig_distrl.add_subplot(
            spec[1, 1], sharex=ax_distrl_neur_signif)

        fig_distrl_reliability = plt.figure(figsize=(3, 3))
        ax_distrl_reliability = fig_distrl_reliability.add_subplot()

        fig_img = plt.figure(figsize=(6, 2))
        spec_img = gs.GridSpec(nrows=1, ncols=3, figure=fig_img)
        ax_img_grab = fig_img.add_subplot(spec_img[0, 0])
        ax_img_neur = fig_img.add_subplot(spec_img[0, 1])
        ax_img_grab_neur = fig_img.add_subplot(spec_img[0, 2])

        # plot all
        # ---------------
        n_sec = self.sector.n_sectors
        rec_px = self.rec.shape[1]

        if zetatest_mask is False:
            optimism = self.psychometrics.grab.optimism
            sort_args_grab = np.argsort(optimism)
            optimism_neur = self.psychometrics.neur.optimism
            sort_args_neur = np.argsort(optimism_neur)

            # plot images
            # ---------
            if ylims is None:
                vmax = np.max(np.abs(optimism))
                vmin = -1 * vmax
            else:
                vmax = ylims[1]
                vmin = ylims[0]

            ax_img_grab.imshow(
                optimism.reshape((n_sec, n_sec)),
                vmin=vmin, vmax=vmax,
                cmap='coolwarm')
            ax_img_grab_neur.imshow(
                optimism.reshape((n_sec, n_sec)),
                vmin=vmin, vmax=vmax,
                cmap='coolwarm')
            ax_img_neur.scatter((self.neur.x*(n_sec/rec_px)-0.5),
                                -1*(self.neur.y*(n_sec/rec_px)-0.5),
                                s=40,
                                c=self.psychometrics.neur.optimism,
                                edgecolors='black', cmap='coolwarm',
                                vmin=vmin, vmax=vmax)
            ax_img_grab_neur.scatter(self.neur.x*(n_sec/rec_px)-0.5,
                                     self.neur.y*(n_sec/rec_px)-0.5,
                                     s=40,
                                     c=self.psychometrics.neur.optimism,
                                     edgecolors='black', cmap='coolwarm',
                                     vmin=vmin, vmax=vmax)
            # ax_img_neur.set_xlim([-0.5, 9.5])
            # ax_img_neur.set_ylim([-0.5, 9.5])

            # plot distributional rl
            # ------------------
            ax_distrl_grab.scatter(np.arange(optimism.shape[0]),
                                   optimism[sort_args_grab],
                                   s=markersize,
                                   c=sns.xkcd_rgb['black'])
            ax_distrl_neur.scatter(np.arange(optimism_neur.shape[0]),
                                   optimism_neur[sort_args_neur],
                                   s=markersize,
                                   c=sns.xkcd_rgb['black'])
            ax_distrl_grab.axhline(self.psychometrics.lick.optimism,
                                   linestyle='dashed',
                                   color=sns.xkcd_rgb['azure'],
                                   linewidth=0.5)
            ax_distrl_neur.axhline(self.psychometrics.lick.optimism,
                                   linestyle='dashed',
                                   color=sns.xkcd_rgb['azure'],
                                   linewidth=0.5)

            for ind, sort_arg in enumerate(sort_args_grab):
                # test for significance: grab
                _sorted_null = np.sort(
                    self.psychometrics.grab_null.optimism[sort_arg])
                _rank = np.argmin(np.abs(
                    _sorted_null - optimism[sort_arg]))
                if _rank <= ind_sig_lower or _rank >= ind_sig_upper:
                    ax_distrl_grab_signif.text(
                        ind, 0, '*', ha='center',
                        va='bottom', fontsize=fontsize)

                # plot
                parts = ax_distrl_grab.violinplot(
                    self.psychometrics.grab_null.optimism[sort_arg],
                    [ind], widths=1,
                    showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(violin_color)

            # plt reliability
            # ---------------
            ax_distrl_reliability.scatter(
                self.psychometrics.grab.optimism_trsplit_a,
                self.psychometrics.grab.optimism_trsplit_b)
            _pearsonr = sp_stats.pearsonr(
                self.psychometrics.grab.optimism_trsplit_a,
                self.psychometrics.grab.optimism_trsplit_b)
            _linreg = sp_stats.linregress(
                self.psychometrics.grab.optimism_trsplit_a,
                self.psychometrics.grab.optimism_trsplit_b)
            _line_x = np.arange(ylims[0], ylims[1], 0.1)
            # _line_y = _linreg.slope * _line_x + _linreg.intercept
            ax_distrl_reliability.plot(_line_x, _line_x,
                                       linestyle='dotted',
                                       color='black')
            ax_distrl_reliability.set_title(
                f'optimism reliability (p={_pearsonr.pvalue:.4f})')

            ax_distrl_reliability.set_xlim(ylims)
            ax_distrl_reliability.set_ylim(ylims)
            ax_distrl_reliability.set_xlabel('optimism (first half of data)')
            ax_distrl_reliability.set_xlabel('optimism (second half of data)')

        elif zetatest_mask is True:
            zetatest_grab_inds = np.where(self.sector.zeta_pval < 0.05)[0]
            optimism_grab = self.psychometrics.grab.optimism[
                zetatest_grab_inds]
            sort_args_grab = np.argsort(optimism_grab)

            optimism_grab_toplt = self.psychometrics.grab.optimism
            optimism_grab_toplt[np.where(self.sector.zeta_pval > 0.05)[0]] = None

            zetatest_neur_inds = np.where(self.neur.zeta_pval < 0.05)[0]
            optimism_neur = self.psychometrics.neur.optimism[
                zetatest_neur_inds]
            sort_args_neur = np.argsort(optimism_neur)

            optimism_neur_toplt = self.psychometrics.neur.optimism[zetatest_neur_inds]
            optimism_neur_x = self.neur.x[zetatest_neur_inds]
            optimism_neur_y = self.neur.y[zetatest_neur_inds]

            # plot images
            # ---------
            if ylims is None:
                vmax = np.max(np.abs(optimism_grab))
                vmin = -1 * vmax
            else:
                vmax = ylims[1]
                vmin = ylims[0]

            ax_img_grab.imshow(
                optimism_grab_toplt.reshape((n_sec, n_sec)),
                vmin=vmin, vmax=vmax,
                cmap='coolwarm')
            ax_img_grab_neur.imshow(
                optimism_grab_toplt.reshape((n_sec, n_sec)),
                vmin=vmin, vmax=vmax,
                cmap='coolwarm')

            ax_img_neur.scatter(optimism_neur_x*(n_sec/rec_px)-0.5,
                                -1 * optimism_neur_y*(n_sec/rec_px)-0.5,
                                s=40,
                                c=optimism_neur_toplt,
                                edgecolors='black', cmap='coolwarm',
                                vmin=vmin, vmax=vmax)

            ax_img_grab_neur.scatter(optimism_neur_x*(n_sec/rec_px)-0.5,
                                     optimism_neur_y*(n_sec/rec_px)-0.5,
                                     s=40,
                                     c=optimism_neur_toplt,
                                     edgecolors='black', cmap='coolwarm',
                                     vmin=vmin, vmax=vmax)

            # ax_img_neur.set_xlim([-0.5, 9.5])
            # ax_img_neur.set_ylim([-0.5, 9.5])
            # plot distributional rl
            # ------------------
            ax_distrl_grab.scatter(np.arange(optimism_grab.shape[0]),
                                   optimism_grab[sort_args_grab],
                                   s=markersize,
                                   c=sns.xkcd_rgb['black'])
            ax_distrl_neur.scatter(np.arange(optimism_neur.shape[0]),
                                   optimism_neur[sort_args_neur],
                                   s=markersize,
                                   c=sns.xkcd_rgb['black'])
            ax_distrl_grab.axhline(self.psychometrics.lick.optimism,
                                   linestyle='dashed',
                                   color=sns.xkcd_rgb['azure'],
                                   linewidth=0.5)
            ax_distrl_neur.axhline(self.psychometrics.lick.optimism,
                                   linestyle='dashed',
                                   color=sns.xkcd_rgb['azure'],
                                   linewidth=0.5)

            for ind, sort_arg in enumerate(sort_args_grab):
                # test for significance: grab
                _ind_in_full_grab = zetatest_grab_inds[sort_arg]
                _sorted_null = np.sort(
                    self.psychometrics.grab_null.optimism[_ind_in_full_grab])
                _rank = np.argmin(np.abs(
                    _sorted_null - optimism_grab[sort_arg]))
                if _rank <= ind_sig_lower or _rank >= ind_sig_upper:
                    ax_distrl_grab_signif.text(
                        ind, 0, '*', ha='center',
                        va='bottom', fontsize=fontsize)

                # plot
                parts = ax_distrl_grab.violinplot(
                    self.psychometrics.grab_null.optimism[_ind_in_full_grab],
                    [ind], widths=1,
                    showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(violin_color)

            # plt reliability
            # ---------------
            ax_distrl_reliability.scatter(
                self.psychometrics.grab.optimism_trsplit_a[zetatest_grab_inds],
                self.psychometrics.grab.optimism_trsplit_b[zetatest_grab_inds])
            _pearsonr = sp_stats.pearsonr(
                self.psychometrics.grab.optimism_trsplit_a[zetatest_grab_inds],
                self.psychometrics.grab.optimism_trsplit_b[zetatest_grab_inds])
            _line_x = np.arange(ylims[0], ylims[1], 0.1)
            # _line_y = _linreg.slope * _line_x + _linreg.intercept
            ax_distrl_reliability.plot(_line_x, _line_x,
                                       linestyle='dotted',
                                       color='black')
            ax_distrl_reliability.set_title(
                f'optimism reliability (p={_pearsonr.pvalue:.4f})')

            ax_distrl_reliability.set_xlim(ylims)
            ax_distrl_reliability.set_ylim(ylims)
            ax_distrl_reliability.set_xlabel('optimism (first half of data)')
            ax_distrl_reliability.set_xlabel('optimism (second half of data)')

            # vmax = np.nanmax(np.abs(optimism_toplt))
            # vmin = -1 * vmax
            # ax_img_grab.imshow(
            #     optimism_toplt.reshape((self.sector.n_sectors,
            #                             self.sector.n_sectors)),
            #     vmin=vmin, vmax=vmax,
            #     cmap='coolwarm')

            # ax_distrl_grab.scatter(np.arange(optimism_masked.shape[0]),
            #                        optimism_masked[sort_args],
            #                        s=markersize,
            #                        c=sns.xkcd_rgb['black'])

            # for ind, sort_arg in enumerate(sort_args):
            #     _ind_in_full_grab = zetatest_mask_inds[sort_arg]

            #     # test for significance
            #     _sorted_null = np.sort(
            #         self.psychometrics.grab_null.optimism[_ind_in_full_grab])
            #     _rank = np.argmin(np.abs(
            #         _sorted_null - optimism_masked[sort_arg]))
            #     if _rank <= ind_sig_lower or _rank >= ind_sig_upper:
            #         ax_distrl_grab_signif.text(ind, 0, '*', ha='center',
            #                        va='bottom', fontsize=fontsize)

            #     # plot
            #     parts = ax_distrl_grab.violinplot(
            #         self.psychometrics.grab_null.optimism[
            #             _ind_in_full_grab], [ind], widths=1,
            #         showextrema=False)
            #     for pc in parts['bodies']:
            #         pc.set_facecolor(violin_color)

        if ylims is not None:
            ax_distrl_grab.set_ylim(ylims)
            ax_distrl_neur.set_ylim(ylims)
            ax_distrl_reliability.set_xlim(ylims)
            ax_distrl_reliability.set_ylim(ylims)

        ax_distrl_grab.axhline(0, linestyle='dotted',
                               color='k',
                               linewidth=1.5)
        ax_distrl_neur.axhline(0, linestyle='dotted',
                               color='k',
                               linewidth=1.5)

        # save figs
        fig_distrl.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{zetatest_mask=}_{ylims=}_optimism_distrl.pdf'))
        fig_distrl_reliability.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{zetatest_mask=}_{ylims=}_optimism_distrl_reliability.pdf'))
        fig_img.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{zetatest_mask=}_{ylims=}_optimism_spatial.pdf'))

        if plt_show is True:
            plt.show()

        return

    def add_null_dists(self, n_null=100, auto_run_methods=True,
                       t_pre=2, t_post=5):
        """
        Makes a 'null' distribution based on the alternate hypothesis
        of no spatial variability in task-related serotonin
        signaling, but instead spatial variability in 1) GRAB sensor
        expression; and 2) imaging-related noise.

        Reconstructs a null version of each sector's response as a
        modified version of the whole-frame response, scaled up/down

        - Must have run plt_stim_aligned_sectors first
        """
        print('creating null distributions')

        # check that sector and frame exist
        print('\trunning checks')

        if not hasattr(self, 'frame'):
            if auto_run_methods is False:
                print('Must have run self.add_frame() first!')
                return
            elif auto_run_methods is True:
                self.add_frame(t_pre=t_pre,
                               t_post=t_post)

        if not hasattr(self, 'sector'):
            if auto_run_methods is False:
                print('Must have run self.add_sectors() first!')
                return
            elif auto_run_methods is True:
                self.add_sectors(t_pre=t_pre,
                                 t_post=t_post)

        # construct the null distribution
        self.sector_null = SimpleNamespace()
        self.sector_null.dff = np.empty(
            (self.sector.n_sectors, self.sector.n_sectors),
            dtype=dict)
        self.sector_null.t = self.sector.t

        _ind_bl_start = 0
        _ind_bl_end = np.argmin(np.abs(self.sector.t-0))
        ampli_frame = np.max(np.mean(self.frame.dff['1'], axis=0))

        print('\tcreating synthetic traces for each sector')
        _n_trace = 0
        for n_x in range(self.sector.n_sectors):
            for n_y in range(self.sector.n_sectors):
                print(f'\t\tsector {_n_trace}/{self.sector.n_sectors**2}...')

                # setup structure
                self.sector_null.dff[n_x, n_y] = {}
                for tr_cond in ['0', '0.5', '1', '0.5_rew', '0.5_norew']:
                    self.sector_null.dff[n_x, n_y][tr_cond] = np.zeros((
                        n_null,
                        self.sector.dff[n_x, n_y][tr_cond].shape[0],
                        self.sector.dff[n_x, n_y][tr_cond].shape[1]))

                # compute scaling variables
                _ampli_sector = np.max(np.mean(
                    self.sector.dff[n_x, n_y]['1'], axis=0))
                _ampli_scaling = _ampli_sector / ampli_frame

                _bl_std_tr = np.empty(0)
                for tr in range(self.sector.dff[n_x, n_y]['1'].shape[0]):
                    _bl_std_tr = np.append(
                        _bl_std_tr,
                        np.std(self.sector.dff[n_x, n_y]['1'][
                            tr, _ind_bl_start:_ind_bl_end]))
                _bl_std = np.mean(_bl_std_tr)

                # generate null traces
                for tr_cond in ['0', '0.5', '1', '0.5_rew', '0.5_norew']:
                    _tr_inds = self.beh.tr_inds[tr_cond]
                    for tr_ind_rel, tr_ind_abs in enumerate(_tr_inds):
                        _frame = self.frame.dff[tr_cond][tr_ind_rel, :]
                        for ind_null in range(n_null):
                            _noise = np.random.normal(
                                scale=_bl_std,
                                size=(_frame.shape[0]))
                            _dff = (_frame * _ampli_scaling) + _noise
                            self.sector_null.dff[n_x, n_y][tr_cond][
                                ind_null, tr_ind_rel, :] = _dff

                _n_trace += 1

        return

    def add_zetatest_neurs(self, frametime_post=2,
                           zeta_type='2samp'):

        print('adding zetatest for neurs...')
        # setup structure and calculate on/off times
        inds_neur = np.where(self.neur.iscell[:, 0] == 1)[0]
        n_neurs = inds_neur.shape[0]

        framerate = 1 / (self.neur.t[1] - self.neur.t[0])
        frames_post = np.ceil(frametime_post * framerate)

        t_on_0 = self.beh.stim.t_start[self.beh.tr_inds['0']]
        t_onoff_0 = np.tile(t_on_0, (2, 1)).T
        t_onoff_0[:, 1] += 2

        t_on_1 = self.beh.stim.t_start[self.beh.tr_inds['1']]
        t_onoff_1 = np.tile(t_on_1, (2, 1)).T
        t_onoff_1[:, 1] += 2

        self.neur.zeta_pval = np.zeros(n_neurs)
        self.neur.mean_pval = np.zeros(n_neurs)
        for ind_neur_rel, ind_neur in enumerate(inds_neur):
            print(f'\tneur={int(ind_neur)}...  ', end='\r')
            _t = self.neur.t
            _f = self.neur.f[ind_neur, :]

            if zeta_type == '2samp':
                _zeta = zetapy.zetatstest2(_t, _f, t_onoff_0,
                                           _t, _f, t_onoff_1,
                                           dblUseMaxDur=frames_post)
            elif zeta_type == '1samp':
                _zeta = zetapy.zetatstest(_t, _f, t_onoff_1,
                                          dblUseMaxDur=frames_post)

            self.neur.zeta_pval[ind_neur_rel] = _zeta[1]['dblZetaP']
            self.neur.mean_pval[ind_neur_rel] = _zeta[1]['dblMeanP']
        print('')

        return

    def add_zetatest_sectors(self, frametime_post=2,
                             zeta_type='2samp'):

        if not hasattr(self, 'sector'):
            self.add_sectors()

        print('adding zetatest for sectors...')
        # setup structure and calculate on/off times
        n_sectors = self.sector.n_sectors**2

        framerate = 1 / (self.neur.t[1] - self.neur.t[0])
        frames_post = np.ceil(frametime_post * framerate)

        t_on_0 = self.beh.stim.t_start[self.beh.tr_inds['0']]
        t_onoff_0 = np.tile(t_on_0, (2, 1)).T
        t_onoff_0[:, 1] += 2

        t_on_1 = self.beh.stim.t_start[self.beh.tr_inds['1']]
        t_onoff_1 = np.tile(t_on_1, (2, 1)).T
        t_onoff_1[:, 1] += 2

        self.sector.zeta_pval = np.zeros(n_sectors)
        self.sector.mean_pval = np.zeros(n_sectors)
        _n_sec = 0
        for n_x in range(self.sector.n_sectors):
            for n_y in range(self.sector.n_sectors):
                print(f'\t\tsector {_n_sec}/{n_sectors}...      ',
                      end='\r')

                _t = self.rec_t
                _f = np.mean(np.mean(
                    self.rec[:,
                             self.sector.x.lower[n_x, n_y]:
                             self.sector.x.upper[n_x, n_y],
                             self.sector.y.lower[n_x, n_y]:
                             self.sector.y.upper[n_x, n_y]],
                    axis=1), axis=1)

                if zeta_type == '2samp':
                    _zeta = zetapy.zetatstest2(_t, _f, t_onoff_0,
                                               _t, _f, t_onoff_1,
                                               dblUseMaxDur=frames_post)
                elif zeta_type == '1samp':
                    _zeta = zetapy.zetatstest(_t, _f, t_onoff_1,
                                              dblUseMaxDur=frames_post)

                self.sector.zeta_pval[_n_sec] = _zeta[1]['dblZetaP']
                self.sector.mean_pval[_n_sec] = _zeta[1]['dblMeanP']
                _n_sec += 1
        print('')

        return

    def plt_null_dist(self,
                      figsize=(6, 3),
                      img_ds_factor=50,
                      img_alpha=0.5,
                      plt_dff={'x': {'gain': 0.6,
                                     'offset': 0.2},
                               'y': {'gain': 10,
                                     'offset': 0.2}}):
        fig = plt.figure(figsize=figsize)
        spec = gs.GridSpec(nrows=1, ncols=2,
                           figure=fig)

        ax_dff = fig.add_subplot(spec[0, 0])
        ax_dff.set_title('dff')
        ax_dff_null = fig.add_subplot(spec[0, 1])
        ax_dff_null.set_title('dff null')

        n_sectors = self.sector.n_sectors

        print('\tcreating max projection image...')
        _rec_max = np.max(self.rec[::img_ds_factor, :, :], axis=0)
        _rec_max[:, ::int(_rec_max.shape[0]/n_sectors)] = 0
        _rec_max[::int(_rec_max.shape[0]/n_sectors), :] = 0

        ax_dff.imshow(_rec_max,
                      extent=[0, n_sectors,
                              n_sectors, 0],
                      alpha=img_alpha)
        ax_dff_null.imshow(_rec_max,
                           extent=[0, n_sectors,
                                   n_sectors, 0],
                           alpha=img_alpha)

        for n_x in range(n_sectors):
            for n_y in range(n_sectors):
                # **************************
                # plotting for this sector
                # **************************

                # make sector plot of stim-aligned traces
                # ----------
                _t_sector = ((self.sector.t / self.sector.t[-1])
                             * plt_dff['x']['gain'])
                _t_sector = _t_sector + n_x + plt_dff['x']['offset']

                for stim_cond in ['0', '0.5', '1']:
                    _dff_mean = np.mean(self.sector.dff[n_x, n_y][stim_cond],
                                        axis=0)
                    _dff_shifted = (-1 * _dff_mean * plt_dff['y']['gain']) \
                        + n_y - plt_dff['y']['offset'] + 1
                    ax_dff.plot(
                        _t_sector, _dff_shifted,
                        color=self.sector.colors[stim_cond],
                        linestyle=self.sector.linestyle[stim_cond])

                    _dff_null_mean = np.mean(
                        self.sector_null.dff[n_x, n_y][stim_cond][0, :, :],
                        axis=0)
                    _dff_null_shifted = (-1 * _dff_null_mean
                                         * plt_dff['y']['gain']) \
                                         + n_y - plt_dff['y']['offset'] + 1
                    ax_dff_null.plot(
                        _t_sector, _dff_null_shifted,
                        color=self.sector.colors[stim_cond],
                        linestyle=self.sector.linestyle[stim_cond])

        fig.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'null_dists.pdf'))

        plt.show()

        return

    def plt_all_trial_resp(self, vmax_neurs=10):
        """
        Plots each trial's response separately
        """

        # GRAB responses
        # -------------
        figs_grab = {}
        specs_grab = {}
        axs_grab = {}

        vmax = np.max(np.abs(self.sector.stim_resp['1']))
        for tr_type in ['0', '0.5', '1']:
            n_trials = self.beh.tr_inds[tr_type].shape[0]
            nrows = np.ceil(np.sqrt(n_trials)).astype(int)

            figs_grab[tr_type] = plt.figure(figsize=(6, 6))
            specs_grab[tr_type] = gs.GridSpec(nrows=nrows, ncols=nrows,
                                              figure=figs_grab[tr_type])
            axs_grab[tr_type] = []

            # plot each trial
            for ind_tr, tr in enumerate(range(n_trials)):
                # setup axis
                _tr_x, _tr_y = np.divmod(ind_tr, nrows)
                if ind_tr == 0:
                    axs_grab[tr_type].append(
                        figs_grab[tr_type].add_subplot(
                            specs_grab[tr_type][_tr_x, _tr_y]))
                elif ind_tr > 0:
                    axs_grab[tr_type].append(figs_grab[tr_type].add_subplot(
                        specs_grab[tr_type][_tr_x, _tr_y],
                        sharex=axs_grab[tr_type][0],
                        sharey=axs_grab[tr_type][0]))

                # plot axis
                axs_grab[tr_type][ind_tr].imshow(
                    self.sector.stim_resp[tr_type][ind_tr, :, :],
                    vmax=vmax, vmin=-1*vmax, cmap='coolwarm')

            figs_grab[tr_type].suptitle(f'stim response, p_rew={tr_type}')
            figs_grab[tr_type].savefig(os.path.join(
                os.getcwd(), 'figs_mbl',
                f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
                + f'_sector_resp_by_trial_prew={tr_type}.pdf'))

        # neur responses
        # -------------
        figs_neur = {}
        specs_neur = {}
        axs_neur = {}

        n_sec = self.sector.n_sectors
        rec_px = self.rec.shape[1]

        if vmax_neurs is None:
            vmax_neurs = np.max(np.abs(self.neur.stim_resp['1']))

        for tr_type in ['0', '0.5', '1']:
            n_trials = self.beh.tr_inds[tr_type].shape[0]
            nrows = np.ceil(np.sqrt(n_trials)).astype(int)

            figs_neur[tr_type] = plt.figure(figsize=(6, 6))
            specs_neur[tr_type] = gs.GridSpec(nrows=nrows, ncols=nrows,
                                              figure=figs_neur[tr_type])
            axs_neur[tr_type] = []

            # plot each trial
            for ind_tr, tr in enumerate(range(n_trials)):
                # setup axis
                _tr_x, _tr_y = np.divmod(ind_tr, nrows)
                if ind_tr == 0:
                    axs_neur[tr_type].append(
                        figs_neur[tr_type].add_subplot(
                            specs_neur[tr_type][_tr_x, _tr_y]))
                elif ind_tr > 0:
                    axs_neur[tr_type].append(figs_neur[tr_type].add_subplot(
                        specs_neur[tr_type][_tr_x, _tr_y],
                        sharex=axs_neur[tr_type][0],
                        sharey=axs_neur[tr_type][0]))

                # plot axis
                axs_neur[tr_type][ind_tr].scatter(
                    self.neur.x*(n_sec/rec_px)-0.5,
                    self.neur.y*(n_sec/rec_px)-0.5,
                    s=40, c=self.neur.stim_resp[tr_type][ind_tr, :],
                    vmax=vmax_neurs, vmin=-1*vmax_neurs, cmap='coolwarm',
                    edgecolors='black')

            axs_neur[tr_type][0].set_xlim([-0.5, 9.5])
            axs_neur[tr_type][0].set_ylim([-0.5, 9.5])
            axs_neur[tr_type][0].yaxis.set_inverted(True)

            figs_neur[tr_type].suptitle(f'stim response, p_rew={tr_type}')
            figs_neur[tr_type].savefig(os.path.join(
                os.getcwd(), 'figs_mbl',
                f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
                + f'_neur_resp_by_trial_prew={tr_type}.pdf'))


        # show
        # -------
        plt.show()

        return

    def plt_frame(self,
                  figsize=(3.43, 2),
                  t_pre=2, t_post=5,
                  colors=sns.cubehelix_palette(
                      n_colors=3,
                      start=2, rot=0,
                      dark=0.1, light=0.6),
                  plot_type=None,
                  plt_show=True):

        """
        Plots the average fluorescence across the whole fov,
        separated by trial-type (eg 0%, 50%, 100% rewarded trials)

        plot_type controls the things plotted.
            None: plots 0, 0.5, 1
            rew_norew: plots 0.5_rew, 0.5_norew
            prelick_noprelick: plots 0.5_prelick, 0.5_noprelick

        """
        print('creating trial-averaged signal (whole-frame)...')

        # setup
        self.add_lickrates()

        self.frame = SimpleNamespace()
        self.frame.params = SimpleNamespace()
        self.frame.params.t_rew_pre = t_pre
        self.frame.params.t_rew_post = t_post

        colors = {'0': colors[0],
                  '0.5': colors[1],
                  '1': colors[2],
                  '0.5_rew': colors[1],
                  '0.5_norew': colors[1],
                  '0.5_prelick': colors[1],
                  '0.5_noprelick': colors[1]}
        linestyles = {'0': 'solid',
                      '0.5': 'solid',
                      '1': 'solid',
                      '0.5_rew': 'solid',
                      '0.5_norew': 'dashed',
                      '0.5_prelick': 'solid',
                      '0.5_noprelick': 'dashed'}

        fig_avg = plt.figure(figsize=figsize)
        spec = gs.GridSpec(nrows=1, ncols=1,
                           figure=fig_avg)
        ax_trace = fig_avg.add_subplot(spec[0, 0])

        # setup stim-aligned traces
        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int((2 + t_post) * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        self.frame.dff = {}
        self.frame.dff['0'] = np.zeros((
            self.beh.tr_inds['0'].shape[0], n_frames_tot))
        self.frame.dff['0.5'] = np.zeros((
            self.beh.tr_inds['0.5'].shape[0], n_frames_tot))
        self.frame.dff['1'] = np.zeros((
            self.beh.tr_inds['1'].shape[0], n_frames_tot))

        self.frame.dff['0.5_rew'] = np.zeros((
            self.beh.tr_inds['0.5_rew'].shape[0], n_frames_tot))
        self.frame.dff['0.5_norew'] = np.zeros((
            self.beh.tr_inds['0.5_norew'].shape[0], n_frames_tot))
        self.frame.dff['0.5_prelick'] = np.zeros((
            self.beh.tr_inds['0.5_prelick'].shape[0], n_frames_tot))
        self.frame.dff['0.5_noprelick'] = np.zeros((
            self.beh.tr_inds['0.5_noprelick'].shape[0], n_frames_tot))

        self.frame.t = np.linspace(-1*t_pre, 2+t_post,
                                   n_frames_tot)

        # store stim-aligned trace
        self.frame.tr_counts = {'0': 0, '0.5': 0, '1': 0,
                                '0.5_rew': 0, '0.5_norew': 0,
                                '0.5_prelick': 0, '0.5_noprelick': 0}
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
                self.rec_t - _t_end)) + 1  # add frames to end

            # extract fluorescence
            _f = np.mean(np.mean(
                self.rec[_ind_t_start:_ind_t_end, :, :], axis=1), axis=1)
            _dff = calc_dff(_f, baseline_frames=n_frames_pre)

            if self.beh.stim.prob[trial] == 0:
                self.frame.dff['0'][
                    self.frame.tr_counts['0'], :] \
                    = _dff[0:n_frames_tot]
                self.frame.tr_counts['0'] += 1
            elif self.beh.stim.prob[trial] == 0.5:
                self.frame.dff['0.5'][
                    self.frame.tr_counts['0.5'], :] \
                    = _dff[0:n_frames_tot]
                self.frame.tr_counts['0.5'] += 1
            elif self.beh.stim.prob[trial] == 1:
                self.frame.dff['1'][
                    self.frame.tr_counts['1'], :] \
                    = _dff[0:n_frames_tot]
                self.frame.tr_counts['1'] += 1

            if self.beh.stim.prob[trial] == 0.5 \
               and self.beh.rew.delivered[trial] == 1:
                self.frame.dff['0.5_rew'][
                    self.frame.tr_counts['0.5_rew'], :] \
                    = _dff[0:n_frames_tot]
                self.frame.tr_counts['0.5_rew'] += 1
            elif self.beh.stim.prob[trial] == 0.5 \
                 and self.beh.rew.delivered[trial] == 0:
                self.frame.dff['0.5_norew'][
                    self.frame.tr_counts['0.5_norew'], :] \
                    = _dff[0:n_frames_tot]
                self.frame.tr_counts['0.5_norew'] += 1

            if self.beh.stim.prob[trial] == 0.5 \
               and self.beh.lick.antic_raw[trial] > 0:
                self.frame.dff['0.5_prelick'][
                    self.frame.tr_counts['0.5_prelick'], :] \
                    = _dff[0:n_frames_tot]
                self.frame.tr_counts['0.5_prelick'] += 1
            elif self.beh.stim.prob[trial] == 0.5 \
                 and self.beh.lick.antic_raw[trial] == 0:
                self.frame.dff['0.5_noprelick'][
                    self.frame.tr_counts['0.5_noprelick'], :] \
                    = _dff[0:n_frames_tot]
                self.frame.tr_counts['0.5_noprelick'] += 1

        # plot stim-aligned traces
        if plt_show is True:
            if plot_type is None:
                stim_conds = ['0', '0.5', '1']
            elif plot_type == 'rew_norew':
                stim_conds = ['0.5_rew', '0.5_norew']
            elif plot_type == 'prelick_noprelick':
                stim_conds = ['0.5_prelick', '0.5_noprelick']

            for stim_cond in stim_conds:
                ax_trace.plot(self.frame.t,
                              np.mean(self.frame.dff[stim_cond], axis=0),
                              color=colors[stim_cond],
                              linestyle=linestyles[stim_cond])
                ax_trace.fill_between(
                    self.frame.t,
                    np.mean(self.frame.dff[stim_cond], axis=0) -
                    np.std(self.frame.dff[stim_cond], axis=0),
                    np.mean(self.frame.dff[stim_cond], axis=0) +
                    np.std(self.frame.dff[stim_cond], axis=0),
                    facecolor=colors[stim_cond],
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
                + f'{plot_type=}_{t_pre=}_{t_post=}_'
                + f'_ch={self.ch_img}_mean_trial_activity.pdf'))

            plt.show()

    def plt_sectors(self,
                    n_sectors=10,
                    t_pre=2, t_post=5,
                    plot_type=None,
                    resid_type='sector',
                    resid_tr_cond='1',
                    resid_alpha=0.5,
                    resid_smooth=True,
                    resid_smooth_windlen=5,
                    resid_smooth_polyorder=3,
                    resid_trial_nbins=30,
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

        plot_type controls the type of plot:
            if plot_type is None:
                stim_list = ['0', '0.5', '1']
            elif plot_type == 'rew_norew':
                stim_list = ['0.5_rew', '0.5_norew']
            elif plot_type == 'prelick_noprelick':
                stim_list = ['0.5_prelick', '0.5_noprelick']
            elif plot_type == 'rew':
                stim_list = ['0', '0.5_rew', '1']
        """
        print('creating trial-averaged signal (sectors)...')
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
        fig_resid = plt.figure(figsize=figsize)
        spec_avg = gs.GridSpec(nrows=1, ncols=1,
                               figure=fig_avg)
        spec_resid = gs.GridSpec(nrows=1, ncols=1,
                                 figure=fig_resid)

        ax_img = fig_avg.add_subplot(spec_avg[0, 0])
        ax_img_resid = fig_resid.add_subplot(spec_resid[0, 0])

        print('\tcreating max projection image...')
        _rec_max = np.max(self.rec[::img_ds_factor, :, :], axis=0)
        _rec_max[:, ::int(_rec_max.shape[0]/n_sectors)] = 0
        _rec_max[::int(_rec_max.shape[0]/n_sectors), :] = 0

        ax_img.imshow(_rec_max,
                      extent=[0, n_sectors,
                              n_sectors, 0],
                      alpha=img_alpha)
        ax_img_resid.imshow(_rec_max,
                            extent=[0, n_sectors,
                                    n_sectors, 0],
                            alpha=img_alpha)
        ax_img_resid.set_title(f'residuals vs mean({resid_type}),'
                               + f' p(rew)={resid_tr_cond}')

        # setup stim-aligned traces
        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int((2 + t_post) * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        self.sector.dff = np.empty((n_sectors, n_sectors), dtype=dict)
        self.sector.dff_resid = np.empty((n_sectors, n_sectors), dtype=dict)

        self.sector.t = np.linspace(
            -1*t_pre, 2+t_post, n_frames_tot)

        # store edges of sectors
        self.sector.x = SimpleNamespace()
        self.sector.x.lower = np.zeros((n_sectors, n_sectors))
        self.sector.x.upper = np.zeros((n_sectors, n_sectors))
        self.sector.y = SimpleNamespace()
        self.sector.y.lower = np.zeros((n_sectors, n_sectors))
        self.sector.y.upper = np.zeros((n_sectors, n_sectors))

        # compute whole-frame avg in case needed
        if not hasattr(self, 'frame'):
            self.plt_stim_aligned_avg(t_pre=t_pre, t_post=t_post, plt_show=False)

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
                self.sector.dff_resid[n_x, n_y] = {}

                for tr_cond in ['0', '0.5', '1', '0.5_rew',
                                '0.5_norew', '0.5_prelick', '0.5_noprelick']:
                    self.sector.dff[n_x, n_y][tr_cond] = np.zeros((
                        self.beh.tr_inds[tr_cond].shape[0], n_frames_tot))
                    self.sector.dff_resid[n_x, n_y][tr_cond] = np.zeros_like(
                        self.sector.dff[n_x, n_y][tr_cond])

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
                       and self.beh.lick.antic_raw[trial] > 0:
                        self.sector.dff[n_x, n_y]['0.5_prelick'][
                            self.sector.tr_counts['0.5_prelick'], :] \
                            = _dff[0:n_frames_tot]
                        self.sector.tr_counts['0.5_prelick'] += 1
                    elif self.beh.stim.prob[trial] == 0.5 \
                         and self.beh.lick.antic_raw[trial] == 0:
                        self.sector.dff[n_x, n_y]['0.5_noprelick'][
                            self.sector.tr_counts['0.5_noprelick'], :] \
                            = _dff[0:n_frames_tot]
                        self.sector.tr_counts['0.5_noprelick'] += 1

                # store dff residuals
                # ----------
                for tr_cond in ['0', '0.5', '1']:
                    for tr in range(
                            self.sector.dff[n_x, n_y][tr_cond].shape[0]):
                        if resid_type == 'sector':
                            _dff_mean = np.mean(
                                self.sector.dff[n_x, n_y][tr_cond], axis=0)
                        elif resid_type == 'trial':
                            _dff_mean = self.frame.dff[tr_cond][tr, :]

                        self.sector.dff_resid[n_x, n_y][tr_cond][tr, :] \
                            = self.sector.dff[n_x, n_y][tr_cond][tr, :] \
                            - _dff_mean

                # **************************
                # plotting for this sector
                # **************************

                # make sector plot of stim-aligned traces
                # ----------
                _t_sector = ((self.sector.t / self.sector.t[-1])
                             * plt_dff['x']['gain'])
                _t_sector = _t_sector + n_x + plt_dff['x']['offset']

                if plot_type is None:
                    stim_list = ['0', '0.5', '1']
                elif plot_type == 'rew_norew':
                    stim_list = ['0.5_rew', '0.5_norew']
                elif plot_type == 'prelick_noprelick':
                    stim_list = ['0.5_prelick', '0.5_noprelick']
                elif plot_type == 'rew':
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

                # make residuals plot
                # ------------------------
                _n_tr_resid = self.sector.dff_resid[n_x, n_y][
                    resid_tr_cond].shape[0]
                for trial in range(_n_tr_resid):
                    _dff = self.sector.dff_resid[n_x, n_y][resid_tr_cond][trial, :]
                    if resid_smooth is True:
                        _dff = sp.signal.savgol_filter(_dff, resid_smooth_windlen,
                                                       resid_smooth_polyorder)
                    _dff_shifted = (-1 * _dff * plt_dff['y']['gain']) \
                        + n_y - plt_dff['y']['offset'] + 1

                    ax_img_resid.plot(
                        _t_sector, _dff_shifted,
                        color=sns.color_palette('rocket', _n_tr_resid)[trial],
                        alpha=resid_alpha,
                        linestyle=self.sector.linestyle[stim_cond])

                # plot rew and stim lines
                # ------------
                _t_stim = (0 * plt_dff['x']['gain']) \
                    + n_x + plt_dff['x']['offset']
                _t_rew = ((2 / self.sector.t[-1])
                          * plt_dff['x']['gain']) \
                    + n_x + plt_dff['x']['offset']
                _dff_zero = n_y - plt_dff['y']['offset'] + 1

                for ax in [ax_img, ax_img_resid]:
                    ax.plot([_t_stim, _t_stim],
                            [n_y + 0.2, n_y + 0.8],
                            color=sns.xkcd_rgb['dark grey'],
                            linewidth=1,
                            linestyle='dashed')
                    ax.plot([_t_rew, _t_rew],
                            [n_y + 0.2, n_y + 0.8],
                            color=sns.xkcd_rgb['bright blue'],
                            linewidth=1,
                            linestyle='dashed')

                ax_img_resid.plot([_t_stim, _t_rew],
                                  [_dff_zero, _dff_zero],
                                  color='k',
                                  linewidth=1,
                                  linestyle='dashed')

                # update _n_trace
                _n_trace += 1

        # compute supplemental plots on data in self.sector.dff_resid
        # ---------------
        n_trs = self.sector.dff_resid[0, 0][resid_tr_cond].shape[0]
        n_rows = np.ceil(np.sqrt(n_trs)).astype(int)

        fig_norm_sec_sig = plt.figure(figsize=(3.43, 1.5))
        spec_norm_sec_sig = gs.GridSpec(nrows=1, ncols=3,
                                        figure=fig_norm_sec_sig)
        ax_norm_sec_sig_prew = {}
        ax_norm_sec_sig_prew['0'] = fig_norm_sec_sig.add_subplot(
            spec_norm_sec_sig[0, 0])
        ax_norm_sec_sig_prew['0.5'] = fig_norm_sec_sig.add_subplot(
            spec_norm_sec_sig[0, 1])
        ax_norm_sec_sig_prew['1'] = fig_norm_sec_sig.add_subplot(
            spec_norm_sec_sig[0, 2])

        fig_resid_corr = plt.figure(figsize=figsize)
        spec_resid_corr = gs.GridSpec(nrows=1, ncols=1,
                                      figure=fig_resid_corr)
        ax_resid_corr = fig_resid_corr.add_subplot(spec_resid_corr[0, 0])

        fig_resid_corr_spatial = plt.figure(figsize=figsize)
        spec_resid_corr_spatial = gs.GridSpec(
            nrows=self.sector.n_sectors,
            ncols=self.sector.n_sectors,
            figure=fig_resid_corr_spatial)

        fig_resid_tr_sep = plt.figure(figsize=figsize)
        spec_resid_tr_sep = gs.GridSpec(nrows=n_rows,
                                        ncols=n_rows,
                                        figure=fig_resid_tr_sep)

        # compute and plot trial-averaged, normalized signals for each sector
        for sec in range(self.sector.n_sectors):
            sec_x, sec_y = np.divmod(sec, self.sector.n_sectors)

            for tr_type in ['0', '0.5', '1']:
                _n_tr = self.sector.dff[0, 0][tr_type].shape[0]
                _sec_max = np.max(self.sector.dff[sec_x, sec_y][tr_type],
                                  axis=1)
                _sec_sig = np.array([self.sector.dff[sec_x, sec_y][tr_type][
                    tr, :] / _sec_max[tr] for tr in range(_n_tr)])
                _sec_sig_travg = np.mean(_sec_sig, axis=0)
                ax_norm_sec_sig_prew[tr_type].plot(
                    self.sector.t,
                    _sec_sig_travg,
                    color=self.sector.colors[tr_type],
                    alpha=0.8)

        # compute and plot cross-correlations
        self.sector.resid_corr_stat = np.zeros(
            (self.sector.n_sectors**2, self.sector.n_sectors**2))
        self.sector.resid_corr_pval = np.zeros(
            (self.sector.n_sectors**2, self.sector.n_sectors**2))

        for sec_1 in range(self.sector.n_sectors**2):
            for sec_2 in range(self.sector.n_sectors**2):
                sec_1_x, sec_1_y = np.divmod(sec_1, self.sector.n_sectors)
                sec_2_x, sec_2_y = np.divmod(sec_2, self.sector.n_sectors)

                _sec_1_sig = np.mean(
                    self.sector.dff_resid[sec_1_x, sec_1_y][resid_tr_cond],
                    axis=1)
                _sec_2_sig = np.mean(
                    self.sector.dff_resid[sec_2_x, sec_2_y][resid_tr_cond],
                    axis=1)

                _corr = sp.stats.pearsonr(_sec_1_sig, _sec_2_sig)
                self.sector.resid_corr_stat[sec_1, sec_2] = _corr.statistic
                self.sector.resid_corr_pval[sec_1, sec_2] = _corr.pvalue

        ax_resid_corr.imshow(self.sector.resid_corr_stat)

        ax_residcorr_spatial = []
        ax_count = 0
        for sec_1 in range(self.sector.n_sectors):
            for sec_2 in range(self.sector.n_sectors):
                ax_residcorr_spatial.append(
                    fig_resid_corr_spatial.add_subplot(
                        spec_resid_corr_spatial[sec_1, sec_2]))
                ax_residcorr_spatial[ax_count].imshow(
                    self.sector.resid_corr_stat[ax_count, :].reshape(
                        self.sector.n_sectors, self.sector.n_sectors),
                    vmax=1, vmin=-1, cmap='coolwarm')
                ax_residcorr_spatial[ax_count].set_xticks([])
                ax_residcorr_spatial[ax_count].set_yticks([])

                ax_count += 1

        fig_resid_corr_spatial.suptitle('residual corrs (spatial), '
                                        + f' {resid_tr_cond=},'
                                        + f' residuals vs mean({resid_type})')

        # compute and plot per-trial residual distributions
        ax = []
        for tr in range(n_trs):
            # compile response of all sectors for this trial
            _sector_dff_resid = []
            for sec_1 in range(self.sector.n_sectors):
                for sec_2 in range(self.sector.n_sectors):
                    _sector_dff_resid.append(np.mean(
                        self.sector.dff_resid[sec_1, sec_2][
                            resid_tr_cond][tr, :]))

            # set up axis
            _tr_x, _tr_y = np.divmod(tr, n_rows)
            if tr == 0:
                ax.append(fig_resid_tr_sep.add_subplot(
                    spec_resid_tr_sep[_tr_x, _tr_y]))
            elif tr > 0:
                ax.append(fig_resid_tr_sep.add_subplot(
                    spec_resid_tr_sep[_tr_x, _tr_y],
                          sharex=ax[0], sharey=ax[0]))

            # plot per-trial distributions
            ax[tr].hist(_sector_dff_resid, bins=resid_trial_nbins,
                        histtype='step',
                        density=True, color=sns.xkcd_rgb['dark grey'])
            ax[tr].axvline(0, color='k',
                           linewidth=1,
                           linestyle='dashed')
            ax[tr].set_xlabel('dff resid.')
            ax[tr].set_ylabel('pdf')
            ax[tr].set_title(f'trial={tr}')
        fig_resid_tr_sep.suptitle(f'residuals vs mean{resid_type} corrs (spatial), '
                                        + f' {resid_tr_cond=},'
                                        + f' residuals vs mean({resid_type})')

        # save all figs
        # -------------
        fig_avg.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            'sectors_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{plt_prefix}_'
            + f'{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}_'
            + f'ch={self.ch_img}.pdf'))

        fig_norm_sec_sig.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            'sectors_normed_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{plt_prefix}_'
            + f'{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}_'
            + f'ch={self.ch_img}.pdf'))

        fig_resid.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            'sectors_resid_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{plt_prefix}_'
            + f'{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}_'
            + f'{resid_tr_cond=}_{resid_type=}_'
            + f'ch={self.ch_img}.pdf'))

        fig_resid_corr.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            'sectors_resid_corr_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{plt_prefix}_'
            + f'{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}_'
            + f'{resid_tr_cond=}_{resid_type=}_'
            + f'ch={self.ch_img}.pdf'))

        fig_resid_corr_spatial.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            'sectors_resid_corr_spatial_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{plt_prefix}_'
            + f'{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}_'
            + f'{resid_tr_cond=}_{resid_type=}_'
            + f'ch={self.ch_img}.pdf'))

        fig_resid_tr_sep.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            'sectors_resid_trial_sep_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{plt_prefix}_'
            + f'{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}_'
            + f'{resid_tr_cond=}_{resid_type=}_'
            + f'ch={self.ch_img}.pdf'))

        # show all plots
        if plt_show is True:
            plt.show()
        elif plt_show is False:
            for _fig in [fig_avg, fig_norm_sec_sig, fig_resid,
                         fig_resid_corr, fig_resid_corr_spatial,
                         fig_resid_tr_sep]:
                plt.close(_fig)

        return

    def plt_neurs(self, t_pre=1, t_post=2,
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
                   and self.beh.lick.antic_raw[trial] > 0:
                    self.neur.dff_aln[ind]['0.5_prelick'][
                        _tr_counts['0.5_prelick'], :] \
                        = _dff[0:n_frames_tot]
                    _tr_counts['0.5_prelick'] += 1
                elif self.beh.stim.prob[trial] == 0.5 \
                     and self.beh.lick.antic_raw[trial] == 0:
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

    def run_and_plt_decoding_trtype_grab(
            self,
            t_start=0, t_end=2,
            t_bl_start=-0.5, t_bl_end=0,
            thresh_n_miss=6,
            thresh_n_neurs=80,
            fold_validation=3,
            n_resamples=20,
            hit_vs_miss=True,
            comp='all',
            summary_stat_fn=np.mean,
            sec_interval=5,
            neur_sampling_rule='random',
            n_jobs=6,
            classifier=sklearn.linear_model.SGDClassifier,
            classifier_kwargs={'class_weight': 'balanced',
                               'loss': 'hinge',
                               'penalty': 'l2',
                               'alpha': 1,
                               'max_iter': 100000},
            scaler=sklearn.preprocessing.StandardScaler,
            scaler_kwargs={},
            classifier_scoring='balanced_accuracy'):

        """
        Attempts to decode trial type from a profile of GRAB sector responses
        for each trial.

        Must have run self.plt_stim_aligned_sectors() first.
        """

        # setup decoder
        # ---------------
        ind_t_start = np.argmin(np.abs(self.sector.t - t_start))
        ind_t_end = np.argmin(np.abs(self.sector.t - t_end))
        # ind_t_bl_start = np.argmin(np.abs(self.sector.t - t_bl_start))
        # ind_t_bl_end = np.argmin(np.abs(self.sector.t - t_bl_end))

        if scaler is None:
            self.clf = make_pipeline(classifier(**classifier_kwargs))
        else:
            self.clf = make_pipeline(scaler(**scaler_kwargs),
                                     classifier(**classifier_kwargs))

        n_tr_prew = {}
        n_trials_all = 0

        if comp == 'all':
            tr_keys = ['0', '0.5', '1']
            tr_labels = {'0': 0, '0.5': 1, '1': 2}
        elif comp == 'rew_norew':
            tr_keys = ['0', '0.5', '1']
            tr_labels = {'0': 0, '0.5': 1, '1': 1}

        for tr_key in tr_keys:
            n_tr_prew[tr_key] = self.sector.dff[0, 0][tr_key].shape[0]
            n_trials_all += n_tr_prew[tr_key]

        self.sector.decoding = SimpleNamespace()
        self.sector.decoding._data = np.zeros((n_trials_all,
                                               self.sector.n_sectors**2))
        self.sector.decoding._labels = np.zeros(n_trials_all)

        # classification: full dataset
        # ---------------
        print('decoder on full dataset\n---------')
        _tr_count = 0
        for tr_key in tr_keys:
            for _tr in range(n_tr_prew[tr_key]):
                for sec in range(self.sector.n_sectors**2):
                    sec_x, sec_y = np.divmod(sec, self.sector.n_sectors)

                    self.sector.decoding._data[
                        _tr_count, sec] = summary_stat_fn(
                        self.sector.dff[sec_x, sec_y][tr_key][
                            _tr, ind_t_start:ind_t_end])

                self.sector.decoding._labels[
                    _tr_count] = tr_labels[tr_key]
                _tr_count += 1

        self.sector.decoding.cvobj_full = cross_validate(
            self.clf, self.sector.decoding._data,
            y=self.sector.decoding._labels,
            cv=fold_validation,
            n_jobs=n_jobs, scoring=classifier_scoring,
            return_estimator=True)

        # classification: changing neuron numbers
        # -------------------
        print('decoder with changing sector numbers\n---------')
        n_sectors_decoding = np.arange(sec_interval,
                                       self.sector.n_sectors**2,
                                       sec_interval)

        self.sector.decoding.cvobj = {}
        self.sector.decoding.perf_means = SimpleNamespace()
        self.sector.decoding.perf_means.keys = n_sectors_decoding
        self.sector.decoding.perf_means.vals = np.zeros_like(
            self.sector.decoding.perf_means.keys, dtype=float)

        for ind_sec_decoder, n_sec_decoder in enumerate(n_sectors_decoding):
            print(f'\tn_sectors={n_sec_decoder}')
            for n_resample in range(n_resamples):
                self.sector.decoding._data = np.zeros((n_trials_all,
                                                       n_sec_decoder))
                self.sector.decoding._labels = np.zeros(n_trials_all)

                _sector_inds_subset = np.random.choice(
                    np.arange(self.sector.n_sectors**2),
                    n_sec_decoder,
                    replace=False)

                _tr_count = 0
                for tr_key in tr_keys:
                    for _tr in range(n_tr_prew[tr_key]):
                        for sec_ind, sec in enumerate(_sector_inds_subset):
                            sec_x, sec_y = np.divmod(
                                sec, self.sector.n_sectors)

                            self.sector.decoding._data[
                                _tr_count, sec_ind] = summary_stat_fn(
                                self.sector.dff[sec_x, sec_y][tr_key][
                                    _tr, ind_t_start:ind_t_end])

                        self.sector.decoding._labels[
                            _tr_count] = tr_labels[tr_key]
                        _tr_count += 1

                # run classification
                self.sector.decoding.cvobj[str(n_sec_decoder)] \
                    = cross_validate(
                        self.clf, self.sector.decoding._data,
                        y=self.sector.decoding._labels,
                        cv=fold_validation,
                        n_jobs=n_jobs, scoring=classifier_scoring,
                        return_estimator=True)

                # store mean value across fold validation
                _temp_perf = np.mean(self.sector.decoding.cvobj[
                    str(n_sec_decoder)]['test_score'])
                self.sector.decoding.perf_means.vals[ind_sec_decoder] \
                    += _temp_perf

            # after all iterations, take mean
            self.sector.decoding.perf_means.vals[ind_sec_decoder] \
                /= n_resamples

        # classification: whole-frame (1p), no variance
        # ---------------
        print('decoder on whole-frame data\n---------')
        self.plt_stim_aligned_avg(t_pre=self.sector.params.t_rew_pre,
                                  t_post=self.sector.params.t_rew_post,
                                  plt_show=False)
        self.sector.decoding._data_1p = np.zeros(n_trials_all)
        self.sector.decoding._labels_1p = np.zeros(n_trials_all)

        _tr_count = 0
        for tr_key in tr_keys:
            for _tr in range(n_tr_prew[tr_key]):
                self.sector.decoding._data_1p[
                    _tr_count] = summary_stat_fn(
                    self.frame.dff[tr_key][
                        _tr, ind_t_start:ind_t_end])

                self.sector.decoding._labels_1p[
                    _tr_count] = tr_labels[tr_key]
                _tr_count += 1

        self.sector.decoding.cvobj_1p = cross_validate(
            self.clf, self.sector.decoding._data_1p.reshape(-1, 1),
            y=self.sector.decoding._labels_1p,
            cv=fold_validation,
            n_jobs=n_jobs, scoring=classifier_scoring,
            return_estimator=True)

        # make plots
        # --------------
        # perf plot
        fig_perf = plt.figure(figsize=(3, 3))
        ax_perf = fig_perf.add_subplot()

        ax_perf.plot(self.sector.decoding.perf_means.keys,
                     self.sector.decoding.perf_means.vals,
                     color=sns.xkcd_rgb['grey'],
                     linewidth=1)
        ax_perf.axhline(np.mean(self.sector.decoding.cvobj_1p['test_score']),
                        color=sns.xkcd_rgb['orange red'],
                        alpha=0.6,
                        linestyle='dotted',
                        linewidth=0.5)
        ax_perf.set_xlabel('sectors')
        ax_perf.set_ylabel('decoding perf')
        ax_perf.set_title(f'decoding: {comp}')

        if comp == 'all':
            ax_perf.set_ylim([0.2, 1])
            ax_perf.axhline(0.3333, linestyle='dashed',
                            linewidth=0.5, color=sns.xkcd_rgb['grey'])
        if comp == 'rew_norew':
            ax_perf.set_ylim([0.4, 1])
            ax_perf.axhline(0.5, linestyle='dashed',
                            linewidth=0.5, color=sns.xkcd_rgb['grey'])

        # weightmap plot
        if comp == 'rew_norew':
            fig_weightmap = plt.figure(figsize=(3, 3))
            ax_weightmap = fig_weightmap.add_subplot()

            self.sector.decoding.final_weights = np.zeros(
                (self.sector.n_sectors, self.sector.n_sectors))
            for fold in range(fold_validation):
                _coefs = self.sector.decoding.cvobj_full['estimator'][
                    fold]._final_estimator.coef_[0, :].reshape(
                    self.sector.n_sectors, self.sector.n_sectors)
                self.sector.decoding.final_weights += _coefs / fold_validation
            vmax_abs = np.max(np.abs(self.sector.decoding.final_weights))

            ax_weightmap.imshow(
                self.sector.decoding.final_weights.reshape(
                    self.sector.n_sectors, self.sector.n_sectors),
                vmax=vmax_abs, vmin=-1*vmax_abs, cmap='coolwarm')
            ax_weightmap.set_title(f'decoding weights: {comp}')
            ax_weightmap.set_xticks([])
            ax_weightmap.set_yticks([])
        elif comp == 'all':
            fig_weightmap = plt.figure(figsize=(8, 3))
            spec_weightmap = gs.GridSpec(nrows=1, ncols=3,
                                         figure=fig_weightmap)

            ax_weightmap = {}
            ax_weightmap['0'] = fig_weightmap.add_subplot(
                spec_weightmap[0, 0])
            ax_weightmap['0.5'] = fig_weightmap.add_subplot(
                spec_weightmap[0, 1])
            ax_weightmap['1'] = fig_weightmap.add_subplot(
                spec_weightmap[0, 2])

            self.sector.decoding.final_weights = {}
            for ind_tr_cond, tr_cond in enumerate(['0', '0.5', '1']):
                self.sector.decoding.final_weights[tr_cond] = np.zeros(
                    (self.sector.n_sectors, self.sector.n_sectors))
                for fold in range(fold_validation):
                    _coefs = self.sector.decoding.cvobj_full['estimator'][
                        fold]._final_estimator.coef_[ind_tr_cond, :].reshape(
                        self.sector.n_sectors, self.sector.n_sectors)
                    self.sector.decoding.final_weights[tr_cond] \
                        += _coefs / fold_validation
                vmax_abs = np.max(np.abs(
                    self.sector.decoding.final_weights[tr_cond]))

                ax_weightmap[tr_cond].imshow(
                    self.sector.decoding.final_weights[tr_cond].reshape(
                        self.sector.n_sectors, self.sector.n_sectors),
                    vmax=vmax_abs, vmin=-1*vmax_abs, cmap='coolwarm')
                ax_weightmap[tr_cond].set_title(f'decoding weights: {tr_cond=}')
                ax_weightmap[tr_cond].set_xticks([])
                ax_weightmap[tr_cond].set_yticks([])

        fig_perf.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            'sectors_decoding_perf_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{comp=}_{t_start=}_{t_end=}_'
            + f'ch={self.ch_img}.pdf'))

        fig_weightmap.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            'sectors_decoding_weightmap_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{comp=}_{t_start=}_{t_end=}_'
            + f'ch={self.ch_img}.pdf'))

        # save the data
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

            self.psychometrics.stim_resp_raw = {'0': [],
                                                '0.5': [],
                                                '1': []}
            self.psychometrics.rew_resp_all = {'0': [],
                                               '0.5': [],
                                               '1': []}            

            _ind_stim_start = np.argmin(np.abs(self.sector.t - 0))
            _ind_stim_end = np.argmin(np.abs(self.sector.t - 2))

            _ind_rew_bl = np.argmin(np.abs(self.sector.t - 1.5))
            _ind_rew_start = np.argmin(np.abs(self.sector.t - 2))
            _ind_rew_end = np.argmin(np.abs(self.sector.t - 5))

            _n_trace = 0
            for n_x in range(self.sector.n_sectors):
                for n_y in range(self.sector.n_sectors):
                    for trial_cond in ['0', '0.5', '1']:
                        _dff = np.mean(
                            self.sector.dff[n_x, n_y][trial_cond],
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
            elif plt_show is False:
                plt.close(fig)
                plt.close(fig_corr)

        except Exception as e:
            print('failed to generate psychometric curve')
            print(e)

    def get_antic_licks_by_trialtype(self, bl_norm=False):
        _tr_counts = {'0': 0, '0.5': 0, '1': 0}
        lickrates_hz = {'0': 0, '0.5': 0, '1': 0}
        for trial in range(self.beh._stimrange.first,
                           self.beh._stimrange.last):
            if bl_norm is True:
                _antic_licks = self.beh.lick.antic_raw[trial] \
                    - self.beh.lick.base[trial]
            elif bl_norm is False:
                _antic_licks = self.beh.lick.antic_raw[trial]

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


class _TwoPRec_DualColor(object):
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

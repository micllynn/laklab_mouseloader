import numpy as np
import scipy as sp
import scipy.stats as sp_stats
from sklearn import decomposition
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from types import SimpleNamespace
import os

from .utils import parse_samp_rate_from_params, find_key_partialmatch, \
    match_region_partial, convert_spktimes_to_spktrain, \
    smooth_spktrain
from .histology import LoadHist


class SpikeTrain(object):
    def __init__(self, arr=None, t=None, spk=None, info=None):
        if spk is None:
            self.arr = arr
            self.t = t
        else:
            self.arr = spk.train
            self.t = spk.t

        self.samp_rate = 1/(self.t[1]-self.t[0])
        self.info = info

    def add_shuff_data(self, n=10):
        self.arr_shuff = np.ndarray(n, dtype=np.ndarray)
        for _n in range(n):
            self.arr_shuff[_n] = np.copy(self.arr)
            np.random.shuffle(self.arr_shuff[_n])

    def plt(self, figsize=(3.43, 2), dotsize=10):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        for neur in range(self.arr.shape[0]):
            _t_spks = self.t[self.arr[neur, :].astype(bool)]
            ax.scatter(_t_spks, np.ones_like(_t_spks)*neur,
                       color='k', s=dotsize)

        ax.set_xlabel('t (sec)')
        ax.set_ylabel('neur')

        plt.show()
        fig.clear()

        return

    def calc_sync_index_dragoi(self, on_shuff=False):
        """
        """
        if on_shuff is False:
            spk_count = np.sum(self.arr, axis=0)
        elif on_shuff is True:
            spk_count = np.sum(self.arr_shuff, axis=0)

        _sd = np.std(spk_count)
        _mean = np.mean(spk_count)
        sync_index = _sd / _mean

        return sync_index

    def calc_sync_index_packer(self, on_shuff=False, shuff_ind=0):
        """
        """
        sync_index = SimpleNamespace()
        sync_index.xcorr_arr = np.ma.zeros(
            (self.arr.shape[0], self.arr.shape[0]))
        for neur1 in range(self.arr.shape[0]):
            for neur2 in range(self.arr.shape[0]):
                if neur2 > neur1:
                    if on_shuff is False:
                        sync_index.xcorr_arr[neur1, neur2] = sp.stats.pearsonr(
                            self.arr[neur1, :],
                            self.arr[neur2, :]).statistic
                    elif on_shuff is True:
                        sync_index.xcorr_arr[neur1, neur2] = sp.stats.pearsonr(
                            self.arr_shuff[shuff_ind][neur1, :],
                            self.arr_shuff[shuff_ind][neur2, :]).statistic
                else:
                    sync_index.xcorr_arr[neur1, neur2] = np.ma.masked
        sync_index.mean_xcorr = np.ma.mean(sync_index.xcorr_arr)

        return sync_index

    def calc_stfr(self):
        return


class EphysData(object):
    def __init__(self, folder, aligner_obj=False, filt_clusts=True):
        """
        Class for loading and working with neuropixel data associated with a
        behavioral dataset. (Blake's ReportOpto dataset here)

        Must be already sorted using **kilosort 4** (kilosort 2.5 will not work
        due to slight differences in how the spike and clust arrays
        are organized.

        Parameters
        ---------------
        folder : str
            Name of the ephys folder with kilosort outputs.
        filt_clusts : bool
            If True, only keeps clusters that were identified by kilosort as
            'good' (ie excludes 'noise' and 'mua'.
        samp_rate_pointproc_hz : int
            Sampling rate for the point process spiketrain, in Hz.
            While the real sampling rate for neuropixels probe is quite high
            (~30k hz), this should be much lower.

        Attributes added
        --------------
        self.spk_clusters
        self.spk_times_raw
        self.spk_info

        self.samp_rate_hz

        self.ch_raw
            self.ch_raw.map
            self.ch_raw.pos
        self.ch
            self.ch.region
            self.ch.region_id
            self.ch.ml
            self.ch.ap
            self.ch.dv
        """

        self.folder = folder
        _start_dir = os.getcwd()

        self.hist = LoadHist('.', ephys_folder=folder)

        print('loading ephys...')
        print('\tloading spike data and channel position data...')

        os.chdir(self.folder)
        # load spk data
        self.spk = SimpleNamespace()
        self.spk_raw = SimpleNamespace()
        self.spk_raw.clusters = np.load('spike_clusters.npy')
        self.spk_raw.clusters = self.spk_raw.clusters

        self.spk_raw.times = np.load('spike_times.npy')
        self.spk_raw.clust_kslabel = pd.read_csv('cluster_KSLabel.tsv', sep='\t')
        self.spk_raw.samp_rate_hz = parse_samp_rate_from_params('params.py')

        # load channel position data
        self.ch_raw = SimpleNamespace()
        self.ch_raw.map = np.load('channel_map.npy')
        self.ch_raw.pos = np.load('channel_positions.npy')

        n_ch = len(self.ch_raw.pos)

        self.ch = SimpleNamespace()
        self.ch.region = np.empty(n_ch, dtype=object)
        self.ch.region_id = np.zeros(n_ch)
        self.ch.ml = np.zeros(n_ch)
        self.ch.ap = np.zeros_like(self.ch.ml)
        self.ch.dv = np.zeros_like(self.ch.ml)
        self.ch.dv_plt = np.zeros_like(self.ch.ml)  # offset

        # setup colors for different brain regions
        self.plt = SimpleNamespace()
        self.plt.colors = {}
        self.plt.colors['primary somatosensory'] = sns.xkcd_rgb['seafoam green']
        self.plt.colors['supplemental somatosensory'] = sns.xkcd_rgb['blue green']
        self.plt.colors['field CA'] = sns.xkcd_rgb['apple green']
        self.plt.colors['caudoputamen'] = sns.xkcd_rgb['powder blue']
        self.plt.colors['striatum'] = sns.xkcd_rgb['powder blue']
        self.plt.colors['pallidum'] = sns.xkcd_rgb['faded blue']
        self.plt.colors['lateral geniculate'] = sns.xkcd_rgb['salmon']
        self.plt.colors['thalamus'] = sns.xkcd_rgb['pinkish']
        self.plt.colors['midbrain'] = sns.xkcd_rgb['fuchsia']
        self.plt.colors['hypothalamus'] = sns.xkcd_rgb['scarlet']
        self.plt.colors['none'] = sns.xkcd_rgb['grey']

        # try to load information about histology-aligned ch locations
        print('\thistology alignment data...')

        for ind_ch in range(n_ch):
            self.ch.region[ind_ch] = self.hist.map_ch_region.ch_to_reg(ind_ch)

            # self.ch.ml[ind_ch] \
            #     = self.ch_raw.histalign_anat[_key_ch]['x']
            # self.ch.ap[ind_ch] \
            #     = self.ch_raw.histalign_anat[_key_ch]['y']
            # self.ch.dv[ind_ch] \
            #     = self.ch_raw.histalign_anat[_key_ch]['z']

        # setup a dv_plt attribute that has dvs that are staggered
        self.ch.dv_plt = np.copy(self.ch.dv)
        self.ch.dv_plt[0:-1] += np.diff(
            self.ch.dv_plt)/2

        # filter data based on phy label of 'good'
        if filt_clusts is True:
            print('\tfiltering clusters...')
            # find key associated with clust id
            _key_id = find_key_partialmatch(self.spk_raw.clust_kslabel, 'id')

            # filter clusts based on quality
            _good_clusts_pd = self.spk_raw.clust_kslabel[
                self.spk_raw.clust_kslabel['KSLabel'] == 'good']
            self.spk_raw._good_clust_ind = np.array(_good_clusts_pd[_key_id])
            self.spk_raw._good_clust_mask = np.isin(
                self.spk_raw.clusters, self.spk_raw._good_clust_ind)

            self.spk_raw.clusters = self.spk_raw.clusters[
                self.spk_raw._good_clust_mask]
            self.spk_raw.times = self.spk_raw.times[
                self.spk_raw._good_clust_mask]

        # extract spiketimes and correct for sampling rate
        print('\textracting spiketimes...')
        self.spk_raw.times = [self.spk_raw.times[i]
                              for i in range(len(self.spk_raw.times))]
        self.spk_raw.times = np.array(self.spk_raw.times, dtype=float)
        self.spk_raw.times /= self.spk_raw.samp_rate_hz

        # align to behavior, if aligner_obj is present
        print('\taligning ephys to behavior...')
        if aligner_obj is False:
            self.aligner_obj = False
            print('\t\tno aligner_obj present.')
        elif aligner_obj is not False:
            self.aligner_obj = aligner_obj
            self.spk_raw.times = self.aligner_obj.correct_ephys_data(
                self.spk_raw.times)
            # self.spk_raw.times = self.deploy_aligner_on_ephys_data(
            #     self.spk_raw.times)
            print('\t\taligned.')

        # go back to original directory
        os.chdir(_start_dir)

        return

    def add_spktrain(self, samp_rate_spktrain_hz=1000):
        """
        Converts a list of spiketimes and cluster IDs
        to a spktrain (timebins with the desired sampling rate), then
        adds this spktrain as an attribute to the obj instance.

        Parameters
        --------------
        samp_rate_spktrain_hz : int
            Sampling rate for spktrain conversion

        Attributes added
        ---------------
        self.spk
            self.spk.t
            self.spk.train
        """
        print('\tadding spktrain...')
        _max_t = np.max(self.spk_raw.times)
        # _n_clusts = np.max(self.spk_raw.clusters)
        _clust_list = np.unique(self.spk_raw.clusters)

        self.spk.t = np.arange(0, _max_t, 1/samp_rate_spktrain_hz)
        # self.spk.train = np.empty(_n_clusts, dtype=np.ndarray)
        self.spk.train = np.empty(len(_clust_list), dtype=np.ndarray)
        self.spk.train_mtx = np.zeros((len(_clust_list), len(self.spk.t)))
        self.spk.raw_clust_id = _clust_list

        self.spk.info = SimpleNamespace()

        self.spk.info.clust_id = self.spk.raw_clust_id
        self.spk.info.ch = []
        self.spk.info.region_full = []
        self.spk.info.region = []
        self.spk.info.dv = []

        for ind_clust, clust_id in enumerate(_clust_list):
            print(f'\tcluster {ind_clust}...', end='\r')
            self.spk.raw_clust_id[ind_clust] = clust_id
            _inds_clust_spkraw = np.where(self.spk_raw.clusters == clust_id)

            self.spk.train[ind_clust] = convert_spktimes_to_spktrain(
                self.spk_raw.times[_inds_clust_spkraw], self.spk.t,
                method='calc')
            self.spk.train_mtx[ind_clust, :] = self.spk.train[ind_clust]

            try:
                # find key associated with clust id
                _ch = self.hist.map_clust_ch.clust_to_ch(clust_id)
                _region_id = self.ch.region[_ch]
                _region_key = match_region_partial(
                    _region_id,
                    list(self.plt.colors.keys()))
                # _dv = self.ch.dv_plt[_ch]

                self.spk.info.ch.append(_ch)
                self.spk.info.region_full.append(_region_id)
                self.spk.info.region.append(_region_key)
                # self.spk.info.dv.append(_dv)

            except:
                self.spk.info.ch.append(None)
                self.spk.info.region_full.append(None)
                self.spk.info.region.append(None)
                self.spk.info.dv.append(None)
        return

    def add_spktrain_kernconv(self, kern_sd=5, samp_rate_spktrain_hz=250):
        key_gauss_stdev = str(kern_sd)+'ms'

        self.add_spktrain(samp_rate_spktrain_hz=samp_rate_spktrain_hz)

        if hasattr(self.spk, 'kernconv') is False:
            self.spk.train_kernconv = SimpleNamespace()
            self.spk.train_kernconv.fr = {}
            self.spk.train_kernconv.fr_zscore = {}
            self.spk.train_kernconv.t = self.spk.t

        self.spk.train_kernconv.fr[key_gauss_stdev] = np.empty_like(
            self.spk.train, dtype=np.ndarray)
        self.spk.train_kernconv.fr_zscore[key_gauss_stdev] = np.empty_like(
            self.spk.train, dtype=np.ndarray)            

        print('\tconvolving with kernel...')
        for ind_clust, clust_id in enumerate(self.spk.raw_clust_id):
            print(f'\tcluster {ind_clust}...', end='\r')

            # convolve with gaussian
            _pointproc_gauss = smooth_spktrain(
                self.spk.train[ind_clust], self.spk.t,
                kern_sd)

            if np.max(_pointproc_gauss) != 0:
                self.spk.train_kernconv.fr[key_gauss_stdev][ind_clust] \
                    = _pointproc_gauss
                self.spk.train_kernconv.fr_zscore[key_gauss_stdev][ind_clust] \
                    = sp_stats.zscore(_pointproc_gauss)
            else:
                self.spk.train_kernconv.fr[key_gauss_stdev][ind_clust] \
                    = np.zeros_like(self.spk.train_kernconv.t, dtype=np.int8)
                self.spk.train_kernconv.fr_zscore[key_gauss_stdev][ind_clust] \
                    = np.zeros_like(self.spk.train_kernconv.t, dtype=np.int8)

        return

    def get_subset_spktrain(self, t_start=None, t_end=None, region=None,
                            get_kernconv=True, get_zscore=True,
                            kernconv_sd_key=None):
        """
        Retrieves a subset of the pointprocess array from a defined
        time window and region.

        Parameters
        ------------
        t_start : None or float
            Start time to extract spikes from. If None, uses full time
        t_end : None or float
            End time to extract spikes from. If None, uses full time.
        region : None or str
            Either None (extract all neurons) or str (extract neurons only
            from that named area, found in ephysobj.spk.info.region)
        """
        spk = SimpleNamespace()

        # parse by region
        # ----------------
        if region is None:
            _inds_neur = np.arange(len(self.spk.train))
        else:
            _inds_neur = np.where(np.array(self.spk.info.region) == region)[0]

        # parse by time
        # ----------------
        if t_start is not None:
            _ind_t_start = np.argmin(np.abs(self.spk.t-t_start))
        elif t_start is None:
            _ind_t_start = 0

        if t_end is not None:
            _ind_t_end = np.argmin(np.abs(self.spk.t-t_end))
        elif t_end is None:
            _ind_t_end = -1

        # save in spk simplenamespace
        # -----------
        if get_kernconv is False:
            spk.train = self.spk.train_mtx[_inds_neur,
                                           _ind_t_start:_ind_t_end]
        elif get_kernconv is True:
            # set sd key if not present
            if kernconv_sd_key is None:
                kernconv_sd_key = list(self.spk.train_kernconv.fr.keys())[0]

            # configure spk.train to be 2d matrix
            _templ = self.spk.train_kernconv.fr[kernconv_sd_key] \
                [_inds_neur[0]][_ind_t_start:_ind_t_end]
            spk.train = np.zeros((len(_inds_neur), _templ.shape[0]))

            # store values
            for _ind_rel_neur, _ind_abs_neur in enumerate(_inds_neur):
                if get_zscore is False:
                    spk.train[_ind_rel_neur, :] \
                        = self.spk.train_kernconv.fr[kernconv_sd_key] \
                        [_ind_abs_neur][_ind_t_start:_ind_t_end]
                elif get_zscore is True:
                    spk.train[_ind_rel_neur, :] \
                        = self.spk.train_kernconv.fr_zscore[kernconv_sd_key] \
                        [_ind_abs_neur][_ind_t_start:_ind_t_end]

        spk.t = self.spk.train_kernconv.t[_ind_t_start:_ind_t_end]

        return spk

    def calc_pca(self, bin_size=100, region=None, t_range=None):
        """
        Parameters
        ----------------
        bin_size : int
            Bin size in ms
        region : None or list
            Regions from which to extract units and calculate pca
        t_range : None or list
            Range of times from which to calculate pca in seconds
        """
        self.add_spktrain(samp_rate_spktrain_hz=1/(bin_size/1000))

        if type(region) == str:
            region = [region]

        if region is None:
            _clust_count = self.spk.raw_clust_id.shape[0]
            _clust_inds = np.arange(self.spk.train.shape[0])
        else:
            _clust_count = 0
            _clust_inds = []
            for ind_clust, raw_clust_id in enumerate(self.spk.raw_clust_id):
                try:
                    _reg = self.spk.info.region[ind_clust]

                    if _reg in region:
                        _clust_inds.append(ind_clust)
                        _clust_count += 1

                except IndexError:
                    pass
                except TypeError:
                    pass

        pointproc_matrix = np.zeros((_clust_count,
                                     self.spk.train[0].shape[0]),
                                    dtype=int)

        for ind_clust_in_matrix, ind_clust_id in enumerate(_clust_inds):
            _t_spk = self.spk.train[ind_clust_id]
            pointproc_matrix[ind_clust_in_matrix, :] = _t_spk

        # run pca
        # ------------
        pca = decomposition.TruncatedSVD(n_components=3)
        if t_range is None:
            pca.fit(pointproc_matrix.T)
            self.pointproc_pca = pca.transform(pointproc_matrix.T)
        else:
            ind_t_min = np.argmin(np.abs(self.spk.t - t_range[0]))
            ind_t_max = np.argmin(np.abs(self.spk.t - t_range[1]))

            pca.fit(pointproc_matrix[ind_t_min:ind_t_max].T)
            self.pointproc_pca = pca.transform(
                pointproc_matrix[ind_t_min:ind_t_max].T)

    def calc_synchrony_index(self, bin_size_count_spks=10,
                             bin_size_synch_index=1000,
                             region=None):

        bin_size_spkcount_sec = bin_size_count_spks/1000
        self.add_spktrain(samp_rate_spktrain_hz=int(1/bin_size_spkcount_sec))

        # setup subset of the spktrain matrix corresponding to the region
        if region is None:
            _clust_count = self.spk.raw_clust_id.shape[0]
            _clust_inds = np.arange(self.spk.train.shape[0])
        else:
            _clust_count = 0
            _clust_inds = []
            for ind_clust, raw_clust_id in enumerate(self.spk.raw_clust_id):
                try:
                    # find key associated with clust id
                    _key_id = find_key_partialmatch(
                        self.spk_raw.clust_info, 'id')

                    _clust_info = self.spk_raw.clust_info[
                        self.spk_raw.clust_info[_key_id] == raw_clust_id]
                    _ch = np.array(_clust_info['ch'])[0]

                    _region_id = self.ch.region[_ch]
                    _region_key = match_region_partial(
                        _region_id,
                        list(self.plt.colors.keys()))

                    if _region_key in region:
                        _clust_inds.append(ind_clust)
                        _clust_count += 1

                except IndexError:
                    pass
                except TypeError:
                    pass

        pointproc_matrix = np.zeros((_clust_count,
                                     self.spk.train[0].shape[0]),
                                    dtype=int)

        for ind_clust_in_matrix, ind_clust_id in enumerate(_clust_inds):
            _t_spk = self.spk.train[ind_clust_id]
            pointproc_matrix[ind_clust_in_matrix, :] = _t_spk

        # Compute synchrony index
        # ----------------
        spk_count = np.sum(pointproc_matrix, axis=0)
        synch_scaling_factor = int(bin_size_synch_index / bin_size_count_spks)
        n_synch = np.floor(spk_count.shape[0]
                           / synch_scaling_factor).astype(int)

        self.psi = SimpleNamespace()
        self.psi.index = np.zeros(n_synch-1)
        self.psi.t = np.zeros_like(self.psi.index)

        for ind_synch in range(n_synch-1):
            _ind_start = int(ind_synch*synch_scaling_factor)
            _ind_end = int((ind_synch+1)*synch_scaling_factor)
            _ind_mid = int((_ind_start+_ind_end)/2)

            _sd = np.std(spk_count[_ind_start:_ind_end])
            _mean = np.mean(spk_count[_ind_start:_ind_end])

            self.psi.index[ind_synch] = _sd / _mean
            self.psi.t[ind_synch] = self.spk.t[_ind_mid]

    def plt_raster(self, scatter_size=2, fig_save=False,
                   t_range=None, region=None):
        fig = plt.figure(figsize=(6.86, 4))
        ax = fig.add_subplot(1, 1, 1)

        max_n_clusts = np.max(self.spk_raw.clusters)

        for clust_id in range(max_n_clusts):
            print(f'plotting {clust_id=}...', end='\r')
            _t_spk = self.spk_raw.times[self.spk_raw.clusters == clust_id]
            try:
                _clust_info = self.spk_raw.clust_info[
                    self.spk_raw.clust_info['id'] == clust_id]
                _ch = np.array(_clust_info['ch'])[0]

                _region_id = self.ch.region[_ch]
                _region_key = match_region_partial(
                    _region_id,
                    list(self.plt.colors.keys()))
                _dv = self.ch.dv_plt[_ch]

                if region is None:
                    ax.scatter(_t_spk, np.ones_like(_t_spk)*_dv,
                               s=scatter_size,
                               color=self.plt.colors[_region_key])
                else:
                    if region == _region_key:
                        ax.scatter(_t_spk, np.ones_like(_t_spk)*_dv,
                                   s=scatter_size,
                                   color=self.plt.colors[_region_key])
            except IndexError:
                pass

        ax.set_xlabel('time (s)')
        ax.set_ylabel('dv (um)')

        if t_range is not None:
            ax.set_xlim(t_range[0], t_range[1])

        if fig_save is True:
            fig.savefig(f'figs/raster_{region=}_{t_range=}.pdf')
        plt.show()

        return

    def plt_raster_and_fr(self, scatter_size=2, fig_save=False,
                          t_range=None, region=None, kern_sd='5ms',
                          plt_areas_separately=True):
        fig = plt.figure(figsize=(6.86, 4))
        spec = gs.GridSpec(nrows=2, ncols=1, height_ratios=[1, 0.2],
                           figure=fig)
        ax_raster = fig.add_subplot(spec[0, 0])
        ax_fr = fig.add_subplot(spec[1, 0], sharex=ax_raster)

        max_n_clusts = np.max(self.spk_raw.clusters)

        for clust_id in range(max_n_clusts):
            print(f'plotting {clust_id=}...', end='\r')
            _t_spk = self.spk_raw.times[self.spk_raw.clusters == clust_id]
            try:
                _clust_info = self.spk_raw.clust_info[
                    self.spk_raw.clust_info['id'] == clust_id]
                _ch = np.array(_clust_info['ch'])[0]

                _region_id = self.ch.region[_ch]
                _region_key = match_region_partial(
                    _region_id,
                    list(self.plt.colors.keys()))
                _dv = self.ch.dv_plt[_ch]

                if region is None:
                    ax_raster.scatter(_t_spk, np.ones_like(_t_spk)*_dv,
                                      s=scatter_size,
                                      color=self.plt.colors[_region_key])
                else:
                    if region == _region_key:
                        ax_raster.scatter(_t_spk, np.ones_like(_t_spk)*_dv,
                                          s=scatter_size,
                                          color=self.plt.colors[_region_key])
            except IndexError:
                pass


        # firing rate plot
        # ------------
        if plt_areas_separately is True:
            _list_areas = list(self.plt.colors.keys())

            # create struct to store region-specific firing rates
            self.plt.fr = {}
            self.plt.fr_neur_counter = {}
            for _area in _list_areas:
                self.plt.fr[_area] = np.zeros_like(self.spk.train_kernconv.t,
                                                   dtype='single')
                self.plt.fr_neur_counter[_area] = 0

            for clust_id in range(max_n_clusts):
                try:
                    _clust_info = self.spk_raw.clust_info[
                        self.spk_raw.clust_info['id'] == clust_id]
                    _ch = np.array(_clust_info['ch'])[0]

                    _region_id = self.ch.region[_ch]
                    _region_key = match_region_partial(
                        _region_id,
                        list(self.plt.colors.keys()))

                    self.plt.fr[_region_key] += self.spk.train_kernconv.fr[kern_sd]\
                        [clust_id].astype(float)
                    self.plt.fr_neur_counter[_region_key] += 1
                except IndexError:
                    pass

            # normalize each region firing rate by number of neurons and plot
            for _area in _list_areas:
                if region is None:
                    if self.plt.fr_neur_counter[_area] != 0:
                        self.plt.fr[_area] /= self.plt.fr_neur_counter[_area]
                    elif self.plt.fr_neur_counter[_area] == 0:
                        pass

                    ax_fr.plot(self.spk.train_kernconv.t, self.plt.fr[_area],
                               color=self.plt.colors[_area], linewidth=0.6)
                else:
                    if region == _area:
                        self.plt.fr[_area] /= self.plt.fr_neur_counter[_area]
                        ax_fr.plot(self.spk.train_kernconv.t, self.plt.fr[_area],
                                   color=self.plt.colors[_area], linewidth=0.6)

        ax_fr.set_ylabel('fr (Hz)')
        ax_fr.set_xlabel('time (s)')
        ax_raster.set_ylabel('dv (um)')

        if t_range is not None:
            ax_raster.set_xlim(t_range[0], t_range[1])

        if fig_save is True:
            fig.savefig('figs/raster_fr.pdf')
        plt.show()

        return



import numpy as np
import scipy as sp
import os
import scipy.signal as sp_signal
import matplotlib.pyplot as plt
from types import SimpleNamespace

try:
    plt.style.use('publication_ml')
except:
    pass


# Tools to load and preprocess dataset
# -------------

def find_event_onsets(events, thresh=5):
    event_diff = np.diff(events)
    event_inds = np.where(event_diff > thresh)[0]
    return event_inds


def find_event_onsets_autothresh(events, n_stdevs=2):
    event_diff = np.diff(events)
    event_diff_mean = np.mean(event_diff)
    event_diff_std = np.std(event_diff)

    event_inds = np.where(
        event_diff > (event_diff_mean + event_diff_std*n_stdevs))[0]

    return event_inds


def remove_event_duplicates(events, abs_refractory_period=0.1):
    events_diff = np.diff(events)
    inds_to_cut = []
    for ind, event_diff in enumerate(events_diff):
        if event_diff < abs_refractory_period:
            inds_to_cut.append(ind)
    events_fixed = np.delete(events, inds_to_cut)
    return events_fixed


def remove_lick_artefact_after_rew(t_licks,
                                   t_interval_after_rew=[0.03, 0.06]):
    """
    Takes a vector of lick times aligned to reward onset (t=0),
    and removes all lick times within a certain time-window after
    reward, in seconds (these are typically artefacts.)
    """
    inds_to_delete = np.argwhere(np.logical_and(
        t_licks > t_interval_after_rew[0],
        t_licks < t_interval_after_rew[1]))
    t_licks_corrected = np.delete(t_licks, inds_to_delete)
    return t_licks_corrected


def calc_alpha(val, val_max, alpha_min=0.2):
    """
    Calculates a normalized transparency value for a datapoint,
    given the max value observed and a target minimum alpha
    (to prevent alpha from going to 0 and becoming invisible.)
    """
    offset = (alpha_min*val_max) / (1-alpha_min)
    alpha = (val+offset) / (val_max+offset)

    return alpha


def convert_spktimes_to_spktrain(spktimes, t,
                                 method='calc'):
    spktrain = np.zeros_like(t, dtype=np.int8)
    _samp_rate_hz = 1/(t[1]-t[0])

    for _t_spk in spktimes:
        if method == 'calc':
            _ind_spk = int(_t_spk * _samp_rate_hz)
        elif method == 'argmin':
            _ind_spk = np.argmin(np.abs(t-_t_spk))
        spktrain[_ind_spk] = 1

    return spktrain


def bin_signal(sig, t,
               t_bin_start=-0.2, t_bin_end=1, sampling_rate=100):
    binned = SimpleNamespace()
    binned.t = np.arange(t_bin_start,
                         t_bin_end+1/1000,
                         1/sampling_rate)
    binned.sig = np.zeros_like(binned.t, dtype=float)

    for ind_bin_t, val in enumerate(binned.t[1:-1]):
        _inds_t = np.where(np.logical_and(
            t > binned.t[ind_bin_t],
            t < binned.t[ind_bin_t+1]))[0]
        if len(_inds_t) == 0:
            binned.sig[ind_bin_t] = binned.sig[ind_bin_t-1]
        elif len(_inds_t) > 0:
            binned.sig[ind_bin_t] = np.mean(sig[_inds_t])

    binned.sig[-1] = binned.sig[-2]
    return binned


def parse_samp_rate_from_params(fname):
    """
    Parameters
    ---------------
    fname: str
        path of the params.py from ephys folder.

    Returns
    -------------
    samp_rate : float
        Sampling rate of the ephys data, in Hz.
    """
    f = open(fname)
    config = f.read()
    samp_rate = float(
        config.partition('sample_rate = ')[2].partition('\n')[0])

    return samp_rate


def dtype_to_list(dtype):
    """Converts a numpy recarray dtype attribute into a parseable list.
    For use when loading block.mat files using sp.io.loadmat.
    """
    dtype_list = [dtype.descr[i][0] for i in range(len(dtype.descr))]
    return dtype_list


def convert_rewvalue_into_floats(rewvalue):
    """Takes totalRewardValue variable (from block.mat), parses into
    individual trials, and converts to floats
    """
    rewvalue_split = rewvalue.split('\N{MICRO SIGN}l')
    rewvalue_floats = [float(rewvalue_split[i])
                       for i in range(len(rewvalue_split)-1)]
    return rewvalue_floats


def find_key_partialmatch(dictionary, str_query):
    for key in list(dictionary.keys()):
        if key.find(str_query) > -1:
            return key
    return None


def match_region_partial(region, all_regions):
    for reg_to_test in all_regions:
        if reg_to_test in region.lower():
            return reg_to_test
    return 'none'


def smooth_squarekern(vec, kern, mode='mean'):
    """
    Smooth a given signal (vec) by convolving with a square kernel of
    a given length (kern).

    Works by first creating a padded vector with a given mode (mode) and
    convolving in the 'valid' mode to avoid edge effects and return a
    smoothed vector of identical length to the input.
    """

    assert np.remainder(kern, 2) == 1, 'kern must be an odd number.'

    vec_pad = np.pad(vec, int((kern-1)/2),
                     mode=mode, stat_length=int((kern-1)/2))

    vec_smooth = sp.signal.convolve(
        vec_pad, np.ones(kern)/kern,
        mode='valid')

    return vec_smooth


def rescale_to_frac(times, beh, n_bins):
    """
    Takes a set of times and associated behavioral variables, rescales the
    times to span [0, 1] (rs.t), and returns a rescaled,
    binned vector of the behavioral variable (rs.beh) over these times.

    Parameters
    ----------
    times
        Vector of timestamps for each behavioral value in beh.
    beh : np.ndarray
        Vector of behavioral values, with the same length as times
    n_bins : int
        Number of bins to tile the behavioral task with.
    """
    t_bins = np.linspace(0, 1, n_bins)

    t_as_frac = (times-times[0]) / (times[-1]-times[0])
    beh_as_frac_binned = np.zeros_like(t_bins)

    _last_ind_with_data = 0
    for ind, bin_frac in enumerate(t_bins[0:-1]):
        _inds_in_times = np.where(np.logical_and(
            t_as_frac > t_bins[ind],
            t_as_frac < t_bins[ind+1]))[0]
        if len(_inds_in_times) > 0:
            _beh = np.mean(beh[_inds_in_times])
            beh_as_frac_binned[ind] = _beh

            _last_ind_with_data = ind
        else:
            if ind != 0:
                beh_as_frac_binned[ind] = beh_as_frac_binned[
                    _last_ind_with_data]
            else:
                pass

    # set last beh bin as second last (as these are bin edges)
    beh_as_frac_binned[-1] = beh_as_frac_binned[-2]

    rs = SimpleNamespace()
    rs.t = t_bins
    rs.beh = beh_as_frac_binned

    return rs


def smooth_spktrain(spktrain, t,
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
    sampling_rate_hz = 1/(t[1]-t[0])
    gauss_stdev_inds = (gauss_stdev_ms/1000) * sampling_rate_hz
    gauss_npoints = int(gauss_stdev_inds*8)

    # define gauss kernel and normalize auc to 1
    gauss_kern = sp_signal.windows.gaussian(
        gauss_npoints, gauss_stdev_inds)
    gauss_kern_integral = np.trapz(gauss_kern,
                                   x=t[0:gauss_kern.shape[0]])
    gauss_kern /= gauss_kern_integral

    # convolve point process with gaussian
    spktrain_smoothed = sp_signal.convolve(
        spktrain,
        gauss_kern, mode='same')

    return spktrain_smoothed


def extract_activ_order(spk_mtx, time, thresh=2):
    """
    Takes a matrix (spk_mtx (neur, t)) of spikes over time for a population
    of neurons in a single trial (aligned to stim or some other event) and a
    time vector (time), and computes the relative onset time for each neuron
    using a threshold (thresh).

    Best if spk_mtx is z-scored smoothed firing rate over time,
    and then a threshold is applied to this.

    Returns
    -------------
    order : SimpleNamespace()
    order.t_onset
        Onset time, of activity for each neuron, calculated using [time]
        (None if no activity)
    order.rank
        Ordered rank of each neuron for that trial, as an integer.
        (None if no activity)
    """

    n_neurs = spk_mtx .shape[0]

    order = SimpleNamespace()
    order.t_onset = np.zeros(n_neurs)
    order.rank = np.zeros(n_neurs)

    # Compute onset times
    for neur in range(n_neurs):
        # try to get the latency
        try:
            _ind_latency = np.where(
                (spk_mtx[neur, :] > thresh) > 0)[0]
            # only take times > 0
            _ind_latency_timethresh = _ind_latency[_ind_latency > np.argwhere(
                time > 0)[0]][0]

            _t_latency = time[_ind_latency_timethresh]
        except IndexError:
            _ind_latency = None
            _t_latency = None

        order.t_onset[neur] = _t_latency

    _rank = np.argsort(order.t_onset)

    # delete all items that have a latency of nan (no activity that trial)
    _rank = np.delete(
        _rank,
        np.arange(-1, -1*(np.sum(np.isnan(order.t_onset)))-1, -1))
    order.rank = _rank

    return order


def pad_rank_with_nans(rank):
    """
    Takes an array of arrays denoting the activation order of each neuron for
    each trial, and convert to a single large array padded with nans (in
    case not every trial contains activation by all neurons
    """
    # find max number of neurons activated in a single trial
    n_tr = rank.shape[0]
    n_neurs_max = 0
    for tr in range(n_tr):
        _size = rank[tr].shape[0]
        if _size > n_neurs_max:
            n_neurs_max = _size

    rank_padded = np.zeros((n_tr, n_neurs_max))

    for tr in range(n_tr):
        _n_to_pad = n_neurs_max - rank[tr].shape[0]
        rank_padded[tr, :] = np.pad(rank[tr].astype(float),
                                    ((0, _n_to_pad)), 'constant',
                                    constant_values=np.nan)

    return rank_padded


def check_folder_exists(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    return

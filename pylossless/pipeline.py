# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
#          James Desjardins <jim.a.desjardins@gmail.com>
#          Tyler Collins <collins.tyler.k@gmail.com>
#
# License: MIT

"""Classes and Functions for running the Lossless Pipeline."""

from copy import deepcopy
from pathlib import Path
from functools import partial
from importlib.metadata import version

# Math and data structures
import numpy as np
import pandas as pd
import xarray as xr
import scipy
from scipy.spatial import distance_matrix
from tqdm import tqdm

# BIDS, MNE, and ICA
import mne
from mne.preprocessing import annotate_break
from mne.preprocessing import ICA
from mne.coreg import Coregistration
from mne.utils import logger, warn
import mne_bids
from mne_bids import get_bids_path_from_fname, BIDSPath

from .config.config import Config
from .flagging import FlaggedChs, FlaggedEpochs, FlaggedICs
from ._logging import lossless_logger, lossless_time
from .utils import _report_flagged_epochs
from .utils.html import _get_ics, _sum_flagged_times, _create_html_details


def epochs_to_xr(epochs, kind="ch", ica=None):
    """Create an Xarray DataArray from an instance of mne.Epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        an instance of mne.Epochs
    kind : string
        The name to be passed into the `coords` argument of xr.DataArray
        corresponding to the channel dimension of the epochs object.
        Must be ``'ch'`` or ``'ic'``.
    ica : mne.preprocessing.ICA
        If not ``None``, should be an instance of mne.preprocessing.ICA
        from which to pull the names of the ICA components.

    Returns
    -------
    xarray.DataArray
        an instance of xarray.DataArray, with dimensions ``'epochs'``,
        ``'time'`` (samples), and either ``'ch'`` (channels) or ``'ic'``
        (independent components).
    """
    if kind == "ch":
        data = epochs.get_data()  # n_epochs, n_channels, n_times
        names = epochs.ch_names
    elif kind == "ic":
        data = ica.get_sources(epochs).get_data()
        names = ica._ica_names

    else:
        raise ValueError("The argument kind must be equal to 'ch' or 'ic'.")

    return xr.DataArray(
        data,
        coords={"epoch": np.arange(data.shape[0]), kind: names, "time": epochs.times},
    )


def get_operate_dim(array, flag_dim):
    """Get the xarray.DataArray dimension to flag for a pipeline method.

    Parameters
    ----------
    array : xarray.DataArray
        An instance of Xarray.DataArray that was constructed from an
        ``mne.Epochs`` object, using ``pylossless.pipeline.epochs_to_xr``.
        The ``array`` must be 2D.
    flag_dim : str
        Name of the dimension to remove in ``xarray.DataArray.dims``.
        Must be one of ``'epoch'``, ``'ch'``, or ``'ic'``.

    Returns
    -------
    list : list
        a list of the dimensions of the xarray.DataArray,
        excluding the dimension that the pipeline will conduct
        flagging operations on.
    """
    dims = list(array.dims)
    assert len(dims) == 2
    dims.remove(flag_dim)
    return dims[0]


def _get_outliers_quantile(array, dim, lower=0.25, upper=0.75, mid=0.5, k=3):
    """Calculate outliers for Epochs or Channels based on the IQR.

    Parameters
    ----------
    array : xr.DataArray
        Array of shape n_channels, n_epochs, representing the stdev across
        time (samples in epoch) for each channel/epoch pair.
    dim : str
        One of 'ch' or 'epoch'. The dimension to operate across.
    lower : float (default 0.75)
        The lower bound of the IQR
    upper : float (default 0.75)
        The upper bound of the IQR
    mid : float (default 0.5)
        The mid-point of the IQR
    k : int | float
        factor to multiply the IQR by.

    Returns
    -------
    Lower value threshold : xr.DataArray
        Vector of values (of size n_channels or n_epochs) to be considered
        as the lower threshold for outliers.
    Upper value threshold : xr.DataArray
        Vector of values (of size n_channels or n_epochs) to be considered the
        upper thresholds for outliers.
    """
    lower_val, mid_val, upper_val = array.quantile([lower, mid, upper], dim=dim)

    # Code below deviates from Tukeys method (Q2 +/- k(Q3-Q1))
    # because we need to account for distribution skewness.
    lower_dist = mid_val - lower_val
    upper_dist = upper_val - mid_val
    return mid_val - lower_dist * k, mid_val + upper_dist * k


def _get_outliers_trimmed(array, dim, trim=0.2, k=3):
    """Calculate outliers for Epochs or Channels based on the trimmed mean."""
    trim_mean = partial(scipy.stats.mstats.trimmed_mean, limits=(trim, trim))
    trim_std = partial(scipy.stats.mstats.trimmed_std, limits=(trim, trim))
    m_dist = array.reduce(trim_mean, dim=dim)
    s_dist = array.reduce(trim_std, dim=dim)
    return m_dist - s_dist * k, m_dist + s_dist * k


def _detect_outliers(
    array,
    flag_dim="epoch",
    outlier_method="quantile",
    flag_crit=0.2,
    init_dir="both",
    outliers_kwargs=None,
    plot_diagnostic=False,
):
    """Mark epochs, channels, or ICs as flagged for artefact.

    Parameters
    ----------
    array : xr.DataArray
        Array of shape n_channels, n_epochs, representing the stdev across
        time (samples in epoch) for each channel/epoch pair.
    dim : str
        One of 'ch' or 'epoch'. The dimension to operate across. For example
        if 'epoch', then detect epochs that are outliers.
    outlier_method : str (default quantile)
        one of 'quantile', 'trimmed', or 'fixed'.
    flag_crit : float
        Threshold (percentage) to consider an epoch or channel as bad. If
        operating across channels using default value, then if more then if
        the channel is an outlier in more than 20% of epochs, it will be
        flagged. if operating across epochs, then if more than 20% of channels
        are outliers in an epoch, it will be flagged as bad.
    init_dir : str
        One of 'pos', 'neg', or 'both'. Direction to test for outliers. If
        'pos', only detect outliers at the upper end of the distribution. If
        'neg', only detect outliers at the lower end of the distribution.
    outliers_kwargs : dict
        Set in the pipeline config. 'k', 'lower', and 'upper' kwargs can be
        passed to _get_outliers_quantile. 'k' can also be passed to
        _get_outliers_trimmed.
    plot_diagnostic : bool
        If True, plot the variance diagnostic plots of the criteria function.

    Returns
    -------
    boolean xr.DataArray of shape n_epochs, n_times, where an epoch x channel
    coordinate is 1 if it is to be flagged as bad.

    """
    if outliers_kwargs is None:
        outliers_kwargs = {}

    # Computing lower and upper bounds for outlier detection
    operate_dim = get_operate_dim(array, flag_dim)

    if outlier_method == "quantile":
        l_out, u_out = _get_outliers_quantile(array, flag_dim, **outliers_kwargs)

    elif outlier_method == "trimmed":
        l_out, u_out = _get_outliers_trimmed(array, flag_dim, **outliers_kwargs)

    elif outlier_method == "fixed":
        l_out, u_out = outliers_kwargs["lower"], outliers_kwargs["upper"]

    else:
        raise ValueError(
            "outlier_method must be 'quantile', 'trimmed'"
            f", or 'fixed'. Got {outlier_method}"
        )

    # Calculating the proportion of outliers along dimension operate_dim
    # and marking items along dimension flag_dim if this number is
    # larger than
    outlier_mask = xr.zeros_like(array, dtype=bool)

    if init_dir == "pos" or init_dir == "both":  # for positive outliers
        outlier_mask = outlier_mask | (array > u_out)

    if init_dir == "neg" or init_dir == "both":  # for negative outliers
        outlier_mask = outlier_mask | (array < l_out)

    # average column of outlier_mask
    # drop quantile coord because it is no longer needed
    prop_outliers = outlier_mask.astype(float).mean(operate_dim)
    if "quantile" in list(prop_outliers.coords.keys()):
        prop_outliers = prop_outliers.drop_vars("quantile")
    flagged_items = prop_outliers[prop_outliers > flag_crit].coords.to_index().values

    # Diagnostic plotting
    if plot_diagnostic:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 8))
        
        # Panel A: Voltage Variance
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        im = ax1.imshow(array.T, aspect='auto', cmap='viridis') # no transpose?
        ax1.set_ylabel('Chs/ ICs')
        ax1.set_ylim(ax1.get_ylim()[::-1])  # Invert y-axis
        
        # Panel B: Voltage Variance scatter plot
        ax2 = plt.subplot2grid((2, 2), (1, 0))
        mid_val = array.quantile(0.5, dim=flag_dim)
        x_coords = np.repeat(array.coords[operate_dim], len(array.coords[flag_dim]))
        y_vals = array.values.flatten()
        ax2.scatter(x_coords, y_vals, color='red', marker='+', alpha=0.5, s=20) # Plot the samples
        ax2.plot(array.coords[operate_dim], mid_val, color='black', label='Median') # Plot median line
        ax2.plot(array.coords[operate_dim], l_out, color='blue', label='Median-Quantile Distance')
        ax2.plot(array.coords[operate_dim], u_out, color='blue')
        ax2.set_ylabel('Variance')
        ax2.set_xlabel('Time')
        
        # Panel C: Flagging Criteria
        ax3 = plt.subplot2grid((2, 2), (0, 1))
        im2 = ax3.imshow(outlier_mask.T, aspect='auto', cmap='YlOrBr', vmin=0, vmax=1)
        ax3.set_ylim(ax3.get_ylim()[::-1])  # Invert y-axis
        
        # Panel D: Critical Cut-off
        ax4 = plt.subplot2grid((2, 2), (1, 1))
        ax4.plot(prop_outliers, range(len(prop_outliers)), 'b-')
        ax4.axvline(flag_crit, color='r', linestyle='--', label=f'Critical Cut-off ({flag_crit})')
        ax4.set_xlabel('Critical Cut-off')

        # TODO: come up with an intelligent figure title showing what things are operating on
        # Seems to be something in the 'kind' part of the xarray dealie
        fig.suptitle(flag_dim, fontsize=14, fontweight='bold') # flagged_items

        plt.tight_layout()
        plt.show()

    return flagged_items


def find_bads_by_threshold(epochs, threshold=5e-5):
    """Return channels with a standard deviation consistently above a fixed threshold.

    Parameters
    ----------
    epochs : mne.Epochs
        an instance of mne.Epochs with a single channel type.
    threshold : float
        the threshold in volts. If the standard deviation of a channel's voltage
        variance at a specific epoch is above the threshold, then that channel x epoch
        will be flagged as an "outlier". If more than 20% of epochs are flagged as
        outliers for a specific channel, then that channel will be flagged as bad.
        Default threshold is 5e-5 (0.00005), i.e. 50 microvolts.

    Returns
    -------
    list
        a list of channel names that are considered outliers.

    Notes
    -----
    If you are having trouble converting between exponential notation and
    decimal notation, you can use the following code to convert between the two:

    >>> import numpy as np
    >>> threshold = 5e-5
    >>> with np.printoptions(suppress=True):
    ...     print(threshold)
    0.00005

    .. seealso::

        :func:`~pylossless.LosslessPipeline.flag_channels_fixed_threshold` to use
        this function within the lossless pipeline.

    Examples
    --------
    >>> import mne
    >>> import pylossless as ll
    >>> fname = mne.datasets.sample.data_path() / "MEG/sample/sample_audvis_raw.fif"
    >>> raw = mne.io.read_raw(fname, preload=True).pick("eeg")
    >>> raw.apply_function(lambda x: x * 3, picks=["EEG 001"]) # Make a noisy channel
    >>> epochs = mne.make_fixed_length_epochs(raw, preload=True)
    >>> bad_chs = ll.pipeline.find_bads_by_threshold(epochs)
    """
    
    ch_types = np.unique(epochs.get_channel_types()).tolist()
    if len(ch_types) > 1:
        warn(
            f"The epochs object contains multiple channel types: {ch_types}.\n"
            " This will likely bias the results of the threshold detection."
            " Use the `mne.Epochs.pick` to select a single channel type."
        )
    bads = _threshold_volt_std(epochs, flag_dim="ch", threshold=threshold)
    return bads


def _threshold_volt_std(epochs, flag_dim, threshold=5e-5):
    """Detect epochs or channels whose voltage std is above threshold.

    Parameters
    ----------
    flag_dim : str
        The dimension to flag outlier in. 'ch' for channels, 'epoch'
        for epochs.
    threshold : float | tuple | list
        The threshold in volts. If the standard deviation of a channel's
        voltage variance at a specific epoch is above the threshold, then
        that channel x epoch will be flagged as an "outlier". If threshold
        is a single int or float, then it is treated as the upper threshold
            and the lower threshold is set to 0. Default is 5e-5, i.e.
            50 microvolts.
    """
    if isinstance(threshold, (tuple, list)):
        assert len(threshold) == 2
        l_out, u_out = threshold
        init_dir = "both"
    elif isinstance(threshold, (float, int)):
        l_out, u_out = (0, threshold)
        init_dir = "pos"
    else:
        raise ValueError(
            "threshold must be an int, float, or a list/tuple"
            f" of 2 int or float values. got {threshold}"
        )

    epochs_xr = epochs_to_xr(epochs, kind="ch")
    data_sd = epochs_xr.std("time")
    # Flag channels or epochs if their std is above
    # a fixed threshold.
    outliers_kwargs = dict(lower=l_out, upper=u_out)
    volt_outlier_inds = _detect_outliers(
        data_sd,
        flag_dim=flag_dim,
        outlier_method="fixed",
        init_dir=init_dir,
        outliers_kwargs=outliers_kwargs,
    )
    return volt_outlier_inds


def chan_neighbour_r(epochs, nneigbr, method):
    """Compute nearest Neighbor R.

    Parameters
    ----------
    epochs : mne.Epochs

    nneigbr : int
        Number of neighbours to compare in open interval

    method : str
        One of 'max', 'mean', or 'trimmean'. This is the function
        which aggregates the neighbours into one value.

    Returns
    -------
    Xarray : Xarray.DataArray
        An instance of Xarray.DataArray
    """
    chan_locs = pd.DataFrame(epochs.get_montage().get_positions()["ch_pos"]).T
    chan_dist = pd.DataFrame(
        distance_matrix(chan_locs, chan_locs),
        columns=chan_locs.index,
        index=chan_locs.index,
    )
    rank = chan_dist.rank("columns", ascending=True) - 1
    rank[rank == 0] = np.nan
    nearest_neighbor = pd.DataFrame(
        {
            ch_name: row.dropna().sort_values()[:nneigbr].index.values
            for ch_name, row in rank.iterrows()
        }
    ).T

    r_list = []
    for name, row in tqdm(list(nearest_neighbor.iterrows())):
        this_ch = epochs.get_data(name)
        nearest_chs = epochs.get_data(list(row.values))
        this_ch_xr = xr.DataArray(
            [this_ch * np.ones_like(nearest_chs)],
            dims=["ref_chan", "epoch", "channel", "time"],
            coords={
                "ref_chan": [name],
                "epoch": np.arange(len(epochs)),
                "channel": row.values.tolist(),
                "time": epochs.times,
            },
        )
        nearest_chs_xr = xr.DataArray(
            [nearest_chs],
            dims=["ref_chan", "epoch", "channel", "time"],
            coords={
                "ref_chan": [name],
                "epoch": np.arange(len(epochs)),
                "channel": row.values.tolist(),
                "time": epochs.times,
            },
        )
        r_list.append(xr.corr(this_ch_xr, nearest_chs_xr, dim=["time"]))

    c_neigbr_r = xr.concat(r_list, dim="ref_chan")

    if method == "max":
        m_neigbr_r = xr.apply_ufunc(np.abs, c_neigbr_r).max(dim="channel")

    elif method == "mean":
        m_neigbr_r = xr.apply_ufunc(np.abs, c_neigbr_r).mean(dim="channel")

    elif method == "trimmean":
        trim_mean_10 = partial(scipy.stats.trim_mean, proportiontocut=0.1)
        m_neigbr_r = xr.apply_ufunc(np.abs, c_neigbr_r).reduce(
            trim_mean_10, dim="channel"
        )

    return m_neigbr_r.rename(ref_chan="ch")


def coregister(
    raw_edf,
    fiducials="estimated",  # get fiducials from fsaverage
    show_coreg=False,
    verbose=False,
):
    """Coregister Raw object to `'fsaverage'`.

    Parameters
    ----------
    raw_edf : mne.io.Raw
        an instance of `mne.io.Raw` to coregister.
    fiducials : str (default 'estimated')
        fiducials to use for coregistration. if `'estimated'`, gets fiducials
        from fsaverage.
    show_coreg : bool (default False)
        If True, shows the coregistration result in a plot.
    verbose : bool | str (default False)
        sets the logging level for `mne.Coregistration`.

    Returns
    -------
    coregistration | numpy.array
        a numpy array containing the coregistration trans values.
    """
    plot_kwargs = dict(
        subject="fsaverage", surfaces="head-dense", dig=True, show_axes=True
    )

    coreg = Coregistration(raw_edf.info, "fsaverage", fiducials=fiducials)
    coreg.fit_fiducials(verbose=verbose)
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=verbose)

    if show_coreg:
        mne.viz.plot_alignment(raw_edf.info, trans=coreg.trans, **plot_kwargs)

    return coreg.trans["trans"][:-1].ravel()


# Warp locations to standard head surface:
def warp_locs(self, raw):
    """Warp locs.

    Parameters
    ----------
    raw : mne.io.Raw
        an instance of mne.io.Raw

    Returns
    -------
    None (operates in place)
    """
    if "montage_info" in self.config["replace_string"]:
        if isinstance(self.config["replace_string"]["montage_info"], str):
            pass
        else:
            pass
            # raw = (warp_locs(raw, c01_config['ref_loc_file'],
            # 'transform',[[montage_info]],
            # 'manual','off'))
            # MNE does not apply the transform to the montage permanently.


class LosslessPipeline:
    """Class used to handle pipeline parameters.

    Parameters
    ----------
    config_path : pathlib.Path
        path to config file specifying the parameters to be used
        in the pipeline.

    Attributes
    ----------
    flags : dict
        A dictionary of detailing the flagged channels, epochs, and ICs.
        keys are ``'ch'``, ``'epoch'``, and ``'ic'``, and values are instances of
        :class:`~pylossless.flagging.FlaggedChs`,
        :class:`~pylossless.flagging.FlaggedEpochs`, and
        :class:`~pylossless.flagging.FlaggedICs`, respectively.
    config_path : pathlib.Path
        path to the config file specifying the parameters to be used in the
        in the pipeline.
    config : dict
        A dictionary containing the pipeline parameters.
    raw : mne.io.Raw
        An instance of :class:`~mne.io.Raw` containing that will
        be processed by the pipeline.
    ica1 : mne.preprocessing.ICA
        An instance of :class:`~mne.preprocessing.ICA`. The result of the initial ICA
        run during the pipeline.
    ica2 : mne.preprocessing.ICA
        An instance of :class:`~mne.preprocessing.ICA`. The result of the final ICA run
        during the pipeline.
    """

    def __init__(self, config_path=None, config=None):
        """Initialize class.

        Parameters
        ----------
        config_path : pathlib.Path | str | None
            Path to config file specifying the parameters to be used in the pipeline.

        config : pylossless.config.Config | None
            :class:`pylossless.config.Config` object for the pipeline.
        """
        self.bids_path = None
        self.flags = {
            "ch": FlaggedChs(self),
            "epoch": FlaggedEpochs(self),
            "ic": FlaggedICs(),
        }
        self._config = None

        if config:
            self.config = config
            if config_path is None:
                self.config_path = "._tmp_pylossless.yaml"
        elif config_path:
            self.config_path = Path(config_path)
            self.load_config()
        else:
            self.config_path = None
        self.raw = None
        self.ica1 = None
        self.ica2 = None

    def _repr_html_(self):
        ch_flags = self.flags.get("ch", None)
        df = self.flags["ic"]

        eog = _get_ics(df, "eog")
        ecg = _get_ics(df, "ecg")
        muscle = _get_ics(df, "muscle")
        line_noise = _get_ics(df, "line_noise")
        channel_noise = _get_ics(df, "channel_noise")

        lossless_flags = [
            "BAD_LL_noisy",
            "BAD_LL_uncorrelated",
            "BAD_LL_noisy_ICs_1",
            "BAD_LL_noisy_ICs_2",
        ]
        flagged_times = _sum_flagged_times(self.raw, lossless_flags)

        config_path = self.config_path
        raw = self.raw.filenames if self.raw else "Not specified"

        html = "<h3>LosslessPipeline</h3>"
        html += "<table>"
        html += f"<tr><td><strong>Raw</strong></td><td>{raw}</td></tr>"
        html += f"<tr><td><strong>Config</strong></td><td>{config_path}</td></tr>"
        html += "</table>"

        # Flagged Channels
        flagged_channels_data = {
            "Noisy": ch_flags.get("noisy", None),
            "Bridged": ch_flags.get("bridged", None),
            "Uncorrelated": ch_flags.get("uncorrelated", None),
            "Rank": ch_flags.get("rank", None),
        }
        html += _create_html_details("Flagged Channels", flagged_channels_data)

        # Flagged ICs
        flagged_ics_data = {
            "EOG (Eye)": eog,
            "ECG (Heart)": ecg,
            "Muscle": muscle,
            "Line Noise": line_noise,
            "Channel Noise": channel_noise,
        }
        html += _create_html_details("Flagged ICs", flagged_ics_data)

        # Flagged Times
        flagged_times_data = flagged_times
        html += _create_html_details(
            "Flagged Times (Total)", flagged_times_data, times=True
        )

        return html

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config
        self._config["version"] = version("pylossless")

    @property
    def config_fname(self):
        warn('config_fname is deprecated and will be removed from future versions.',
             DeprecationWarning)
        return self.config_path

    @config_fname.setter
    def config_fname(self, config_path):
        warn('config_fname is deprecated and will be removed from future versions.',
             DeprecationWarning)
        self.config_path = config_path

    def load_config(self):
        """Load the config file."""
        self.config = Config().read(self.config_path)

    def _check_sfreq(self):
        """Make sure sampling frequency is an integer.

        If the sfreq is a float, it will cause slightly incorrect mappings
        from epochs to raw annotations. For example the annotation might start
        at 0.98 when it should start at 1, which will result in 2 epochs
        being dropped the next time data are epoched.
        """
        sfreq = self.raw.info["sfreq"]
        if not sfreq.is_integer():
            # we can't use f-strings in the logging module
            msg = (
                f"The Raw sampling frequency is {sfreq:.2f}. a non-integer "
                f"sampling frequency can cause incorrect mapping of epochs "
                f"to annotations. downsampling to {int(sfreq)}"
            )
            warn(msg)
            self.raw.resample(int(sfreq))
        return self.raw

    def set_montage(self):
        """Set the montage."""
        analysis_montage = self.config["project"]["analysis_montage"]
        if analysis_montage == "" and self.raw.get_montage() is not None:
            # No analysis montage has been specified and raw already has
            # a montage. Nothing to do; just return. This can happen
            # with a BIDS dataset automatically loaded with its corresponding
            # montage.
            return

        if analysis_montage in mne.channels.montage.get_builtin_montages():
            # If chanlocs is a string of one the standard MNE montages
            montage = mne.channels.make_standard_montage(analysis_montage)
            montage_kwargs = self.config["project"]["set_montage_kwargs"]
            self.raw.set_montage(montage, **montage_kwargs)
        else:  # If the montage is a filepath of a custom montage
            raise ValueError(
                'self.config["project"]["analysis_montage"]'
                " should be one of the default MNE montages as"
                " specified by"
                " mne.channels.get_builtin_montages()."
            )
            # montage = read_custom_montage(chan_locs)

    def add_pylossless_annotations(self, inds, event_type, epochs):
        """Add annotations for flagged epochs.

        Parameters
        ----------
        inds : list | tuple
            indices corresponding to artefactual epochs
        event_type : str
            One of 'ch_sd', 'low_r', 'ic_sd1'
        epochs : mne.Epochs
            an instance of mne.Epochs
        """
        # Concatenate epoched data back to continuous data
        t_onset = epochs.events[inds, 0] / epochs.info["sfreq"]
        desc = f"BAD_LL_{event_type}"
        df = pd.DataFrame(t_onset, columns=["onset"])
        # We exclude the last sample from the duration because
        # if the annot lasts the whole duration of the epoch
        # it's end will coincide with the first sample of the
        # next epoch, causing it to erroneously be rejected.
        df["duration"] = 1 / epochs.info["sfreq"] * len(epochs.times[:-1])
        df["description"] = desc

        # Merge close onsets to prevent a bunch of 1-second annotations of the same name
        # find onsets close enough to be considered the same
        df["close"] = df.sort_values("onset")["onset"].diff().le(1)
        df["group"] = ~df["close"]
        df["group"] = df["group"].cumsum()
        # group the close onsets and merge them
        df["onset"] = df.groupby("group")["onset"].transform("first")
        df["duration"] = df.groupby("group")["duration"].transform("sum")
        df = df.drop_duplicates(subset=["onset", "duration"])

        annotations = mne.Annotations(
            df["onset"],
            df["duration"],
            df["description"],
            orig_time=self.raw.annotations.orig_time,
        )
        self.raw.set_annotations(self.raw.annotations + annotations)
        _report_flagged_epochs(self.raw, desc)

    def get_events(self):
        """Make an MNE events array of fixed length events."""
        tmin = self.config["epoching"]["epochs_args"]["tmin"]
        tmax = self.config["epoching"]["epochs_args"]["tmax"]
        overlap = self.config["epoching"]["overlap"]
        return mne.make_fixed_length_events(
            self.raw, duration=tmax - tmin, overlap=overlap
        )

    def get_epochs(self, detrend=None, preload=True,
                   rereference=True, picks="eeg"):
        """Create mne.Epochs according to user arguments.

        Parameters
        ----------
        detrend : int | None (default None)
            If 0 or 1, the data channels (MEG and EEG) will be detrended when
            loaded. 0 is a constant (DC) detrend, 1 is a linear detrend.None is
            no detrending. Note that detrending is performed before baseline
            correction. If no DC offset is preferred (zeroth order detrending),
            either turn off baseline correction, as this may introduce a DC
            shift, or set baseline correction to use the entire time interval
            (will yield equivalent results but be slower).
        preload : bool (default True)
            Load epochs from disk when creating the object or wait before
            accessing each epoch (more memory efficient but can be slower).
        picks : str (default "eeg")
            Type of channels to pick.

        Returns
        -------
        Epochs : mne.Epochs
            an instance of mne.Epochs
        """
        # TODO: automatically load detrend/preload description from MNE.
        logger.info("🧹 Epoching..")
        events = self.get_events()
        epoching_kwargs = deepcopy(self.config["epoching"]["epochs_args"])

        # MNE epoching is end-inclusive, causing an extra time
        # sample be included. This removes that extra sample:
        # https://github.com/mne-tools/mne-python/issues/6932
        epoching_kwargs["tmax"] -= 1 / self.raw.info["sfreq"]
        if detrend is not None:
            epoching_kwargs["detrend"] = detrend
        epochs = mne.Epochs(self.raw, events=events, preload=preload, **epoching_kwargs)
        epochs = epochs.pick(picks=picks, exclude="bads").pick(
            picks=None, exclude=list(self.flags["ch"].get_flagged())
        )
        if rereference and picks=="eeg":
            self.flags["ch"].rereference(epochs)

        return epochs

    def run_staging_script(self):
        """Run a staging script if specified in config."""

        if "staging_script" in self.config:
            staging_script = Path(self.config["staging_script"])
            if staging_script.exists():
                exec(staging_script.open().read())

    @lossless_logger
    def find_breaks(self):
        """Find breaks using mne.preprocessing.annotate_break.

        Parameters
        ----------
        kwargs : dict
            a dict with keys that are valid keyword arguments for
            mne.preprocessing.annotate_break, and with values that
            are valid for their respective arguments.

        Notes
        -----
        ``kwargs=dict(min_break_duration=15.0, t_start_after_previous=5.0)``
        for example would unpack two keyword arguments from
        mne.preprocessing.annotate_break
        """
        if "find_breaks" not in self.config or not self.config["find_breaks"]:
            return
        if not self.raw.annotations:
            logger.debug("No annotations found in raw object. Skipping find_breaks.")
            return
        breaks = annotate_break(self.raw, **self.config["find_breaks"])
        self.raw.set_annotations(breaks + self.raw.annotations)

    def _flag_volt_std(self, flag_dim, threshold=5e-5, picks="eeg"):
        """Determine if voltage standard deviation is above threshold.

        Parameters
        ----------
        flag_dim : str
            Whether to flag epochs or channels. 'ch' for channels, 'epoch'
            for epochs.
        threshold : float
            threshold, in volts. If the standard deviation across time in
            any channel x epoch indice is above this threshold, then the
            channel x epoch indices will considered an outlier. Defaults
            to 5e-5, or 50 microvolts. Note that here, 'time' refers to
            the samples in an epoch.
        picks : str (default "eeg")
            Type of channels to pick.

        Notes
        -----
        This method takes an array of shape n_channels x n_epochs x n_times
        and calculates the standard deviation across the time dimension (i.e.
        across the samples in each epoch, for each channel) - which returns
        an array of shape n_channels x n_epochs, where each element of the
        array is the std value of that channel x epoch indice. For each
        channel, if its std value is above the given threshold for more than
        20% of the epochs, it is flagged. For each epoch, if the std value of
        more than 20% of channels (in that epoch) is above the threshold, it
        is flagged. A cutoff threshold other than 20% can be provided, if set
        in the config.

        WARNING: the default threshold of 50 microvolts may not be appropriate
        for a particular dataset or data file, as the baseline voltage variance
        is affected by the impedance of the system that the data was recording
        on. You may need to assess a more appropriate value for your own data.
        """
        epochs = self.get_epochs(picks=picks)
        if flag_dim == "ch":
            above_threshold = find_bads_by_threshold(epochs, threshold=threshold)
            if above_threshold.any():
                logger.info(
                    f"🚩 Found {len(above_threshold)} channels with "
                    f"voltage variance above {threshold} volts: {above_threshold}"
                )
            else:
                msg = f"No channels with standard deviation above {threshold} volts."
                logger.info(msg)
        else:
            above_threshold = _threshold_volt_std(
                epochs, flag_dim=flag_dim, threshold=threshold
            )
        self.flags[flag_dim].add_flag_cat("volt_std", above_threshold, epochs)

    def find_outlier_chs(self, epochs=None, picks="eeg"):
        """Detect outlier Channels to leave out of rereference.

        Parameters
        ----------
        epochs : mne.Epochs | None
            An instance of :class:`mne.Epochs`, or ``None``. If ``None``, then
            :attr:`pylossless.LosslessPipeline.raw` should be set, and this
            method will call :meth:`pylossless.LosslessPipeline.get_epochs`
            to create epochs to use for outlier detection.
        picks : str (default "eeg")
            Channels to include in the outlier detection process. You can pass any
            argument that is valid for the :meth:`~mne.Epochs.pick` method, but
            you should avoid passing a mix of channel types with differing units of
            measurement (e.g. EEG and MEG), as this would likely lead to incorrect
            outlier detection (e.g. all EEG channels would be flagged as outliers).

        Returns
        -------
        list
            a list of channel names that are considered outliers.

        Notes
        -----
        - This method is used to detect channels that are so noisy that they
          should be left out of the robust average rereference process.

        Examples
        --------
        >>> import mne
        >>> import pylossless as ll
        >>> config = ll.Config().load_default()
        >>> pipeline = ll.LosslessPipeline(config=config)
        >>> fname = mne.datasets.sample.data_path() / "MEG/sample/sample_audvis_raw.fif"
        >>> raw = mne.io.read_raw(fname)
        >>> epochs = mne.make_fixed_length_epochs(raw, preload=True)
        >>> chs_to_leave_out = pipeline.find_outlier_chs(epochs=epochs)
        """
        
        logger.info("🔍 Detecting channels to leave out of reference.")
        if epochs is None:
            epochs = self.get_epochs(rereference=False)
        epochs = epochs.copy().pick(picks=picks)
        epochs_xr = epochs_to_xr(epochs, kind="ch")

        # Determines comically bad channels,
        # and leaves them out of average rereference
        trim_ch_sd = epochs_xr.std("time")
        # Measure how diff the std of 1 channel is with respect
        # to other channels (nonparametric z-score)
        ch_dist = trim_ch_sd - trim_ch_sd.median(dim="ch")
        perc_30 = trim_ch_sd.quantile(0.3, dim="ch")
        perc_70 = trim_ch_sd.quantile(0.7, dim="ch")
        ch_dist /= perc_70 - perc_30  # shape (chans, epoch)

        mean_ch_dist = ch_dist.mean(dim="epoch")  # shape (chans)

        # find the median and 30 and 70 percentiles
        # of the mean of the channel distributions
        mdn = np.median(mean_ch_dist)
        deviation = np.diff(np.quantile(mean_ch_dist, [0.3, 0.7]))

        return mean_ch_dist.ch[mean_ch_dist > mdn + 6 * deviation].values.tolist()

    @lossless_logger
    def flag_channels_fixed_threshold(self, threshold=5e-5, picks="eeg"):
        """Flag channels based on the stdev value across the time dimension.

        Flags channels if the voltage-variance standard deviation is above
        the given threshold in n_percent of epochs (default: 20%).

        Parameters
        ----------
        threshold : float
            threshold, in volts. If the standard deviation across time in
            any channel x epoch indice is above this threshold, then the
            channel x epoch indices will be considered an outlier. Defaults
            to 5e-5, or 50 microvolts. Note that here, 'time' refers to
            the samples in an epoch. For each channel, if its std value is
            above the given threshold in more than 20% of the epochs, it
            is flagged.
        picks : str (default "eeg")
            Type of channels to pick.

        Returns
        -------
        None
            If any channels are flagged, those channel names will be logged
            in the `flags` attribute of the `LosslessPipeline` object,
            under the key ``'volt_std'``, e.g.
            ``my_pipeline.flags["ch"]["volt_std"]``.

        Notes
        -----
        .. warning::

            the default threshold of 50 microvolts may not be appropriate
            for a particular dataset or data file, as the baseline voltage variance
            is affected by the impedance of the system that the data was recorded
            with. You may need to assess a more appropriate value for your own
            data. You can use the :func:`~pylossless.pipeline.find_bads_by_threshold`
            function to quickly assess a more appropriate threshold.

        .. seealso::

            :func:`~pylossless.pipeline.find_bads_by_threshold`

        Examples
        --------
        >>> import mne
        >>> import pylossless as ll
        >>> config = ll.Config().load_default()
        >>> config["flag_channels_fixed_threshold"] = {"threshold": 5e-5}
        >>> pipeline = ll.LosslessPipeline(config=config)
        >>> sample_fpath = mne.datasets.sample.data_path()
        >>> fpath = sample_fpath / "MEG" / "sample" / "sample_audvis_raw.fif"
        >>> raw = mne.io.read_raw(fpath).pick("eeg")
        >>> pipeline.raw = raw
        >>> pipeline.flag_channels_fixed_threshold()
        """
        self._flag_volt_std(flag_dim="ch", threshold=threshold, picks=picks)

    def flag_epochs_fixed_threshold(self, threshold=5e-5, picks="eeg"):
        """Flag epochs based on the stdev value across the time dimension.

        Flags an epoch if the voltage-variance standard deviation is above
        the given threshold in n_percent of channels (default: 20%).

        Parameters
        ----------
        threshold : float
            threshold, in volts. If the standard deviation across time in
            any channel x epoch indice is above this threshold, then the
            channel x epoch indices will considered an outlier. Defaults
            to 5e-5, or 50 microvolts. Note that here, 'time' refers to
            the samples in an epoch. For each epoch, if the std value of
            more than 20% of channels (in that epoch) are above the given
            threshold, the epoch is flagged.
        picks : str (default "eeg")
            Type of channels to pick.

        Notes
        -----
        WARNING: the default threshold of 50 microvolts may not be appropriate
        for a particular dataset or data file, as the baseline voltage variance
        is affected by the impedance of the system that the data was recorded
        with. You may need to assess a more appropriate value for your own
        data.
        """
        if "flag_epochs_fixed_threshold" not in self.config:
            return
        if "threshold" in self.config["flag_epochs_fixed_threshold"]:
            threshold = self.config["flag_epochs_fixed_threshold"]["threshold"]
        self._flag_volt_std(flag_dim="epoch", threshold=threshold, picks="eeg")

    @lossless_logger
    def flag_noisy_channels(self, picks="eeg"):
        """Flag channels with outlying standard deviation.

        Calculates the standard deviation of the voltage-variance for
        each channel at each epoch (default: 1-second epochs). Then, for each
        epoch, creates a distribution of the stdev values of all channels.
        Then, for each epoch, estimates a stdev outlier threshold, where
        any channel that has an stdev value higher than the threshold (in the
        current epoch) is flagged. If a channel is flagged as an outlier in
        more than n_percent of epochs (default: 20%), the channel is flagged
        for removal.

        Parameters
        ----------
        picks : str (default "eeg")
            Type of channels to pick.
        """

        epochs_xr = epochs_to_xr(self.get_epochs(picks=picks), kind="ch")
        data_sd = epochs_xr.std("time")

        # flag noisy channels
        bad_ch_names = _detect_outliers(
            data_sd, flag_dim="ch", init_dir="pos", **self.config["noisy_channels"]
        )
        logger.info(f"📋 LOSSLESS: Noisy channels: {bad_ch_names}")

        self.flags["ch"].add_flag_cat(kind="noisy", bad_ch_names=bad_ch_names)

    @lossless_logger
    def flag_noisy_epochs(self, picks="eeg"):
        """Flag epochs with outlying standard deviation.

        Parameters
        ----------
        picks : str (default "eeg")
            Type of channels to pick.
        """
        outlier_methods = ("quantile", "trimmed", "fixed")
        epochs = self.get_epochs(picks=picks)
        epochs_xr = epochs_to_xr(epochs, kind="ch")
        data_sd = epochs_xr.std("time")

        # flag noisy epochs
        if "noisy_epochs" in self.config:
            config_epoch = self.config["noisy_epochs"]
            if "outlier_method" in config_epoch:
                if config_epoch["outlier_method"] is None:
                    del config_epoch["outlier_method"]
                elif config_epoch["outlier_method"] not in outlier_methods:
                    raise NotImplementedError
        bad_epoch_inds = _detect_outliers(
            data_sd, flag_dim="epoch", init_dir="pos", **config_epoch
        )
        self.flags["epoch"].add_flag_cat("noisy", bad_epoch_inds, epochs)

    def get_n_nbr(self, picks="eeg"):
        """Calculate nearest neighbour correlation for channels.

        Parameters
        ----------
        picks : str (default "eeg")
            Type of channels to pick.
        """
        # Calculate nearest neighbour correlation on
        # non-flagged channels and epochs...
        epochs = self.get_epochs(picks=picks)
        n_nbr_ch = self.config["nearest_neighbors"]["n_nbr_ch"]
        return chan_neighbour_r(epochs, n_nbr_ch, "max"), epochs

    @lossless_logger
    def flag_uncorrelated_channels(self, picks="eeg"):
        """Check neighboring channels for too high or low of a correlation.

        Parameters
        ----------
        picks : str (default "eeg")
            Type of channels to pick.

        Returns
        -------
        data array : `numpy.array`
            an instance of `numpy.array`
        """
        # Calculate nearest neighbour correlation on
        # non-flagged channels and epochs...
        data_r_ch = self.get_n_nbr(picks=picks)[0]

        # Create the window criteria vector for flagging low_r chan_info...
        bad_ch_names = _detect_outliers(
            data_r_ch,
            flag_dim="ch",
            init_dir="neg",
            **self.config["uncorrelated_channels"],
        )
        logger.info(f"📋 LOSSLESS: Uncorrelated channels: {bad_ch_names}")
        # Edit the channel flag info structure
        self.flags["ch"].add_flag_cat(kind="uncorrelated", bad_ch_names=bad_ch_names)
        return data_r_ch

    @lossless_logger
    def flag_bridged_channels(self, data_r_ch):
        """Flag bridged channels.

        Parameters
        ----------
        data_r_ch : `numpy.array`
            an instance of `numpy.array`
        """
        # Uses the correlation of neighbours
        # calculated to flag bridged channels.

        msr = data_r_ch.median("epoch") / data_r_ch.reduce(scipy.stats.iqr, dim="epoch")

        trim = self.config["bridged_channels"]["bridge_trim"]
        if trim >= 1:
            trim /= 100
        trim /= 2

        trim_mean = partial(scipy.stats.mstats.trimmed_mean, limits=(trim, trim))
        trim_std = partial(scipy.stats.mstats.trimmed_std, limits=(trim, trim))

        z_val = self.config["bridged_channels"]["bridge_z"]
        mask = msr > msr.reduce(trim_mean, dim="ch") + z_val * msr.reduce(
            trim_std, dim="ch"
        )

        bad_ch_names = data_r_ch.ch.values[mask]
        logger.info(f"📋 LOSSLESS: Bridged channels: {bad_ch_names}")
        self.flags["ch"].add_flag_cat(kind="bridged", bad_ch_names=bad_ch_names)

    @lossless_logger
    def flag_rank_channel(self, data_r_ch):
        """Flag the channel that is the least unique.

        Flags the channel that is the least unique, the channel to remove prior
        to ICA in order to account for the rereference rank deficiency.

        Parameters
        ----------
        data_r_ch : `numpy.array`.
            an instance of `numpy.array`.
        """
        if len(self.flags["ch"].get_flagged()):
            ch_sel = [
                ch
                for ch in data_r_ch.ch.values
                if ch not in self.flags["ch"].get_flagged()
            ]
            data_r_ch = data_r_ch.sel(ch=ch_sel)

        bad_ch_names = [str(data_r_ch.median("epoch").idxmax(dim="ch").to_numpy())]
        logger.info(f"📋 LOSSLESS: Rank channel: {bad_ch_names}")
        self.flags["ch"].add_flag_cat(kind="rank", bad_ch_names=bad_ch_names)

    @lossless_logger
    def flag_uncorrelated_epochs(self, picks="eeg"):
        """Flag epochs where too many channels are uncorrelated.

        Parameters
        ----------
        picks : str (default "eeg")
            Type of channels to pick.

        Notes
        -----
        Similarly to the neighbor r calculation done between channels this
        section looks at the correlation, but between all channels and for
        epochs of time. Time segments are flagged for removal.
        """
        # Calculate nearest neighbour correlation on
        # non-flagged channels and epochs...
        data_r_ch, epochs = self.get_n_nbr(picks=picks)

        bad_epoch_inds = _detect_outliers(
            data_r_ch,
            flag_dim="epoch",
            init_dir="neg",
            **self.config["uncorrelated_epochs"],
        )
        self.flags["epoch"].add_flag_cat("uncorrelated", bad_epoch_inds, epochs)

    @lossless_logger
    def run_ica(self, run, picks="eeg"):
        """Run ICA.

        Parameters
        ----------
        run : str
            Must be 'run1' or 'run2'. 'run1' is the initial ICA use to flag
            epochs, 'run2' is the final ICA used to classify components with
            `mne_icalabel`.
        picks : str (default "eeg")
            Type of channels to pick.
        """
        ica_kwargs = self.config["ica"]["ica_args"][run]
        if "max_iter" not in ica_kwargs:
            ica_kwargs["max_iter"] = "auto"
        if "random_state" not in ica_kwargs:
            ica_kwargs["random_state"] = 97

        epochs = self.get_epochs(picks=picks)
        if run == "run1":
            self.ica1 = ICA(**ica_kwargs)
            self.ica1.fit(epochs)

        elif run == "run2":
            self.ica2 = ICA(**ica_kwargs)
            self.ica2.fit(epochs)
            if picks == "eeg":
                self.flags["ic"].label_components(epochs, self.ica2)
        else:
            raise ValueError("The `run` argument must be 'run1' or 'run2'")

    @lossless_logger
    def flag_noisy_ics(self, run_id, picks="eeg"):
        """Calculate the IC standard Deviation by epoch window.

        Flags windows with too many ICs with outlying standard deviations.

        Parameters
        ----------
        run_id : int
            Which pass of noisy_ic, i.e. ic_s_sd1 vs ic_s_sd2
        picks : str (default "eeg")
            Type of channels to pick.
        """
        # Calculate IC sd by window
        epochs = self.get_epochs(picks=picks)
        epochs_xr = epochs_to_xr(epochs, kind="ic", ica=self.ica1)
        data_sd = epochs_xr.std("time")

        # Create the windowing sd criteria
        kwargs = self.config["ica"]["noisy_ic_epochs"]
        bad_epoch_inds = _detect_outliers(data_sd, flag_dim="epoch", **kwargs)

        self.flags["epoch"].add_flag_cat(f"noisy_ICs_{run_id}", bad_epoch_inds, epochs)

        # icsd_epoch_flags=padflags(raw, icsd_epoch_flags,1,'value',.5);

    def non_bids_save(self, subject_label, root_save_dir, overwrite=False, format="EDF", event_id=None):
        """Forcibly saves the pipeline output skipping the derivative BIDS
        path requirement. Will still write a passing derivative based on
        given parameters.
    
        Parameters
        ----------
        subject_label : str
            Subject ID to save the recording under.
        root_save_dir : str
            Where to begin creating the "derivatives" folder from
        """
        
        forced_path = mne_bids.BIDSPath(subject=subject_label, task='pyl', root=root_save_dir, datatype='eeg')
        forced_path = self.get_derivative_path(forced_path)
        self.save(derivatives_path=forced_path, overwrite=overwrite, format=format, event_id=event_id)

    def save(self, derivatives_path=None, overwrite=False, format="EDF", event_id=None):
        """Save the file at the end of the pipeline.

        Parameters
        ----------
        derivatives_path : None | mne_bids.BIDSPath
            path of the derivatives folder to save the file to.
        overwrite : bool (default False)
            whether to overwrite existing files with the same name.
        format : str (default "EDF")
            The format to use for saving the raw data. Can be ``"auto"``,
            ``"FIF"``, ``"EDF"``, ``"BrainVision"``, ``"EEGLAB"``.
        event_id : dict | None (default None)
            Dictionary mapping annotation descriptions to event codes.
        """
        if derivatives_path is None:
            derivatives_path = self.get_derivative_path(self.bids_path)

        mne_bids.write_raw_bids(
            self.raw,
            derivatives_path,
            overwrite=overwrite,
            format=format,
            allow_preload=True,
            event_id=event_id,
        )

        # Save ICAs
        bpath = derivatives_path.copy()
        for (
            this_ica,
            self_ica,
        ) in zip(["ica1", "ica2"], [self.ica1, self.ica2]):
            suffix = this_ica + "_ica"
            ica_bidspath = bpath.update(extension=".fif", suffix=suffix, check=False)
            self_ica.save(ica_bidspath, overwrite=overwrite)

        # Save IC labels
        iclabels_bidspath = bpath.update(
            extension=".tsv", suffix="iclabels", check=False
        )
        self.flags["ic"].save_tsv(iclabels_bidspath)

        # raw.save(derivatives_path, overwrite=True, split_naming='bids')
        config_bidspath = bpath.update(
            extension=".yaml", suffix="ll_config", check=False
        )

        self.config.save(config_bidspath)

        # Save flag["ch"]
        flagged_chs_fpath = bpath.update(
            extension=".tsv", suffix="ll_FlaggedChs", check=False
        )
        self.flags["ch"].save_tsv(flagged_chs_fpath.fpath)

    @lossless_logger
    def filter(self):
        """Run filter procedure based on structured config args."""
        # 5.a. Filter lowpass/highpass
        self.raw.filter(**self.config["filtering"]["filter_args"])

        # 5.b. Filter notch
        if "notch_filter_args" in self.config["filtering"]:
            notch_args = self.config["filtering"]["notch_filter_args"]
            spectrum_fit_method = (
                "method" in notch_args and notch_args["method"] == "spectrum_fit"
            )
            if notch_args["freqs"] or spectrum_fit_method:
                # in raw.notch_filter, freqs=None is ok if method=='spectrum_fit'
                self.raw.notch_filter(**notch_args)
            else:
                logger.info("No notch filter arguments provided. Skipping")
        else:
            logger.info("No notch filter arguments provided. Skipping")

    def run(self, bids_path, save=True, overwrite=False):
        """Run the pylossless pipeline.

        Parameters
        ----------
        bids_path : `pathlib.Path`
            Path of the individual file `bids_root`.
        save : bool (default True).
            Whether to save the files after completing the pipeline. Defaults
            to `True`. if `False`, files are not saved.
        overwrite : bool (default False).
            Whether to overwrite existing files of the same name.
        """
        # Linter ID'd below as bad practice - likely need a structure fix
        self.bids_path = bids_path
        self.raw = mne_bids.read_raw_bids(self.bids_path)
        self.raw.load_data()
        self._run()

        if save:
            self.save(self.get_derivative_path(bids_path), overwrite=overwrite)

    def run_with_raw(self, raw):
        """Execute pipeline on a raw object."""
        self.raw = raw
        self._run()
        return self.raw

    @lossless_time
    def _run(self):

        # Make sure sampling frequency is an integer
        self._check_sfreq()
        self.set_montage()

        if "modality" not in self.config:
            self.config["modality"] = ["eeg"]
        if isinstance(self.config["modality"], str):
            self.config["modality"] = [self.config["modality"]]

        for picks in self.config["modality"]:
            # 1. Execute the staging script if specified.
            self.run_staging_script()

            # find breaks
            self.find_breaks(message="Looking for break periods between tasks")

            # OPTIONAL: Flag chs/epochs based off fixed std threshold of time axis
            self.flag_epochs_fixed_threshold(picks=picks)
            if "flag_channels_fixed_threshold" in self.config:
                msg = "Flagging Channels by fixed threshold"
                kwargs = dict(picks=picks, message=msg)
                if "threshold" in self.config["flag_channels_fixed_threshold"]:
                    threshold = self.config["flag_channels_fixed_threshold"][
                        "threshold"
                    ]
                    kwargs["threshold"] = threshold
                self.flag_channels_fixed_threshold(**kwargs)

            # 3.flag channels based on large Stdev. across time
            msg = "Flagging Noisy Channels"
            self.flag_noisy_channels(message=msg, picks=picks)

            # 4.flag epochs based on large Channel Stdev. across time
            msg = "Flagging Noisy Time periods"
            self.flag_noisy_epochs(message=msg, picks=picks)

            # 5. Filtering
            self.filter(message="Filtering")

            if picks == "eeg":
                # These steps are relevant only for EEG. For example,
                # MEG channels don't get bridged or high impedance.
                # Further, MEG doesn't have a montage so
                # flag_uncorrelated_channels would crash. We could use
                # mne.channels.read_layout() for MEG channel if we need
                # at some point to implement such a functionality. But it
                # seems irrelevant.

                # 6. calculate nearest neighbort r values
                msg = "Flagging uncorrelated channels"
                data_r_ch = self.flag_uncorrelated_channels(message=msg, picks=picks)

                # 7. Identify bridged channels
                msg = "Flagging Bridged channels"
                self.flag_bridged_channels(data_r_ch, message=msg)

            # 8. Flag rank channels
            self.flag_rank_channel(data_r_ch, message="Flagging the rank channel")

            if picks == "eeg":
                # 9. Calculate nearest neighbour R values for epochs
                msg = "Flagging Uncorrelated epochs"
                self.flag_uncorrelated_epochs(message=msg, picks=picks)

            if self.config["ica"] is None:
                # Skip ICA steps.
                continue

            # 10. Run ICA
            self.run_ica("run1", message="Running Initial ICA", picks=picks)

            # 11. Calculate IC SD 1
            msg = "Flagging time periods with noisy IC's in first ICA."
            self.flag_noisy_ics(message=msg, run_id=1, picks=picks)

            # 12. Run second ICA
            msg = "Running Final ICA and ICLabel."
            self.run_ica("run2", message=msg, picks=picks)

            # 13. Calculate IC SD 2
            msg = "Flagging time periods with noisy IC's in second ICA."
            self.flag_noisy_ics(message=msg, run_id=2, picks=picks)

    def run_dataset(self, paths):
        """Run a full dataset.

        Parameters
        ----------
        paths : list | tuple
            a list of the bids_paths for all recordings in the dataset that
            should be run.
        """
        for path in paths:
            self.run(path)

    def load_ll_derivative(self, derivatives_path):
        """Load a completed pylossless derivative state.

        Parameters
        ----------
        derivatives_path : str | mne_bids.BIDSPath
            Path to a saved pylossless derivatives.

        Returns
        -------
        :class:`~pylossless.pipeline.LosslessPipeline`
            Returns an instance of :class:`~pylossless.pipeline.LosslessPipeline`
            for the loaded pylossless derivative state.
        """
        if not isinstance(derivatives_path, BIDSPath):
            derivatives_path = get_bids_path_from_fname(derivatives_path)
        self.raw = mne_bids.read_raw_bids(derivatives_path)
        bpath = derivatives_path.copy()
        # Load ICAs
        for this_ica in ["ica1", "ica2"]:
            suffix = this_ica + "_ica"
            ica_bidspath = bpath.update(extension=".fif", suffix=suffix, check=False)
            setattr(self, this_ica, mne.preprocessing.read_ica(ica_bidspath.fpath))

        # Load IC labels
        iclabels_bidspath = bpath.update(
            extension=".tsv", suffix="iclabels", check=False
        )
        self.flags["ic"].load_tsv(iclabels_bidspath.fpath)

        self.config_path = bpath.update(
            extension=".yaml", suffix="ll_config", check=False
        )
        self.load_config()

        # Load Flagged Chs
        flagged_chs_fpath = bpath.update(
            extension=".tsv", suffix="ll_FlaggedChs", check=False
        )
        self.flags["ch"].load_tsv(flagged_chs_fpath.fpath)

        # Load Flagged Epochs
        self.flags["epoch"].load_from_raw(self.raw, self.get_events(), self.config)

        return self

    def get_derivative_path(self, bids_path, derivative_name="pylossless"):
        """Build derivative path for file."""
        lossless_suffix = bids_path.suffix if bids_path.suffix else ""
        lossless_suffix += "_ll"
        lossless_root = bids_path.root / "derivatives" / derivative_name
        return bids_path.copy().update(
            suffix=lossless_suffix, root=lossless_root, check=False
        )

    def get_all_event_ids(self):
        """
        Get a combined event ID dictionary from existing markers and raw annotations.

        Returns
        -------
        dict or None
            A combined dictionary of event IDs, including both existing markers
            and new ones from annotations.
            Returns ``None`` if no events or annotations are found.
        """
        try:
            # Get existing events and their IDs
            event_id = mne.events_from_annotations(self.raw)[1]
        except ValueError as e:
            warn(f"Warning: No events found in raw data. Error: {e}")
            event_id = {}

        # Check if there are any annotations
        if len(self.raw.annotations) == 0 and not event_id:
            warn("Warning: No events or annotations found in the raw data.")
            return

        # Initialize the combined event ID dictionary with existing events
        combined_event_id = event_id.copy()

        # Determine the starting ID for new annotations
        start_id = max(combined_event_id.values()) + 1 if combined_event_id else 1

        # Get unique annotations and add new event IDs
        for desc in set(self.raw.annotations.description):
            if desc not in combined_event_id:
                combined_event_id[desc] = start_id
                start_id += 1

        # Final check to ensure we have at least one event
        if not combined_event_id:
            warn("Warning: No valid events or annotations could be processed.")
            return

        return combined_event_id
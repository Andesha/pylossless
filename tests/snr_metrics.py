"""
SNR (Signal-to-Noise Ratio) metrics for EEG data quality assessment.

This module provides functions to calculate various SNR metrics for EEG data
processed with MNE-Python, allowing quantification of signal improvement
after preprocessing steps like filtering, ICA, and artifact rejection.
"""

import numpy as np
import mne
from typing import Tuple, Optional, Union, Dict


def time_domain_snr(raw_before: mne.io.Raw, 
                    raw_after: mne.io.Raw,
                    picks: str = 'eeg',
                    tmin: Optional[float] = None,
                    tmax: Optional[float] = None) -> Dict[str, float]:
    """
    Calculate time-domain SNR improvement after processing.
    
    This function compares the variance of EEG signals before and after
    processing to quantify the improvement in signal quality.
    
    Parameters
    ----------
    raw_before : mne.io.Raw
        Raw data before processing
    raw_after : mne.io.Raw
        Raw data after processing
    picks : str
        Channel types to include (default: 'eeg')
    tmin : float | None
        Start time for analysis in seconds (default: None, uses beginning of data)
    tmax : float | None
        End time for analysis in seconds (default: None, uses end of data)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'snr_before': SNR estimate before processing
        - 'snr_after': SNR estimate after processing
        - 'improvement_ratio': Ratio of after/before SNR
        - 'improvement_db': Improvement in dB
    """
    # Get data from before processing
    data_before, times = raw_before.get_data(picks=picks, 
                                            tmin=tmin, 
                                            tmax=tmax, 
                                            return_times=True)
    
    # Get data from after processing
    data_after, _ = raw_after.get_data(picks=picks, 
                                      tmin=tmin, 
                                      tmax=tmax, 
                                      return_times=True)
    
    # Calculate variance across time for each channel
    var_before = np.var(data_before, axis=1)
    var_after = np.var(data_after, axis=1)
    
    # Estimate noise as the difference between before and after
    noise_estimate = np.var(data_before - data_after, axis=1)
    
    # Calculate SNR for before and after
    # SNR = signal variance / noise variance
    snr_before = np.mean(var_before / noise_estimate)
    snr_after = np.mean(var_after / noise_estimate)
    
    # Calculate improvement
    improvement_ratio = snr_after / snr_before
    improvement_db = 10 * np.log10(improvement_ratio)
    
    return {
        'snr_before': float(snr_before),
        'snr_after': float(snr_after),
        'improvement_ratio': float(improvement_ratio),
        'improvement_db': float(improvement_db)
    }


def frequency_domain_snr(raw_before: mne.io.Raw, 
                         raw_after: mne.io.Raw,
                         fmin: float = 8.0,
                         fmax: float = 12.0,
                         noise_fmin: float = 50.0,
                         noise_fmax: float = 60.0,
                         picks: str = 'eeg') -> Dict[str, float]:
    """
    Calculate frequency-domain SNR improvement after processing.
    
    This function compares the power in a frequency band of interest relative
    to a noise frequency band, before and after processing.
    
    Parameters
    ----------
    raw_before : mne.io.Raw
        Raw data before processing
    raw_after : mne.io.Raw
        Raw data after processing
    fmin : float
        Lower frequency bound of interest (default: 8.0 Hz, alpha band)
    fmax : float
        Upper frequency bound of interest (default: 12.0 Hz, alpha band)
    noise_fmin : float
        Lower frequency bound for noise (default: 50.0 Hz)
    noise_fmax : float
        Upper frequency bound for noise (default: 60.0 Hz)
    picks : str
        Channel types to include (default: 'eeg')
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'snr_before': SNR estimate before processing
        - 'snr_after': SNR estimate after processing
        - 'improvement_ratio': Ratio of after/before SNR
        - 'improvement_db': Improvement in dB
    """
    # Calculate PSD before processing
    psd_before = raw_before.compute_psd(picks=picks)
    freqs = psd_before.freqs
    psd_data_before = psd_before.get_data()
    
    # Calculate PSD after processing
    psd_after = raw_after.compute_psd(picks=picks)
    psd_data_after = psd_after.get_data()
    
    # Find frequency indices for signal and noise bands
    signal_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    noise_idx = np.logical_and(freqs >= noise_fmin, freqs <= noise_fmax)
    
    # Calculate mean power in signal and noise bands
    signal_power_before = np.mean(psd_data_before[:, signal_idx], axis=1)
    noise_power_before = np.mean(psd_data_before[:, noise_idx], axis=1)
    
    signal_power_after = np.mean(psd_data_after[:, signal_idx], axis=1)
    noise_power_after = np.mean(psd_data_after[:, noise_idx], axis=1)
    
    # Calculate SNR for before and after
    # SNR = signal power / noise power
    snr_before = np.mean(signal_power_before / noise_power_before)
    snr_after = np.mean(signal_power_after / noise_power_after)
    
    # Calculate improvement
    improvement_ratio = snr_after / snr_before
    improvement_db = 10 * np.log10(improvement_ratio)
    
    return {
        'snr_before': float(snr_before),
        'snr_after': float(snr_after),
        'improvement_ratio': float(improvement_ratio),
        'improvement_db': float(improvement_db)
    }


def evoked_response_snr(evoked_before: mne.Evoked, 
                        evoked_after: mne.Evoked,
                        tmin_baseline: float = -0.2,
                        tmax_baseline: float = 0.0,
                        tmin_signal: float = 0.05,
                        tmax_signal: float = 0.2,
                        picks: str = 'eeg') -> Dict[str, float]:
    """
    Calculate SNR of evoked responses before and after processing.
    
    This function compares the amplitude of the evoked response relative
    to the baseline variability, before and after processing.
    
    Parameters
    ----------
    evoked_before : mne.Evoked
        Evoked data before processing
    evoked_after : mne.Evoked
        Evoked data after processing
    tmin_baseline : float
        Start time of baseline period in seconds (default: -0.2)
    tmax_baseline : float
        End time of baseline period in seconds (default: 0.0)
    tmin_signal : float
        Start time of signal period in seconds (default: 0.05)
    tmax_signal : float
        End time of signal period in seconds (default: 0.2)
    picks : str
        Channel types to include (default: 'eeg')
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'snr_before': SNR estimate before processing
        - 'snr_after': SNR estimate after processing
        - 'improvement_ratio': Ratio of after/before SNR
        - 'improvement_db': Improvement in dB
    """
    # Get data and times
    data_before = evoked_before.copy().pick(picks).data
    times_before = evoked_before.times
    
    data_after = evoked_after.copy().pick(picks).data
    times_after = evoked_after.times
    
    # Find time indices for baseline and signal
    baseline_idx_before = np.logical_and(times_before >= tmin_baseline, 
                                         times_before <= tmax_baseline)
    signal_idx_before = np.logical_and(times_before >= tmin_signal, 
                                       times_before <= tmax_signal)
    
    baseline_idx_after = np.logical_and(times_after >= tmin_baseline, 
                                        times_after <= tmax_baseline)
    signal_idx_after = np.logical_and(times_after >= tmin_signal, 
                                      times_after <= tmax_signal)
    
    # Calculate baseline standard deviation (noise estimate)
    baseline_std_before = np.std(data_before[:, baseline_idx_before], axis=1)
    baseline_std_after = np.std(data_after[:, baseline_idx_after], axis=1)
    
    # Calculate peak amplitude in signal window (signal estimate)
    peak_amplitude_before = np.max(np.abs(data_before[:, signal_idx_before]), axis=1)
    peak_amplitude_after = np.max(np.abs(data_after[:, signal_idx_after]), axis=1)
    
    # Calculate SNR for before and after
    # SNR = peak amplitude / baseline standard deviation
    snr_before = np.mean(peak_amplitude_before / baseline_std_before)
    snr_after = np.mean(peak_amplitude_after / baseline_std_after)
    
    # Calculate improvement
    improvement_ratio = snr_after / snr_before
    improvement_db = 10 * np.log10(improvement_ratio)
    
    return {
        'snr_before': float(snr_before),
        'snr_after': float(snr_after),
        'improvement_ratio': float(improvement_ratio),
        'improvement_db': float(improvement_db)
    }


def global_field_power_snr(evoked_before: mne.Evoked, 
                          evoked_after: mne.Evoked,
                          tmin_baseline: float = -0.2,
                          tmax_baseline: float = 0.0,
                          tmin_signal: float = 0.05,
                          tmax_signal: float = 0.2) -> Dict[str, float]:
    """
    Calculate SNR using Global Field Power (GFP) before and after processing.
    
    GFP is the standard deviation across channels at each time point.
    This function compares the GFP in the signal window to the GFP in the
    baseline window, before and after processing.
    
    Parameters
    ----------
    evoked_before : mne.Evoked
        Evoked data before processing
    evoked_after : mne.Evoked
        Evoked data after processing
    tmin_baseline : float
        Start time of baseline period in seconds (default: -0.2)
    tmax_baseline : float
        End time of baseline period in seconds (default: 0.0)
    tmin_signal : float
        Start time of signal period in seconds (default: 0.05)
    tmax_signal : float
        End time of signal period in seconds (default: 0.2)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'snr_before': SNR estimate before processing
        - 'snr_after': SNR estimate after processing
        - 'improvement_ratio': Ratio of after/before SNR
        - 'improvement_db': Improvement in dB
    """
    # Calculate GFP for before and after
    gfp_before = np.std(evoked_before.data, axis=0)
    gfp_after = np.std(evoked_after.data, axis=0)
    
    times = evoked_before.times
    
    # Find time indices for baseline and signal
    baseline_idx = np.logical_and(times >= tmin_baseline, times <= tmax_baseline)
    signal_idx = np.logical_and(times >= tmin_signal, times <= tmax_signal)
    
    # Calculate mean GFP in baseline and signal windows
    baseline_gfp_before = np.mean(gfp_before[baseline_idx])
    signal_gfp_before = np.mean(gfp_before[signal_idx])
    
    baseline_gfp_after = np.mean(gfp_after[baseline_idx])
    signal_gfp_after = np.mean(gfp_after[signal_idx])
    
    # Calculate SNR for before and after
    # SNR = signal GFP / baseline GFP
    snr_before = signal_gfp_before / baseline_gfp_before
    snr_after = signal_gfp_after / baseline_gfp_after
    
    # Calculate improvement
    improvement_ratio = snr_after / snr_before
    improvement_db = 10 * np.log10(improvement_ratio)
    
    return {
        'snr_before': float(snr_before),
        'snr_after': float(snr_after),
        'improvement_ratio': float(improvement_ratio),
        'improvement_db': float(improvement_db)
    }


def example_usage():
    """
    Example usage of the SNR metrics functions.
    """
    # Load sample data
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif'
    
    # Read the raw data
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)
    
    # Create a copy and add some noise to simulate "before" processing
    raw_noisy = raw.copy()
    noise = np.random.normal(0, 5e-6, raw_noisy.get_data().shape)
    raw_noisy._data += noise
    
    # Calculate time-domain SNR
    time_snr = time_domain_snr(raw_noisy, raw)
    print("Time-domain SNR:")
    print(f"  Before: {time_snr['snr_before']:.2f}")
    print(f"  After: {time_snr['snr_after']:.2f}")
    print(f"  Improvement: {time_snr['improvement_db']:.2f} dB")
    
    # Calculate frequency-domain SNR
    freq_snr = frequency_domain_snr(raw_noisy, raw)
    print("\nFrequency-domain SNR (alpha band vs. high frequency noise):")
    print(f"  Before: {freq_snr['snr_before']:.2f}")
    print(f"  After: {freq_snr['snr_after']:.2f}")
    print(f"  Improvement: {freq_snr['improvement_db']:.2f} dB")
    
    # Find events and create epochs
    events = mne.find_events(raw)
    event_id = {'auditory/left': 1}
    
    # Create epochs before and after
    epochs_before = mne.Epochs(raw_noisy, events, event_id, tmin=-0.2, tmax=0.5,
                              baseline=(None, 0), preload=True)
    epochs_after = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5,
                             baseline=(None, 0), preload=True)
    
    # Create evoked responses
    evoked_before = epochs_before.average()
    evoked_after = epochs_after.average()
    
    # Calculate evoked response SNR
    evoked_snr = evoked_response_snr(evoked_before, evoked_after)
    print("\nEvoked response SNR:")
    print(f"  Before: {evoked_snr['snr_before']:.2f}")
    print(f"  After: {evoked_snr['snr_after']:.2f}")
    print(f"  Improvement: {evoked_snr['improvement_db']:.2f} dB")
    
    # Calculate GFP SNR
    gfp_snr = global_field_power_snr(evoked_before, evoked_after)
    print("\nGlobal Field Power SNR:")
    print(f"  Before: {gfp_snr['snr_before']:.2f}")
    print(f"  After: {gfp_snr['snr_after']:.2f}")
    print(f"  Improvement: {gfp_snr['improvement_db']:.2f} dB")


if __name__ == "__main__":
    example_usage()

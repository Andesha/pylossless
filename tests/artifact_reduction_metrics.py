"""
Artifact reduction metrics for EEG data quality assessment.

This module provides functions to quantify the effectiveness of artifact reduction
in EEG data processed with MNE-Python, allowing measurement of improvement
after preprocessing steps like ICA and filtering.
"""

import numpy as np
import mne
from typing import Dict, List, Optional, Union, Tuple
import scipy.stats


def artifact_amplitude_reduction(raw_before: mne.io.Raw, 
                                raw_after: mne.io.Raw,
                                picks: str = 'eeg') -> Dict[str, float]:
    """
    Calculate the reduction in extreme amplitude values after processing.
    
    This function compares the amplitude distribution before and after
    processing to quantify the reduction in outlier values.
    
    Parameters
    ----------
    raw_before : mne.io.Raw
        Raw data before processing
    raw_after : mne.io.Raw
        Raw data after processing
    picks : str
        Channel types to include (default: 'eeg')
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'peak_reduction_percent': Percentage reduction in peak amplitudes
        - 'variance_reduction_percent': Percentage reduction in signal variance
        - 'kurtosis_before': Kurtosis of signal before (higher values indicate more outliers)
        - 'kurtosis_after': Kurtosis of signal after
    """
    # Get data
    data_before = raw_before.get_data(picks=picks)
    data_after = raw_after.get_data(picks=picks)
    
    # Calculate peak amplitudes (99.9th percentile to avoid extreme outliers)
    peak_before = np.percentile(np.abs(data_before), 99.9)
    peak_after = np.percentile(np.abs(data_after), 99.9)
    
    # Calculate variance
    var_before = np.var(data_before)
    var_after = np.var(data_after)
    
    # Calculate kurtosis (measure of outliers)
    kurtosis_before = scipy.stats.kurtosis(data_before.ravel())
    kurtosis_after = scipy.stats.kurtosis(data_after.ravel())
    
    # Calculate reductions
    peak_reduction = (peak_before - peak_after) / peak_before * 100
    var_reduction = (var_before - var_after) / var_before * 100
    
    return {
        'peak_reduction_percent': float(peak_reduction),
        'variance_reduction_percent': float(var_reduction),
        'kurtosis_before': float(kurtosis_before),
        'kurtosis_after': float(kurtosis_after)
    }


def artifact_frequency_reduction(raw_before: mne.io.Raw, 
                                raw_after: mne.io.Raw,
                                artifact_bands: Dict[str, Tuple[float, float]] = None,
                                picks: str = 'eeg') -> Dict[str, Dict[str, float]]:
    """
    Calculate the reduction in power in frequency bands associated with artifacts.
    
    Parameters
    ----------
    raw_before : mne.io.Raw
        Raw data before processing
    raw_after : mne.io.Raw
        Raw data after processing
    artifact_bands : dict
        Dictionary mapping artifact names to frequency bands (default: muscle, eye, line noise)
    picks : str
        Channel types to include (default: 'eeg')
        
    Returns
    -------
    dict
        Dictionary containing for each artifact band:
        - 'power_before': Power in band before processing
        - 'power_after': Power in band after processing
        - 'reduction_percent': Percentage reduction in power
        - 'reduction_db': Reduction in decibels
    """
    if artifact_bands is None:
        artifact_bands = {
            'muscle': (30.0, 100.0),  # Muscle artifact band
            'eye': (1.0, 4.0),        # Eye movement band
            'line_noise': (49.0, 51.0)  # Line noise (adjust for your region)
        }
    
    # Calculate PSD before and after
    psd_before = raw_before.compute_psd(picks=picks)
    psd_after = raw_after.compute_psd(picks=picks)
    
    freqs = psd_before.freqs
    psd_data_before = psd_before.get_data()
    psd_data_after = psd_after.get_data()
    
    results = {}
    
    # Calculate power reduction for each artifact band
    for artifact_name, (fmin, fmax) in artifact_bands.items():
        # Find frequency indices for the band
        band_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        
        # Calculate mean power in the band
        power_before = np.mean(psd_data_before[:, band_idx])
        power_after = np.mean(psd_data_after[:, band_idx])
        
        # Calculate reduction
        reduction_percent = (power_before - power_after) / power_before * 100
        
        # Calculate reduction in dB
        if power_after > 0 and power_before > 0:
            reduction_db = 10 * np.log10(power_before / power_after)
        else:
            reduction_db = 0.0
        
        results[artifact_name] = {
            'power_before': float(power_before),
            'power_after': float(power_after),
            'reduction_percent': float(reduction_percent),
            'reduction_db': float(reduction_db)
        }
    
    return results


def eog_correlation_reduction(raw_before: mne.io.Raw, 
                             raw_after: mne.io.Raw,
                             eog_chs: List[str] = None) -> Dict[str, float]:
    """
    Calculate the reduction in correlation between EEG and EOG channels.
    
    Parameters
    ----------
    raw_before : mne.io.Raw
        Raw data before processing
    raw_after : mne.io.Raw
        Raw data after processing
    eog_chs : list
        List of EOG channel names (default: None, auto-detect)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'mean_correlation_before': Mean correlation with EOG before processing
        - 'mean_correlation_after': Mean correlation with EOG after processing
        - 'reduction_percent': Percentage reduction in correlation
    """
    # Auto-detect EOG channels if not provided
    if eog_chs is None:
        eog_chs = [ch_name for ch_name in raw_before.ch_names 
                  if ch_name.lower().startswith('eog') or 
                  raw_before.get_channel_types([ch_name])[0] == 'eog']
        
        if not eog_chs:
            raise ValueError("No EOG channels found. Please specify EOG channels.")
    
    # Get EEG and EOG data
    eeg_data_before = raw_before.get_data(picks='eeg')
    eog_data_before = raw_before.get_data(picks=eog_chs)
    
    eeg_data_after = raw_after.get_data(picks='eeg')
    eog_data_after = raw_after.get_data(picks=eog_chs)
    
    # Calculate correlation between each EEG channel and each EOG channel
    correlations_before = []
    correlations_after = []
    
    for eeg_idx in range(eeg_data_before.shape[0]):
        for eog_idx in range(eog_data_before.shape[0]):
            corr_before = np.abs(np.corrcoef(eeg_data_before[eeg_idx], 
                                           eog_data_before[eog_idx])[0, 1])
            corr_after = np.abs(np.corrcoef(eeg_data_after[eeg_idx], 
                                          eog_data_after[eog_idx])[0, 1])
            
            correlations_before.append(corr_before)
            correlations_after.append(corr_after)
    
    # Calculate mean correlations
    mean_corr_before = np.mean(correlations_before)
    mean_corr_after = np.mean(correlations_after)
    
    # Calculate reduction
    reduction_percent = (mean_corr_before - mean_corr_after) / mean_corr_before * 100
    
    return {
        'mean_correlation_before': float(mean_corr_before),
        'mean_correlation_after': float(mean_corr_after),
        'reduction_percent': float(reduction_percent)
    }


def artifact_epoch_rejection_rate(epochs_before: mne.Epochs, 
                                 epochs_after: mne.Epochs) -> Dict[str, float]:
    """
    Calculate the reduction in epochs that would be rejected based on amplitude criteria.
    
    Parameters
    ----------
    epochs_before : mne.Epochs
        Epochs before processing
    epochs_after : mne.Epochs
        Epochs after processing
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'rejection_rate_before': Percentage of epochs that would be rejected before
        - 'rejection_rate_after': Percentage of epochs that would be rejected after
        - 'improvement_percent': Percentage improvement in rejection rate
    """
    # Define rejection criteria (typical values, can be adjusted)
    reject_criteria = dict(eeg=100e-6)  # 100 µV for EEG
    
    # Count epochs that would be rejected before processing
    epochs_before_clean = epochs_before.copy()
    epochs_before_clean.drop_bad(reject=reject_criteria)
    n_rejected_before = len(epochs_before) - len(epochs_before_clean)
    rejection_rate_before = n_rejected_before / len(epochs_before) * 100
    
    # Count epochs that would be rejected after processing
    epochs_after_clean = epochs_after.copy()
    epochs_after_clean.drop_bad(reject=reject_criteria)
    n_rejected_after = len(epochs_after) - len(epochs_after_clean)
    rejection_rate_after = n_rejected_after / len(epochs_after) * 100
    
    # Calculate improvement
    if rejection_rate_before > 0:
        improvement_percent = (rejection_rate_before - rejection_rate_after) / rejection_rate_before * 100
    else:
        improvement_percent = 0.0
    
    return {
        'rejection_rate_before': float(rejection_rate_before),
        'rejection_rate_after': float(rejection_rate_after),
        'improvement_percent': float(improvement_percent)
    }


def example_usage():
    """
    Example usage of the artifact reduction metrics functions.
    """
    # Load sample data
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif'
    
    # Read the raw data
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)
    
    # Create a copy and add some noise to simulate "before" processing
    raw_noisy = raw.copy()
    
    # Add different types of noise to simulate artifacts
    data = raw_noisy.get_data()
    
    # Add high-frequency noise (muscle-like)
    muscle_noise = np.random.normal(0, 2e-6, data.shape)
    muscle_noise = mne.filter.filter_data(muscle_noise, raw.info['sfreq'], 20, None)
    
    # Add slow drift (eye movement-like)
    n_times = data.shape[1]
    t = np.arange(n_times) / raw.info['sfreq']
    eye_noise = 5e-6 * np.sin(2 * np.pi * 0.3 * t)
    eye_noise = np.tile(eye_noise, (data.shape[0], 1))
    
    # Add line noise
    line_freq = 50.0  # Hz
    line_noise = 3e-6 * np.sin(2 * np.pi * line_freq * t)
    line_noise = np.tile(line_noise, (data.shape[0], 1))
    
    # Combine all noise
    raw_noisy._data = data + muscle_noise + eye_noise + line_noise
    
    # Calculate amplitude reduction
    amp_reduction = artifact_amplitude_reduction(raw_noisy, raw)
    print("Artifact Amplitude Reduction:")
    print(f"  Peak amplitude reduction: {amp_reduction['peak_reduction_percent']:.2f}%")
    print(f"  Variance reduction: {amp_reduction['variance_reduction_percent']:.2f}%")
    print(f"  Kurtosis before: {amp_reduction['kurtosis_before']:.2f}")
    print(f"  Kurtosis after: {amp_reduction['kurtosis_after']:.2f}")
    
    # Calculate frequency reduction
    freq_reduction = artifact_frequency_reduction(raw_noisy, raw)
    print("\nArtifact Frequency Reduction:")
    for band, metrics in freq_reduction.items():
        print(f"  {band.capitalize()} band:")
        print(f"    Reduction: {metrics['reduction_percent']:.2f}%")
        print(f"    Reduction: {metrics['reduction_db']:.2f} dB")
    
    # Find events and create epochs
    events = mne.find_events(raw)
    event_id = {'auditory/left': 1}
    
    # Create epochs before and after
    epochs_before = mne.Epochs(raw_noisy, events, event_id, tmin=-0.2, tmax=0.5,
                              baseline=(None, 0), preload=True)
    epochs_after = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5,
                             baseline=(None, 0), preload=True)
    
    # Calculate epoch rejection rate
    rejection_rate = artifact_epoch_rejection_rate(epochs_before, epochs_after)
    print("\nEpoch Rejection Rate:")
    print(f"  Before processing: {rejection_rate['rejection_rate_before']:.2f}%")
    print(f"  After processing: {rejection_rate['rejection_rate_after']:.2f}%")
    print(f"  Improvement: {rejection_rate['improvement_percent']:.2f}%")


if __name__ == "__main__":
    example_usage()

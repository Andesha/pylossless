"""
ICA component quality metrics for EEG data quality assessment.

This module provides functions to evaluate the quality of ICA components
and quantify the effectiveness of ICA-based artifact removal in EEG data
processed with MNE-Python.
"""

import numpy as np
import mne
from typing import Dict, List, Optional, Union, Tuple
import scipy.stats


def ica_component_metrics(ica: mne.preprocessing.ICA, 
                         raw: mne.io.Raw) -> Dict[str, Dict[str, float]]:
    """
    Calculate quality metrics for each ICA component.
    
    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.io.Raw
        Raw data used for ICA
        
    Returns
    -------
    dict
        Dictionary containing metrics for each component:
        - 'kurtosis': Kurtosis of component (higher values often indicate artifacts)
        - 'variance_explained': Percentage of variance explained by component
        - 'autocorrelation': Temporal autocorrelation (lag=1)
    """
    # Get the ICA components (sources)
    sources = ica.get_sources(raw).get_data()
    
    # Get the mixing matrix
    mixing_matrix = ica.mixing_matrix_
    
    # Calculate metrics for each component
    n_components = sources.shape[0]
    results = {}
    
    for comp_idx in range(n_components):
        # Get component data
        comp_data = sources[comp_idx]
        
        # Calculate kurtosis
        kurtosis = scipy.stats.kurtosis(comp_data)
        
        # Calculate variance explained
        # The column of the mixing matrix corresponds to the spatial pattern
        # The squared sum of the pattern gives the relative variance explained
        comp_pattern = mixing_matrix[:, comp_idx]
        variance_explained = np.sum(comp_pattern ** 2) / np.sum(mixing_matrix ** 2) * 100
        
        # Calculate autocorrelation (lag=1)
        autocorr = np.corrcoef(comp_data[:-1], comp_data[1:])[0, 1]
        
        results[f'component_{comp_idx}'] = {
            'kurtosis': float(kurtosis),
            'variance_explained': float(variance_explained),
            'autocorrelation': float(autocorr)
        }
    
    return results


def ica_artifact_probability(ica: mne.preprocessing.ICA, 
                            raw: mne.io.Raw,
                            threshold: float = 0.8,
                            method: str = 'correlation') -> Dict[str, Dict[str, float]]:
    """
    Estimate the probability that each ICA component represents an artifact.
    
    This function uses correlation with EOG and ECG channels, or spatial and
    temporal characteristics to classify components.
    
    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.io.Raw
        Raw data used for ICA
    threshold : float
        Threshold for classifying a component as an artifact (default: 0.8)
    method : str
        Method for estimating artifact probability ('correlation' or 'features')
        
    Returns
    -------
    dict
        Dictionary containing for each component:
        - 'eog_correlation': Correlation with EOG channels
        - 'ecg_correlation': Correlation with ECG channels
        - 'artifact_probability': Estimated probability of being an artifact
        - 'is_artifact': Boolean indicating if component exceeds threshold
    """
    # Get the ICA components (sources)
    sources = ica.get_sources(raw).get_data()
    n_components = sources.shape[0]
    
    results = {}
    
    if method == 'correlation':
        # Find EOG and ECG channels
        eog_chs = mne.pick_types(raw.info, eog=True, meg=False, eeg=False)
        ecg_chs = mne.pick_types(raw.info, ecg=True, meg=False, eeg=False)
        
        # If no EOG/ECG channels, try to create virtual ones
        if len(eog_chs) == 0:
            # Create EOG epochs
            eog_epochs = mne.preprocessing.create_eog_epochs(raw)
            if len(eog_epochs) > 0:
                eog_evoked = eog_epochs.average()
                eog_data = eog_evoked.data
            else:
                eog_data = None
        else:
            eog_data = raw.get_data(picks=eog_chs)
        
        if len(ecg_chs) == 0:
            # Create ECG epochs
            ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
            if len(ecg_epochs) > 0:
                ecg_evoked = ecg_epochs.average()
                ecg_data = ecg_evoked.data
            else:
                ecg_data = None
        else:
            ecg_data = raw.get_data(picks=ecg_chs)
        
        # Calculate correlations for each component
        for comp_idx in range(n_components):
            comp_data = sources[comp_idx]
            
            # Calculate correlation with EOG
            if eog_data is not None:
                if eog_data.ndim == 3:  # Evoked data
                    eog_corr = np.max([np.abs(np.corrcoef(comp_data, eog_ch)[0, 1]) 
                                     for eog_ch in eog_data])
                else:  # Raw data
                    eog_corr = np.max([np.abs(np.corrcoef(comp_data, eog_ch)[0, 1]) 
                                     for eog_ch in eog_data])
            else:
                eog_corr = 0.0
            
            # Calculate correlation with ECG
            if ecg_data is not None:
                if ecg_data.ndim == 3:  # Evoked data
                    ecg_corr = np.max([np.abs(np.corrcoef(comp_data, ecg_ch)[0, 1]) 
                                     for ecg_ch in ecg_data])
                else:  # Raw data
                    ecg_corr = np.max([np.abs(np.corrcoef(comp_data, ecg_ch)[0, 1]) 
                                     for ecg_ch in ecg_data])
            else:
                ecg_corr = 0.0
            
            # Estimate artifact probability as maximum correlation
            artifact_prob = max(eog_corr, ecg_corr)
            
            results[f'component_{comp_idx}'] = {
                'eog_correlation': float(eog_corr),
                'ecg_correlation': float(ecg_corr),
                'artifact_probability': float(artifact_prob),
                'is_artifact': artifact_prob > threshold
            }
    
    elif method == 'features':
        # Calculate component metrics
        metrics = ica_component_metrics(ica, raw)
        
        for comp_idx in range(n_components):
            comp_metrics = metrics[f'component_{comp_idx}']
            
            # Use kurtosis and autocorrelation to estimate artifact probability
            # High kurtosis and high autocorrelation often indicate artifacts
            kurtosis_score = min(abs(comp_metrics['kurtosis']) / 5.0, 1.0)
            autocorr_score = min(abs(comp_metrics['autocorrelation']), 1.0)
            
            # Combine scores (simple average)
            artifact_prob = (kurtosis_score + autocorr_score) / 2.0
            
            results[f'component_{comp_idx}'] = {
                'kurtosis_score': float(kurtosis_score),
                'autocorr_score': float(autocorr_score),
                'artifact_probability': float(artifact_prob),
                'is_artifact': artifact_prob > threshold
            }
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'correlation' or 'features'.")
    
    return results


def ica_cleaning_effectiveness(raw_before: mne.io.Raw, 
                              raw_after: mne.io.Raw,
                              ica: mne.preprocessing.ICA) -> Dict[str, float]:
    """
    Evaluate the effectiveness of ICA cleaning by comparing data before and after.
    
    Parameters
    ----------
    raw_before : mne.io.Raw
        Raw data before ICA cleaning
    raw_after : mne.io.Raw
        Raw data after ICA cleaning
    ica : mne.preprocessing.ICA
        Fitted ICA object used for cleaning
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'variance_removed_percent': Percentage of variance removed
        - 'artifact_components_percent': Percentage of components identified as artifacts
        - 'data_rank_reduction': Reduction in effective rank of the data
    """
    # Calculate variance before and after
    var_before = np.var(raw_before.get_data())
    var_after = np.var(raw_after.get_data())
    
    # Calculate variance removed
    var_removed = var_before - var_after
    var_removed_percent = var_removed / var_before * 100
    
    # Calculate percentage of artifact components
    n_components = ica.n_components_
    n_excluded = len(ica.exclude)
    artifact_components_percent = n_excluded / n_components * 100 if n_components > 0 else 0
    
    # Calculate effective rank before and after
    rank_before = mne.compute_rank(raw_before)
    rank_after = mne.compute_rank(raw_after)
    
    # Get the rank values for EEG
    if 'eeg' in rank_before:
        rank_before_val = rank_before['eeg']
        rank_after_val = rank_after['eeg']
        rank_reduction = rank_before_val - rank_after_val
    else:
        # If no EEG, use the first modality available
        key = list(rank_before.keys())[0]
        rank_before_val = rank_before[key]
        rank_after_val = rank_after[key]
        rank_reduction = rank_before_val - rank_after_val
    
    return {
        'variance_removed_percent': float(var_removed_percent),
        'artifact_components_percent': float(artifact_components_percent),
        'data_rank_reduction': float(rank_reduction)
    }


def ica_component_dipole_fit(ica: mne.preprocessing.ICA, 
                            raw: mne.io.Raw,
                            sphere: Optional[Union[str, Dict, Tuple]] = None) -> Dict[str, Dict[str, float]]:
    """
    Fit dipoles to ICA components and evaluate the goodness of fit.
    
    Note: This requires a head model and sensor positions.
    
    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.io.Raw
        Raw data used for ICA
    sphere : str | dict | tuple | None
        Sphere model for dipole fitting (default: None, auto-detect)
        
    Returns
    -------
    dict
        Dictionary containing for each component:
        - 'residual_variance': Residual variance of dipole fit (lower is better)
        - 'pos': Position of the dipole (x, y, z)
        - 'is_brain_source': Boolean indicating if likely a brain source
    """
    try:
        # Check if we have sensor positions
        if not hasattr(raw.info, 'dig') or raw.info['dig'] is None:
            raise ValueError("Raw data does not have sensor positions.")
        
        # Create a sphere model if not provided
        if sphere is None:
            sphere = mne.make_sphere_model('auto', 'auto', raw.info)
        
        # Fit dipoles to ICA components
        dipoles = mne.fit_dipole(ica.get_components(), raw.info, sphere)[0]
        
        results = {}
        n_components = ica.n_components_
        
        for comp_idx in range(n_components):
            # Get dipole for this component
            dipole = dipoles[comp_idx]
            
            # Get residual variance (lower is better)
            residual_variance = dipole.residual / 100.0  # Convert from percentage
            
            # Get position
            pos = dipole.pos[0].tolist()
            
            # Determine if likely a brain source (RV < 0.15 is often used as threshold)
            is_brain_source = residual_variance < 0.15
            
            results[f'component_{comp_idx}'] = {
                'residual_variance': float(residual_variance),
                'pos': pos,
                'is_brain_source': bool(is_brain_source)
            }
        
        return results
    
    except (ValueError, RuntimeError) as e:
        # Return a placeholder if dipole fitting fails
        print(f"Dipole fitting failed: {str(e)}")
        return {f'component_{i}': {
            'residual_variance': 1.0,
            'pos': [0, 0, 0],
            'is_brain_source': False
        } for i in range(ica.n_components_)}


def example_usage():
    """
    Example usage of the ICA component quality metrics functions.
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
    
    # Set up and fit ICA
    ica = mne.preprocessing.ICA(n_components=15, random_state=42)
    ica.fit(raw_noisy, picks='eeg')
    
    # Find artifact components (for example, using correlation with EOG)
    eog_indices, eog_scores = ica.find_bads_eog(raw_noisy)
    ica.exclude = eog_indices
    
    # Apply ICA to clean the data
    raw_cleaned = raw_noisy.copy()
    ica.apply(raw_cleaned)
    
    # Calculate component metrics
    comp_metrics = ica_component_metrics(ica, raw_noisy)
    print("ICA Component Metrics:")
    for comp_name, metrics in list(comp_metrics.items())[:3]:  # Show first 3 components
        print(f"  {comp_name}:")
        print(f"    Kurtosis: {metrics['kurtosis']:.2f}")
        print(f"    Variance explained: {metrics['variance_explained']:.2f}%")
        print(f"    Autocorrelation: {metrics['autocorrelation']:.2f}")
    
    # Calculate artifact probabilities
    artifact_probs = ica_artifact_probability(ica, raw_noisy, method='features')
    print("\nICA Artifact Probabilities (feature-based):")
    for comp_name, probs in list(artifact_probs.items())[:3]:  # Show first 3 components
        print(f"  {comp_name}:")
        print(f"    Artifact probability: {probs['artifact_probability']:.2f}")
        print(f"    Is artifact: {probs['is_artifact']}")
    
    # Calculate cleaning effectiveness
    effectiveness = ica_cleaning_effectiveness(raw_noisy, raw_cleaned, ica)
    print("\nICA Cleaning Effectiveness:")
    print(f"  Variance removed: {effectiveness['variance_removed_percent']:.2f}%")
    print(f"  Artifact components: {effectiveness['artifact_components_percent']:.2f}%")
    print(f"  Data rank reduction: {effectiveness['data_rank_reduction']:.2f}")


if __name__ == "__main__":
    example_usage()

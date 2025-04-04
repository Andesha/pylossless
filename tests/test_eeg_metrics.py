"""
Pytest integration examples for EEG quality metrics.

This module demonstrates how to integrate the EEG quality metrics into pytest
for automated testing of EEG processing pipelines.
"""

import pytest
import numpy as np
import mne
from pathlib import Path
import sys
import os
import pylossless as ll

# Add the code_examples directory to the path so we can import our metrics modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snr_metrics import time_domain_snr, frequency_domain_snr, evoked_response_snr
from artifact_reduction_metrics import artifact_amplitude_reduction, artifact_frequency_reduction
from ica_quality_metrics import ica_component_metrics, ica_cleaning_effectiveness

@pytest.fixture(scope="session")
def sample_raw_data(request):
    """Fixture that loads a small dataset by default, but a large dataset if --big-data is passed."""
    if request.config.getoption("--full-size"):
        return load_large_dataset()
    return load_small_dataset()

def load_small_dataset():
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = (
        sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
    )
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True).pick('eeg')
    raw.info['bads'] = [] # EEG 053 should get detected
    return raw

def load_large_dataset():
    # TODO: Add logic for downloading the large dataset
    raw = mne.io.read_raw('tests/data/sub-1_face13_eeg.fif', preload=True)
    raw = raw.set_montage('biosemi128')
    return raw


@pytest.fixture(scope="session")
def sample_data_config(request, sample_raw_data):
    config_object = ll.Config().load_default()

    # Have to change the filtering as the sfreq of the sample is low
    if not request.config.getoption("--full-size"): 
        config_object['filtering']['filter_args']['h_freq'] = 60
        config_object['filtering']['notch_filter_args']['freqs'] = []

    return config_object

@pytest.fixture(scope="session")
def lossless_cleaned_data(sample_raw_data, sample_data_config):
    original_raw = sample_raw_data.copy()
    pipeline = ll.LosslessPipeline(config=sample_data_config)
    pipeline.run_with_raw(sample_raw_data)
    assert pipeline.flags is not None, 'Pipeline flags somehow empty after basic invocation'

    rejection_policy = ll.RejectionPolicy()
    rejection_policy['ch_cleaning_mode'] = 'interpolate'
    cleaned_raw = rejection_policy.apply(pipeline)

    clean_matrix = cleaned_raw.get_data() 
    old_matrix = original_raw.get_data()

    if clean_matrix.shape == old_matrix.shape:
        diff = np.abs(clean_matrix - old_matrix)
        assert np.any(diff > 0), "Cleaning did not provide a difference in data"

    return cleaned_raw, pipeline.ica2


# Define test functions
def test_time_domain_snr_improvement(sample_raw_data, lossless_cleaned_data):
    """Test that time-domain SNR improves after ICA cleaning."""
    raw_noisy = sample_raw_data
    raw_cleaned = lossless_cleaned_data[0]
    
    # Calculate SNR improvement
    snr_results = time_domain_snr(raw_noisy, raw_cleaned)
    
    # Assert that SNR improved
    assert snr_results['snr_after'] > snr_results['snr_before'], \
        f"SNR did not improve: before={snr_results['snr_before']}, after={snr_results['snr_after']}"
    
    # Assert that improvement is above threshold
    assert snr_results['improvement_db'] > 3.0, \
        f"SNR improvement below threshold: {snr_results['improvement_db']} dB"


def test_artifact_amplitude_reduction(sample_raw_data, lossless_cleaned_data):
    """Test that artifact amplitudes are reduced after ICA cleaning."""
    raw_noisy = sample_raw_data
    raw_cleaned = lossless_cleaned_data[0]
    
    # Calculate amplitude reduction
    amp_results = artifact_amplitude_reduction(raw_noisy, raw_cleaned)
    
    # Assert that peak amplitudes were reduced
    assert amp_results['peak_reduction_percent'] > 20.0, \
        f"Peak amplitude reduction below threshold: {amp_results['peak_reduction_percent']}%"
    
    # Assert that variance was reduced
    assert amp_results['variance_reduction_percent'] > 20.0, \
        f"Variance reduction below threshold: {amp_results['variance_reduction_percent']}%"
    
    # Assert that kurtosis decreased (fewer outliers)
    assert amp_results['kurtosis_after'] < amp_results['kurtosis_before'], \
        f"Kurtosis did not decrease: before={amp_results['kurtosis_before']}, after={amp_results['kurtosis_after']}"


def test_ica_cleaning_effectiveness(sample_raw_data, lossless_cleaned_data):
    """Test the effectiveness of ICA cleaning."""
    raw_noisy = sample_raw_data
    raw_cleaned, ica = lossless_cleaned_data
    
    # Calculate cleaning effectiveness
    effectiveness = ica_cleaning_effectiveness(raw_noisy, raw_cleaned, ica)
    
    # Assert that sufficient variance was removed
    assert effectiveness['variance_removed_percent'] > 10.0, \
        f"Variance removal below threshold: {effectiveness['variance_removed_percent']}%"
    
    # Assert that a reasonable percentage of components were identified as artifacts
    assert 5.0 < effectiveness['artifact_components_percent'] < 50.0, \
        f"Unusual percentage of artifact components: {effectiveness['artifact_components_percent']}%"


def test_ica_component_metrics(sample_raw_data, lossless_cleaned_data):
    """Test that ICA component metrics are calculated correctly."""
    raw_noisy = sample_raw_data
    _, ica = lossless_cleaned_data
    
    # Calculate component metrics
    metrics = ica_component_metrics(ica, raw_noisy)
    
    # Assert that metrics were calculated for all components
    assert len(metrics) == ica.n_components_, \
        f"Expected metrics for {ica.n_components_} components, got {len(metrics)}"
    
    # Check that at least one component has high kurtosis (likely artifact)
    high_kurtosis_components = [comp for comp, vals in metrics.items() 
                               if abs(vals['kurtosis']) > 2.0]
    assert len(high_kurtosis_components) > 0, \
        "No components with high kurtosis found, expected at least one artifact component"
    
    # Check that variance explained sums to a reasonable value
    total_variance = sum(vals['variance_explained'] for vals in metrics.values())
    assert 80.0 < total_variance < 120.0, \
        f"Total variance explained ({total_variance}%) outside expected range"


# Example of a test that generates a report
def test_generate_quality_report(sample_raw_data, lossless_cleaned_data, tmp_path):
    """Generate a quality report with metrics for different processing methods."""
    raw_noisy = sample_raw_data
    raw_ica_cleaned, _ = lossless_cleaned_data
    
    # Calculate metrics for different processing methods
    ica_time_snr = time_domain_snr(raw_noisy, raw_ica_cleaned)
    ica_freq_snr = frequency_domain_snr(raw_noisy, raw_ica_cleaned)
    ica_amp_reduction = artifact_amplitude_reduction(raw_noisy, raw_ica_cleaned)
    
    # Create report file
    report_file = tmp_path / "quality_report.txt"
    with open(report_file, 'w') as f:
        f.write("EEG Quality Metrics Report\n")
        f.write("=========================\n\n")
        
        f.write("Time-domain SNR Improvement:\n")
        f.write(f"  ICA: {ica_time_snr['improvement_db']:.2f} dB\n")
        
        f.write("Frequency-domain SNR Improvement (alpha band):\n")
        f.write(f"  ICA: {ica_freq_snr['improvement_db']:.2f} dB\n")
        
        f.write("Artifact Amplitude Reduction:\n")
        f.write(f"  ICA: {ica_amp_reduction['peak_reduction_percent']:.2f}%\n")
    
    # Assert that the report was created
    assert report_file.exists(), "Quality report was not created"

    # Print report path for reference
    print(f"\nQuality report generated at: {report_file}")

# EEG Quality Metrics for pylossless Pipeline

This documentation provides a comprehensive overview of the quality metrics developed for quantifying EEG signal improvement in the pylossless pipeline. These metrics can be integrated into pytest to automatically verify the effectiveness of processing steps like ICA, filtering, and artifact rejection.

## Table of Contents

1. [Introduction](#introduction)
2. [Signal-to-Noise Ratio (SNR) Metrics](#signal-to-noise-ratio-snr-metrics)
3. [Artifact Reduction Metrics](#artifact-reduction-metrics)
4. [ICA Component Quality Metrics](#ica-component-quality-metrics)
5. [Pytest Integration](#pytest-integration)
6. [Usage Examples](#usage-examples)
7. [Implementation Details](#implementation-details)
8. [References](#references)

## Introduction

The pylossless pipeline performs various processing steps to improve EEG data quality, including:
- Independent Component Analysis (ICA) for artifact removal
- Filtering to remove frequency bands associated with noise
- Rejection of bad channels and time segments

To quantify the improvement in signal quality after these processing steps, we've developed a set of metrics that can be integrated into automated tests. These metrics provide objective measures of how much "cleaner" the signal becomes after processing.

## Signal-to-Noise Ratio (SNR) Metrics

SNR metrics quantify the ratio of signal power to noise power, with higher values indicating cleaner data.

### Time-domain SNR

The `time_domain_snr` function calculates SNR in the time domain by comparing the variance of signals before and after processing:

```python
time_snr = time_domain_snr(raw_before, raw_after)
print(f"SNR improvement: {time_snr['improvement_db']:.2f} dB")
```

**Key metrics returned:**
- `snr_before`: SNR estimate before processing
- `snr_after`: SNR estimate after processing
- `improvement_ratio`: Ratio of after/before SNR
- `improvement_db`: Improvement in decibels

### Frequency-domain SNR

The `frequency_domain_snr` function calculates SNR in the frequency domain by comparing power in frequency bands of interest relative to noise bands:

```python
freq_snr = frequency_domain_snr(raw_before, raw_after, 
                               fmin=8.0, fmax=12.0,  # Alpha band
                               noise_fmin=50.0, noise_fmax=60.0)  # Line noise band
print(f"Alpha band SNR improvement: {freq_snr['improvement_db']:.2f} dB")
```

**Key metrics returned:**
- `snr_before`: SNR estimate before processing
- `snr_after`: SNR estimate after processing
- `improvement_ratio`: Ratio of after/before SNR
- `improvement_db`: Improvement in decibels

### Evoked Response SNR

The `evoked_response_snr` function calculates SNR of evoked responses by comparing the amplitude of the response relative to baseline variability:

```python
evoked_snr = evoked_response_snr(evoked_before, evoked_after)
print(f"Evoked response SNR improvement: {evoked_snr['improvement_db']:.2f} dB")
```

**Key metrics returned:**
- `snr_before`: SNR estimate before processing
- `snr_after`: SNR estimate after processing
- `improvement_ratio`: Ratio of after/before SNR
- `improvement_db`: Improvement in decibels

### Global Field Power SNR

The `global_field_power_snr` function calculates SNR using Global Field Power (GFP), which is the standard deviation across channels at each time point:

```python
gfp_snr = global_field_power_snr(evoked_before, evoked_after)
print(f"GFP SNR improvement: {gfp_snr['improvement_db']:.2f} dB")
```

**Key metrics returned:**
- `snr_before`: SNR estimate before processing
- `snr_after`: SNR estimate after processing
- `improvement_ratio`: Ratio of after/before SNR
- `improvement_db`: Improvement in decibels

## Artifact Reduction Metrics

Artifact reduction metrics quantify how effectively artifacts have been removed from the data.

### Artifact Amplitude Reduction

The `artifact_amplitude_reduction` function calculates the reduction in extreme amplitude values after processing:

```python
amp_reduction = artifact_amplitude_reduction(raw_before, raw_after)
print(f"Peak amplitude reduction: {amp_reduction['peak_reduction_percent']:.2f}%")
```

**Key metrics returned:**
- `peak_reduction_percent`: Percentage reduction in peak amplitudes
- `variance_reduction_percent`: Percentage reduction in signal variance
- `kurtosis_before`: Kurtosis of signal before (higher values indicate more outliers)
- `kurtosis_after`: Kurtosis of signal after

### Artifact Frequency Reduction

The `artifact_frequency_reduction` function calculates the reduction in power in frequency bands associated with artifacts:

```python
artifact_bands = {
    'muscle': (30.0, 100.0),  # Muscle artifact band
    'eye': (1.0, 4.0),        # Eye movement band
    'line_noise': (49.0, 51.0)  # Line noise
}
freq_reduction = artifact_frequency_reduction(raw_before, raw_after, artifact_bands)
print(f"Muscle artifact reduction: {freq_reduction['muscle']['reduction_percent']:.2f}%")
```

**Key metrics returned for each band:**
- `power_before`: Power in band before processing
- `power_after`: Power in band after processing
- `reduction_percent`: Percentage reduction in power
- `reduction_db`: Reduction in decibels

### EOG Correlation Reduction

The `eog_correlation_reduction` function calculates the reduction in correlation between EEG and EOG channels:

```python
eog_reduction = eog_correlation_reduction(raw_before, raw_after)
print(f"EOG correlation reduction: {eog_reduction['reduction_percent']:.2f}%")
```

**Key metrics returned:**
- `mean_correlation_before`: Mean correlation with EOG before processing
- `mean_correlation_after`: Mean correlation with EOG after processing
- `reduction_percent`: Percentage reduction in correlation

### Artifact Epoch Rejection Rate

The `artifact_epoch_rejection_rate` function calculates the reduction in epochs that would be rejected based on amplitude criteria:

```python
rejection_rate = artifact_epoch_rejection_rate(epochs_before, epochs_after)
print(f"Improvement in rejection rate: {rejection_rate['improvement_percent']:.2f}%")
```

**Key metrics returned:**
- `rejection_rate_before`: Percentage of epochs that would be rejected before
- `rejection_rate_after`: Percentage of epochs that would be rejected after
- `improvement_percent`: Percentage improvement in rejection rate

## ICA Component Quality Metrics

ICA component quality metrics evaluate the quality of ICA decomposition and the effectiveness of ICA-based artifact removal.

### ICA Component Metrics

The `ica_component_metrics` function calculates quality metrics for each ICA component:

```python
comp_metrics = ica_component_metrics(ica, raw)
for comp_name, metrics in comp_metrics.items():
    print(f"{comp_name} - Kurtosis: {metrics['kurtosis']:.2f}, Variance explained: {metrics['variance_explained']:.2f}%")
```

**Key metrics returned for each component:**
- `kurtosis`: Kurtosis of component (higher values often indicate artifacts)
- `variance_explained`: Percentage of variance explained by component
- `autocorrelation`: Temporal autocorrelation (lag=1)

### ICA Artifact Probability

The `ica_artifact_probability` function estimates the probability that each ICA component represents an artifact:

```python
artifact_probs = ica_artifact_probability(ica, raw, method='features')
for comp_name, probs in artifact_probs.items():
    print(f"{comp_name} - Artifact probability: {probs['artifact_probability']:.2f}")
```

**Key metrics returned for each component:**
- `artifact_probability`: Estimated probability of being an artifact
- `is_artifact`: Boolean indicating if component exceeds threshold
- Additional metrics depending on method ('correlation' or 'features')

### ICA Cleaning Effectiveness

The `ica_cleaning_effectiveness` function evaluates the effectiveness of ICA cleaning by comparing data before and after:

```python
effectiveness = ica_cleaning_effectiveness(raw_before, raw_after, ica)
print(f"Variance removed: {effectiveness['variance_removed_percent']:.2f}%")
```

**Key metrics returned:**
- `variance_removed_percent`: Percentage of variance removed
- `artifact_components_percent`: Percentage of components identified as artifacts
- `data_rank_reduction`: Reduction in effective rank of the data

### ICA Component Dipole Fit

The `ica_component_dipole_fit` function fits dipoles to ICA components and evaluates the goodness of fit:

```python
dipole_fits = ica_component_dipole_fit(ica, raw)
for comp_name, fit in dipole_fits.items():
    print(f"{comp_name} - Residual variance: {fit['residual_variance']:.2f}")
```

**Key metrics returned for each component:**
- `residual_variance`: Residual variance of dipole fit (lower is better)
- `pos`: Position of the dipole (x, y, z)
- `is_brain_source`: Boolean indicating if likely a brain source

## Pytest Integration

The metrics can be integrated into pytest for automated testing of EEG processing pipelines. Here's how to set up pytest tests:

### Fixtures

Define fixtures for test data:

```python
@pytest.fixture
def sample_raw_data():
    """Fixture to load sample raw data."""
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif'
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)
    return raw

@pytest.fixture
def noisy_raw_data(sample_raw_data):
    """Fixture to create noisy version of the sample data."""
    raw_noisy = sample_raw_data.copy()
    # Add noise...
    return raw_noisy

@pytest.fixture
def ica_cleaned_data(noisy_raw_data):
    """Fixture to create ICA-cleaned version of the noisy data."""
    # Apply ICA...
    return raw_cleaned, ica
```

### Test Functions

Create test functions that use the metrics to verify processing quality:

```python
def test_time_domain_snr_improvement(noisy_raw_data, ica_cleaned_data):
    """Test that time-domain SNR improves after ICA cleaning."""
    raw_noisy = noisy_raw_data
    raw_cleaned = ica_cleaned_data[0]
    
    # Calculate SNR improvement
    snr_results = time_domain_snr(raw_noisy, raw_cleaned)
    
    # Assert that SNR improved
    assert snr_results['snr_after'] > snr_results['snr_before']
    
    # Assert that improvement is above threshold
    assert snr_results['improvement_db'] > 3.0
```

### Parametrized Tests

Use parametrized tests to optimize processing parameters:

```python
@pytest.mark.parametrize("l_freq,h_freq,expected_improvement", [
    (1.0, 40.0, 3.0),  # Standard filtering
    (0.5, 30.0, 2.0),  # Wider band
    (4.0, 30.0, 4.0),  # Narrower band
])
def test_filter_parameter_optimization(noisy_raw_data, l_freq, h_freq, expected_improvement):
    """Test different filtering parameters to find optimal settings."""
    # Apply filter with specified parameters...
    # Assert that improvement meets or exceeds expected value...
```

### Quality Reports

Generate quality reports with metrics for different processing methods:

```python
def test_generate_quality_report(noisy_raw_data, ica_cleaned_data, filtered_data, tmp_path):
    """Generate a quality report with metrics for different processing methods."""
    # Calculate metrics for different processing methods...
    # Create report file...
    # Assert that the report was created...
```

## Usage Examples

Here are some examples of how to use these metrics in your pylossless pipeline:

### Basic Usage

```python
import mne
from eeg_metrics.snr_metrics import time_domain_snr
from eeg_metrics.artifact_reduction_metrics import artifact_amplitude_reduction
from eeg_metrics.ica_quality_metrics import ica_cleaning_effectiveness

# Load data
raw = mne.io.read_raw_fif('your_data.fif', preload=True)

# Run your pylossless pipeline
# ...

# Calculate metrics
snr_results = time_domain_snr(raw_before, raw_after)
amp_reduction = artifact_amplitude_reduction(raw_before, raw_after)
ica_effectiveness = ica_cleaning_effectiveness(raw_before, raw_after, ica)

# Print results
print(f"SNR improvement: {snr_results['improvement_db']:.2f} dB")
print(f"Peak amplitude reduction: {amp_reduction['peak_reduction_percent']:.2f}%")
print(f"Variance removed by ICA: {ica_effectiveness['variance_removed_percent']:.2f}%")
```

### Integration with pylossless

```python
import pylossless as ll
import mne
from eeg_metrics.snr_metrics import time_domain_snr

# Load data
raw = mne.io.read_raw_fif('your_data.fif', preload=True)
raw_before = raw.copy()

# Run pylossless pipeline
config = ll.config.Config()
config.load_default()
pipeline = ll.LosslessPipeline(config=config)
pipeline.run_with_raw(raw)

# Get cleaned data
rejection_policy = ll.RejectionPolicy()
raw_after = rejection_policy.apply(pipeline)

# Calculate metrics
snr_results = time_domain_snr(raw_before, raw_after)
print(f"SNR improvement: {snr_results['improvement_db']:.2f} dB")
```

## Implementation Details

### Dependencies

- MNE-Python (>=1.0.0)
- NumPy (>=1.20.0)
- SciPy (>=1.7.0)
- pytest (>=6.0.0) for test integration

### Performance Considerations

- Most metrics are computationally efficient and suitable for automated testing
- Dipole fitting can be computationally intensive and may be skipped in routine testing
- For large datasets, consider downsampling or selecting a subset of channels/time points

### Customization

The metrics can be customized for specific needs:
- Adjust frequency bands for frequency-domain metrics
- Modify thresholds for artifact detection
- Add custom metrics for specific artifact types

## References

1. Delorme, A., & Makeig, S. (2004). EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics including independent component analysis. Journal of neuroscience methods, 134(1), 9-21.

2. Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., ... & Hämäläinen, M. S. (2014). MNE software for processing MEG and EEG data. Neuroimage, 86, 446-460.

3. Nolan, H., Whelan, R., & Reilly, R. B. (2010). FASTER: fully automated statistical thresholding for EEG artifact rejection. Journal of neuroscience methods, 192(1), 152-162.

4. Pion-Tonachini, L., Kreutz-Delgado, K., & Makeig, S. (2019). ICLabel: An automated electroencephalographic independent component classifier, dataset, and website. NeuroImage, 198, 181-197.

5. Winkler, I., Haufe, S., & Tangermann, M. (2011). Automatic classification of artifactual ICA-components for artifact removal in EEG signals. Behavioral and Brain Functions, 7(1), 1-15.

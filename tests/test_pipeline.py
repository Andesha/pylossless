import pytest
import mne
import numpy as np
import pylossless as ll

@pytest.fixture
def mne_sample_data():
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = (
        sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
    )
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True).pick('eeg')
    raw.info['bads'] = [] # EEG 053 should get detected

    return raw

@pytest.fixture
def mne_sample_data_config():
    # Have to change the filtering as the sfreq of the sample is low
    config_object = ll.Config().load_default()
    config_object['filtering']['filter_args']['h_freq'] = 60
    config_object['filtering']['notch_filter_args']['freqs'] = []

    return config_object

def test_sum(mne_sample_data, mne_sample_data_config):
    original_raw = mne_sample_data.copy()
    pipeline = ll.LosslessPipeline(config=mne_sample_data_config)
    pipeline.run_with_raw(mne_sample_data)
    assert pipeline.flags is not None, 'Pipeline flags somehow empty after basic invocation'

    rejection_policy = ll.RejectionPolicy()
    rejection_policy['ch_cleaning_mode'] = 'interpolate'
    cleaned_raw = rejection_policy.apply(pipeline)

    clean_matrix = cleaned_raw.get_data() 
    old_matrix = original_raw.get_data()

    if clean_matrix.shape == old_matrix.shape:
        diff = np.abs(clean_matrix - old_matrix)
        assert np.any(diff > 0), "Cleaning did not provide a difference in data"

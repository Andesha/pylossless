![logo](https://github.com/scott-huberty/wip_pipeline-figures/blob/main/logo/Logo_neutral.png)


## Introduction to the Lossless Pipeline

This EEG processing pipeline is especially useful for the following scenarios:

- You want to keep your EEG data in a continuous state, allowing you the flexibility to
  epoch your data at a later stage.
- You are part of a research team or community that shares a common dataset, and you
  want to process the data once in a way that can be used for multiple analyses (i.e.,
  one analysis can segment the cleaned data into 10-second epochs and filter the data
  betweeen 1-30Hz, while another analysis can use 1-second epochs with no filter, etc.)
- You want to be able to do a hands on review of the pre-processing results for each file.

## Fork Description

This fork is maintained as a lightweight HPC-ready version of the original implementation.

All credit to the original Authors.


## 📘 Installation and usage instructions

The development version can be installed from GitHub with
```bash
$ git clone https://github.com/Andesha/pylossless.git
$ pip install ./pylossless
```

Please find the full documentation at
[**pylossless.readthedocs.io**](https://pylossless.readthedocs.io/en/latest/index.html).


## ▶️ Running the pyLossless Pipeline
Below is a minimal example that runs the pipeline one of MNE's sample files.  
```python
import pylossless as ll 
import mne
fname = mne.datasets.sample.data_path() / 'MEG' / 'sample' /  'sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(fname, preload=True)

config = ll.config.Config()
config.load_default()

pipeline = ll.LosslessPipeline(config=config)
pipeline.run_with_raw(raw)
```

Once it is completed, You can see what channels and times were flagged:
```python
print(pipeline.flagged_chs)
print(pipeline.flagged_epochs)
```

Once you are ready, you can save your file in its lossless state:
```python
pipeline.save(pipeline.get_derivative_path(bids_path), overwrite=True)
```

To get a **cleaned** version, you can use a `RejectionPolicy` object to apply
these annotations to your raw object. This is a lossy operation:
```python
rejection_policy = ll.RejectionPolicy()
cleaned_raw = rejection_policy.apply(pipeline)
```

## ▶️ Example HPC Environment Setup

If you are a Canadian researcher working on an HPC system such as [Narval](https://docs.alliancecan.ca/wiki/Narval/en):

```bash
# Build the virtualenv in your homedir
virtualenv --no-download eeg-env
source eeg-env/bin/activate

pip install --no-index mne
pip install --no-index pandas
pip install --no-index xarray
pip install --no-index pyyaml
pip install --no-index sklearn
pip install mne_bids

# Clone down mne-iclabel and switch to the right version and install it locally
git clone https://github.com/mne-tools/mne-icalabel.git
cd mne-icalabel
git checkout maint/0.4
pip install .

# Clone down pipeline and install without reading dependencies
git clone git@github.com:lina-usc/pylossless.git
cd pylossless
pip install --no-deps .

# Verify that the package has installed correct with an import
python -c 'import pylossless'
```

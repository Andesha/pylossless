![logo](https://github.com/scott-huberty/wip_pipeline-figures/blob/main/logo/Logo_neutral.png)


## Introduction to the Lossless Pipeline

The pyLossless pre-processing pipeline focuses on isolation of cortical signals from noise while retaining maximal information from the raw EEG data. The pipeline’s goal is to ensure minimal data manipulation and enhance signal quality annotations. This provides researchers with a scalable solution for handling large datasets across multiple EEG recording conditions. Particular attention has also been given to making the package easy to run for researchers just beginning their journey into working with large datasets in a High Performance Computing (HPC) environment.

### Critical assumptions of the pipeline's approach

The following are the theoretical assumptions and foundations:

* Independent component analysis (ICA) for removal of artifacts is extremely effective at cleaning data
* ICA algorithms fall appart when there is non-stationarity (wild variance) in the data
* The pipeline should seek to eliminate these times/sources to improve ICA outcomes
* Raw voltage values does not tell the complete story of an artifact
* Examining distributions of the variances of voltages within small windows allows for rejection of bad sources or time

The following are implementation requirements of the pipeline that meets the above assumptions:

* Expert manual review is important and must be possible
* Researcher must be able to free to manipulate their data before and after the pipeline easily
* The pipeline must be able to be effective on large datasets

### Pipeline stages

The key stages of the pipeline include:

1. Special average reference calculation for leaving out comically bad channels
2. Rejection of bad channels followed by rejection of bad time
3. High and low pass filtering
4. Bridged channel rejection (and rejection of channels that are too unlike their neighbours) as well as rank channel
5. Compute first ICA
6. Rejection of time where ICA failed to decompose signal into minimal number of components
7. Second pass of ICA
8. Post processing and classication of components using ICLabel

For a formal breakdown of each step, please see the documentation.

### Output state

Users may choose to export data from the pipeline in any way they see fit. This includes BIDS, MNE's fif raw objects, EEGLAB files, or just cleaned EDFs.

## Documentation
 
Please find the full documentation at
[**pylossless.readthedocs.io**](https://pylossless.readthedocs.io/en/latest/index.html).

**NOTE**: This documentation refers to the original implementation and may have some slight differences compared to this fork. For now, issues can be opened here on GitHub if things appear to be incorrect/broken.

## Fork Description

This fork is maintained as a lightweight HPC-ready version of the original implementation.

All credit to the original Authors and their repository [here](https://github.com/lina-usc/pylossless).


## 📘 Installation and usage instructions

To begin using the latest version, it is recommended to create a new virtual environment to install the package in. Assuming you have done so, proceed as follows:
```bash
git clone https://github.com/Andesha/pylossless.git
pip install ./pylossless
```

### Installing the Quality Control dependencies

This package also of course supports expert review via a Quality Control (QC) process.

To install the package so that QCing is possible do the following at install time:
```bash
git clone https://github.com/Andesha/pylossless.git
pip install ./pylossless[qc]
```

NOTE: This can also be done after installing just the basic version as above.

### Running a simple build test

If you are unsure if the pipeline has been set up correctly, you can run a simple test via the CLI from within the pylossless folder via:
```bash
pytest -W ignore tests/test_eeg_metrics.py::test_pipeline_running
```

**NOTE**: You may need to manually make sure that the MNE sample data has a directory to download to. This issue was run into on MacOS and was fixed simply by running `mkdir -p ~/mne_data` in the terminal app.

If this passes, you can assume the pipeline is working and ready to go! If not, please open an issue on the issue tracker.

## ▶️ Running the pyLossless Pipeline
Below is a minimal example that runs the pipeline one of MNE's sample files. This example will return a warning that the sampling rate of the data is too low for ICLabel. This is an expected property of the sample data.

```python
import pylossless as ll 
import mne
fname = mne.datasets.sample.data_path() / 'MEG' / 'sample' /  'sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(fname, preload=True).pick('eeg')

config = ll.config.Config()
config.load_default()

pipeline = ll.LosslessPipeline(config=config)
pipeline.run_with_raw(raw)
```

Once it is completed, You can see what channels and times were flagged:
```python
print(pipeline.flags['ch'])
print(pipeline.flags['ic'])
print(pipeline.flags['epoch'])
```

Saving the pipeline can be done as a BIDS derivative if you are working within that structure, or more simply, you can forcibly save the subject at any given directory level via:

```python
pipeline.non_bids_save('1', '.', overwrite=True)
```

This example will save the subject as "sub-1" and create a new derivatives folder at `'.'`, meaning the current working directory.

To get a **cleaned** version, you can use a `RejectionPolicy` object to apply
these annotations to your raw object. This is a lossy operation:
```python
rejection_policy = ll.RejectionPolicy()
cleaned_raw = rejection_policy.apply(pipeline)
```

### Launching the Quality Control procedure

The following example will launch the beta version of QCing from this fork. Replace the path to the sample with your own file.

Components can be clicked on the topo plot to zoom in, and when rejecting components from the scroll plot, the effect on the scalp data will be illustrated.

```python
pipeline = ll.LosslessPipeline()
pipeline = pipeline.load_ll_derivative('derivatives/pylossless/sub-1/eeg/sub-1_task-pyl_eeg.edf')

rejection_policy = ll.RejectionPolicy()
review = ll.QC(pipeline, rejection_policy)
review.run()
cleaned_raw = review.apply_qc()
```

### ▶️ Example HPC Environment Setup

If you are a Canadian researcher working on an HPC system such as [Narval](https://docs.alliancecan.ca/wiki/Narval/en):

```bash
# Load python and build the virtualenv in your homedir
module load python/3.12
virtualenv --no-download ~/eeg-env
source ~/eeg-env/bin/activate

# Clone this also in your homedir and install
git clone https://github.com/Andesha/pylossless.git
pip install ./pylossless
```

#### Suggested Sample HPC Workflow

The following is a list of steps with scripts suggesting how to run any given study on an HPC resource, assuming you have first created the above virtual environment.

1. Upload your data through [Globus](https://docs.alliancecan.ca/wiki/Globus) or another tool like `rsync`.
2. Create your pipeline configuration file via the following snippet somewhere. This can be inside of the Python interpreter or in a notebook.:

```python
import pylossless as ll
config = ll.config.Config()
config.load_default()
# Make optional changes here like below:
config['project']['analysis_montage'] = 'biosemi128'
# Save to file. You can also edit it manually after.
config.save('project_specific_name.yaml')
```

3. Create a "main" file that will stage the recordings and execute the pipeline. If you are familiar with older versions of the pipeline you may do staging script activities here. The following is a sample template that is intended to be run as `python main.py /path/to/recording/ subject_id`. The `/path/to/recording/` argument should be replaced with something akin to `sub-001/eeg/sub-001_task-pyl_eeg.edf` that matches the project. The `subject_id` argument should be, in this case, `001`. The purpose of this is to give the pipeline script a chance to rename subjects given the chance. This is entirely optional and potentially redundant. The file should be saved at the root of the project.

```python
import pylossless as ll
import mne
import sys

# Take path and subject information from command line call
subject_file = sys.argv[1]
subject_id = sys.argv[2]

# Load data
raw = mne.io.read_raw(subject_file, preload=True)

# Automatically mark break periods via MNE helper
events = mne.find_events(raw)
anno = mne.annotations_from_events(events, raw.info['sfreq'])
raw.set_annotations(anno)
break_annots = mne.preprocessing.annotate_break(
    raw=raw,
    min_break_duration=10,
    t_start_after_previous=1,
    t_stop_before_next=1,
)
raw.set_annotations(raw.annotations + break_annots) 

# Load pipeline configuration and execute
pipeline = ll.LosslessPipeline('project_specific_name.yaml')
pipeline.run_with_raw(raw)

# Save pipeline output
pipeline.non_bids_save(subject_id, '.', overwrite=True)
```

4. Create a job script, `job.sh` that will prep the HPC environment and call `main.py`. The intention of this format is to allow for changes to be made to the shape of the job without having to have dozens of different scripts per subject. To know how to modify the `#SBATCH` fields, please see the Alliance specific documentation page [here](https://docs.alliancecan.ca/wiki/Running_jobs).

```bash
#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --account=def-your-name-here
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Assuming you have created the virtualenv with the same name as the above example
source ~/eeg-env/bin/activate

cd $YOUR_PROJECT_LOCATION

# The $@ convention is what allows us to pass arguments into this job script externally
python main.py $@
```

5. Create the final submission bash script, typically called `run_all.sh`. This script can be generated by spreadsheet, or other tool and calls `job.sh` on each subject file. As seen below, information after the `job.sh` argument is passed into the `main.py` call. The intention being that changing all job parameters is only done once. Further, this names all jobs consistently to better monitor potential failures. Other arguments name the jobs, and redirect their output and errors to one helpful location. Make sure this is created at the root level with `mkdir logs`. Note that the subject ids, file names, etc are intended to be changed to match a given study.

```
# Sample run_all.sh
sbatch --output=logs/sub-001_task-Face13_eeg.out --error=logs/sub-001_task-Face13_eeg.err --job-name=sub-001_task-Face13_eeg job.sh sub-001/eeg/sub-001_task-Face13_eeg.edf 001
sbatch --output=logs/sub-002_task-Face13_eeg.out --error=logs/sub-002_task-Face13_eeg.err --job-name=sub-002_task-Face13_eeg job.sh sub-002/eeg/sub-002_task-Face13_eeg.edf 002
sbatch --output=logs/sub-003_task-Face13_eeg.out --error=logs/sub-003_task-Face13_eeg.err --job-name=sub-003_task-Face13_eeg job.sh sub-003/eeg/sub-003_task-Face13_eeg.edf 003
sbatch --output=logs/sub-004_task-Face13_eeg.out --error=logs/sub-004_task-Face13_eeg.err --job-name=sub-004_task-Face13_eeg job.sh sub-004/eeg/sub-004_task-Face13_eeg.edf 004
sbatch --output=logs/sub-005_task-Face13_eeg.out --error=logs/sub-005_task-Face13_eeg.err --job-name=sub-005_task-Face13_eeg job.sh sub-005/eeg/sub-005_task-Face13_eeg.edf 005

```

6. Below are some helpful commands to help view job submission and completion status. Note that many of these are only applicable to the Alliance systems.
    * `sq`: prints the current user's jobs in the queue
    * `sacct -aX -u $USER -o jobid,jobname%50,state,elapsed`: print a simplified output of recent jobs are their state for the current user.
    * `sacct -aX -u $USER -S 2025-05-01`: look for all jobs submitted since May 1st 2025 by the current user.
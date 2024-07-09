"""PyLossless setup file.

Authors:
Christian O'Reilly <christian.oreilly@sc.edu>
Scott Huberty <seh33@uw.edu>
James Desjardins <jim.a.desjardins@gmail.com>
Tyler Collins <collins.tyler.k@gmail.com>
License: MIT
"""
from pathlib import Path
from setuptools import setup, find_packages

with Path("requirements.txt").open() as f:
    requirements = f.read().splitlines()[1:] # Trimming off that first line to avoid the torch cpu thing

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

qc_entry_point = ["pylossless_qc=pylossless.dash.pylossless_qc:main"]
setup(
    name="pylossless",
    version="0.1.0",
    description="Lossless EEG Processing Pipeline Built on MNE and Dash",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Scott Huberty",
    author_email="seh33@uw.edu",
    url="https://github.com/lina-usc/pylossless",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    entry_points={"console_scripts": qc_entry_point},
    dependency_links=['https://download.pytorch.org/whl/cpu'],
)

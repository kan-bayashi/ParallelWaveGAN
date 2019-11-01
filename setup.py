#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pip
import sys

from distutils.version import LooseVersion
from setuptools import find_packages
from setuptools import setup

if LooseVersion(sys.version) < LooseVersion("3.6"):
    raise RuntimeError(
        "ESPnet requires Python>=3.6, "
        "but your Python is {}".format(sys.version))
if LooseVersion(pip.__version__) < LooseVersion("19"):
    raise RuntimeError(
        "pip>=19.0.0 is required, but your pip is {}. "
        "Try again after \"pip install -U pip\"".format(pip.__version__))

requirements = {
    "install": [
        "torch>=1.0.1",  # Installation from anaconda is recommended for PyTorch
        "setuptools>=38.5.1",
        "librosa>=0.7.0",
        "soundfile>=0.10.2",
        "tensorboardX>=1.8",
        "matplotlib>=3.1.0",
        "PyYAML>=3.12",
        "tqdm>=4.26.1",
    ],
    "setup": [
        "numpy",
        "pytest-runner"
    ],
    "test": [
        "pytest>=3.3.0",
        "pytest-pythonpath>=0.7.3",
        "pytest-cov>=2.7.1",
        "hacking>=1.1.0",
        "mock>=2.0.0",
        "autopep8>=1.3.3",
        "jsondiff>=1.2.0",
        "flake8>=3.7.8",
        "flake8-docstrings>=1.3.1"
    ]}
install_requires = requirements["install"]
setup_requires = requirements["setup"]
tests_require = requirements["test"]
extras_require = {k: v for k, v in requirements.items()
                  if k not in ["install", "setup"]}

dirname = os.path.dirname(__file__)
setup(name="parallel_wavegan",
      version="0.0.1",
      url="http://github.com/kan-bayashi/ParalellWaveGAN",
      author="Tomoki Hayashi",
      description="Parallel WaveGAN implementation",
      long_description=open(os.path.join(dirname, "README.md"),
                            encoding="utf-8").read(),
      license="MIT License",
      packages=find_packages(include=["parallel_wavegan*"]),
      install_requires=install_requires,
      setup_requires=setup_requires,
      tests_require=tests_require,
      extras_require=extras_require,
      classifiers=[
          "Programming Language :: Python :: 3",
          "Intended Audience :: Science/Research",
          "Operating System :: POSIX :: Linux",
          "License :: OSI Approved :: Apache Software License",
          "Topic :: Software Development :: Libraries :: Python Modules"],
      )

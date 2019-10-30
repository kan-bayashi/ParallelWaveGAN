# Parallel WaveGAN implementation with Pytorch

This repository provides Parallel WaveGAN implementation with Pytorch.  
Now under construction.

## Requirements

- Python 3.6+
- Cuda 10.0

## Setup

### A. Use pip

```bash
git clone https://github.com/kan-bayashi/ParallelWaveGAN.git
cd ParallelWaveGAN
pip install -e .
pip install -e .[test]
```

### B. Make virtualenv

```bash
git clone https://github.com/kan-bayashi/ParallelWaveGAN.git
cd ParallelWaveGAN/tools
make
source venv/bin/activate
```

## TODO

- [x] implement generator
- [x] implement discriminator
- [x] implement STFT-based losss
- [x] implement training script
    - [x] data loader
    - [x] yaml style configuration
    - [x] optimizer
    - [x] lr schedular
    - [x] trainer
    - [x] resume function
    - [x] intermediate result checker
- [ ] train model and check the performance
- [ ] implement evaluation script
- [ ] implement pre-processing scripts (or reuse @r9y9's scripts?)

## References

- https://arxiv.org/pdf/1910.11480.pdf
- https://github.com/r9y9/wavenet_vocoder

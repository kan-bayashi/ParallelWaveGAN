# Parallel WaveGAN implementation with Pytorch

This repository provides **UNOFFICIAL** Parallel WaveGAN implementation with Pytorch.  
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

## Run

Currently this repository provides LJSPeech dataset recipe.

```bash
cd egs/ljspeech/voc1
./run.sh
```

The training is on going. Please check the #1.  
Once I finished the training, I will upload samples and pre-pretrained model.

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
- [x] implement decoding script
- [x] implement pre-processing scripts
    - [x] audio preprocessing
    - [x] feature extraction
    - [x] normalization
- [x] implement several recipes
    - [x] ljspeech
    - [ ] jsut
    - [ ] csmsc
- [ ] train model and check the performance

## References

- https://arxiv.org/pdf/1910.11480.pdf
- https://github.com/r9y9/wavenet_vocoder

## Contact

Tomoki Hayashi @ Nagoya University  
hayashi.tomoki@g.sp.m.is.nagoya-u.ac.jp

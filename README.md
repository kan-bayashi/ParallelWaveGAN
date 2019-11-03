# Parallel WaveGAN implementation with Pytorch

This repository provides **UNOFFICIAL** [Parallel WaveGAN](https://arxiv.org/abs/1910.11480) implementation with Pytorch.

![](https://user-images.githubusercontent.com/22779813/68081503-4b8fcf00-fe52-11e9-8791-e02851220355.png)

The goal of this repository is to provide the real-time neural vocoder which is compatible with [ESPnet-TTS](https://github.com/espnet/espnet).  
Audio samples and pretrained models will be available at [our google drive](https://drive.google.com/open?id=1sd_QzcUNnbiaWq7L0ykMP7Xmk-zOuxTi).

> Source of the figure: https://arxiv.org/pdf/1910.11480.pdf

## Requirements

This repository is tested on Ubuntu 16.04 with a GPU Titan V.

- Python 3.6+
- Cuda 10.0
- CuDNN 7+

All of the codes are tested on Pytorch 1.0.1, 1.1, 1.2, and 1.3.

## Setup

You can select the installation method from two alternatives.

### A. Use pip

```bash
$ git clone https://github.com/kan-bayashi/ParallelWaveGAN.git
$ cd ParallelWaveGAN
$ pip install -e .
```

### B. Make virtualenv

```bash
$ git clone https://github.com/kan-bayashi/ParallelWaveGAN.git
$ cd ParallelWaveGAN/tools
$ make
$ source venv/bin/activate
```

## Run

This repository provides [Kaldi](https://github.com/kaldi-asr/kaldi)-style recipes, as the same as [ESPnet](https://github.com/espnet/espnet).  
Currently, three recipes are supported.

- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/): English female speaker
- [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut): Japanese female speaker
- [CSMSC](https://www.data-baker.com/open_source.html): Mandarin female speaker

To run the recipe, please follow the below instruction.

```bash
# Let us move on the recipe directory
$ cd egs/ljspeech/voc1

# Run the recipe from scratch
$ ./run.sh

# You can select the stage to start and stop
$ ./run.sh --stage 2 --stop_stage 2
```

All of the hyperparameters is written in a single yaml format configuration file.  
Please check [this example](https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/egs/ljspeech/voc1/conf/parallel_wavegan.v1.yaml) in ljspeech recipe.

The training is still on going. Please check the (https://github.com/kan-bayashi/ParallelWaveGAN/issues/1).

## References

- [Parallel WaveGAN](https://arxiv.org/abs/1910.11480)
- [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)
- [LiyuanLucasLiu/RAdam](https://github.com/LiyuanLucasLiu/RAdam)

## Acknowledgement

The author would like to thank Ryuichi Yamamoto ([@r9y9](https://github.com/r9y9)) for his great repository, paper and valuable discussions.

## Author

Tomoki Hayashi ([@kan-bayashi](https://github.com/kan-bayashi))  
E-mail: `hayashi.tomoki<at>g.sp.m.is.nagoya-u.ac.jp`

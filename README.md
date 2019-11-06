# Parallel WaveGAN implementation with Pytorch

![](https://img.shields.io/pypi/l/parallel-wavegan?color=green) [![](https://img.shields.io/pypi/v/parallel-wavegan?color=blue)](https://pypi.org/project/parallel-wavegan/) ![](https://img.shields.io/github/last-commit/kan-bayashi/ParallelWaveGAN?color=red)

This repository provides **UNOFFICIAL** [Parallel WaveGAN](https://arxiv.org/abs/1910.11480) implementation with Pytorch.

You can check our samples in [our demo HP](https://kan-bayashi.github.io/ParallelWaveGAN)!

![](https://user-images.githubusercontent.com/22779813/68081503-4b8fcf00-fe52-11e9-8791-e02851220355.png)

The goal of this repository is to provide the real-time neural vocoder which is compatible with [ESPnet-TTS](https://github.com/espnet/espnet).  

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
Currently, four recipes are supported.

- [CMU Arctic](http://www.festvox.org/cmu_arctic/): English speakers
- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/): English female speaker
- [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut): Japanese female speaker
- [CSMSC](https://www.data-baker.com/open_source.html): Mandarin female speaker

To run the recipe, please follow the below instruction.

```bash
# Let us move on the recipe directory
$ cd egs/ljspeech/voc1

# Run the recipe from scratch
$ ./run.sh

# You can change config via command line
$ ./run.sh --conf <your_customized_yaml_config>

# You can select the stage to start and stop
$ ./run.sh --stage 2 --stop_stage 2

# If you want to specify the gpu
$ CUDA_VISIBLE_DEVICES=1 ./run.sh --stage 2
```

The integration with job schedulers such as [slurm](https://slurm.schedmd.com/documentation.html) can be done via `cmd.sh` and  `conf/slurm.conf`.  
If you want to use it, please check [this page](https://kaldi-asr.org/doc/queue.html).

All of the hyperparameters is written in a single yaml format configuration file.  
Please check [this example](https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/egs/ljspeech/voc1/conf/parallel_wavegan.v1.yaml) in ljspeech recipe.

The training requires ~3 days with a single GPU (TITAN V).  
The speed of the training is 0.5 seconds per an iteration, in total ~ 200000 sec (= 2.31 days).  
You can monitor the training progress via tensorboard.

```bash
$ tensorboard --logdir exp
```

![](https://user-images.githubusercontent.com/22779813/68100080-58bbc500-ff09-11e9-9945-c835186fd7c2.png)

The decoding speed is RTF = 0.016 with TITAN V, much faster than the real-time.

```bash
[decode]: 100%|██████████| 250/250 [00:30<00:00,  8.31it/s, RTF=0.0156]
2019-11-03 09:07:40,480 (decode:127) INFO: finished generation of 250 utterances (RTF = 0.016).
```

Even on the CPU (Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz 16 threads), it can generate less than the real-time.

```bash
[decode]: 100%|██████████| 250/250 [22:16<00:00,  5.35s/it, RTF=0.841]
2019-11-06 09:04:56,697 (decode:129) INFO: finished generation of 250 utterances (RTF = 0.734).
```

## Results
You can listen to the samples and download pretrained models at [our google drive](https://drive.google.com/open?id=1sd_QzcUNnbiaWq7L0ykMP7Xmk-zOuxTi).  

The training is still on going. Please check the latest progress at https://github.com/kan-bayashi/ParallelWaveGAN/issues/1.

## References

- [Parallel WaveGAN](https://arxiv.org/abs/1910.11480)
- [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)
- [LiyuanLucasLiu/RAdam](https://github.com/LiyuanLucasLiu/RAdam)

## Acknowledgement

The author would like to thank Ryuichi Yamamoto ([@r9y9](https://github.com/r9y9)) for his great repository, paper and valuable discussions.

## Author

Tomoki Hayashi ([@kan-bayashi](https://github.com/kan-bayashi))  
E-mail: `hayashi.tomoki<at>g.sp.m.is.nagoya-u.ac.jp`

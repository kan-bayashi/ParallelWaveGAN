---
date: 2019-11-05T00:00:00+09:00
type: "index"
---

# Unofficial Parallel WaveGAN implementation demo

This is the demonstration page of **UNOFFICIAL** Parallel WaveGAN implementation.

Github: https://github.com/kan-bayashi/ParallelWaveGAN


## Audio samples (English)


Here is the comparison in the analysis-synthesis condition.

- Groundtruth: Target speech
- Parallel WaveGAN (official): Official samples provided in [the official demo HP](https://r9y9.github.io/demos/projects/icassp2020).
- Parallel WaveGAN (ours): Our samples based [this config](https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/egs/ljspeech/voc1/conf/parallel_wavegan.v1.yaml).


| Groundtruth | ParallelWaveGAN (official) | ParallelWaveGAN (ours) |
| --- | --- | --- |
|<audio controls="" ><source src="wav/ljspeech/raw/LJ050-0029.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/r9y9_wavegan/LJ050-0029.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/kan-bayashi_wavegan/LJ050-0029.wav"/></audio>|

| Groundtruth | ParallelWaveGAN (official) | ParallelWaveGAN (ours) |
| --- | --- | --- |
|<audio controls="" ><source src="wav/ljspeech/raw/LJ050-0030.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/r9y9_wavegan/LJ050-0030.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/kan-bayashi_wavegan/LJ050-0030.wav"/></audio>|

| Groundtruth | ParallelWaveGAN (official) | ParallelWaveGAN (ours) |
| --- | --- | --- |
|<audio controls="" ><source src="wav/ljspeech/raw/LJ050-0031.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/r9y9_wavegan/LJ050-0031.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/kan-bayashi_wavegan/LJ050-0031.wav"/></audio>|

| Groundtruth | ParallelWaveGAN (official) | ParallelWaveGAN (ours) |
| --- | --- | --- |
|<audio controls="" ><source src="wav/ljspeech/raw/LJ050-0032.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/r9y9_wavegan/LJ050-0032.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/kan-bayashi_wavegan/LJ050-0032.wav"/></audio>|

| Groundtruth | ParallelWaveGAN (official) | ParallelWaveGAN (ours) |
| --- | --- | --- |
|<audio controls="" ><source src="wav/ljspeech/raw/LJ050-0033.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/r9y9_wavegan/LJ050-0033.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/kan-bayashi_wavegan/LJ050-0033.wav"/></audio>|


## References

- [Parallel WaveGAN](https://arxiv.org/abs/1910.11480)
- [Official Parallel WaveGAN demo](https://r9y9.github.io/demos/projects/icassp2020)


## Author

Tomoki Hayashi  
e-mail: hayashi.tomoki@g.sp.m.is.nagoya-u.ac.jp


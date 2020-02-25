---
date: 2019-11-05T00:00:00+09:00
type: "index"
---

# Unofficial Parallel WaveGAN (+ MelGAN) implementation demo

This is the demonstration page of **UNOFFICIAL** Parallel WaveGAN and MelGAN implementations.

Github: https://github.com/kan-bayashi/ParallelWaveGAN


## Audio samples (English)


Here is the comparison in the analysis-synthesis condition using [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).  
Note that we limit the frequency range from 80 to 7600 Hz in Mel spectrogram calculation.

- **Groundtruth**: Target speech.
- **Parallel WaveGAN (official)**: Official samples provided in [the official demo HP](https://r9y9.github.io/demos/projects/icassp2020).
- **Parallel WaveGAN (ours)**: Our samples based [this config](https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/egs/ljspeech/voc1/conf/parallel_wavegan.v1.yaml).
- **MelGAN + STFT-loss (ours)**: Our samples based [this config](https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/egs/ljspeech/voc1/conf/melgan.v3.long.yaml).

|     |     |
| --- | --- |
| **Groundtruth** | **ParallelWaveGAN (official)** |
|<audio controls="" ><source src="wav/ljspeech/raw/LJ050-0029.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/r9y9_wavegan/LJ050-0029.wav"/></audio>|
| **ParallelWaveGAN (ours)** | **MelGAN + STFT-loss (ours)** |
|<audio controls="" ><source src="wav/ljspeech/kan-bayashi_wavegan/LJ050-0029.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/kan-bayashi_melgan/LJ050-0029.wav"/></audio>|
|     |     |
| **Groundtruth** | **ParallelWaveGAN (official)** |
|<audio controls="" ><source src="wav/ljspeech/raw/LJ050-0030.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/r9y9_wavegan/LJ050-0030.wav"/></audio>|
| **ParallelWaveGAN (ours)** | **MelGAN + STFT-loss (ours)** |
|<audio controls="" ><source src="wav/ljspeech/kan-bayashi_wavegan/LJ050-0030.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/kan-bayashi_melgan/LJ050-0030.wav"/></audio>|
|     |     |
| **Groundtruth** | **ParallelWaveGAN (official)** |
|<audio controls="" ><source src="wav/ljspeech/raw/LJ050-0031.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/r9y9_wavegan/LJ050-0031.wav"/></audio>|
| **ParallelWaveGAN (ours)** | **MelGAN + STFT-loss (ours)** |
|<audio controls="" ><source src="wav/ljspeech/kan-bayashi_wavegan/LJ050-0031.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/kan-bayashi_melgan/LJ050-0031.wav"/></audio>|
|     |     |
| **Groundtruth** | **ParallelWaveGAN (official)** |
|<audio controls="" ><source src="wav/ljspeech/raw/LJ050-0032.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/r9y9_wavegan/LJ050-0032.wav"/></audio>|
| **ParallelWaveGAN (ours)** | **MelGAN + STFT-loss (ours)** |
|<audio controls="" ><source src="wav/ljspeech/kan-bayashi_wavegan/LJ050-0032.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/kan-bayashi_melgan/LJ050-0032.wav"/></audio>|
|     |     |
| **Groundtruth** | **ParallelWaveGAN (official)** |
|<audio controls="" ><source src="wav/ljspeech/raw/LJ050-0033.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/r9y9_wavegan/LJ050-0033.wav"/></audio>|
| **ParallelWaveGAN (ours)** | **MelGAN + STFT-loss (ours)** |
|<audio controls="" ><source src="wav/ljspeech/kan-bayashi_wavegan/LJ050-0033.wav"/></audio>|<audio controls="" ><source src="wav/ljspeech/kan-bayashi_melgan/LJ050-0033.wav"/></audio>|

## Audio samples (Japanese)

Audio sampels trained on [JSUT dataset](https://sites.google.com/site/shinnosuketakamichi/publication/jsut).  
Note that groundtruth samples are 48 kHz and we downsampled to 24 kHz and we limit the frequency range from 80 to 7600 Hz in Mel spectrogram calculation.

- **Groundtruth**: Target speech.
- **Parallel WaveGAN (ours)**: Our samples based [this config](https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/egs/jsut/voc1/conf/parallel_wavegan.v1.yaml).

|     |     |
| --- | --- |
| **Groundtruth** | **ParallelWaveGAN (ours)** |
|<audio controls="" ><source src="wav/jsut/raw/BASIC5000_0001.wav"/></audio>|<audio controls="" ><source src="wav/jsut/kan-bayashi_wavegan.v1/BASIC5000_0001.wav"/></audio>|
| **Groundtruth** | **ParallelWaveGAN (ours)** |
|<audio controls="" ><source src="wav/jsut/raw/BASIC5000_0002.wav"/></audio>|<audio controls="" ><source src="wav/jsut/kan-bayashi_wavegan.v1/BASIC5000_0002.wav"/></audio>|
| **Groundtruth** | **ParallelWaveGAN (ours)** |
|<audio controls="" ><source src="wav/jsut/raw/BASIC5000_0003.wav"/></audio>|<audio controls="" ><source src="wav/jsut/kan-bayashi_wavegan.v1/BASIC5000_0003.wav"/></audio>|


## Audio samples (Mandarin)

Audio sampels trained on [CSMSC dataset](https://www.data-baker.com/open_source.html).  
Note that groundtruth samples are 48 kHz and we downsampled to 24 kHz and we limit the frequency range from 80 to 7600 Hz in Mel spectrogram calculation.

- **Groundtruth**: Target speech.
- **Parallel WaveGAN (ours)**: Our samples based [this config](https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/egs/csmsc/voc1/conf/parallel_wavegan.v1.yaml).

|     |     |
| --- | --- |
| **Groundtruth** | **ParallelWaveGAN (ours)** |
|<audio controls="" ><source src="wav/csmsc/raw/009901.wav"/></audio>|<audio controls="" ><source src="wav/csmsc/kan-bayashi_wavegan.v1/009901.wav"/></audio>|
| **Groundtruth** | **ParallelWaveGAN (ours)** |
|<audio controls="" ><source src="wav/csmsc/raw/009902.wav"/></audio>|<audio controls="" ><source src="wav/csmsc/kan-bayashi_wavegan.v1/009902.wav"/></audio>|
| **Groundtruth** | **ParallelWaveGAN (ours)** |
|<audio controls="" ><source src="wav/csmsc/raw/009903.wav"/></audio>|<audio controls="" ><source src="wav/csmsc/kan-bayashi_wavegan.v1/009903.wav"/></audio>|


## References

- [Parallel WaveGAN](https://arxiv.org/abs/1910.11480)
- [MelGAN](https://arxiv.org/abs/1910.06711)
- [Official Parallel WaveGAN demo](https://r9y9.github.io/demos/projects/icassp2020)


## Author

Tomoki Hayashi  
e-mail: hayashi.tomoki@g.sp.m.is.nagoya-u.ac.jp


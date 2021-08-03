#!/usr/bin/env python3

# Copyright 2021 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Test code for Mel-spectrogram loss modules."""


import numpy as np
import torch

from parallel_wavegan.bin.preprocess import logmelfilterbank
from parallel_wavegan.losses import MelSpectrogram


def test_mel_spectrogram_is_equal():
    x = np.random.randn(22050)
    x = np.abs(x) / np.max(np.abs(x))
    mel_npy = logmelfilterbank(
        x,
        22050,
        fft_size=1024,
        hop_size=256,
        win_length=None,
        window="hann",
        num_mels=80,
        fmin=80,
        fmax=7600,
        eps=1e-10,
    )
    mel_spectrogram = MelSpectrogram(
        fs=22050,
        fft_size=1024,
        hop_size=256,
        win_length=None,
        window="hann",
        num_mels=80,
        fmin=80,
        fmax=7600,
        eps=1e-10,
    ).to(dtype=torch.double)
    mel_torch = mel_spectrogram(torch.from_numpy(x).unsqueeze(0))
    np.testing.assert_array_almost_equal(
        mel_npy.transpose(1, 0).astype(np.float32),
        mel_torch[0].numpy().astype(np.float32),
    )

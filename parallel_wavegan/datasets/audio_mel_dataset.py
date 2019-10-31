# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset related classes."""

import logging

import numpy as np

from torch.utils.data import Dataset

from parallel_wavegan.utils import find_files


class AudioMelDataset(Dataset):
    """PyTorch compatible dataset."""

    def __init__(self,
                 root_dir,
                 audio_query="*-wave.npy",
                 mel_query="*-feats.npy",
                 audio_length_threshold=None,
                 mel_length_threshold=None,
                 audio_load_fn=np.load,
                 mel_load_fn=np.load,
                 return_filename=False,
                 ):
        """Initialize pytorch dataset."""
        # find all of audio and mel files
        audio_files = sorted(find_files(root_dir, audio_query))
        mel_files = sorted(find_files(root_dir, mel_query))

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [idx for idx in range(len(audio_files)) if audio_lengths[idx] >= audio_length_threshold]
            if len(audio_files) != len(idxs):
                logging.info(f"some files are filtered by audio length threshold "
                             f"({len(audio_files)} -> {len(idxs)}).")
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
            idxs = [idx for idx in range(len(mel_files)) if mel_lengths[idx] >= mel_length_threshold]
            if len(mel_files) != len(idxs):
                logging.info(f"some files are filtered by mel length threshold "
                             f"({len(mel_files)} -> {len(idxs)}).")
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."
        assert len(audio_files) == len(mel_files), \
            f"Number of audio and mel files are different ({len(audio_files)} vs {len(mel_files)})."

        self.audio_files = audio_files
        self.mel_files = mel_files
        self.audio_load_fn = audio_load_fn
        self.mel_load_fn = mel_load_fn
        self.return_filename = return_filename

    def __getitem__(self, idx):
        """Get specifed idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            ndarray: Audio signal (T,).
            ndarray: Mel spec feature (T', C).

        """
        audio = self.audio_load_fn(self.audio_files[idx])
        mel = self.mel_load_fn(self.mel_files[idx])

        if self.return_filename:
            return self.audio_files[idx], self.mel_files[idx], audio, mel
        else:
            return audio, mel

    def __len__(self):
        """Return dataset length."""
        return len(self.audio_files)


class MelDataset(Dataset):
    """PyTorch compatible dataset."""

    def __init__(self,
                 root_dir,
                 mel_query="*-feats.npy",
                 mel_length_threshold=None,
                 mel_load_fn=np.load,
                 return_filename=False,
                 ):
        """Initialize pytorch dataset."""
        # find all of the mel files
        mel_files = sorted(find_files(root_dir, mel_query))

        # filter by threshold
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
            idxs = [idx for idx in range(len(mel_files)) if mel_lengths[idx] >= mel_length_threshold]
            if len(mel_files) != len(idxs):
                logging.info(f"some files are filtered by mel length threshold "
                             f"({len(mel_files)} -> {len(idxs)}).")
            mel_files = [mel_files[idx] for idx in idxs]

        # assert the number of files
        assert len(mel_files) != 0, f"Not found any mel files in ${root_dir}."

        self.mel_files = mel_files
        self.mel_load_fn = mel_load_fn
        self.return_filename = return_filename

    def __getitem__(self, idx):
        """Get specifed idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Filename.
            ndarray: Mel spec feature (T', C).

        """
        if self.return_filename:
            return self.mel_files[idx], self.mel_load_fn(self.mel_files[idx])
        else:
            return self.mel_load_fn(self.mel_files[idx])

    def __len__(self):
        """Return dataset length."""
        return len(self.mel_files)

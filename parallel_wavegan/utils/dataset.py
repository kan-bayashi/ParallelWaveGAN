# -*- coding: utf-8 -*-

"""Dataset related classes."""

import glob
import logging
import os

import numpy as np

from nnmnkwii.datasets import FileDataSource
from nnmnkwii.datasets import FileSourceDataset


class BaseDataSource(FileDataSource):
    """Base class of data source."""

    def __init__(self, dump_root, query="*.npy", length_threshold=None):
        """Initialize base data source."""
        self.dump_root = dump_root
        self.query = query
        self.length_threshold = length_threshold

    def collect_files(self):
        """Collect all of the files."""
        paths = sorted(glob.glob(os.path.join(self.dump_root, self.query)))
        if self.length_threshold is not None:
            new_paths = [p for p in paths if self.collect_features(p).shape[0] > self.length_threshold]
            if len(paths) != len(new_paths):
                logging.warn(f"short samples are remvoed ({len(paths)} -> {len(new_paths)}).")
            return new_paths
        else:
            return paths

    def collect_features(self, path):
        """Collect feature from path."""
        return np.load(path)


class RawAudioDataSource(BaseDataSource):
    """Raw audio data source."""

    def __init__(self, dump_root, audio_length_threshold=None):
        """Initialize Raw audio data source."""
        super(RawAudioDataSource, self).__init__(dump_root, "*-wave.npy", audio_length_threshold)


class MelSpecDataSource(BaseDataSource):
    """Mel spectrogram data source."""

    def __init__(self, dump_root, mel_length_threshold=None):
        """Initialize Mel spectrogram data source."""
        super(MelSpecDataSource, self).__init__(dump_root, "*-feats.npy", mel_length_threshold)


class RawAudioDataset(FileSourceDataset):
    """Raw audio dataset."""

    def __init__(self, dump_root, audio_length_threshold=None):
        """Initialize Raw audio dataset."""
        data_source = RawAudioDataSource(dump_root, audio_length_threshold)
        super(RawAudioDataset, self).__init__(data_source)


class MelSpecDataset(FileSourceDataset):
    """Raw audio dataset."""

    def __init__(self, dump_root, mel_length_threshold=None):
        """Initialize Raw audio dataset."""
        data_source = MelSpecDataSource(dump_root, mel_length_threshold)
        super(MelSpecDataset, self).__init__(data_source)


class PyTorchDataset(object):
    """PyTorch compatible dataset."""

    def __init__(self, dump_root, audio_length_threshold=None, mel_length_threshold=None):
        """Initialize pytroch dataset."""
        audio_dataset = RawAudioDataset(dump_root, audio_length_threshold)
        mel_dataset = MelSpecDataset(dump_root, mel_length_threshold)
        assert len(audio_dataset) == len(mel_dataset)
        self.audio_dataset = audio_dataset
        self.mel_dataset = mel_dataset

    def __getitem__(self, idx):
        """Get specifed idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            ndarray: Audio signal (T,).
            ndarray: Mel spec feature (T', C).

        """
        audio = self.audio_dataset[idx]
        mel = self.mel_dataset[idx]
        return audio, mel

    def __len__(self):
        """Return dataset length."""
        return len(self.audio_dataset)

# -*- coding: utf-8 -*-

"""Dataset related classes."""

import glob
import os

import numpy as np

from nnmnkwii.datasets import FileDataSource
from nnmnkwii.datasets import FileSourceDataset


class BaseDataSource(FileDataSource):
    """Base class of data source."""

    def __init__(self, dump_root, query="*.npy"):
        """Initialize base data source."""
        self.dump_root = dump_root
        self.query = query

    def collect_files(self):
        """Collect all of the files."""
        paths = sorted(glob.glob(os.path.join(self.dump_root, self.query)))
        return paths

    def collect_features(self, path):
        """Collect feature from path."""
        return np.load(path)


class RawAudioDataSource(BaseDataSource):
    """Raw audio data source."""

    def __init__(self, dump_root):
        """Initialize Raw audio data source."""
        super(RawAudioDataSource, self).__init__(dump_root, "*-wave.npy")


class MelSpecDataSource(BaseDataSource):
    """Mel spectrogram data source."""

    def __init__(self, dump_root):
        """Initialize Mel spectrogram data source."""
        super(MelSpecDataSource, self).__init__(dump_root, "*-feats.npy")


class RawAudioDataset(FileSourceDataset):
    """Raw audio dataset."""

    def __init__(self, dump_root):
        """Initialize Raw audio dataset."""
        data_source = RawAudioDataSource(dump_root)
        super(RawAudioDataset, self).__init__(data_source)


class MelSpecDataset(FileSourceDataset):
    """Raw audio dataset."""

    def __init__(self, dump_root):
        """Initialize Raw audio dataset."""
        data_source = MelSpecDataSource(dump_root)
        super(MelSpecDataset, self).__init__(data_source)


class PyTorchDataset(object):
    """PyTorch compatible dataset."""

    def __init__(self, dump_root):
        """Initialize pytroch dataset."""
        audio_dataset = RawAudioDataset(dump_root)
        mel_dataset = MelSpecDataset(dump_root)
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

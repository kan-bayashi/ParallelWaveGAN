# -*- coding: utf-8 -*-

"""Dataset related classes."""

import glob
import os

import numpy as np

from nnmnkwii.datasets import FileDataSource
from nnmnkwii.datasets import FileSourceDataset


class BaseDataSource(FileDataSource):
    """Base class of data source."""

    def __init__(self, dump_root, query="*.npy", min_length=None):
        """Initialize base data source."""
        self.dump_root = dump_root
        self.query = query
        self.min_length = min_length

    def collect_files(self):
        """Collect all of the files."""
        paths = sorted(glob.glob(os.path.join(self.dump_root, self.query)))
        lengths = [self.collect_features(path).shape[0] for path in paths]
        # remove short utterances
        if self.min_length is not None:
            idxs = [idx for idx, l in enumerate(lengths) if l >= self.min_length]
            if len(idxs) != len(paths):
                print(f"short utterances are removed ({len(paths)} -> {len(idxs)}).")
            paths = [paths[idx] for idx in idxs]
            lengths = [lengths[idx] for idx in idxs]
        self.lengths = lengths
        return paths

    def collect_features(self, path):
        """Collect feature from path."""
        return np.load(path)


class RawAudioDataSource(BaseDataSource):
    """Raw audio data source."""

    def __init__(self, dump_root, min_length=None):
        """Initialize Raw audio data source."""
        super(RawAudioDataSource, self).__init__(dump_root, "*-wave.npy", min_length)


class MelSpecDataSource(BaseDataSource):
    """Mel spectrogram data source."""

    def __init__(self, dump_root, min_length=None):
        """Initialize Mel spectrogram data source."""
        super(MelSpecDataSource, self).__init__(dump_root, "*-feats.npy", min_length)


class RawAudioDataset(FileSourceDataset):
    """Raw audio dataset."""

    def __init__(self, dump_root, min_length=None):
        """Initialize Raw audio dataset."""
        data_source = RawAudioDataSource(dump_root, min_length)
        super(RawAudioDataset, self).__init__(data_source)


class MelSpecDataset(FileSourceDataset):
    """Raw audio dataset."""

    def __init__(self, dump_root, min_length=None):
        """Initialize Raw audio dataset."""
        data_source = MelSpecDataSource(dump_root, min_length)
        super(MelSpecDataset, self).__init__(data_source)


class PyTorchDataset(object):
    """PyTorch compatible dataset."""

    def __init__(self, dump_root, audio_min_length=None, mel_min_length=None):
        """Initialize pytroch dataset."""
        audio_dataset = RawAudioDataset(dump_root, audio_min_length)
        mel_dataset = MelSpecDataset(dump_root, mel_min_length)
        assert len(audio_dataset) == len(mel_dataset)
        self.audio_dataset = audio_dataset
        self.mel_dataset = mel_dataset
        self.audio_lengths = np.array(audio_dataset.file_data_source.lengths)
        self.mel_lengths = np.array(mel_dataset.file_data_source.lengths)

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

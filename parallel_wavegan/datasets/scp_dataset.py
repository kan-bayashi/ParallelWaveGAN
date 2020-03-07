# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules based on kaldi-style scp files."""

import logging

from multiprocessing import Manager

import kaldiio
import numpy as np

from torch.utils.data import Dataset


class AudioMelSCPDataset(Dataset):
    """PyTorch compatible audio and mel dataset based on kaldi-stype scp files."""

    def __init__(self,
                 wavscp,
                 featscp,
                 segments=None,
                 audio_length_threshold=None,
                 mel_length_threshold=None,
                 return_utt_id=False,
                 allow_cache=False,
                 ):
        """Initialize dataset.

        Args:
            wavscp (str): Kaldi-style wav.scp file.
            featscp (str): Kaldi-style fests.scp file.
            segments (str): Kaldi-style segments file.
            audio_length_threshold (int): Threshold to remove short audio files.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return utterance id.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # load scp as lazy dict
        audio_loader = kaldiio.load_scp(wavscp, segments=segments)
        mel_loader = kaldiio.load_scp(featscp)
        audio_keys = list(audio_loader.keys())
        mel_keys = list(mel_loader.keys())

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio.shape[0] for _, audio in audio_loader.values()]
            idxs = [idx for idx in range(len(audio_keys)) if audio_lengths[idx] > audio_length_threshold]
            if len(audio_keys) != len(idxs):
                logging.warning(f"Some files are filtered by audio length threshold "
                                f"({len(audio_keys)} -> {len(idxs)}).")
            audio_keys = [audio_keys[idx] for idx in idxs]
            mel_keys = [mel_keys[idx] for idx in idxs]
        if mel_length_threshold is not None:
            mel_lengths = [mel.shape[0] for mel in mel_loader.values()]
            idxs = [idx for idx in range(len(mel_keys)) if mel_lengths[idx] > mel_length_threshold]
            if len(mel_keys) != len(idxs):
                logging.warning(f"Some files are filtered by mel length threshold "
                                f"({len(mel_keys)} -> {len(idxs)}).")
            audio_keys = [audio_keys[idx] for idx in idxs]
            mel_keys = [mel_keys[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_keys) == len(mel_keys), \
            f"Number of audio and mel files are different ({len(audio_keys)} vs {len(mel_keys)})."

        self.audio_loader = audio_loader
        self.mel_loader = mel_loader
        self.utt_ids = audio_keys
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.utt_ids))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio signal (T,).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio = self.audio_loader[utt_id]
        mel = self.mel_loader[utt_id]

        # normalize audio signal to be [-1, 1]
        audio = audio.astype(np.float32)
        audio /= (1 << (16 - 1))  # assume that wav is PCM 16 bit

        if self.return_utt_id:
            items = utt_id, audio, mel
        else:
            items = audio, mel

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.utt_ids)


class AudioSCPDataset(Dataset):
    """PyTorch compatible audio dataset based on kaldi-stype scp files."""

    def __init__(self,
                 wavscp,
                 segments=None,
                 audio_length_threshold=None,
                 return_utt_id=False,
                 allow_cache=False,
                 ):
        """Initialize dataset.

        Args:
            wavscp (str): Kaldi-style wav.scp file.
            featscp (str): Kaldi-style fests.scp file.
            segments (str): Kaldi-style segments file.
            audio_length_threshold (int): Threshold to remove short audio files.
            return_utt_id (bool): Whether to return utterance id.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # load scp as lazy dict
        audio_loader = kaldiio.load_scp(wavscp, segments=segments)
        audio_keys = list(audio_loader.keys())

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio.shape[0] for _, audio in audio_loader.values()]
            idxs = [idx for idx in range(len(audio_keys)) if audio_lengths[idx] > audio_length_threshold]
            if len(audio_keys) != len(idxs):
                logging.warning(f"Some files are filtered by audio length threshold "
                                f"({len(audio_keys)} -> {len(idxs)}).")
            audio_keys = [audio_keys[idx] for idx in idxs]

        self.audio_loader = audio_loader
        self.utt_ids = audio_keys
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.utt_ids))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio signal (T,).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio = self.audio_loader[utt_id]

        # normalize audio signal to be [-1, 1]
        audio = audio.astype(np.float32)
        audio /= (1 << (16 - 1))  # assume that wav is PCM 16 bit

        if self.return_utt_id:
            items = utt_id, audio
        else:
            items = audio

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.utt_ids)


class MelSCPDataset(Dataset):
    """PyTorch compatible mel dataset based on kaldi-stype scp files."""

    def __init__(self,
                 featscp,
                 mel_length_threshold=None,
                 return_utt_id=False,
                 allow_cache=False,
                 ):
        """Initialize dataset.

        Args:
            featscp (str): Kaldi-style fests.scp file.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return utterance id.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # load scp as lazy dict
        mel_loader = kaldiio.load_scp(featscp)
        mel_keys = list(mel_loader.keys())

        # filter by threshold
        if mel_length_threshold is not None:
            mel_lengths = [mel.shape[0] for mel in mel_loader.values()]
            idxs = [idx for idx in range(len(mel_keys)) if mel_lengths[idx] > mel_length_threshold]
            if len(mel_keys) != len(idxs):
                logging.warning(f"Some files are filtered by mel length threshold "
                                f"({len(mel_keys)} -> {len(idxs)}).")
            mel_keys = [mel_keys[idx] for idx in idxs]

        self.mel_loader = mel_loader
        self.utt_ids = mel_keys
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.utt_ids))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        mel = self.mel_loader[utt_id]

        if self.return_utt_id:
            items = utt_id, mel
        else:
            items = mel

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.utt_ids)

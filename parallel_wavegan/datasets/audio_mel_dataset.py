# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules."""

import logging
import os
from multiprocessing import Manager

import numpy as np
from torch.utils.data import Dataset

from parallel_wavegan.utils import find_files, read_hdf5


class AudioMelDataset(Dataset):
    """PyTorch compatible audio and mel (+global conditioning feature) dataset."""

    def __init__(
        self,
        root_dir,
        audio_query="*.h5",
        audio_load_fn=lambda x: read_hdf5(x, "wave"),
        mel_query="*.h5",
        mel_load_fn=lambda x: read_hdf5(x, "feats"),
        local_query=None,
        local_load_fn=None,
        global_query=None,
        global_load_fn=None,
        audio_length_threshold=None,
        mel_length_threshold=None,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            audio_load_fn (func): Function to load audio file.
            mel_query (str): Query to find feature files in root_dir.
            mel_load_fn (func): Function to load feature file.
            local_query (str): Query to find local conditioning feature files in root_dir.
            local_load_fn (func): Function to load local conditioning feature file.
            global_query (str): Query to find global conditioning feature files in root_dir.
            global_load_fn (func): Function to load global conditioning feature file.
            audio_length_threshold (int): Threshold to remove short audio files.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        audio_files = sorted(find_files(root_dir, audio_query))
        mel_files = sorted(find_files(root_dir, mel_query))
        self.use_local = local_query is not None
        if self.use_local:
            local_files = sorted(find_files(root_dir, local_query))
        self.use_global = global_query is not None
        if self.use_global:
            global_files = sorted(find_files(root_dir, global_query))

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [
                idx
                for idx in range(len(audio_files))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_files) != len(idxs):
                logging.warning(
                    "Some files are filtered by audio length threshold "
                    f"({len(audio_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]
            if self.use_local:
                local_files = [local_files[idx] for idx in idxs]
            if self.use_global:
                global_files = [global_files[idx] for idx in idxs]
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
            idxs = [
                idx
                for idx in range(len(mel_files))
                if mel_lengths[idx] > mel_length_threshold
            ]
            if len(mel_files) != len(idxs):
                logging.warning(
                    "Some files are filtered by mel length threshold "
                    f"({len(mel_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]
            if self.use_local:
                local_files = [local_files[idx] for idx in idxs]
            if self.use_global:
                global_files = [global_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."
        assert len(audio_files) == len(mel_files), (
            f"Number of audio and mel files are different ({len(audio_files)} vs"
            f" {len(mel_files)})."
        )
        if self.use_local:
            assert len(audio_files) == len(local_files), (
                f"Number of audio and local files are different ({len(audio_files)} vs"
                f" {len(local_files)})."
            )
        if self.use_global:
            assert len(audio_files) == len(global_files), (
                f"Number of audio and global files are different ({len(audio_files)} vs"
                f" {len(global_files)})."
            )

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        self.mel_files = mel_files
        self.mel_load_fn = mel_load_fn
        if self.use_local:
            self.local_files = local_files
            self.local_load_fn = local_load_fn
        if self.use_global:
            self.global_files = global_files
            self.global_load_fn = global_load_fn
        if ".npy" in audio_query:
            self.utt_ids = [
                os.path.basename(f).replace("-wave.npy", "") for f in audio_files
            ]
        else:
            self.utt_ids = [
                os.path.splitext(os.path.basename(f))[0] for f in audio_files
            ]
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio signal (T,).
            ndarray: Feature (T', C).
            ndarray: Local feature (T' C').
            ndarray: Global feature (1,).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio = self.audio_load_fn(self.audio_files[idx])
        mel = self.mel_load_fn(self.mel_files[idx])
        items = (audio, mel)

        if self.use_local:
            l_ = self.local_load_fn(self.local_files[idx])
            items = items + (l_,)

        if self.use_global:
            g = self.global_load_fn(self.global_files[idx]).reshape(-1)
            items = items + (g,)

        if self.return_utt_id:
            items = (utt_id,) + items

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class AudioMelF0ExcitationDataset(Dataset):
    """PyTorch compatible audio and mel dataset."""

    def __init__(
        self,
        root_dir,
        audio_query="*.h5",
        mel_query="*.h5",
        f0_query="*.h5",
        excitation_query="*.h5",
        audio_load_fn=lambda x: read_hdf5(x, "wave"),
        mel_load_fn=lambda x: read_hdf5(x, "feats"),
        f0_load_fn=lambda x: read_hdf5(x, "f0"),
        excitation_load_fn=lambda x: read_hdf5(x, "excitation"),
        audio_length_threshold=None,
        mel_length_threshold=None,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            mel_query (str): Query to find mel feature files in root_dir.
            f0_query (str): Query to find f0 feature files in root_dir.
            excitation_query (str): Query to find excitation feature files in root_dir.
            audio_load_fn (func): Function to load audio file.
            mel_load_fn (func): Function to load mel feature file.
            audio_load_fn (func): Function to load audio file.
            mel_load_fn (func): Function to load mel feature file.
            f0_load_fn (func): Function to load f0 feature file.
            excitation_load_fn (func): Function to load excitation feature file.
            audio_length_threshold (int): Threshold to remove short audio files.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        audio_files = sorted(find_files(root_dir, audio_query))
        mel_files = sorted(find_files(root_dir, mel_query))
        f0_files = sorted(find_files(root_dir, f0_query))
        excitation_files = sorted(find_files(root_dir, excitation_query))

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [
                idx
                for idx in range(len(audio_files))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_files) != len(idxs):
                logging.warning(
                    "Some files are filtered by audio length threshold "
                    f"({len(audio_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]
            f0_files = [f0_files[idx] for idx in idxs]
            excitation_files = [excitation_files[idx] for idx in idxs]
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
            idxs = [
                idx
                for idx in range(len(mel_files))
                if mel_lengths[idx] > mel_length_threshold
            ]
            if len(mel_files) != len(idxs):
                logging.warning(
                    "Some files are filtered by mel length threshold "
                    f"({len(mel_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]
            f0_files = [f0_files[idx] for idx in idxs]
            excitation_files = [excitation_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."
        assert len(audio_files) == len(mel_files), (
            f"Number of audio and mel files are different ({len(audio_files)} vs"
            f" {len(mel_files)})."
        )
        assert len(audio_files) == len(f0_files), (
            f"Number of audio and f0 files are different ({len(audio_files)} vs"
            f" {len(f0_files)})."
        )
        assert len(audio_files) == len(excitation_files), (
            f"Number of audio and excitation files are different ({len(audio_files)} vs"
            f" {len(excitation_files)})."
        )

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        self.mel_files = mel_files
        self.mel_load_fn = mel_load_fn
        self.f0_files = f0_files
        self.f0_load_fn = f0_load_fn
        self.excitation_files = excitation_files
        self.excitation_load_fn = excitation_load_fn
        if ".npy" in audio_query:
            self.utt_ids = [
                os.path.basename(f).replace("-wave.npy", "") for f in audio_files
            ]
        else:
            self.utt_ids = [
                os.path.splitext(os.path.basename(f))[0] for f in audio_files
            ]

        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio signal (T,).
            ndarray: Feature (T', C).
            ndarray: Feature (T', ).
            ndarray: Feature (T', C').

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio = self.audio_load_fn(self.audio_files[idx])
        mel = self.mel_load_fn(self.mel_files[idx])
        f0 = self.f0_load_fn(self.f0_files[idx])
        excitation = self.excitation_load_fn(self.excitation_files[idx])

        if self.return_utt_id:
            items = utt_id, audio, mel, f0, excitation
        else:
            items = audio, mel, f0, excitation

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class AudioDataset(Dataset):
    """PyTorch compatible audio dataset."""

    def __init__(
        self,
        root_dir,
        audio_query="*-wave.npy",
        audio_length_threshold=None,
        audio_load_fn=np.load,
        local_query=None,
        local_load_fn=None,
        global_query=None,
        global_load_fn=None,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            audio_load_fn (func): Function to load audio file.
            audio_length_threshold (int): Threshold to remove short audio files.
            local_query (str): Query to find local conditioning feature files in root_dir.
            local_load_fn (func): Function to load local conditioning feature file.
            global_query (str): Query to find global conditioning feature files in root_dir.
            global_load_fn (func): Function to load global conditioning feature file.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        audio_files = sorted(find_files(root_dir, audio_query))
        self.use_local = local_query is not None
        self.use_global = global_query is not None
        if self.use_local:
            local_files = sorted(find_files(root_dir, local_query))
        if self.use_global:
            global_files = sorted(find_files(root_dir, global_query))

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [
                idx
                for idx in range(len(audio_files))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_files) != len(idxs):
                logging.warning(
                    "some files are filtered by audio length threshold "
                    f"({len(audio_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            if self.use_local:
                local_files = [local_files[idx] for idx in idxs]
            if self.use_global:
                global_files = [global_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."
        if self.use_local:
            assert len(audio_files) == len(local_files), (
                f"Number of audio and local files are different ({len(audio_files)} vs"
                f" {len(local_files)})."
            )
        if self.use_global:
            assert len(audio_files) == len(global_files), (
                f"Number of audio and global files are different ({len(audio_files)} vs"
                f" {len(global_files)})."
            )

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        if self.use_local:
            self.local_files = local_files
            self.local_load_fn = local_load_fn
        if self.use_global:
            self.global_files = global_files
            self.global_load_fn = global_load_fn
        if ".npy" in audio_query:
            self.utt_ids = [
                os.path.basename(f).replace("-wave.npy", "") for f in audio_files
            ]
        else:
            self.utt_ids = [
                os.path.splitext(os.path.basename(f))[0] for f in audio_files
            ]
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio (T,).
            ndarray: Feature (1,).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio = self.audio_load_fn(self.audio_files[idx])
        items = (audio,)
        if self.use_local:
            l_ = self.local_load_fn(self.local_files[idx])
            items = items + (l_,)
        if self.use_global:
            g = self.global_load_fn(self.global_files[idx]).reshape(-1)
            items = items + (g,)

        if self.return_utt_id:
            items = (utt_id,) + items

        # NOTE(kan-bayashi): if the return item is one, do not return as tuple
        if len(items) == 1:
            items = items[0]

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class MelDataset(Dataset):
    """PyTorch compatible mel (+global conditioning feature) dataset."""

    def __init__(
        self,
        root_dir,
        mel_query="*.h5",
        mel_load_fn=lambda x: read_hdf5(x, "feats"),
        local_query=None,
        local_load_fn=None,
        global_query=None,
        global_load_fn=None,
        mel_length_threshold=None,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            mel_query (str): Query to find feature files in root_dir.
            mel_load_fn (func): Function to load feature file.
            local_query (str): Query to find local conditioning feature files in root_dir.
            local_load_fn (func): Function to load local conditioning feature file.
            global_query (str): Query to find global conditioning feature files in root_dir.
            global_load_fn (func): Function to load global conditioning feature file.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        mel_files = sorted(find_files(root_dir, mel_query))
        self.use_local = local_query is not None
        self.use_global = global_query is not None
        if self.use_local:
            local_files = sorted(find_files(root_dir, local_query))
        if self.use_global:
            global_files = sorted(find_files(root_dir, global_query))

        # filter by threshold
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
            idxs = [
                idx
                for idx in range(len(mel_files))
                if mel_lengths[idx] > mel_length_threshold
            ]
            if len(mel_files) != len(idxs):
                logging.warning(
                    "Some files are filtered by mel length threshold "
                    f"({len(mel_files)} -> {len(idxs)})."
                )
            mel_files = [mel_files[idx] for idx in idxs]
            if self.use_local:
                local_files = [local_files[idx] for idx in idxs]
            if self.use_global:
                global_files = [global_files[idx] for idx in idxs]

        # assert the number of files
        if self.use_local:
            assert len(mel_files) == len(local_files), (
                f"Number of audio and local files are different ({len(mel_files)} vs"
                f" {len(local_files)})."
            )
        if self.use_global:
            assert len(mel_files) == len(global_files), (
                f"Number of audio and global files are different ({len(mel_files)} vs"
                f" {len(global_files)})."
            )

        self.mel_files = mel_files
        self.mel_load_fn = mel_load_fn
        if self.use_local:
            self.local_files = local_files
            self.local_load_fn = local_load_fn
        if self.use_global:
            self.global_files = global_files
            self.global_load_fn = global_load_fn
        if ".npy" in mel_query:
            self.utt_ids = [
                os.path.basename(f).replace("-feats.npy", "") for f in mel_files
            ]
        else:
            self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in mel_files]
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(mel_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Feature (T', C).
            ndarray: Feature (1,).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        mel = self.mel_load_fn(self.mel_files[idx])
        items = (mel,)

        if self.use_local:
            l_ = self.local_load_fn(self.local_files[idx])
            items = items + (l_,)

        if self.use_global:
            g = self.global_load_fn(self.global_files[idx]).reshape(-1)
            items = items + (g,)

        if self.return_utt_id:
            items = (utt_id,) + items

        # NOTE(kan-bayashi): if the return item is one, do not return as tuple
        if len(items) == 1:
            items = items[0]

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.mel_files)


class MelF0ExcitationDataset(Dataset):
    """PyTorch compatible mel dataset."""

    def __init__(
        self,
        root_dir,
        mel_query="*-feats.npy",
        f0_query="*-f0.npy",
        excitation_query="*-excitation.npy",
        mel_length_threshold=None,
        mel_load_fn=np.load,
        f0_load_fn=np.load,
        excitation_load_fn=np.load,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            mel_query (str): Query to find feature files in root_dir.
            mel_load_fn (func): Function to load feature file.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of the mel files
        mel_files = sorted(find_files(root_dir, mel_query))
        f0_files = sorted(find_files(root_dir, f0_query))
        excitation_files = sorted(find_files(root_dir, excitation_query))

        # filter by threshold
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
            idxs = [
                idx
                for idx in range(len(mel_files))
                if mel_lengths[idx] > mel_length_threshold
            ]
            if len(mel_files) != len(idxs):
                logging.warning(
                    "Some files are filtered by mel length threshold "
                    f"({len(mel_files)} -> {len(idxs)})."
                )
            mel_files = [mel_files[idx] for idx in idxs]
            f0_files = [f0_files[idx] for idx in idxs]
            excitation_files = [excitation_files[idx] for idx in idxs]

        # assert the number of files
        assert len(mel_files) != 0, f"Not found any mel files in ${root_dir}."
        assert len(f0_files) != 0, f"Not found any f0 files in ${root_dir}."
        assert (
            len(excitation_files) != 0
        ), f"Not found any excitation files in ${root_dir}."

        self.mel_files = mel_files
        self.mel_load_fn = mel_load_fn
        self.f0_files = f0_files
        self.f0_load_fn = f0_load_fn
        self.excitation_files = excitation_files
        self.excitation_load_fn = excitation_load_fn

        self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in mel_files]
        if ".npy" in mel_query:
            self.utt_ids = [
                os.path.basename(f).replace("-feats.npy", "") for f in mel_files
            ]
        else:
            self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in mel_files]
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(mel_files))]

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
        mel = self.mel_load_fn(self.mel_files[idx])
        f0 = self.f0_load_fn(self.f0_files[idx])
        excitation = self.excitation_load_fn(self.excitation_files[idx])

        if self.return_utt_id:
            items = utt_id, mel, f0, excitation
        else:
            items = mel, f0, excitation

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.mel_files)

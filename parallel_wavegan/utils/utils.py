# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Utility functions."""

import fnmatch
import logging
import os
import sys
import tarfile

from distutils.version import LooseVersion

import h5py
import numpy as np
import torch
import yaml

PRETRAINED_MODEL_LIST = {
    "ljspeech_parallel_wavegan.v1": "1PdZv37JhAQH6AwNh31QlqruqrvjTBq7U",
    "ljspeech_parallel_wavegan.v1.long": "1A9TsrD9fHxFviJVFjCk5W6lkzWXwhftv",
    "ljspeech_parallel_wavegan.v1.no_limit": "1CdWKSiKoFNPZyF1lo7Dsj6cPKmfLJe72",
    "ljspeech_parallel_wavegan.v3": "1-oZpwpWZMMolDYsCqeL12dFkXSBD9VBq",
    "ljspeech_full_band_melgan.v2": "1Kb7q5zBeQ30Wsnma0X23G08zvgDG5oen",
    "ljspeech_multi_band_melgan.v2": "1b70pJefKI8DhGYz4SxbEHpxm92tj1_qC",
    "ljspeech_hifigan.v1": "1XQD9pCBahhAbTVx5gvJ0q29iPZzcZOsN",
    "ljspeech_style_melgan.v1": "1oKN8wcN9FAQdG8tatUlqSaM2-MwUzkmd",
    "jsut_parallel_wavegan.v1": "1qok91A6wuubuz4be-P9R2zKhNmQXG0VQ",
    "jsut_multi_band_melgan.v2": "1chTt-76q2p69WPpZ1t1tt8szcM96IKad",
    "csmsc_parallel_wavegan.v1": "1QTOAokhD5dtRnqlMPTXTW91-CG7jf74e",
    "csmsc_multi_band_melgan.v2": "1G6trTmt0Szq-jWv2QDhqglMdWqQxiXQT",
    "arctic_slt_parallel_wavegan.v1": "1_MXePg40-7DTjD0CDVzyduwQuW_O9aA1",
    "jnas_parallel_wavegan.v1": "1D2TgvO206ixdLI90IqG787V6ySoXLsV_",
    "vctk_parallel_wavegan.v1": "1bqEFLgAroDcgUy5ZFP4g2O2MwcwWLEca",
    "vctk_parallel_wavegan.v1.long": "1tO4-mFrZ3aVYotgg7M519oobYkD4O_0-",
    "vctk_multi_band_melgan.v2": "10PRQpHMFPE7RjF-MHYqvupK9S0xwBlJ_",
    "libritts_parallel_wavegan.v1": "1zHQl8kUYEuZ_i1qEFU6g2MEu99k3sHmR",
    "libritts_parallel_wavegan.v1.long": "1b9zyBYGCCaJu0TIus5GXoMF8M3YEbqOw",
    "libritts_multi_band_melgan.v2": "1kIDSBjrQvAsRewHPiFwBZ3FDelTWMp64",
    "kss_parallel_wavegan.v1": "1mLtQAzZHLiGSWguKCGG0EZa4C_xUO5gX",
    "hui_acg_hokuspokus_parallel_wavegan.v1": "1irKf3okMLau56WNeOnhr2ZfSVESyQCGS",
    "ruslan_parallel_wavegan.v1": "1M3UM6HN6wrfSe5jdgXwBnAIl_lJzLzuI",
}


def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        list: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logging.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """Write dataset to hdf5.

    Args:
        hdf5_name (str): Hdf5 dataset filename.
        hdf5_path (str): Dataset path in hdf5.
        write_data (ndarray): Data to write.
        is_overwrite (bool): Whether to overwrite dataset.

    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                logging.warning(
                    "Dataset in hdf5 file already exists. " "recreate dataset in hdf5."
                )
                hdf5_file.__delitem__(hdf5_path)
            else:
                logging.error(
                    "Dataset in hdf5 file already exists. "
                    "if you want to overwrite, please set is_overwrite = True."
                )
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()


class HDF5ScpLoader(object):
    """Loader class for a fests.scp file of hdf5 file.

    Examples:
        key1 /some/path/a.h5:feats
        key2 /some/path/b.h5:feats
        key3 /some/path/c.h5:feats
        key4 /some/path/d.h5:feats
        ...
        >>> loader = HDF5ScpLoader("hdf5.scp")
        >>> array = loader["key1"]

        key1 /some/path/a.h5
        key2 /some/path/b.h5
        key3 /some/path/c.h5
        key4 /some/path/d.h5
        ...
        >>> loader = HDF5ScpLoader("hdf5.scp", "feats")
        >>> array = loader["key1"]

        key1 /some/path/a.h5:feats_1,feats_2
        key2 /some/path/b.h5:feats_1,feats_2
        key3 /some/path/c.h5:feats_1,feats_2
        key4 /some/path/d.h5:feats_1,feats_2
        ...
        >>> loader = HDF5ScpLoader("hdf5.scp")
        # feats_1 and feats_2 will be concatenated
        >>> array = loader["key1"]

    """

    def __init__(self, feats_scp, default_hdf5_path="feats"):
        """Initialize HDF5 scp loader.

        Args:
            feats_scp (str): Kaldi-style feats.scp file with hdf5 format.
            default_hdf5_path (str): Path in hdf5 file. If the scp contain the info, not used.

        """
        self.default_hdf5_path = default_hdf5_path
        with open(feats_scp) as f:
            lines = [line.replace("\n", "") for line in f.readlines()]
        self.data = {}
        for line in lines:
            key, value = line.split()
            self.data[key] = value

    def get_path(self, key):
        """Get hdf5 file path for a given key."""
        return self.data[key]

    def __getitem__(self, key):
        """Get ndarray for a given key."""
        p = self.data[key]
        if ":" in p:
            if len(p.split(",")) == 1:
                return read_hdf5(*p.split(":"))
            else:
                p1, p2 = p.split(":")
                feats = [read_hdf5(p1, p) for p in p2.split(",")]
                return np.concatenate(
                    [f if len(f.shape) != 1 else f.reshape(-1, 1) for f in feats], 1
                )
        else:
            return read_hdf5(p, self.default_hdf5_path)

    def __len__(self):
        """Return the length of the scp file."""
        return len(self.data)

    def __iter__(self):
        """Return the iterator of the scp file."""
        return iter(self.data)

    def keys(self):
        """Return the keys of the scp file."""
        return self.data.keys()

    def values(self):
        """Return the values of the scp file."""
        for key in self.keys():
            yield self[key]


class NpyScpLoader(object):
    """Loader class for a fests.scp file of npy file.

    Examples:
        key1 /some/path/a.npy
        key2 /some/path/b.npy
        key3 /some/path/c.npy
        key4 /some/path/d.npy
        ...
        >>> loader = NpyScpLoader("feats.scp")
        >>> array = loader["key1"]

    """

    def __init__(self, feats_scp):
        """Initialize npy scp loader.

        Args:
            feats_scp (str): Kaldi-style feats.scp file with npy format.

        """
        with open(feats_scp) as f:
            lines = [line.replace("\n", "") for line in f.readlines()]
        self.data = {}
        for line in lines:
            key, value = line.split()
            self.data[key] = value

    def get_path(self, key):
        """Get npy file path for a given key."""
        return self.data[key]

    def __getitem__(self, key):
        """Get ndarray for a given key."""
        return np.load(self.data[key])

    def __len__(self):
        """Return the length of the scp file."""
        return len(self.data)

    def __iter__(self):
        """Return the iterator of the scp file."""
        return iter(self.data)

    def keys(self):
        """Return the keys of the scp file."""
        return self.data.keys()

    def values(self):
        """Return the values of the scp file."""
        for key in self.keys():
            yield self[key]


def load_model(checkpoint, config=None):
    """Load trained model.

    Args:
        checkpoint (str): Checkpoint path.
        config (dict): Configuration dict.

    Return:
        torch.nn.Module: Model instance.

    """
    # load config if not provided
    if config is None:
        dirname = os.path.dirname(checkpoint)
        config = os.path.join(dirname, "config.yml")
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)

    # lazy load for circular error
    import parallel_wavegan.models

    # get model and load parameters
    model_class = getattr(
        parallel_wavegan.models,
        config.get("generator_type", "ParallelWaveGANGenerator"),
    )
    model = model_class(**config["generator_params"])
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu")["model"]["generator"]
    )

    # add pqmf if needed
    if config["generator_params"]["out_channels"] > 1:
        # lazy load for circular error
        from parallel_wavegan.layers import PQMF

        pqmf_params = {}
        if LooseVersion(config.get("version", "0.1.0")) <= LooseVersion("0.4.2"):
            # For compatibility, here we set default values in version <= 0.4.2
            pqmf_params.update(taps=62, cutoff_ratio=0.15, beta=9.0)
        model.pqmf = PQMF(
            subbands=config["generator_params"]["out_channels"],
            **config.get("pqmf_params", pqmf_params),
        )

    return model


def download_pretrained_model(tag, download_dir=None):
    """Download pretrained model form google drive.

    Args:
        tag (str): Pretrained model tag.
        download_dir (str): Directory to save downloaded files.

    Returns:
        str: Path of downloaded model checkpoint.

    """
    assert tag in PRETRAINED_MODEL_LIST, f"{tag} does not exists."
    id_ = PRETRAINED_MODEL_LIST[tag]
    if download_dir is None:
        download_dir = os.path.expanduser("~/.cache/parallel_wavegan")
    output_path = f"{download_dir}/{tag}.tar.gz"
    os.makedirs(f"{download_dir}", exist_ok=True)
    if not os.path.exists(output_path):
        # lazy load for compatibility
        import gdown

        gdown.download(
            f"https://drive.google.com/uc?id={id_}", output_path, quiet=False
        )
        with tarfile.open(output_path, "r:*") as tar:
            for member in tar.getmembers():
                if member.isreg():
                    member.name = os.path.basename(member.name)
                    tar.extract(member, f"{download_dir}/{tag}")
    checkpoint_path = find_files(f"{download_dir}/{tag}", "checkpoint*.pkl")

    return checkpoint_path[0]

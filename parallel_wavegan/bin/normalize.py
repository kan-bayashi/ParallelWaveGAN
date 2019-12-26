#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Normalize feature files and dump them."""

import argparse
import logging
import os

import numpy as np
import yaml

from joblib import delayed
from joblib import Parallel
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from parallel_wavegan.datasets import AudioMelDataset
from parallel_wavegan.utils import read_hdf5
from parallel_wavegan.utils import write_hdf5

# make sure each process use single thread
os.environ["OMP_NUM_THREADS"] = "1"


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Normalize dumped raw features (See detail in parallel_wavegan/bin/normalize.py).")
    parser.add_argument("--rootdir", type=str, required=True,
                        help="directory including feature files to be normalized.")
    parser.add_argument("--dumpdir", type=str, required=True,
                        help="directory to dump normalized feature files.")
    parser.add_argument("--stats", type=str, required=True,
                        help="statistics file.")
    parser.add_argument("--config", type=str, required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--n_jobs", type=int, default=16,
                        help="number of parallel jobs. (default=16)")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning('Skip DEBUG/INFO messages')

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # check directory existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir)

    # get dataset
    if config["format"] == "hdf5":
        audio_query, mel_query = "*.h5", "*.h5"
        audio_load_fn = lambda x: read_hdf5(x, "wave")  # NOQA
        mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA
    elif config["format"] == "npy":
        audio_query, mel_query = "*-wave.npy", "*-feats.npy"
        audio_load_fn = np.load
        mel_load_fn = np.load
    else:
        raise ValueError("support only hdf5 or npy format.")
    dataset = AudioMelDataset(
        root_dir=args.rootdir,
        audio_query=audio_query,
        mel_query=mel_query,
        audio_load_fn=audio_load_fn,
        mel_load_fn=mel_load_fn,
        return_filename=True,
    )
    logging.info(f"The number of files = {len(dataset)}.")

    # restore scaler
    scaler = StandardScaler()
    if config["format"] == "hdf5":
        scaler.mean_ = read_hdf5(args.stats, "mean")
        scaler.scale_ = read_hdf5(args.stats, "scale")
    elif config["format"] == "npy":
        scaler.mean_ = np.load(args.stats)[0]
        scaler.scale_ = np.load(args.stats)[1]
    else:
        raise ValueError("support only hdf5 or npy format.")

    def _process_single_file(data):
        # parse inputs
        audio_name, mel_name, audio, mel = data

        # normalize
        mel = scaler.transform(mel)

        # save
        if config["format"] == "hdf5":
            write_hdf5(os.path.join(args.dumpdir, f"{os.path.basename(audio_name)}"),
                       "wave", audio.astype(np.float32))
            write_hdf5(os.path.join(args.dumpdir, f"{os.path.basename(mel_name)}"),
                       "feats", mel.astype(np.float32))
        elif config["format"] == "npy":
            np.save(os.path.join(args.dumpdir, f"{os.path.basename(audio_name)}"),
                    audio.astype(np.float32), allow_pickle=False)
            np.save(os.path.join(args.dumpdir, f"{os.path.basename(mel_name)}"),
                    mel.astype(np.float32), allow_pickle=False)
        else:
            raise ValueError("support only hdf5 or npy format.")

    # process in parallel
    Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
        [delayed(_process_single_file)(data) for data in tqdm(dataset)])


if __name__ == "__main__":
    main()

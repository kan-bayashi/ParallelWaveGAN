#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Calculate statistics of feature files."""

import argparse
import logging
import os

import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from parallel_wavegan.datasets import MelDataset, MelSCPDataset
from parallel_wavegan.utils import read_hdf5, write_hdf5


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean and variance of dumped raw features "
            "(See detail in parallel_wavegan/bin/compute_statistics.py)."
        )
    )
    parser.add_argument(
        "--feats-scp",
        "--scp",
        default=None,
        type=str,
        help=(
            "kaldi-style feats.scp file. "
            "you need to specify either feats-scp or rootdir."
        ),
    )
    parser.add_argument(
        "--rootdir",
        type=str,
        required=True,
        help=(
            "directory including feature files. "
            "you need to specify either feats-scp or rootdir."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--dumpdir",
        default=None,
        type=str,
        required=True,
        help=(
            "directory to save statistics. if not provided, "
            "stats will be saved in the above root directory."
        ),
    )
    parser.add_argument(
        "--target-feats",
        type=str,
        default="feats",
        choices=["feats", "local"],
        help="target name to compute statistics.",
    )
    parser.add_argument(
        "--utt2spk",
        default=None,
        type=str,
        help=(
            "kaldi-style spk2utt file. if given, calculate statistics of each speaker."
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging.",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # check arguments
    if (args.feats_scp is not None and args.rootdir is not None) or (
        args.feats_scp is None and args.rootdir is None
    ):
        raise ValueError("Please specify either --rootdir or --feats-scp.")

    # check directory existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir)

    # get dataset
    if args.feats_scp is None:
        if config["format"] == "hdf5":
            mel_query = "*.h5"
            mel_load_fn = lambda x: read_hdf5(x, args.target_feats)  # NOQA
        elif config["format"] == "npy":
            mel_query = f"*-{args.target_feats}.npy"
            mel_load_fn = np.load
        else:
            raise ValueError("support only hdf5 or npy format.")
        dataset = MelDataset(
            args.rootdir,
            mel_query=mel_query,
            mel_load_fn=mel_load_fn,
            return_utt_id=False if args.utt2spk is None else True,
        )
    else:
        if args.target_feats != "feats":
            raise NotImplementedError("Not supported.")
        dataset = MelSCPDataset(
            args.feats_scp,
            return_utt_id=False if args.utt2spk is None else True,
        )
    logging.info(f"The number of files = {len(dataset)}.")

    if args.utt2spk is None:
        # calculate global statistics
        logging.info("Caluculate global statistics.")
        scaler = StandardScaler()
        for mel in tqdm(dataset):
            scaler.partial_fit(mel)

        if config["format"] == "hdf5":
            write_hdf5(
                os.path.join(args.dumpdir, "stats.h5"),
                "mean",
                scaler.mean_.astype(np.float32),
            )
            write_hdf5(
                os.path.join(args.dumpdir, "stats.h5"),
                "scale",
                scaler.scale_.astype(np.float32),
            )
        else:
            stats = np.stack([scaler.mean_, scaler.scale_], axis=0)
            np.save(
                os.path.join(args.dumpdir, "stats.npy"),
                stats.astype(np.float32),
                allow_pickle=False,
            )
    else:
        # calculate statistics of each speaker
        logging.info("Caluculate each speaker statistics.")
        with open(args.utt2spk) as f:
            lines = [line.replace("\n", "") for line in f.readlines()]
        utt2spk = {line.split()[0]: line.split()[1] for line in lines}
        spks = list(set(utt2spk.values()))
        spk2scaler = {spk: StandardScaler() for spk in spks}
        for utt_id, mel in tqdm(dataset):
            spk = utt2spk[utt_id]
            spk2scaler[spk].partial_fit(mel)

        for spk, scaler in spk2scaler.items():
            if config["format"] == "hdf5":
                write_hdf5(
                    os.path.join(args.dumpdir, "stats.h5"),
                    f"{spk}/mean",
                    scaler.mean_.astype(np.float32),
                )
                write_hdf5(
                    os.path.join(args.dumpdir, "stats.h5"),
                    f"{spk}/scale",
                    scaler.scale_.astype(np.float32),
                )
            else:
                stats = np.stack([scaler.mean_, scaler.scale_], axis=0)
                np.save(
                    os.path.join(args.dumpdir, f"stats-{spk}.npy"),
                    stats.astype(np.float32),
                    allow_pickle=False,
                )


if __name__ == "__main__":
    main()

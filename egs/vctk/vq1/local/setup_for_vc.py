#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Setup features for voice conversion."""

import argparse
import logging
import os

import numpy as np
import yaml
from tqdm import tqdm

from parallel_wavegan.datasets import AudioDataset
from parallel_wavegan.utils import read_hdf5, write_hdf5


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Normalize dumped raw features (See detail in"
            " parallel_wavegan/bin/normalize.py)."
        )
    )
    parser.add_argument(
        "--rootdir",
        default=None,
        type=str,
        help=(
            "directory including feature files to be normalized. "
            "you need to specify either feats-scp or rootdir."
        ),
    )
    parser.add_argument(
        "--feats-scp",
        default=None,
        type=str,
        help=(
            "kaldi-style feats.scp file. "
            "you need to specify either feats-scp or rootdir."
        ),
    )
    parser.add_argument(
        "--dumpdir", type=str, required=True, help="directory to dump feature files."
    )
    parser.add_argument(
        "--statdir",
        type=str,
        default=None,
        help="direcotry including statistics files.",
    )
    parser.add_argument(
        "--utt2spk", default=None, type=str, help="kaldi-style utt2spk file."
    )
    parser.add_argument(
        "--spk2idx", default=None, type=str, help="kaldi-style spk2idx file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
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

    # load utt2spk and spk2idx
    with open(args.utt2spk) as f:
        lines = [l_.replace("\n", "") for l_ in f.readlines()]
    utt2spk = {l_.split()[0]: l_.split()[1] for l_ in lines}
    with open(args.spk2idx) as f:
        lines = [l_.replace("\n", "") for l_ in f.readlines()]
    spk2idx = {l_.split()[0]: int(l_.split()[1]) for l_ in lines}
    spks = list(spk2idx.keys())

    # get dataset
    if args.rootdir is not None:
        local_query = None
        local_load_fn = None
        if config["format"] == "hdf5":
            audio_query = "*.h5"
            audio_load_fn = lambda x: read_hdf5(x, "wave")  # NOQA
            if config.get("use_local_condition", False):
                local_query = "*.h5"
                local_load_fn = lambda x: read_hdf5(x, "local")  # NOQA
        elif config["format"] == "npy":
            audio_query = "*-wave.npy"
            audio_load_fn = np.load
            if config.get("use_local_condition", False):
                local_query = "*-local.npy"
                local_load_fn = np.load
        else:
            raise ValueError("support only hdf5 or npy format.")
        dataset = AudioDataset(
            root_dir=args.rootdir,
            audio_query=audio_query,
            audio_load_fn=audio_load_fn,
            local_query=local_query,
            local_load_fn=local_load_fn,
            return_utt_id=True,
        )
    else:
        raise NotImplementedError("Not supported.")
    logging.info(f"The number of files = {len(dataset)}.")

    # load statistics
    if config.get("use_local_condition", False):
        spk2lf0avg = {}
        for spk in spks:
            # FIXME: it is better to remove this hard coding
            if config["format"] == "hdf5":
                lf0avg = read_hdf5(f"{args.statdir}/stats.h5", f"{spk}/mean")[0]
            elif config["format"] == "npy":
                lf0avg = np.load(f"{args.statdir}/stats-${spk}.h5")[0][0]
            else:
                raise ValueError("support only hdf5 or npy format.")
            spk2lf0avg[spk] = lf0avg

    # process each file
    for items in tqdm(dataset):
        if config.get("use_local_condition", False):
            utt_id, audio, l_ = items
        else:
            utt_id, audio = items
        source_spk = utt2spk[utt_id]

        # replace global and adjust f0 range
        for target_spk in spks:
            if target_spk == source_spk:
                continue

            # adjust f0 range
            if config.get("use_local_condition", False):
                l_ = l_.copy()
                target_lf0avg = spk2lf0avg[target_spk]
                source_lf0avg = spk2lf0avg[source_spk]
                l_[:, 0] = l_[:, 0] + target_lf0avg - source_lf0avg

            # get global
            g = np.array(spk2idx[target_spk])

            # save
            if config["format"] == "hdf5":
                write_hdf5(
                    os.path.join(args.dumpdir, f"{utt_id}_to_{target_spk}.h5"),
                    "wave",
                    audio.astype(np.float32),
                )
                write_hdf5(
                    os.path.join(args.dumpdir, f"{utt_id}_to_{target_spk}.h5"),
                    "global",
                    g.reshape(-1),
                )
                if config.get("use_local_condition", False):
                    write_hdf5(
                        os.path.join(args.dumpdir, f"{utt_id}_to_{target_spk}.h5"),
                        "local",
                        l_.astype(np.float32),
                    )
            elif config["format"] == "npy":
                np.save(
                    os.path.join(args.dumpdir, f"{utt_id}_to_{target_spk}-wave.npy"),
                    audio.astype(np.float32),
                    allow_pickle=False,
                )
                np.save(
                    os.path.join(args.dumpdir, f"{utt_id}_to_{target_spk}-global.npy"),
                    g.reshape(-1),
                    allow_pickle=False,
                )
                if config.get("use_local_condition", False):
                    np.save(
                        os.path.join(
                            args.dumpdir, f"{utt_id}_to_{target_spk}-local.npy"
                        ),
                        l_.astype(np.float32),
                        allow_pickle=False,
                    )
            else:
                raise ValueError("support only hdf5 or npy format.")


if __name__ == "__main__":
    main()

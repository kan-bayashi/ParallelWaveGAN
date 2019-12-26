#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode with trained Parallel WaveGAN Generator."""

import argparse
import logging
import os
import time

import kaldiio
import numpy as np
import soundfile as sf
import torch
import yaml

from tqdm import tqdm

from parallel_wavegan.datasets import MelDataset
from parallel_wavegan.models import ParallelWaveGANGenerator
from parallel_wavegan.utils import read_hdf5


def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description="Decode dumped features with trained Parallel WaveGAN Generator "
                    "(See detail in parallel_wavegan/bin/decode.py).")
    parser.add_argument("--scp", default=None, type=str,
                        help="kaldi-style feats.scp file. you need to specify either scp or dumpdir.")
    parser.add_argument("--dumpdir", default=None, type=str,
                        help="directory including feature files. you need to specify either scp or dumpdir.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save generated speech.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="checkpoint file to be loaded.")
    parser.add_argument("--config", default=None, type=str,
                        help="yaml format configuration file. if not explicitly provided, "
                             "it will be searched in the checkpoint directory. (default=None)")
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
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.checkpoint)
        args.config = os.path.join(dirname, "config.yml")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # check arguments
    if (args.scp is not None and args.dumpdir is not None) or \
            (args.scp is None and args.dumpdir is None):
        raise ValueError("Please specify either dumpdir or scp.")

    # get dataset
    if args.scp is None:
        if config["format"] == "hdf5":
            mel_query = "*.h5"
            mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA
        elif config["format"] == "npy":
            mel_query = "*-feats.npy"
            mel_load_fn = np.load
        else:
            raise ValueError("support only hdf5 or npy format.")
        dataset = MelDataset(
            args.dumpdir,
            mel_query=mel_query,
            mel_load_fn=mel_load_fn,
            return_filename=True)
        logging.info(f"The number of features to be decoded = {len(dataset)}.")
    else:
        dataset = kaldiio.ReadHelper(f"scp:{args.scp}")
        logging.info(f"The feature loaded from {args.scp}.")

    # setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = ParallelWaveGANGenerator(**config["generator_params"])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"]["generator"])
    model.remove_weight_norm()
    model = model.eval().to(device)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")

    # start generation
    pad_size = (config["generator_params"]["aux_context_window"],
                config["generator_params"]["aux_context_window"])
    total_rtf = 0.0
    with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
        for idx, (feat_path, c) in enumerate(pbar, 1):
            # generate each utterance
            z = torch.randn(1, 1, c.shape[0] * config["hop_size"]).to(device)
            c = np.pad(c, (pad_size, (0, 0)), "edge")
            c = torch.FloatTensor(c).unsqueeze(0).transpose(2, 1).to(device)
            start = time.time()
            y = model(z, c).view(-1).cpu().numpy()
            rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

            # save as PCM 16 bit wav file
            utt_id = os.path.splitext(os.path.basename(feat_path))[0]
            sf.write(os.path.join(config["outdir"], f"{utt_id}_gen.wav"),
                     y, config["sampling_rate"], "PCM_16")

    # report average RTF
    logging.info(f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f}).")


if __name__ == "__main__":
    main()

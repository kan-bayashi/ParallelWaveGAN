#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode with trained Parallel WaveGAN Generator."""

import argparse
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from tqdm import tqdm

import parallel_wavegan.models

from parallel_wavegan.datasets import MelDataset
from parallel_wavegan.datasets import MelSCPDataset
from parallel_wavegan.layers import PQMF
from parallel_wavegan.utils import read_hdf5


def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description="Decode dumped features with trained Parallel WaveGAN Generator "
                    "(See detail in parallel_wavegan/bin/decode.py).")
    parser.add_argument("--feats-scp", "--scp", default=None, type=str,
                        help="kaldi-style feats.scp file. "
                             "you need to specify either feats-scp or dumpdir.")
    parser.add_argument("--dumpdir", default=None, type=str,
                        help="directory including feature files. "
                             "you need to specify either feats-scp or dumpdir.")
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
    if (args.feats_scp is not None and args.dumpdir is not None) or \
            (args.feats_scp is None and args.dumpdir is None):
        raise ValueError("Please specify either --dumpdir or --feats-scp.")

    # get dataset
    if args.dumpdir is not None:
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
            return_utt_id=True,
        )
    else:
        dataset = MelSCPDataset(
            feats_scp=args.feats_scp,
            return_utt_id=True,
        )
    logging.info(f"The number of features to be decoded = {len(dataset)}.")

    # setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model_class = getattr(
        parallel_wavegan.models,
        config.get("generator_type", "ParallelWaveGANGenerator"))
    model = model_class(**config["generator_params"])
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu")["model"]["generator"])
    logging.info(f"Loaded model parameters from {args.checkpoint}.")
    model.remove_weight_norm()
    model = model.eval().to(device)
    use_noise_input = not isinstance(
        model, parallel_wavegan.models.MelGANGenerator)
    pad_fn = torch.nn.ReplicationPad1d(
        config["generator_params"].get("aux_context_window", 0))
    if config["generator_params"]["out_channels"] > 1:
        pqmf = PQMF(config["generator_params"]["out_channels"]).to(device)

    # start generation
    total_rtf = 0.0
    with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
        for idx, (utt_id, c) in enumerate(pbar, 1):
            # setup input
            x = ()
            if use_noise_input:
                z = torch.randn(1, 1, len(c) * config["hop_size"]).to(device)
                x += (z,)
            c = pad_fn(torch.tensor(c, dtype=torch.float).unsqueeze(0).transpose(2, 1)).to(device)
            x += (c,)

            # generate
            start = time.time()
            if config["generator_params"]["out_channels"] == 1:
                y = model(*x).view(-1).cpu().numpy()
            else:
                y = pqmf.synthesis(model(*x)).view(-1).cpu().numpy()
            rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

            # save as PCM 16 bit wav file
            sf.write(os.path.join(config["outdir"], f"{utt_id}_gen.wav"),
                     y, config["sampling_rate"], "PCM_16")

    # report average RTF
    logging.info(f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f}).")


if __name__ == "__main__":
    main()

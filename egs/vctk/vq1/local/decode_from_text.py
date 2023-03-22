#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode text with trained VQ-VAE Generator or discrete symbol vocoder."""

import argparse
import logging
import os
import time

import soundfile as sf
import torch
import yaml
from tqdm import tqdm

import parallel_wavegan.models
from parallel_wavegan.utils import load_model


def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description=(
            "Decode text with trained VQ-VAE decoder "
            "(See detail in parallel_wavegan/bin/decode.py)."
        )
    )
    parser.add_argument(
        "--text",
        required=True,
        type=str,
        help="kaldi-style text file.",
    )
    parser.add_argument(
        "--utt2spk",
        default=None,
        type=str,
        help="kaldi-style utt2spk file.",
    )
    parser.add_argument(
        "--spk2idx",
        default=None,
        type=str,
        help="kaldi-style spk2idx file.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save generated speech.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="checkpoint file to be loaded.",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help=(
            "yaml format configuration file. if not explicitly provided, "
            "it will be searched in the checkpoint directory. (default=None)"
        ),
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

    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = load_model(args.checkpoint, config)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")
    model.remove_weight_norm()
    model = model.eval().to(device)
    is_vqvae = isinstance(model, parallel_wavegan.models.VQVAE)

    # setup dataset
    with open(args.text) as f:
        lines = [l_.replace("\n", "") for l_ in f.readlines()]
    text = {l_.split()[0]: list(map(int, l_.split()[1:])) for l_ in lines}

    utt2spk = None
    if args.utt2spk is not None:
        assert args.spk2idx is not None
        with open(args.utt2spk) as f:
            lines = [l_.replace("\n", "") for l_ in f.readlines()]
        utt2spk = {l_.split()[0]: str(l_.split()[1]) for l_ in lines}
        with open(args.spk2idx) as f:
            lines = [l_.replace("\n", "") for l_ in f.readlines()]
        spk2idx = {l_.split()[0]: int(l_.split()[1]) for l_ in lines}

    # start generation
    total_rtf = 0.0
    with torch.no_grad(), tqdm(text.items(), desc="[decode]") as pbar:
        for idx, items in enumerate(pbar, 1):
            utt_id, indices = items
            z = torch.LongTensor(indices).view(1, -1).to(device)
            g = None
            if utt2spk is not None:
                spk_idx = spk2idx[utt2spk[utt_id]]
                g = torch.tensor(spk_idx).long().view(1).to(device)
            if is_vqvae:
                # VQVAE case
                start = time.time()
                y = model.decode(z, None, g).view(-1).cpu().numpy()
                rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
                pbar.set_postfix({"RTF": rtf})
                total_rtf += rtf
            else:
                # Discrete symbol vocoder case
                start = time.time()
                g = int(g.item()) if g is not None else None
                y = model.inference(z.view(-1, 1), g=g).view(-1).cpu().numpy()
                rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
                pbar.set_postfix({"RTF": rtf})
                total_rtf += rtf

            # save as PCM 16 bit wav file
            sf.write(
                os.path.join(config["outdir"], f"{utt_id}_gen.wav"),
                y,
                config["sampling_rate"],
                "PCM_16",
            )

    # report average RTF
    logging.info(
        f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f})."
    )


if __name__ == "__main__":
    main()

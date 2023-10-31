#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Perform preprocessing and raw feature extraction."""

import argparse
import logging
import os

import librosa
import numpy as np
import resampy
import soundfile as sf
import yaml
import torch
from tqdm import tqdm
from scipy.interpolate import interp1d
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from parallel_wavegan.datasets import AudioDataset, AudioSCPDataset
from parallel_wavegan.utils import write_hdf5

def f0_torchyin(
    audio,
    sampling_rate,
    hop_size=256,
    frame_length=None,
    pitch_min=40,
    pitch_max=10000,
    use_log_f0=True,
):
    """Compute F0 with Yin.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        hop_size (int): Hop size.
        pitch_min (int): Minimum pitch in pitch extraction.
        pitch_max (int): Maximum pitch in pitch extraction.

    Returns:
        ndarray: f0 feature (#frames, ).

    Note:
        Unvoiced frame has value = 0.

    """
    torch_wav = torch.from_numpy(audio).float()
    if frame_length is not None:
        pitch_min = sampling_rate / (frame_length / 2)

    import torchyin
    pitch = torchyin.estimate(
        torch_wav,
        sample_rate=sampling_rate,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        frame_stride=hop_size / sampling_rate,
    )
    f0 = pitch.cpu().numpy()
    # midi_number = np.zeros(shape=f0.shape, dtype=float)
    # midi_number[nonzeros_idxs] = 69 + 12 * np.log2(f0[nonzeros_idxs] / 440)
    # midi_number = midi_number.astype(int)
    if use_log_f0:
        nonzeros_idxs = np.where(f0 != 0)[0]
        f0[nonzeros_idxs] = np.log(f0[nonzeros_idxs])
    return f0


def f0_dio(
    audio,
    sampling_rate,
    hop_size=256,
    pitch_min=40,
    pitch_max=10000,
    use_log_f0=True,
):
    """Compute F0 with pyworld.dio

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        hop_size (int): Hop size.
        pitch_min (int): Minimum pitch in pitch extraction.
        pitch_max (int): Maximum pitch in pitch extraction.

    Returns:
        ndarray: f0 feature (#frames, ).

    Note:
        Unvoiced frame has value = 0.

    """
    if torch.is_tensor(audio):
        x = audio.cpu().numpy().astype(np.double)
    else:
        x = audio.astype(np.double)
    frame_period = 1000 * hop_size / sampling_rate
    import pyworld
    f0, timeaxis = pyworld.dio(
        x,
        sampling_rate,
        f0_floor=pitch_min,
        f0_ceil=pitch_max,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(x, f0, timeaxis, sampling_rate)
    if use_log_f0:
        nonzero_idxs = np.where(f0 != 0)[0]
        f0[nonzero_idxs] = np.log(f0[nonzero_idxs])
    return f0


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess audio and then extract features (See detail in"
            " parallel_wavegan/bin/preprocess.py)."
        )
    )
    parser.add_argument(
        "--wav-scp",
        "--scp",
        default=None,
        type=str,
        help="kaldi-style wav.scp file. you need to specify either scp or rootdir.",
    )
    parser.add_argument(
        "--segments",
        default=None,
        type=str,
        help=(
            "kaldi-style segments file. if use, you must to specify both scp and"
            " segments."
        ),
    )
    parser.add_argument(
        "--text",
        default=None,
        type=str,
        help="kaldi-style text format hubert embedding index.",
    )
    parser.add_argument(
        "--utt2spk",
        default=None,
        type=str,
        help=(
            "kaldi-style utt2spk file. If you want to add global conditionning with "
            "speaker id, you need to specify this argument."
        ),
    )
    parser.add_argument(
        "--spk2idx",
        default=None,
        type=str,
        help=(
            "kaldi-style spk2idx file. If you want to add global conditionning with "
            "speaker id, you need to specify this argument."
        ),
    )
    parser.add_argument(
        "--rootdir",
        default=None,
        type=str,
        help=(
            "directory including wav files. you need to specify either scp or rootdir."
        ),
    )
    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump feature files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--use-pretrain-feature",
        default=False,
        action="store_true",
        help="whether to use pretrain model to get feature.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--use-f0",
        default=False,
        action="store_true",
        help="whether to use f0 sequence.",
    )
    parser.add_argument(
        "--layer-num",
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
    if (args.wav_scp is not None and args.rootdir is not None) or (
        args.wav_scp is None and args.rootdir is None
    ):
        raise ValueError("Please specify either --rootdir or --wav-scp.")

    # get dataset
    if args.rootdir is not None:
        dataset = AudioDataset(
            args.rootdir,
            "*.wav",
            audio_load_fn=sf.read,
            return_utt_id=True,
        )
    else:
        dataset = AudioSCPDataset(
            args.wav_scp,
            segments=args.segments,
            return_utt_id=True,
            return_sampling_rate=True,
        )

    # get text
    if not os.path.isdir(args.text):
        with open(args.text) as f:
            lines = [line.strip() for line in f.readlines()]
        text = {
            line.split(maxsplit=1)[0]: line.split(maxsplit=1)[1].split() for line in lines
        }
    else: 
        text = {}
        for fname in os.listdir(args.text):
            fpath = os.path.join(args.text, fname)
            with open(fpath, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
            for line in lines:
                l0, l1 = line.split(maxsplit=1) # name, list
                l1 = l1.split() 
                if text.get(l0) is None:
                    text[l0] = []
                text[l0].append(line.split(maxsplit=1)[1].split())

    # load spk2utt file
    if args.utt2spk is not None:
        with open(args.utt2spk) as f:
            lines = [l.replace("\n", "") for l in f.readlines()]
        utt2spk = {l.split()[0]: l.split()[1] for l in lines}
        with open(args.spk2idx) as f:
            lines = [l.replace("\n", "") for l in f.readlines()]
        spk2idx = {l.split()[0]: int(l.split()[1]) for l in lines}

    # check directly existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir, exist_ok=True)

    # process each data
    for utt_id, (audio, fs) in tqdm(dataset):
        # check
        assert len(audio.shape) == 1, f"{utt_id} seems to be multi-channel signal."
        assert (
            np.abs(audio).max() <= 1.0
        ), f"{utt_id} seems to be different from 16 bit PCM."

        # downsample
        if fs != config["sampling_rate"]:
            audio = resampy.resample(audio, fs, config["sampling_rate"], axis=0)

        # trim silence
        if config["trim_silence"]:
            audio, _ = librosa.effects.trim(
                audio,
                top_db=config["trim_threshold_in_db"],
                frame_length=config["trim_frame_size"],
                hop_length=config["trim_hop_size"],
            )
        
        if args.use_pretrain_feature:
            pretrained_model = "facebook/hubert-base-ls960"
            model = HubertModel.from_pretrained(pretrained_model)
            processor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model, trust_remote_code=True) 
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True)
            features = outputs.hidden_states
            mel = features[args.layer_num].squeeze(0).detach().numpy()
        else:
            # use hubert index instead of mel
            mel = np.array(text[utt_id]).astype(np.int64)
            if mel.ndim > 1: 
                mel = mel.transpose(1, 0)
            else:
                mel = mel.reshape(-1, 1)
        
        if args.spk2idx is not None:
            spk = utt2spk[utt_id]
            if spk in spk2idx:
                idx = spk2idx[spk]
            else:
                logging.warn(f"{spk} is unknown speaker.")
                max_idx = max(spk2idx.values()) + 1
                idx = max_idx

            # concatenate with mel
            idx = np.repeat(np.array(idx).reshape(1, 1), len(mel), axis=0)
            mel = np.concatenate([mel, idx], axis=1)

        # make sure the audio length and feature length are matched
        logging.info(f"Mod: {len(audio) - len(mel) * config['hop_size']}")
        if len(mel) * config["hop_size"] <= len(audio):
            logging.warning(
                "len(mel) * config['hop_size'] <= len(audio), may be errors"
            )
        mel = mel[: len(audio) // config["hop_size"]]
        audio = audio[: len(mel) * config["hop_size"]]
        assert len(mel) * config["hop_size"] == len(audio)
        
        # use f0
        if args.use_f0:
            # f0 = f0_torchyin(
            #     audio,
            #     sampling_rate=config["sampling_rate"],
            #     hop_size=config["hop_size"],
            #     frame_length=config["win_length"],
            # ).reshape(-1, 1)
            f0 = f0_dio(
                audio,
                sampling_rate=config["sampling_rate"],
                hop_size=config["hop_size"],
            ) # (#frames,)
            if len(f0) > len(mel):
                f0 = f0[: len(mel)]
            else:
                f0 = np.pad(f0, (0, len(mel) - len(f0)), mode="edge")
            # logging.info(f'f0: {f0}')

        # apply global gain
        if config["global_gain_scale"] > 0.0:
            audio *= config["global_gain_scale"]
        if np.abs(audio).max() >= 1.0:
            logging.warn(
                f"{utt_id} causes clipping. "
                "it is better to re-consider global gain scale."
            )
            continue

        # save
        if config["format"] == "hdf5":
            # logging.info(f'mel: {mel.shape} f0: {f0.shape}')
            write_hdf5(
                os.path.join(args.dumpdir, f"{utt_id}.h5"),
                "wave",
                audio.astype(np.float32),
            )
            write_hdf5(
                os.path.join(args.dumpdir, f"{utt_id}.h5"),
                "feats",
                mel.astype(np.float32),
            )
            if args.use_f0:
                write_hdf5(
                    os.path.join(args.dumpdir, f"{utt_id}.h5"),
                    "f0",
                    f0.astype(np.float32),
                )
                
        elif config["format"] == "npy":
            np.save(
                os.path.join(args.dumpdir, f"{utt_id}-wave.npy"),
                audio.astype(np.float32),
                allow_pickle=False,
            )
            np.save(
                os.path.join(args.dumpdir, f"{utt_id}-feats.npy"),
                mel.astype(np.float32),
                allow_pickle=False,
            )
            if args.use_f0:
                np.save(
                    os.path.join(args.dumpdir, f"{utt_id}-f0.npy"),
                    f0.astype(np.float32),
                    allow_pickle=False,
                )
                
        else:
            raise ValueError("support only hdf5 or npy format.")


if __name__ == "__main__":
    main()

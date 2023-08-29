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
from tqdm import tqdm
from scipy.interpolate import interp1d


from parallel_wavegan.datasets import AudioDataset, AudioSCPDataset
from parallel_wavegan.utils import write_hdf5

def logf0_and_vuv_pyreaper(audio, fs, hop_size=64, f0min=40.0, f0max=500.0):
    """Extract continuous log f0 and uv sequences.

    Args:
        audio (ndarray): Audio sequence in float (-1, 1).
        fs (ndarray): Sampling rate.
        hop_size (int): Hop size in point.
        f0min (float): Minimum f0 value.
        f0max (float): Maximum f0 value.

    Returns:
        ndarray: Continuous log f0 sequence (#frames, 1).
        ndarray: Voiced (=1) / unvoiced (=0) sequence (#frames, 1).

    """
    # delayed import
    import pyreaper

    # convert to 16 bit interger and extract f0
    logging.info(f'old audio2: {len(audio)}')
    audio = np.array([round(x * np.iinfo(np.int16).max) for x in audio], dtype=np.int16)
    logging.info(f'new audio2: {len(audio)}')
    _, _, f0_times, f0, _ = pyreaper.reaper(audio, fs, frame_period=hop_size / fs)
    logging.info(f'new f0: {max(f0)}')

    # get vuv
    vuv = np.float32(f0 != -1)

    if vuv.sum() == 0:
        logging.warn("All of the frames are unvoiced.")
        return

    # get start and end of f0
    start_f0 = f0[f0 != -1][0]
    end_f0 = f0[f0 != -1][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    voiced_frame_idxs = np.where(f0 != -1)[0]

    # perform linear interpolation
    f = interp1d(f0_times[voiced_frame_idxs], f0[voiced_frame_idxs])
    f0 = f(f0_times)

    # convert to log domain
    lf0 = np.log(f0)

    return lf0.reshape(-1, 1), vuv.reshape(-1, 1)


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
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--extract-f0",
        default=False,
        action="store_true",
        help="whether to extract f0 sequence.",
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
    with open(args.text) as f:
        lines = [line.strip() for line in f.readlines()]
    text = {
        line.split(maxsplit=1)[0]: line.split(maxsplit=1)[1].split() for line in lines
    }

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

        # use hubert index instead of mel
        mel = np.array(text[utt_id]).astype(np.int64).reshape(-1, 1)
        
        logging.info(f'mel: {mel.shape}')
        logging.info(f'audio: {audio.shape}')

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
        logging.info(f"old audio: {len(audio)}")
        logging.info(f"old mel: {len(mel)}")
        if len(mel) * config["hop_size"] <= len(audio):
            logging.warning(
                "len(mel) * config['hop_size'] <= len(audio), may be errors"
            )
        mel = mel[: len(audio) // config["hop_size"]]
        audio = audio[: len(mel) * config["hop_size"]]
        logging.info(f"new audio: {len(audio)}")
        logging.info(f"new mel: {len(mel)}")
        assert len(mel) * config["hop_size"] == len(audio)

        logging.info(args.extract_f0)
        if args.extract_f0:                
            l_ = logf0_and_vuv_pyreaper(audio, config["sampling_rate"], config["hop_size"])
            if l_ is None:
                continue
            l_ = np.concatenate(l_, axis=-1)
            logging.info(f'len(f0)={l_.shape}')
            if len(audio) > len(l_) * config["hop_size"]:
                audio = audio[: len(l_) * config["hop_size"]]
            if len(audio) < len(l_) * config["hop_size"]:
                audio = np.pad(
                    audio, (0, len(l_) * config["hop_size"] - len(audio)), mode="edge"
                )

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
            if args.extract_f0:
                write_hdf5(
                    os.path.join(args.dumpdir, f"{utt_id}.h5"),
                    "local",
                    l_.astype(np.float32),
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
            if args.extract_f0:
                np.save(
                    os.path.join(args.dumpdir, f"{utt_id}-local.npy"),
                    l_.astype(np.float32),
                    allow_pickle=False,
                )
        else:
            raise ValueError("support only hdf5 or npy format.")


if __name__ == "__main__":
    main()

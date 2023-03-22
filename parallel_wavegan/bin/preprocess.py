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
import soundfile as sf
import torch
import yaml
from scipy.interpolate import interp1d
from tqdm import tqdm

from parallel_wavegan.datasets import AudioDataset, AudioSCPDataset
from parallel_wavegan.utils import write_hdf5


def logmelfilterbank(
    audio,
    sampling_rate,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window="hann",
    num_mels=80,
    fmin=None,
    fmax=None,
    eps=1e-10,
    log_base=10.0,
):
    """Compute log-Mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.
        log_base (float): Log base. If set to None, use np.log.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="reflect",
    )
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(
        sr=sampling_rate,
        n_fft=fft_size,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax,
    )

    mel = np.maximum(eps, np.dot(spc, mel_basis.T))

    if log_base is None:
        return np.log(mel)
    elif log_base == 10.0:
        return np.log10(mel)
    elif log_base == 2.0:
        return np.log2(mel)
    else:
        raise ValueError(f"{log_base} is not supported.")


def f0_torchyin(
    audio,
    sampling_rate,
    hop_size=256,
    frame_length=None,
    pitch_min=40,
    pitch_max=10000,
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

    nonzeros_idxs = np.where(f0 != 0)[0]
    f0[nonzeros_idxs] = np.log(f0[nonzeros_idxs])
    return f0


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
    audio = np.array([round(x * np.iinfo(np.int16).max) for x in audio], dtype=np.int16)
    _, _, f0_times, f0, _ = pyreaper.reaper(audio, fs, frame_period=hop_size / fs)

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
        "--skip-mel-ext",
        default=False,
        action="store_true",
        help="whether to skip the extraction of mel features.",
    )
    parser.add_argument(
        "--extract-f0",
        default=False,
        action="store_true",
        help="whether to extract f0 sequence.",
    )
    parser.add_argument(
        "--allow-different-sampling-rate",
        default=False,
        action="store_true",
        help="whether to allow different sampling rate in config.",
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

    # check model architecture
    generator_type = config.get("generator_type", "ParallelWaveGANGenerator")
    use_f0_and_excitation = generator_type == "UHiFiGANGenerator"

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

    # check directly existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir, exist_ok=True)

    if "sampling_rate_for_feats" not in config:
        sampling_rate = config["sampling_rate"]
    else:
        sampling_rate = config["sampling_rate_for_feats"]

    if use_f0_and_excitation:
        from parallel_wavegan.layers import SineGen

        ExcitationExtractor = SineGen(samp_rate=sampling_rate)

    # load spk2utt file
    if args.utt2spk is not None:
        with open(args.utt2spk) as f:
            lines = [line.replace("\n", "") for line in f.readlines()]
        utt2spk = {line.split()[0]: line.split()[1] for line in lines}
        with open(args.spk2idx) as f:
            lines = [line.replace("\n", "") for line in f.readlines()]
        spk2idx = {line.split()[0]: int(line.split()[1]) for line in lines}

    # process each data
    for utt_id, (audio, fs) in tqdm(dataset):
        # check
        assert len(audio.shape) == 1, f"{utt_id} seems to be multi-channel signal."
        assert (
            np.abs(audio).max() <= 1.0
        ), f"{utt_id} seems to be different from 16 bit PCM."
        assert (
            fs == config["sampling_rate"]
        ), f"{utt_id} seems to have a different sampling rate."

        # trim silence
        if config["trim_silence"]:
            audio, _ = librosa.effects.trim(
                audio,
                top_db=config["trim_threshold_in_db"],
                frame_length=config["trim_frame_size"],
                hop_length=config["trim_hop_size"],
            )

        if not args.skip_mel_ext:
            if "sampling_rate_for_feats" not in config:
                x = audio
                sampling_rate = config["sampling_rate"]
                hop_size = config["hop_size"]
            else:
                # NOTE(kan-bayashi): this procedure enables to train the model with different
                #   sampling rate for feature and audio, e.g., training with mel extracted
                #   using 16 kHz audio and 24 kHz audio as a target waveform
                x = librosa.resample(
                    audio, orig_sr=fs, target_sr=config["sampling_rate_for_feats"]
                )
                sampling_rate = config["sampling_rate_for_feats"]
                assert (
                    config["hop_size"] * config["sampling_rate_for_feats"] % fs == 0
                ), (
                    "hop_size must be int value. please check sampling_rate_for_feats"
                    " is correct."
                )
                hop_size = config["hop_size"] * config["sampling_rate_for_feats"] // fs

            # extract feature
            mel = logmelfilterbank(
                x,
                sampling_rate=sampling_rate,
                hop_size=hop_size,
                fft_size=config["fft_size"],
                win_length=config["win_length"],
                window=config["window"],
                num_mels=config["num_mels"],
                fmin=config["fmin"],
                fmax=config["fmax"],
            )

            # make sure the audio length and feature length are matched
            audio = np.pad(audio, (0, config["fft_size"]), mode="edge")
            audio = audio[: len(mel) * config["hop_size"]]
            assert len(mel) * config["hop_size"] == len(audio)

        # extract f0 sequence
        if args.extract_f0:
            l_ = logf0_and_vuv_pyreaper(audio, fs, config["hop_size"])
            if l_ is None:
                continue
            l_ = np.concatenate(l_, axis=-1)
            if len(audio) > len(l_) * config["hop_size"]:
                audio = audio[: len(l_) * config["hop_size"]]
            if len(audio) < len(l_) * config["hop_size"]:
                audio = np.pad(
                    audio, (0, len(l_) * config["hop_size"] - len(audio)), mode="edge"
                )

        if use_f0_and_excitation:
            f0 = f0_torchyin(
                audio,
                sampling_rate=sampling_rate,
                hop_size=hop_size,
                frame_length=config["win_length"],
            ).reshape(-1, 1)
            if len(f0) > len(mel):
                f0 = f0[: len(mel)]
            else:
                f0 = np.pad(f0, (0, len(mel) - len(f0)), mode="edge")
            extended_f0 = (
                torch.from_numpy(f0)
                .reshape(1, 1, -1)
                .repeat(1, config["hop_size"], 1)
                .reshape(1, -1, 1)
            )
            sine_waves, _, _ = ExcitationExtractor(extended_f0)
            excitation = sine_waves.squeeze(0).squeeze(-1).cpu().numpy()
            excitation = excitation[: len(mel) * config["hop_size"]]
            excitation = excitation.reshape(-1, config["hop_size"])
            f0 = np.squeeze(f0)  # (#frames,)
            excitation = np.squeeze(excitation)  # (#frames, hop_size)

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
            if not args.skip_mel_ext:
                write_hdf5(
                    os.path.join(args.dumpdir, f"{utt_id}.h5"),
                    "feats",
                    mel.astype(np.float32),
                )
            if use_f0_and_excitation:
                write_hdf5(
                    os.path.join(args.dumpdir, f"{utt_id}.h5"),
                    "f0",
                    f0.astype(np.float32),
                )
                write_hdf5(
                    os.path.join(args.dumpdir, f"{utt_id}.h5"),
                    "excitation",
                    excitation.astype(np.float32),
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
            if not args.skip_mel_ext:
                np.save(
                    os.path.join(args.dumpdir, f"{utt_id}-feats.npy"),
                    mel.astype(np.float32),
                    allow_pickle=False,
                )
            if use_f0_and_excitation:
                np.save(
                    os.path.join(args.dumpdir, f"{utt_id}-f0.npy"),
                    f0.astype(np.float32),
                    allow_pickle=False,
                )
                np.save(
                    os.path.join(args.dumpdir, f"{utt_id}-excitation.npy"),
                    excitation.astype(np.float32),
                )
            if args.extract_f0:
                np.save(
                    os.path.join(args.dumpdir, f"{utt_id}-local.npy"),
                    l_.astype(np.float32),
                    allow_pickle=False,
                )
        else:
            raise ValueError("support only hdf5 or npy format.")

        # save global embedding
        if config.get("use_global_condition", False):
            spk = utt2spk[utt_id]
            idx = spk2idx[spk]
            if config["format"] == "hdf5":
                write_hdf5(
                    os.path.join(args.dumpdir, f"{utt_id}.h5"), "global", int(idx)
                )
            elif config["format"] == "npy":
                np.save(
                    os.path.join(args.dumpdir, f"{utt_id}-global.npy"),
                    int(idx),
                    allow_pickle=False,
                )


if __name__ == "__main__":
    main()

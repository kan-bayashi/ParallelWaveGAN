import argparse
import os

import librosa
import numpy as np


def process_utterance(wavscp, utt2spk, audio_dir, wav_dumpdir, segment, tgt_sr=24000):
    uid, _, phns, notes, syb_dur, phn_dur, keep = segment.strip().split("|")

    utt2spk.write("{} {}\n".format(uid, "opencpop"))

    # apply bit convert, there is a known issue in direct convert in format wavscp
    cmd = (
        f"sox {os.path.join(audio_dir, uid)}.wav -c 1 -t wavpcm -b 16 -r"
        f" {tgt_sr} {os.path.join(wav_dumpdir, uid)}_bits16.wav"
    )
    os.system(cmd)

    wavscp.write("{} {}_bits16.wav\n".format(uid, os.path.join(wav_dumpdir, uid)))


def process_subset(args, set_name):
    if not os.path.exists(args.tgt_dir):
        os.makedirs(args.tgt_dir)
    if not os.path.exists(os.path.join(args.tgt_dir, set_name)):
        os.makedirs(os.path.join(args.tgt_dir, set_name))
    wavscp = open(
        os.path.join(args.tgt_dir, set_name, "wav.scp"), "w", encoding="utf-8"
    )
    utt2spk = open(
        os.path.join(args.tgt_dir, set_name, "utt2spk"), "w", encoding="utf-8"
    )

    with open(
        os.path.join(args.src_data, "segments", set_name + ".txt"),
        "r",
        encoding="utf-8",
    ) as f:
        segments = f.read().strip().split("\n")
        for segment in segments:
            process_utterance(
                wavscp,
                utt2spk,
                os.path.join(args.src_data, "segments", "wavs"),
                args.wav_dumpdir,
                segment,
                tgt_sr=args.sr,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Data for Opencpop Database")
    parser.add_argument("src_data", type=str, help="source data directory")
    parser.add_argument("--tgt_dir", type=str, default="data")
    parser.add_argument(
        "--wav_dumpdir", type=str, help="wav dump directory (rebit)", default="wav_dump"
    )
    parser.add_argument("--sr", type=int, help="sampling rate (Hz)")
    args = parser.parse_args()

    for name in ["train", "test"]:
        process_subset(args, name)

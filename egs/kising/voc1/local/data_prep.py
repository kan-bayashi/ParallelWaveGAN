import argparse
import os

import librosa
import numpy as np

UTT_PREFIX = "kising"
DEV_LIST = ["435_all.wav"]
TEST_LIST = ["434_all.wav"]


def pack_zero(string, size=20):
    if len(string) < size:
        string = "0" * (size - len(string)) + string
    return string


def train_check(song):
    return (song not in DEV_LIST) and (song not in TEST_LIST)


def dev_check(song):
    return song in DEV_LIST


def test_check(song):
    return song in TEST_LIST


def process_text_info(text):
    info = open(text, "r", encoding="utf-8")
    label_info = []
    text_info = []
    for line in info.readlines():
        line = line.strip().split()
        label_info.append(
            "{} {} {}".format(float(line[0]), float(line[1]), line[2].strip())
        )
        text_info.append(line[2].strip())
    return " ".join(label_info), " ".join(text_info)


def process_subset(args, set_name, check_func):
    if not os.path.exists(os.path.join(args.tgt_dir, set_name)):
        os.makedirs(os.path.join(args.tgt_dir, set_name))

    wavscp = open(
        os.path.join(args.tgt_dir, set_name, "wav.scp"), "w", encoding="utf-8"
    )
    label = open(os.path.join(args.tgt_dir, set_name, "label"), "w", encoding="utf-8")
    utt2spk = open(
        os.path.join(args.tgt_dir, set_name, "utt2spk"), "w", encoding="utf-8"
    )

    src_wavdir = os.path.join(args.src_data, "segmented-wav", "clean")
    src_labeldir = os.path.join(args.src_data, "segmented-label")
    # Note: not using midi, cause the midi is not accurate

    for song in os.listdir(src_wavdir):
        if not check_func(song):
            continue

        utt = song.split("_")[0]
        utt_id = "{}_{}".format(UTT_PREFIX, pack_zero(utt))

        cmd = (
            f"sox {os.path.join(src_wavdir, song)} -c 1 -t wavpcm -b 16 -r"
            f" {args.sr} {os.path.join(args.wav_dumpdir, utt_id)}_bits16.wav"
        )
        os.system(cmd)

        wavscp.write(
            "{} {}\n".format(
                utt_id, os.path.join(args.wav_dumpdir, utt_id) + "_bits16.wav"
            )
        )

        utt2spk.write("{} {}\n".format(utt_id, UTT_PREFIX))
        label_info, text_info = process_text_info(
            os.path.join(src_labeldir, "0{}_align_all.txt".format(utt))
        )
        label.write("{} {}\n".format(utt_id, label_info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Data for Opencpop Database")
    parser.add_argument("src_data", type=str, help="source data directory")
    parser.add_argument("--tgt_dir", type=str, default="data")
    parser.add_argument(
        "--wav_dumpdir", type=str, help="wav dump directoyr (rebit)", default="wav_dump"
    )
    parser.add_argument("--sr", type=int, help="sampling rate (Hz)")
    args = parser.parse_args()

    if not os.path.exists(args.wav_dumpdir):
        os.makedirs(args.wav_dumpdir)

    process_subset(args, "tr_no_dev", train_check)
    process_subset(args, "dev", dev_check)
    process_subset(args, "eval", test_check)

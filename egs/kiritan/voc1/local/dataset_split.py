#!/usr/bin/env python3
import argparse
import os
import random
import shutil
import sys
from shutil import copyfile

UTT_PREFIX = "kiritan"
DEV_LIST = ["13", "14", "26", "28", "39"]
TEST_LIST = ["01", "16", "17", "27", "44"]


def train_check(song):
    return (song not in DEV_LIST) and (song not in TEST_LIST)


def dev_check(song):
    return song in DEV_LIST


def test_check(song):
    return song in TEST_LIST


def pack_zero(string, size=4):
    if len(string) < size:
        string = "0" * (size - len(string)) + string
    return string


def makedir(data_url):
    if os.path.exists(data_url):
        shutil.rmtree(data_url)

    os.makedirs(data_url)


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


def process_subset(src_data, subset, check_func, fs):
    subfolder = os.listdir(src_data)
    makedir(subset)
    wavscp = open(os.path.join(subset, "wav.scp"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(subset, "utt2spk"), "w", encoding="utf-8")
    label_scp = open(os.path.join(subset, "label"), "w", encoding="utf-8")
    fixed_data = os.path.join(subset, "fix_byte")
    makedir(fixed_data)

    for song_index in range(1, 51):
        song_index = pack_zero(str(song_index), size=2)
        if not check_func(song_index):
            continue
        utt_id = "{}_{}".format(UTT_PREFIX, pack_zero(song_index))

        cmd = (
            f"sox {os.path.join(src_data, 'wav', song_index)}.wav -c 1 -t wavpcm -b 16"
            f" -r {fs} {os.path.join(fixed_data, song_index)}_bits16.wav"
        )
        print(f"cmd: {cmd}")
        os.system(cmd)

        wavscp.write(
            "{} {}\n".format(
                utt_id, os.path.join(fixed_data, "{}_bits16.wav".format(song_index))
            )
        )
        utt2spk.write("{} {}\n".format(utt_id, UTT_PREFIX))
        label_info, text_info = process_text_info(
            os.path.join(src_data, "mono_label", "{}.lab".format(song_index))
        )
        label_scp.write("{} {}\n".format(utt_id, label_info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Data for Natsume Database")
    parser.add_argument("src_data", type=str, help="source data directory")
    parser.add_argument("train", type=str, help="train set")
    parser.add_argument("dev", type=str, help="development set")
    parser.add_argument("test", type=str, help="test set")
    parser.add_argument("--fs", type=int, help="frame rate (Hz)")
    args = parser.parse_args()

    process_subset(args.src_data, args.train, train_check, args.fs)
    process_subset(args.src_data, args.dev, dev_check, args.fs)
    process_subset(args.src_data, args.test, test_check, args.fs)

import argparse
import json
import os
import shutil

import random
import librosa
import miditoolkit
import numpy as np

from espnet2.fileio.score_scp import SingingScoreWriter

"""Generate segments according to structured annotation."""
"""Transfer music score into 'score' format."""


def makedir(data_url):
    if os.path.exists(data_url):
        shutil.rmtree(data_url)
    os.makedirs(data_url)


def process_utterance(
    wavscp,
    audio_dir,
    wav_dumpdir,
    segment,
    tgt_sr=24000,
):
    name = segment["item_name"]

    # load tempo from midi
    uid = name.encode("unicode_escape").decode().replace("\\u", "#U")
    song_name = uid[: uid.rindex("#")]
    uid = uid.replace(" ", "+")

    # apply bit convert, there is a known issue in direct convert in format wavscp
    cmd = "sox {}.wav -c 1 -t wavpcm -b 16 -r {} {}/m4singer_{}.wav".format(
        os.path.join(
            audio_dir,
            song_name.replace(" ", "\\ "),
            uid.replace("+", "\\ ").split("#")[-1],
        ),
        tgt_sr,
        wav_dumpdir,
        uid,
    )
    os.system(cmd)
    wavscp.write("m4singer_{} {}/m4singer_{}.wav\n".format(uid, wav_dumpdir, uid))


def process_subset(args, set_name, data):
    makedir(os.path.join(args.tgt_dir, set_name))
    wavscp = open(
        os.path.join(args.tgt_dir, set_name, "wav.scp"), "w", encoding="utf-8"
    )

    for segment in data:
        process_utterance(
            wavscp,
            os.path.join(args.src_data),
            args.wav_dumpdir,
            segment,
            tgt_sr=args.sr,
        )


def split_subset(args, meta):
    overall_data = {}
    item_names = []
    for i in range(len(meta)):
        item_name = meta[i]["item_name"]
        overall_data[item_name] = meta[i]
        item_names.append(item_name)

    item_names = sorted(item_names)

    # Refer to https://github.com/M4Singer/M4Singer
    random.seed(1234)
    random.shuffle(item_names)
    valid_num, test_num = 100, 100

    test_names = item_names[:test_num]
    # NOTE(jiatong): the valid set is different from M4Singer
    # As they include test set in the validation set
    # but we do not.
    valid_names = item_names[test_num : valid_num + test_num]
    train_names = item_names[valid_num + test_num :]

    data = {"tr_no_dev": [], "dev": [], "eval": []}

    for key in overall_data.keys():
        if key in test_names:
            data["eval"].append(overall_data[key])
        elif key in valid_names:
            data["dev"].append(overall_data[key])
        else:
            data["tr_no_dev"].append(overall_data[key])
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Data for M4Singer Database")
    parser.add_argument("src_data", type=str, help="source data directory")
    parser.add_argument("--tgt_dir", type=str, default="data")
    parser.add_argument(
        "--wav_dumpdir", type=str, help="wav dump directoyr (rebit)", default="wav_dump"
    )
    parser.add_argument("--sr", type=int, help="sampling rate (Hz)")
    args = parser.parse_args()

    with open(os.path.join(args.src_data, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    data = split_subset(args, meta)

    for name in ["tr_no_dev", "dev", "eval"]:
        process_subset(args, name, data[name])

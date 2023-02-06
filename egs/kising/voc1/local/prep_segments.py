#!/usr/bin/env python3
import argparse
import math
import os
import sys


class LabelInfo(object):
    def __init__(self, start, end, label_id):
        self.label_id = label_id
        self.start = start
        self.end = end


class SegInfo(object):
    def __init__(self):
        self.segs = []
        self.start = -1
        self.end = -1

    def add(self, start, end, label):
        start = float(start)
        end = float(end)
        if self.start < 0 or self.start > start:
            self.start = start
        if self.end < end:
            self.end = end
        self.segs.append((start, end, label))


def pack_zero(file_id, number, length=4):
    number = str(number)
    return file_id + "_" + "0" * (length - len(number)) + number


def get_parser():
    parser = argparse.ArgumentParser(
        description="Prepare segments from HTS-style alignment files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp", type=str, help="scp folder")
    return parser


def make_segment(file_id, labels):
    segments = []
    segment = SegInfo()
    running_end = 0
    for label in labels:
        if float(running_end) + 1 < float(label.start):
            if len(segment.segs) > 0:
                segments.append(segment.segs)
                segment = SegInfo()
        segment.add(label.start, label.end, label.label_id)
        running_end = label.end

    if len(segment.segs) > 0:
        segments.append(segment.segs)

    segments_w_id = {}
    id = 0
    for seg in segments:
        if len(seg) == 0:
            continue
        segments_w_id[pack_zero(file_id, id)] = seg
        id += 1
    return segments_w_id


if __name__ == "__main__":
    # os.chdir(sys.path[0]+'/..')
    args = get_parser().parse_args()
    segments = []

    wavscp = open(os.path.join(args.scp, "wav.scp"), "r", encoding="utf-8")
    label = open(os.path.join(args.scp, "label"), "r", encoding="utf-8")

    update_segments = open(
        os.path.join(args.scp, "segments.tmp"), "w", encoding="utf-8"
    )

    for wav_line in wavscp:
        label_line = label.readline()
        if not label_line:
            raise ValueError("not match label and wav.scp in {}".format(args.scp))

        wavline = wav_line.strip().split(" ")
        recording_id = wavline[0]
        path = " ".join(wavline[1:])
        phn_info = label_line.strip().split()[1:]
        temp_info = []
        for i in range(len(phn_info) // 3):
            temp_info.append(
                LabelInfo(phn_info[i * 3], phn_info[i * 3 + 1], phn_info[i * 3 + 2])
            )
        segments.append(make_segment(recording_id, temp_info))

    for file in segments:
        for key, val in file.items():
            segment_begin = "{:.3f}".format(val[0][0])
            segment_end = "{:.3f}".format(val[-1][1])

            update_segments.write(
                "{} {} {} {}\n".format(
                    key, "_".join(key.split("_")[:-1]), segment_begin, segment_end
                )
            )

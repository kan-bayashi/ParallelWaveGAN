#!/bin/bash

# Make subset files located in data direcoty.

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;


if [ $# -ne 3 ]; then
    echo "Usage: $0 <src_dir> <num_split> <dst_dir>"
    echo "e.g.: $0 data/train_nodev 16 data/train_nodev/split16"
    exit 1
fi

set -eu

src_dir=$1
num_split=$2
dst_dir=$3

src_scp=${src_dir}/wav.scp
num_src_utts=$(wc -l < "${src_scp}")
has_utt2spk=false
has_segments=false

if [ -e "${src_dir}/segments" ]; then
    has_segments=true
    src_segments=${src_dir}/segments
    num_src_utts=$(wc -l < "${src_segments}")
fi

if [ -e "${src_dir}/utt2spk" ]; then
    has_utt2spk=true
    src_utt2spk=${src_dir}/utt2spk
fi

if ${has_utt2spk}; then
    num_src_utt2spk=$(wc -l < "${src_utt2spk}")
    if [ "${num_src_utt2spk}" -ne "${num_src_utts}" ]; then
        echo "ERROR: wav.scp and utt2spk has different #lines (${num_src_utts} vs ${num_src_utt2spk})." >&2
        exit 1;
    fi
fi

if ! ${has_segments}; then
    split_scps=""
    for i in $(seq 1 "${num_split}"); do
        split_scps+=" ${dst_dir}/wav.${i}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${src_scp}" ${split_scps}
else
    split_scps=""
    for i in $(seq 1 "${num_split}"); do
        split_scps+=" ${dst_dir}/segments.${i}"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${src_segments}" ${split_scps}
    for i in $(seq 1 "${num_split}"); do
        awk '{print $2}' < "${dst_dir}/segments.${i}" | sort | uniq | while read -r wav_id; do
            grep "^${wav_id} " < "${src_scp}" >> "${dst_dir}/wav.${i}.scp"
        done
    done
fi
if ${has_utt2spk}; then
    split_utt2spks=""
    for i in $(seq 1 "${num_split}"); do
        split_utt2spks+=" ${dst_dir}/utt2spk.${i}"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${src_utt2spk}" ${split_utt2spks}
fi
echo "Successfully make subsets."

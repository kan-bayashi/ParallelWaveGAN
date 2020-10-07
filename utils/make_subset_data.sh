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
if [ -e "${src_dir}/segments" ]; then
    has_segments=true
    src_segments=${src_dir}/segments
else
    has_segments=false
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
echo "Successfully make subsets."

#!/bin/bash

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

# shellcheck disable=SC1091
. parse_options.sh || exit 1;

db_root=$1
part=$2
data_dir=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 [Options] <db_root> <part> <data_dir>"
    echo "e.g.: $0 downloads/LibriTTS train-clean-100 data"
    exit 1
fi

set -euo pipefail

# check spk existence
[ ! -e "${db_root}/${part}" ] && \
    echo "${part} does not exist." >&2 && exit 1;

[ ! -e "${data_dir}/${part}" ] && mkdir -p "${data_dir}/${part}"

# set filenames
scp="${data_dir}/${part}/wav.scp"
segments="${data_dir}/${part}/segments"

# check file existence
[ -e "${scp}" ] && rm "${scp}"
[ -e "${segments}" ] && rm "${segments}"

# make scp and segments
find "${db_root}/${part}" -follow -name "*.wav" | sort | while read -r wav; do
    id=$(basename "${wav}" | sed -e "s/\.[^\.]*$//g")
    lab=${wav//.wav/.lab}

    # check lab existence
    if [ ! -e "${lab}" ]; then
        echo "${id} has not label file. skipped."
        continue
    fi

    echo "${id} ${wav}" >> "${scp}"

    # parse label
    idx=1
    while true; do
        next_idx=$((idx+1))
        next_symbol=$(sed -n "${next_idx}p" "${lab}" | awk '{print $3}')
        if [ "${next_symbol}" != "pau" ]; then
            start_nsec=$(sed -n "${idx}p" "${lab}" | awk '{print $2}')
            break
        fi
        idx=${next_idx}
    done
    idx=$(wc -l < "${lab}")
    while true; do
        prev_idx=$((idx-1))
        prev_symbol=$(sed -n "${prev_idx}p" "${lab}" | awk '{print $3}')
        if [ "${prev_symbol}" != "pau" ]; then
            end_nsec=$(sed -n "${idx}p" "${lab}" | awk '{print $1}')
            break
        fi
        idx=${prev_idx}
    done
    start_sec=$(echo "${start_nsec}*0.0000001" | bc | sed "s/^\./0./")
    end_sec=$(echo "${end_nsec}*0.0000001" | bc | sed "s/^\./0./")
    echo "${id} ${id} ${start_sec} ${end_sec}" >> "${segments}"
done

echo "Successfully prepared ${part} data."

#!/bin/bash

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db_root=$1
part=$2
data_dir=$3
db_label_root=$4

# check arguments
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "Usage: $0 [Options] <db_root> <part> <data_dir> [<db_label_root>]"
    echo "e.g.: $0 downloads/LibriTTS train-clean-100 data"
    echo "e.g.: $0 downloads/LibriTTS train-clean-100 data downloads/LibriTTSLabel"
    exit 1
fi

set -euo pipefail

# check spk existence
[ ! -e "${db_root}/${part}" ] && \
    echo "${part} does not exist." >&2 && exit 1;

[ ! -e "${data_dir}/${part}" ] && mkdir -p "${data_dir}/${part}"

# set filenames
scp="${data_dir}/${part}/wav.scp"
if [ -n "${db_label_root}" ]; then
    use_segments=true
    segments="${data_dir}/${part}/segments"
else
    use_segments=false
fi

# check file existence
[ -e "${scp}" ] && rm "${scp}"
if "${use_segments}"; then
    [ -e "${segments}" ] && rm "${segments}"
fi

# make scp and segments
find "${db_root}/${part}" -follow -name "*.wav" | sort | while read -r wav; do
    id=$(basename "${wav}" | sed -e "s/\.[^\.]*$//g")
    lab=$(echo "${wav}" | sed -e "s;${db_root}/${part};${db_label_root}/lab/phone/${part};g" -e "s/.wav/.lab/g")

    # check lab existence
    if "${use_segments}" && [ ! -e "${lab}" ]; then
        echo "${id} does not have a label file. skipped."
        continue
    fi

    echo "${id} ${wav}" >> "${scp}"

    if "${use_segments}"; then
        # parse label
        idx=1
        while true; do
            symbol=$(sed -n "${idx}p" "${lab}" | awk '{print $3}')
            if [ "${symbol}" != "sil" ]; then
                start_sec=$(sed -n "${idx}p" "${lab}" | awk '{print $1}')
                break
            fi
            idx=$((idx+1))
        done
        idx=$(wc -l < "${lab}")
        while true; do
            symbol=$(sed -n "${idx}p" "${lab}" | awk '{print $3}')
            if [ -n "${symbol}" ] && [ "${symbol}" != "sp" ]; then
                end_sec=$(sed -n "${idx}p" "${lab}" | awk '{print $2}')
                break
            fi
            idx=$((idx-1))
        done
        echo "${id} ${id} ${start_sec} ${end_sec}" >> "${segments}"
    fi
done

echo "Successfully prepared ${part} data."

#!/bin/bash

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

fs=24000
num_dev=10
num_eval=10
train_set="train_nodev"
dev_set="dev"
eval_set="eval"
shuffle=false

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db_root=$1
spk=$2
data_dir=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 [Options] <db_root> <spk> <data_dir>"
    echo "e.g.: $0 downloads/VCTK-Corpus p225 data"
    echo ""
    echo "Options:"
    echo "    --fs: target sampling rate (default=24000)."
    echo "    --num_dev: number of development uttreances (default=10)."
    echo "    --num_eval: number of evaluation uttreances (default=10)."
    echo "    --train_set: name of train set (default=train_nodev)."
    echo "    --dev_set: name of dev set (default=dev)."
    echo "    --eval_set: name of eval set (default=eval)."
    echo "    --shuffle: whether to perform shuffle in making dev / eval set (default=false)."
    exit 1
fi

set -euo pipefail

# check spk existence
[ ! -e "${db_root}/lab/mono/${spk}" ] && \
    echo "${spk} does not exist." >&2 && exit 1;

[ ! -e "${data_dir}/all" ] && mkdir -p "${data_dir}/all"

# set filenames
scp="${data_dir}/all/wav.scp"
segments="${data_dir}/all/segments"

# check file existence
[ -e "${scp}" ] && rm "${scp}"
[ -e "${segments}" ] && rm "${segments}"

# make scp and segments
find "${db_root}/wav48/${spk}" -follow -name "*.wav" | sort | while read -r wav; do
    id=$(basename "${wav}" | sed -e "s/\.[^\.]*$//g")
    lab=${db_root}/lab/mono/${spk}/${id}.lab

    # check lab existence
    if [ ! -e "${lab}" ]; then
        echo "${id} does not have a label file. skipped."
        continue
    fi

    echo "${id} cat ${wav} | sox -t wav - -c 1 -b 16 -t wav - rate ${fs} |" >> "${scp}"

    # parse start and end time from HTS-style mono label
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

# split
num_all=$(wc -l < "${scp}")
num_deveval=$((num_dev + num_eval))
num_train=$((num_all - num_deveval))
utils/split_data.sh \
    --num_first "${num_train}" \
    --num_second "${num_deveval}" \
    --shuffle "${shuffle}" \
    "${data_dir}/all" \
    "${data_dir}/${train_set}" \
    "${data_dir}/deveval"
utils/split_data.sh \
    --num_first "${num_dev}" \
    --num_second "${num_eval}" \
    --shuffle "${shuffle}" \
    "${data_dir}/deveval" \
    "${data_dir}/${dev_set}" \
    "${data_dir}/${eval_set}"

# remove tmp directories
rm -rf "${data_dir}/all"
rm -rf "${data_dir}/deveval"

echo "Successfully prepared data."

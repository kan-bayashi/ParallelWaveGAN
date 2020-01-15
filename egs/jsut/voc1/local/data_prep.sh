#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

fs=24000
num_dev=250
num_eval=250
train_set="train_nodev"
dev_set="dev"
eval_set="eval"

# shellcheck disable=SC1091
. parse_options.sh || exit 1;

db_root=$1
data_dir=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 [Options] <db_root> <data_dir>"
    echo "e.g.: $0 downloads/jsut_ver1.1 data"
    echo ""
    echo "Options:"
    echo "    --num_dev: number of development uttreances (default=250)."
    echo "    --num_eval: number of evaluation uttreances (default=250)."
    echo "    --train_set: name of train set (default=train_nodev)."
    echo "    --dev_set: name of dev set (default=dev)."
    echo "    --eval_set: name of eval set (default=eval)."
    exit 1
fi

set -euo pipefail

# make dirs
for name in all "${train_set}" "${dev_set}" "${eval_set}"; do
    [ ! -e "${data_dir}/${name}" ] && mkdir -p "${data_dir}/${name}"
done

# set filenames
scp="${data_dir}/all/wav.scp"
segments="${data_dir}/all/segments"

# check file existence
[ -e "${scp}" ] && rm "${scp}"
[ -e "${segments}" ] && rm "${segments}"

# make scp
find "${db_root}" -follow -name "*.wav" | sort | while read -r filename; do
    id=$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")
    echo "${id} cat ${filename} | sox -t wav - -c 1 -b 16 -t wav - rate ${fs} |" >> "${scp}"
done

# make segments
find "${db_root}" -name "*.lab" -follow | sort | while read -r filename;do
    id=$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")
    start_nsec=$(head -n 1 "${filename}" | cut -d " " -f 2)
    end_nsec=$(tail -n 1 "${filename}" | cut -d " " -f 1)
    start_sec=$(echo "${start_nsec}*0.0000001" | bc | sed "s/^\./0./")
    end_sec=$(echo "${end_nsec}*0.0000001" | bc | sed "s/^\./0./")
    echo "${id} ${id} ${start_sec} ${end_sec}" >> "${segments}"
done

# split
num_all=$(wc -l < "${scp}")
num_deveval=$((num_dev + num_eval))
num_train=$((num_all - num_deveval))
tail -n "${num_train}" "${scp}" > "${data_dir}/${train_set}/wav.scp"
head -n "${num_deveval}" "${scp}" | tail -n "${num_dev}" > "${data_dir}/${dev_set}/wav.scp"
head -n "${num_deveval}" "${scp}" | head -n "${num_eval}" > "${data_dir}/${eval_set}/wav.scp"
tail -n "${num_train}" "${segments}" > "${data_dir}/${train_set}/segments"
head -n "${num_deveval}" "${segments}" | tail -n "${num_dev}" > "${data_dir}/${dev_set}/segments"
head -n "${num_deveval}" "${segments}" | head -n "${num_eval}" > "${data_dir}/${eval_set}/segments"

# remove all
rm -rf "${data_dir}/all"

echo "Successfully prepared data."

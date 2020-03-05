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
. parse_options.sh || exit 1;

db_root=$1
spk=$2
data_dir=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 [Options] <db_root> <spk> <data_dir>"
    echo "e.g.: $0 downloads/VCTK-Corpus p225 data"
    echo ""
    echo "Options:"
    echo "    --fs: sampling frequency (default=${fs})."
    echo "    --num_dev: number of development uttreances (default=${num_dev})."
    echo "    --num_eval: number of evaluation uttreances (default=${num_eval})."
    echo "    --train_set: name of train set (default=${train_set})."
    echo "    --dev_set: name of dev set (default=${dev_set})."
    echo "    --eval_set: name of eval set (default=${eval_set})."
    echo "    --shuffle: whether to perform shuffle in making dev / eval set (default=${shuffle})."
    exit 1
fi

set -euo pipefail

# check spk existence
[ ! -e "${db_root}/wav48/${spk}" ] && \
    echo "${spk} does not exist." >&2 && exit 1;

[ ! -e "${data_dir}/all" ] && mkdir -p "${data_dir}/all"

# set filenames
scp="${data_dir}/all/wav.scp"

# check file existence
[ -e "${scp}" ] && rm "${scp}"

# make scp
find "${db_root}/wav48/${spk}" -follow -name "*.wav" | sort | while read -r filename; do
    id=$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")
    echo "${id} cat ${filename} | sox -t wav - -c 1 -b 16 -t wav - rate ${fs} |" >> "${scp}"
done

# split
num_all=$(wc -l < "${scp}")
num_deveval=$((num_dev + num_eval))
num_train=$((num_all - num_deveval))
split_data.sh \
    --num_first "${num_train}" \
    --num_second "${num_deveval}" \
    --shuffle "${shuffle}" \
    "${data_dir}/all" \
    "${data_dir}/${train_set}" \
    "${data_dir}/deveval"
split_data.sh \
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

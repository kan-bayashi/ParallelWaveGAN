#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

fs=24000
num_dev=100
num_eval=100
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
    echo "e.g.: $0 downloads/CSMSC data"
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

[ ! -e "${data_dir}/all" ] && mkdir -p "${data_dir}/all"

# set filenames
scp="${data_dir}/all/wav.scp"
segments="${data_dir}/all/segments"

# check file existence
[ -e "${scp}" ] && rm "${scp}"
[ -e "${segments}" ] && rm "${segments}"

# make wav.scp
find "${db_root}/Wave" -name "*.wav" -follow | sort | while read -r filename; do
    id="$(basename "${filename}" .wav)"
    echo "csmsc_${id} cat ${filename} | sox -t wav - -c 1 -b 16 -t wav - rate ${fs} |" >> "${scp}"
done

# make segments
find "${db_root}/PhoneLabeling" -name "*.interval" -follow | sort | while read -r filename; do
    id="$(basename "${filename}" .interval)"
    start_sec=$(tail -n +14 "${filename}" | head -n 1)
    end_sec=$(head -n -2 "${filename}" | tail -n 1)
    echo "csmsc_${id} csmsc_${id} ${start_sec} ${end_sec}" >> "${segments}"
done

# check
diff -q <(awk '{print $1}' "${scp}") <(awk '{print $1}' "${segments}") > /dev/null

# split
num_all=$(wc -l < "${scp}")
num_deveval=$((num_dev + num_eval))
num_train=$((num_all - num_deveval))
split_data.sh \
    --num_first "${num_train}" \
    --num_second "${num_deveval}" \
    "${data_dir}/all" \
    "${data_dir}/${train_set}" \
    "${data_dir}/deveval"
split_data.sh \
    --num_first "${num_dev}" \
    --num_second "${num_eval}" \
    "${data_dir}/deveval" \
    "${data_dir}/${dev_set}" \
    "${data_dir}/${eval_set}"

# remove tmp directories
rm -rf "${data_dir}/all"
rm -rf "${data_dir}/deveval"

echo "Successfully prepared data."

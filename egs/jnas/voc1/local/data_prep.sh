#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

num_dev=500
train_set="train_nodev"
dev_set="dev"
eval_set="eval"
shuffle=false

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db_root> <data_dir> <spk_list>"
    echo "e.g.: $0 /database/JNAS data conf/train_speakers.txt"
    echo ""
    echo "Options:"
    echo "    --num_dev: number of development uttreances (default=500)."
    echo "    --train_set: name of train set (default=train_nodev)."
    echo "    --dev_set: name of dev set (default=dev)."
    echo "    --eval_set: name of eval set (default=eval)."
    echo "    --shuffle: whether to perform shuffle in making dev / eval set (default=false)."
    exit 1
fi

set -euo pipefail

db_root=$1  # database root directory
data_dir=$2
spk_list=$3

eval_db_root=${db_root}/DOCS/Test_set
wav_type=HS  # DT or HS

# make directories
for name in train "${eval_set}"; do
    [ ! -e "${data_dir}/${name}" ] && mkdir -p "${data_dir}/${name}"
done

# make training & development data
scp="${data_dir}/train/wav.scp"

# check file existence
[ -e "${scp}" ] && rm "${scp}"

# shellcheck disable=SC2013
for spk in $(cat "${spk_list}"); do
    wavdir=${db_root}/WAVES_${wav_type}/${spk}
    [ ! -e "${wavdir}" ] && echo "There are no such a directory (${wavdir})" && exit 1
    find "${wavdir}" -follow -name "*.wav" | sort | while read -r filename; do
        id=$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")
        echo "${spk}_${id} ${filename}" >> "${scp}"
    done
done

# shuffle
cp "${scp}" "${scp}.tmp"
sort -R "${scp}.tmp" > "${scp}"
rm -r "${scp}.tmp"

# split
utils/split_data.sh \
    --num_second ${num_dev} \
    --shuffle "${shuffle}" \
    "${data_dir}/train" \
    "${data_dir}/${train_set}" \
    "${data_dir}/${dev_set}"

# make evaluation data
scp="${data_dir}/${eval_set}/wav.scp"

# check file existence
[ -e "${scp}" ] && rm "${scp}"

for name in JNAS_testset_100 JNAS_testset_500; do
    find "${eval_db_root}/${name}/WAVES" -follow -name "*.wav" | sort | while read -r filename; do
        id=$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")
        dirname=$(basename "$(dirname "${filename}")")
        echo "${name}_${dirname}_${id} ${filename}" >> "${scp}"
    done
done

echo "Successfully prepared data."

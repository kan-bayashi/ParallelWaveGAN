#!/usr/bin/env bash

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

num_dev=250
num_eval=250
train_set="train_nodev"
dev_set="dev"
eval_set="eval"
shuffle=false

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db_root=$1
data_dir=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 [Options] <db_root> <data_dir>"
    echo "e.g.: $0 downloads/LJSpeech-1.1 data"
    echo ""
    echo "Options:"
    echo "    --num_dev: number of development uttreances (default=250)."
    echo "    --num_eval: number of evaluation uttreances (default=250)."
    echo "    --train_set: name of train set (default=train_nodev)."
    echo "    --dev_set: name of dev set (default=dev)."
    echo "    --eval_set: name of eval set (default=eval)."
    echo "    --shuffle: whether to perform shuffle in making dev / eval set (default=false)."
    exit 1
fi

set -euo pipefail

# check directory existence
[ ! -e "${data_dir}" ] && mkdir -p "${data_dir}"

# set filenames
mkdir -p "${data_dir}/all"
scp=${data_dir}/all/wav.scp

# check file existence
[ -e "${scp}" ] && rm "${scp}"

# make scp, utt2spk, and spk2utt
find "${db_root}/kss" -name "*.wav" | sort | while read -r filename; do
    id=kss_$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")
    # NOTE(kan-bayashi): Some wav files are stereo
    echo "${id} sox ${filename} -t wav -c 1 - |" >> "${scp}"
done
echo "Successfully finished making wav.scp."

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

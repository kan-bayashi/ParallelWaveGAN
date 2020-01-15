#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

num_dev=100
num_eval=100
train_set="train_nodev"
dev_set="dev"
eval_set="eval"

# shellcheck disable=SC1091
. parse_options.sh || exit 1;

db_root=$1
spk=$2
data_dir=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db_root> <spk> <data_dir>"
    echo "e.g.: $0 downloads/cms_us_slt_arctic slt data"
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

# check speaker
available_spks=(
    "slt" "clb" "bdl" "rms" "jmk" "awb" "ksp"
)
if ! echo "${available_spks[*]}" | grep -q "${spk}"; then
    echo "Specified speaker ${spk} is not available."
    echo "Available speakers: ${available_spks[*]}"
    exit 1
fi

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
find "${db_root}" -name "*.wav" -follow | sort | while read -r filename;do
    id="${spk}_$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")"
    echo "${id} ${filename}" >> "${scp}"
done

# make segments
find "${db_root}/lab" -name "*.lab" -follow | sort | while read -r filename; do
    # get start time
    while read -r line; do
        phn=$(echo "${line}" | cut -d " " -f 3)
        if [ "${phn}" != "pau" ]; then
            break
        fi
        start=$(echo "${line}" | cut -d " " -f 1)
    done < <(tail -n +2 "$filename")
    # get end time
    while read -r line; do
        end=$(echo "${line}" | cut -d " " -f 1)
        phn=$(echo "${line}" | cut -d " " -f 3)
        if [ "${phn}" != "pau" ]; then
            break
        fi
    done < <(tail -n +2 "$filename" | tac)
    echo "${spk}_$(basename "${filename}" .lab) ${spk}_$(basename "${filename}" .lab) ${start} ${end}" >> "${segments}"
done

# split
num_all=$(wc -l < "${scp}")
num_deveval=$((num_dev + num_eval))
num_train=$((num_all - num_deveval))
head -n "${num_train}" "${scp}" > "${data_dir}/${train_set}/wav.scp"
tail -n "${num_deveval}" "${scp}" | head -n "${num_dev}" > "${data_dir}/${dev_set}/wav.scp"
tail -n "${num_deveval}" "${scp}" | tail -n "${num_eval}" > "${data_dir}/${eval_set}/wav.scp"
head -n "${num_train}" "${segments}" > "${data_dir}/${train_set}/segments"
tail -n "${num_deveval}" "${segments}" | head -n "${num_dev}" > "${data_dir}/${dev_set}/segments"
tail -n "${num_deveval}" "${segments}" | tail -n "${num_eval}" > "${data_dir}/${eval_set}/segments"

# remove all
rm -rf "${data_dir}/all"

echo "Successfully prepared data."

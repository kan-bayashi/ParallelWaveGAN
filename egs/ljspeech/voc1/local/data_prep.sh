#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

. ./path.sh || exit 1;

num_dev=250
num_eval=250

. parse_options.sh || exit 1;

db=$1
data_dir=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 [Options] <db> <data_dir>"
    echo "Options:"
    echo "    --num_dev: number of development uttreances (default=250)."
    echo "    --num_eval: number of evaluation uttreances (default=250)."
    exit 1
fi

set -euo pipefail

# make dirs
for name in all train_nodev dev eval; do
    [ ! -e ${data_dir}/${name} ] && mkdir -p ${data_dir}/${name}
done

# set filenames
scp=${data_dir}/all/wav.scp

# check file existence
[ -e ${scp} ] && rm ${scp}

# make all scp
find ${db} -follow -name "*.wav" | sort | while read -r filename;do
    id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
    echo "${id} ${filename}" >> ${scp}
done

# split
num_all=$(cat ${scp} | wc -l)
num_deveval=$((num_dev + num_eval))
num_train=$((num_all - num_deveval))
head -n ${num_train} ${scp} > ${data_dir}/train_nodev/wav.scp
tail -n ${num_deveval} ${scp} | head -n ${num_dev} > ${data_dir}/dev/wav.scp
tail -n ${num_deveval} ${scp} | tail -n ${num_eval} > ${data_dir}/eval/wav.scp

# remove all
rm -rf ${data_dir}/all

echo "successfully prepared data."

#!/bin/bash

# Copyright 2022 Jiatong Shi
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

fs=22050

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db_root=$1
data_dir=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 [Options] <db_root> <data_dir>"
    exit 1
fi

set -euo pipefail

for subset in "train" "dev" "test"; do
    mkdir -p "${data_dir}/${subset}"
    scp="${data_dir}/${subset}/wav.scp"
    [ -e "${scp}" ] && rm "${scp}"
    find "${db_root}/${subset}" -follow -name "*.wav" | sort | while read -r filename; do
        id=$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")
        echo "${id} cat ${filename} | sox -t wav - -c 1 -b 16 -t wav - rate ${fs} |" >> "${scp}"
    done
done

echo "Successfully prepared data."

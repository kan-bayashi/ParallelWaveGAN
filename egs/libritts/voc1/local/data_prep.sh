#!/bin/bash

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

# shellcheck disable=SC1091
. parse_options.sh || exit 1;

db_root=$1
part=$2
data_dir=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 [Options] <db_root> <part> <data_dir>"
    echo "e.g.: $0 downloads/LibriTTS train-clean-100 data"
    exit 1
fi

set -euo pipefail

# check spk existence
[ ! -e "${db_root}/${part}" ] && \
    echo "${part} does not exist." >&2 && exit 1;

[ ! -e "${data_dir}/${part}" ] && mkdir -p "${data_dir}/${part}"

# set filenames
scp="${data_dir}/${part}/wav.scp"

# check file existence
[ -e "${scp}" ] && rm "${scp}"

# make scp and segments
find "${db_root}/${part}" -follow -name "*.wav" | sort | while read -r wav; do
    id=$(basename "${wav}" | sed -e "s/\.[^\.]*$//g")
    echo "${id} ${wav}" >> "${scp}"
done

echo "Successfully prepared ${part} data."

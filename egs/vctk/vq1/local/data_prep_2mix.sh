#!/bin/bash

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db_root=$1
data_dir_root=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <db_root> <data_dir>"
    echo "e.g.: $0 downloads/VCTK-Corpus data"
    exit 1
fi

set -euo pipefail

for mix_type in min max; do
    for dset in tr tt cv; do
        for src_spk in s1 s2; do
            echo "${mix_type}_${dset}_${src_spk}"
            # set filenames
            data_dir=${data_dir_root}/${mix_type}_${dset}_${src_spk}
            scp="${data_dir}/wav.scp"
            utt2spk="${data_dir}/utt2spk"

            # check file existence
            [ ! -e "${data_dir}" ] && mkdir -p "${data_dir}"
            [ -e "${scp}" ] && rm "${scp}"
            [ -e "${utt2spk}" ] && rm "${utt2spk}"

            # make scp, utt2spk, and segments
            find "${db_root}/${mix_type}/${dset}/${src_spk}" -follow -name "*.wav" | sort | while read -r wav; do
                id=$(basename "${wav}" | sed -e "s/\.[^\.]*$//g")
                if [ ${src_spk} = "s1" ]; then
                    spk=$(echo "${id}" | cut -f 1 -d "_")
                else
                    spk=$(echo "${id}" | cut -f 4 -d "_")
                fi
                echo "${id} ${wav}" >> "${scp}"
                echo "${id} ${spk}" >> "${utt2spk}"
            done
        done
    done
done


echo "Successfully prepared data."

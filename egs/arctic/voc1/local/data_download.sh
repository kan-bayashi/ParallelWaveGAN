#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

download_dir=$1
spk=$2

available_spks=(
    "slt" "clb" "bdl" "rms" "jmk" "awb" "ksp"
)

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <download_dir> <spk>"
    echo "Available speakers: ${available_spks[*]}"
    exit 1
fi

set -euo pipefail

# check speakers
if ! echo "${available_spks[*]}" | grep -q "${spk}"; then
    echo "Specified spk (${spk}) is not available or not supported." >&2
    exit 1
fi

# download dataset
cwd=$(pwd)
if [ ! -e "${download_dir}/cmu_us_${spk}_arctic" ]; then
    mkdir -p "${download_dir}"
    cd "${download_dir}"
    wget "http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_${spk}_arctic-0.95-release.tar.bz2"
    tar xf "cmu_us_${spk}_arctic-0.95-release.tar.bz2"
    rm "cmu_us_${spk}_arctic-0.95-release.tar.bz2"
    cd "${cwd}"
    echo "Successfully finished download."
else
    echo "Already exists. Skip download."
fi

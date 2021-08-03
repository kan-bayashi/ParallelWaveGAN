#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

download_dir=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <download_dir>"
    exit 1
fi

set -euo pipefail

cwd=$(pwd)
if [ ! -e "${download_dir}/speech_commands" ]; then
    mkdir -p "${download_dir}/speech_commands"
    cd "${download_dir}"
    wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
    tar -vxf ./*.tar.gz -C "${download_dir}/speech_commands"
    rm ./*.tar.gz
    cd "${cwd}"
    echo "Successfully downloaded data."
else
    echo "Already exists. Skipped."
fi

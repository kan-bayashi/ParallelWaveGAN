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
if [ ! -e "${download_dir}/jsut_ver1.1" ]; then
    mkdir -p "${download_dir}"
    cd "${download_dir}" || exit 1;
    wget http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip
    unzip -o ./*.zip
    rm ./*.zip
    cd "${cwd}" || exit 1;
    echo "Successfully downloaded data."
else
    echo "Already exists. Skipped."
fi

if [ ! -e "${download_dir}/jsut_lab" ]; then
    cd "${download_dir}" || exit 1;
    git clone https://github.com/r9y9/jsut-lab
    for name in loanword128 repeat500 voiceactress100 basic5000 onomatopee300 travel1000 countersuffix26 precedent130 utparaphrase512; do
        cp -vr "jsut-lab/${name}" jsut_ver1.1/
    done
    cd - || exit 1;
    echo "Successfully downloaded context label."
else
    echo "Already exists. Skipped."
fi

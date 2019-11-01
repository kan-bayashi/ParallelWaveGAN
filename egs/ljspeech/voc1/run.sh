#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

. ./path.sh || exit 1;

# basic settings
stage=-1
stop_stage=100
verbose=1
nj=16

config=conf/hparam.yml

# directory path setting
download_dir=downloads
dumpdir=dump

# training related setting
resume=""

# decoding related setting
checkpoint=""

. parse_options.sh || exit 1;

set -euo pipefail

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "Stage -1: Data download"
    local/data_download.sh ${download_dir}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Data preparation"
    local/data_prep.sh ${download_dir}/LJSpeech-1.1 data
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Feature extraction"
    # extract raw features
    for name in train_nodev dev eval; do
        preprocessing.py \
            --config ${config} \
            --wavscp data/${name}/wav.scp \
            --dumpdir ${dumpdir}/${name}/raw \
            --n_jobs ${nj} \
            --verbose ${verbose}
        echo "successfully finished feature extraction of ${name} set."
    done
    echo "successfully finished feature extraction."

    # calculate statistics for normalization
    compute_statistics.py \
        --config ${config} \
        --rootdir ${dumpdir}/train_nodev/raw \
        --dumpdir ${dumpdir}/train_nodev \
        --verbose ${verbose}
    echo "successfully finished calculation of statistics."

    # normalize and dump them
    for name in train_nodev dev eval; do
        normalize.py \
            --config ${config} \
            --stats ${dumpdir}/train_nodev/stats.h5 \
            --rootdir ${dumpdir}/${name}/raw \
            --dumpdir ${dumpdir}/${name}/norm \
            --n_jobs ${nj} \
            --verbose ${verbose}
        echo "successfully finished normalization of ${name} set."
    done
    echo "successfully finished normalization."
fi

expdir=exp/train_ljspeech_$(basename ${config} .yml)
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Network training"
    train.py \
        --config ${config} \
        --train-dumpdir ${dumpdir}/train_nodev/norm \
        --dev-dumpdir ${dumpdir}/dev/norm \
        --outdir ${expdir} \
        --resume ${resume} \
        --verbose ${verbose}
    echo "successfully finished training."
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 2: Network decoding"
    [ ! -n "${checkpoint}" ] && checkpoint=$(find ${expdir} -name "*.pkl" | xargs ls -t | head -n 1)
    outdir=${expdir}/wav/$(basename ${checkpoint} .pkl)
    for name in dev eval; do
        decode.py \
            --config ${config} \
            --dumpdir ${dumpdir}/train_nodev/norm \
            --checkpoint ${checkpoint} \
            --outdir ${outdir} \
            --resume ${resume} \
            --verbose ${verbose}
    done
    echo "successfully finished decoding."
fi
echo "finished."

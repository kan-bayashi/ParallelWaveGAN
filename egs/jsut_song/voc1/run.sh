#!/bin/bash

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# basic settings
stage=-1        # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=4       # number of parallel jobs in feature extraction

# NOTE(kan-bayashi): renamed to conf to avoid conflict in parse_options.sh
conf=conf/hifigan.v1.yaml

# directory path setting
db_root=downloads # direcotry to download data and labels from https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song
dumpdir=dump # directory to dump features

# subset setting
shuffle=false # whether to shuffle the data to create subset

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)
pretrain="" # checkpoint path to load pretrained parameters
            # (e.g. ../../jsut/<path>/<to>/checkpoint-400000steps.pkl)

# decoding related setting
checkpoint="" # checkpoint path to be used for decoding
              # if not provided, the latest one will be used
              # (e.g. <path>/<to>/checkpoint-400000steps.pkl)

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

train_set="train_nodev" # name of training data directory
dev_set="dev"           # name of development data direcotry
eval_set="eval"         # name of evaluation data direcotry

set -euo pipefail

if [ "${stage}" -le -1 ] && [ "${stop_stage}" -ge -1 ]; then
    echo "Stage -1: Data download"
    if [ -e "${db_root}/todai_child" ] && [ -e "${db_root}/jsut-song_ver1/child_song/wav" ]; then
        echo "The JSUT-song corpus exists. Skip downloading."
    
    elif [ -e "${db_root}/jsut-song_ver1.zip" ] && [ -e "${db_root}/jsut-song_label.zip" ]; then
        echo "Unzipping downloaded zip files for JSUT-song corpus."
        unzip ${db_root}/jsut-song_ver1.zip -d ${db_root}
        unzip ${db_root}/jsut-song_label.zip -d ${db_root}
        rm ${db_root}/jsut-song_ver1.zip
        rm ${db_root}/jsut-song_label.zip

    else
    	echo "ERROR: The JSUT-song corpus does not exist."
    	echo "ERROR: Please download from https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song"
        echo "and locate it at ${db_root}"
        echo "Please ensure that you've downloaded songs (jsut-song_ver1.zip) and labels (jsut-song_label.zip) to ${db_root} before proceeding"
        # Terms from https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song
    	exit 1
    fi
fi

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Data preparation"
    mkdir -p score_dump
    mkdir -p wav_dump
    python local/data_prep.py \
        --lab_srcdir ${db_root}/todai_child \
        --wav_srcdir ${db_root}/jsut-song_ver1/child_song/wav \
        --score_dump score_dump \
        --wav_dumpdir wav_dump \
        --sr "$(yq ".sampling_rate" "${conf}")" \
        --train_set "${train_set}" \
        --dev_set "${dev_set}" \
        --eval_set "${eval_set}"
    for src_data in ${train_set} ${dev_set} ${eval_set}; do
        utils/utt2spk_to_spk2utt.pl < data/${src_data}/utt2spk > data/${src_data}/spk2utt
        utils/fix_data_dir.sh --utt_extra_files "label score.scp" data/${src_data}
    done
fi

stats_ext=$(grep -q "hdf5" <(yq ".format" "${conf}") && echo "h5" || echo "npy")
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Feature extraction"
    # extract raw features
    pids=()
    for name in "${train_set}" "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        utils/make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/raw"
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            parallel-wavegan-preprocess \
                --config "${conf}" \
                --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --verbose "${verbose}"
        echo "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished feature extraction."

    # calculate statistics for normalization
    if [ -z "${pretrain}" ]; then
        # calculate statistics for normalization
        echo "Statistics computation start. See the progress via ${dumpdir}/${train_set}/compute_statistics.log."
        ${train_cmd} "${dumpdir}/${train_set}/compute_statistics.log" \
            parallel-wavegan-compute-statistics \
                --config "${conf}" \
                --rootdir "${dumpdir}/${train_set}/raw" \
                --dumpdir "${dumpdir}/${train_set}" \
                --verbose "${verbose}"
        echo "Successfully finished calculation of statistics."
    else
        echo "Use statistics of pretrained model. Skip statistics computation."
        cp "$(dirname "${pretrain}")/stats.${stats_ext}" "${dumpdir}/${train_set}"
    fi

    # normalize and dump them
    pids=()
    for name in "${train_set}" "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/norm" ] && mkdir -p "${dumpdir}/${name}/norm"
        echo "Nomalization start. See the progress via ${dumpdir}/${name}/norm/normalize.*.log."
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/norm/normalize.JOB.log" \
            parallel-wavegan-normalize \
                --config "${conf}" \
                --stats "${dumpdir}/${train_set}/stats.${stats_ext}" \
                --rootdir "${dumpdir}/${name}/raw/dump.JOB" \
                --dumpdir "${dumpdir}/${name}/norm/dump.JOB" \
                --verbose "${verbose}"
        echo "Successfully finished normalization of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished normalization."
fi

if [ -z "${tag}" ]; then
    expdir="exp/${train_set}_jsut_song_$(basename "${conf}" .yaml)"
    if [ -n "${pretrain}" ]; then
        pretrain_tag=$(basename "$(dirname "${pretrain}")")
        expdir+="_${pretrain_tag}"
    fi
else
    expdir="exp/${train_set}_jsut_song_${tag}"
fi
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    cp "${dumpdir}/${train_set}/stats.${stats_ext}" "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        train="python -m parallel_wavegan.distributed.launch --nproc_per_node ${n_gpus} -c parallel-wavegan-train"
    else
        train="parallel-wavegan-train"
    fi
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --config "${conf}" \
            --train-dumpdir "${dumpdir}/${train_set}/norm" \
            --dev-dumpdir "${dumpdir}/${dev_set}/norm" \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --pretrain "${pretrain}" \
            --verbose "${verbose}"
    echo "Successfully finished training."
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
    pids=()
    for name in "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
            parallel-wavegan-decode \
                --dumpdir "${dumpdir}/${name}/norm" \
                --checkpoint "${checkpoint}" \
                --outdir "${outdir}/${name}" \
                --verbose "${verbose}"
        echo "Successfully finished decoding of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished decoding."
fi
echo "Finished."

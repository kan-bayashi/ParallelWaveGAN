#!/bin/bash

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# basic settings
stage=2        # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=4       # number of parallel jobs in feature extraction

# NOTE(kan-bayashi): renamed to conf to avoid conflict in parse_options.sh
conf=conf/uhifigan.v1.yaml

# directory path setting
db_root= # direcotry including spk name directory (MODIFY BY YOURSELF)
# e.g.
# /path/to/database
# ├── spk_1
# │   ├── utt1.wav
# ├── spk_2
# │   ├── utt1.wav
# │   ...
# └── spk_N
#     ├── utt1.wav
#     ...
dumpdir=dump # directory to dump features

# subset setting
spks="all"    # speaker name to be used (e.g. "spk1 spk2")
              # it must be matched the name under the ${db_root}
              # if set to "all", all of the speakers in ${db_root} will be used
shuffle=false # whether to shuffle the data to create subset
num_dev=5    # the number of development data for each speaker
num_eval=5   # the number of evaluation data for each speaker
              # (if set to 0, the same dev set is used as eval set)

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

train_set="train_nodev_$(echo "${spks}" | tr " " "_")" # name of training data directory
dev_set="dev_$(echo "${spks}" | tr " " "_")"           # name of development data directory
eval_set="eval_$(echo "${spks}" | tr " " "_")"         # name of evaluation data directory

set -euo pipefail

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Data preparation"
    train_data_dirs=""
    dev_data_dirs=""
    eval_data_dirs=""
    if [ "${spks}" = "all" ]; then
        spks=$(find "${db_root}" -maxdepth 1 ! -path "${db_root}" \
            -follow -type d -print0 -name "[^.]*" | xargs -0 -I{} basename {})
    fi
    for spk in ${spks}; do
        local/data_prep.sh \
            --fs "$(yq ".sampling_rate" "${conf}")" \
            --shuffle "${shuffle}" \
            --num_dev "${num_dev}" \
            --num_eval "${num_eval}" \
            --train_set "train_nodev_${spk}" \
            --dev_set "dev_${spk}" \
            --eval_set "eval_${spk}" \
            "${db_root}" "${spk}" data
        train_data_dirs+=" data/train_nodev_${spk}"
        dev_data_dirs+=" data/dev_${spk}"
        eval_data_dirs+=" data/eval_${spk}"
    done
    # shellcheck disable=SC2086
    utils/combine_data.sh "data/${train_set}" ${train_data_dirs}
    # shellcheck disable=SC2086
    utils/combine_data.sh "data/${dev_set}" ${dev_data_dirs}
    # shellcheck disable=SC2086
    utils/combine_data.sh "data/${eval_set}" ${eval_data_dirs}
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
    expdir="exp/${train_set}_$(basename "${conf}" .yaml)"
    if [ -n "${pretrain}" ]; then
        pretrain_tag=$(basename "$(dirname "${pretrain}")")
        expdir+="_${pretrain_tag}"
    fi
else
    expdir="exp/${train_set}_${tag}"
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


if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: Obejctive evaluation"
    for dset in "${dev_set}" "${eval_set}"; do
        _data="data/${dset}"
        _gt_wavscp="${_data}/wav.scp"
        # shellcheck disable=SC2012
        [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
        _dir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
        _gen_wavdir="${_dir}/${dset}"

        # Objective Evaluation - MCD
        echo "Begin Scoring for MCD metrics on ${dset}, results are written under ${_dir}/${dset}/MCD_res"
        mkdir -p "${_dir}/${dset}/MCD_res"
        python -m parallel_wavegan.bin.evaluate_mcd \
            --gen_wavdir_or_wavscp "${_gen_wavdir}" \
            --gt_wavdir_or_wavscp "${_gt_wavscp}" \
            --outdir "${_dir}/${dset}/MCD_res"

        # Objective Evaluation - log-F0 RMSE & Semitone ACC & VUV Error Rate
        echo "Begin Scoring for F0 related metrics on ${dset}, results are written under ${_dir}/${dset}/F0_res"
        mkdir -p "${_dir}/${dset}/F0_res"
        python -m parallel_wavegan.bin.evaluate_f0 \
            --gen_wavdir_or_wavscp "${_gen_wavdir}" \
            --gt_wavdir_or_wavscp "${_gt_wavscp}" \
            --outdir "${_dir}/${dset}/F0_res"

    done
    echo "Successfully finished objective evaluation."
fi

echo "Finished."

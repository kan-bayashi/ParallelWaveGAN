#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# basic settings
stage=2       # stage to start
stop_stage=4 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=8      # number of parallel jobs in feature extraction

# NOTE(kan-bayashi): renamed to conf to avoid conflict in parse_options.sh
conf=conf/uhifigan.v1.yaml

# directory path setting
download_dir=/data3/qt # set the directory to your database
dumpdir=dump           # directory to dump features
python=python3 

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)

# decoding related setting
checkpoint="" # checkpoint path to be used for decoding
              # if not provided, the latest one will be used
              # (e.g. <path>/<to>/checkpoint-400000steps.pkl)

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

train_set="tr_no_dev" # name of training data directory
dev_set="dev"           # name of development data direcotry
eval_set="test"         # name of evaluation data direcotry

set -euo pipefail

if [ "${stage}" -le -1 ] && [ "${stop_stage}" -ge -1 ]; then
    echo "Stage -1: Data download"
    if [ ! -e "${download_dir}/Opencpop" ]; then
    	echo "ERROR: Opencpop data does not exist."
    	echo "ERROR: Please download https://wenet.org.cn/opencpop/download/ and locate it at ${download_dir}"
    	exit 1
    fi
fi

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Data preparation"
    mkdir -p wav_dump
    python local/data_prep.py ${download_dir}/Opencpop \
        --wav_dumpdir wav_dump \
        --sr 24000

    sort -o data/train/wav.scp data/train/wav.scp

    dev_num=50
    train_num=$(( $(wc -l < data/train/wav.scp) - dev_num ))
    mkdir -p data/${dev_set}
    mkdir -p data/${train_set}
    head -n $train_num data/train/wav.scp > data/${train_set}/wav.scp
    tail -n $dev_num data/train/wav.scp > data/${dev_set}/wav.scp

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
    echo "Statistics computation start. See the progress via ${dumpdir}/${train_set}/compute_statistics.log."
    ${train_cmd} "${dumpdir}/${train_set}/compute_statistics.log" \
        parallel-wavegan-compute-statistics \
            --config "${conf}" \
            --rootdir "${dumpdir}/${train_set}/raw" \
            --dumpdir "${dumpdir}/${train_set}" \
            --verbose "${verbose}"
    echo "Successfully finished calculation of statistics."

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
    expdir="exp_unet/${train_set}_opencpop_$(basename "${conf}" .yaml)"
else
    expdir="exp_unet/${train_set}_opencpop_${tag}"
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
            --verbose "${verbose}"
    echo "Successfully finished training."
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
    pids=()
    for name in "${train_set}" ; do
    # for name in "${dev_set}" "${eval_set}"; do
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
        [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
        _dir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
        _gen_wavdir="${_dir}/${dset}"


        # Objective Evaluation - MCD
        echo "Begin Scoring for MCD metrics on ${dset}, results are written under ${_dir}/${dset}/MCD_res"

        mkdir -p "${_dir}/${dset}/MCD_res"
        ${python} local/evaluate_mcd.py \
            --gen_wavdir_or_wavscp ${_gen_wavdir} \
            --gt_wavdir_or_wavscp ${_gt_wavscp} \
            --outdir "${_dir}/${dset}/MCD_res"
        
        # Objective Evaluation - log-F0 RMSE & Semitone ACC & VUV Error Rate
        echo "Begin Scoring for F0 related metrics on ${dset}, results are written under ${_dir}/${dset}/F0_res"

        mkdir -p "${_dir}/${dset}/F0_res"
        ${python} local/evaluate_f0.py \
            --gen_wavdir_or_wavscp ${_gen_wavdir} \
            --gt_wavdir_or_wavscp ${_gt_wavscp} \
            --outdir "${_dir}/${dset}/F0_res"

    done
    # ${train_cmd} JOB=1:${n_jobs} "${expdir}/wav/evaluating.JOB.log" \
    #         parallel-wavegan-preprocess \
    #             --config "${conf}" \
    #             --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
    #             --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
    #             --verbose "${verbose}"
    #     echo "Successfully finished feature extraction of ${name} set."


    # shellcheck disable=SC2012
    # [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    # outdir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
    # pids=()
    # for name in "${dev_set}" "${eval_set}"; do
    # (
    #     [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
    #     [ "${n_gpus}" -gt 1 ] && n_gpus=1
    #     echo "Decoding start. See the progress via ${outdir}/${name}/decode.log."
    #     ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
    #         parallel-wavegan-decode \
    #             --dumpdir "${dumpdir}/${name}/norm" \
    #             --checkpoint "${checkpoint}" \
    #             --outdir "${outdir}/${name}" \
    #             --verbose "${verbose}"
    #     echo "Successfully finished decoding of ${name} set."
    # ) &
    # pids+=($!)
    # done
    # i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    # [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished scoring."
fi

echo "Finished."

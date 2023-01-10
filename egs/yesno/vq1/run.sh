#!/bin/bash

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# basic settings
stage=-1       # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=0       # number of gpus in training
n_jobs=2       # number of parallel jobs in feature extraction

# NOTE(kan-bayashi): renamed to conf to avoid conflict in parse_options.sh
conf=conf/conditioned_melgan_vae.v3.debug.yaml

# directory path setting
download_dir=downloads # directory to save database
dumpdir=dump           # directory to dump features

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

train_set="train_nodev" # name of training data directory
dev_set="dev"           # name of development data direcotry
eval_set="eval"         # name of evaluation data direcotry

set -euo pipefail

if [ "${stage}" -le -1 ] && [ "${stop_stage}" -ge -1 ]; then
    echo "Stage -1: Data download"
    local/data_download.sh "${download_dir}"
fi

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Data preparation"
    local/data_prep.sh \
        --train_set "${train_set}" \
        --dev_set "${dev_set}" \
        --eval_set "${eval_set}" \
        "${download_dir}/waves_yesno" data

    # make utt2spk
    for dset in "${train_set}" "${dev_set}" "${eval_set}"; do
        awk < "data/${dset}/wav.scp" '{print $1 " yesno"}' > "data/${dset}/utt2spk"
    done

    # make spk2idx
    cut -f 2 -d " " "data/${train_set}/utt2spk" | sort | uniq | \
        awk '{print $1 " " NR -1}' > "data/${train_set}/spk2idx"
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
        if [ "$(yq ".use_local_condition" "${conf}")" = "true" ]; then
            opts="--extract-f0 --skip-mel-ext"
        else
            opts="--skip-mel-ext"
        fi
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            parallel-wavegan-preprocess \
                --config "${conf}" \
                --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --utt2spk "${dumpdir}/${name}/raw/utt2spk.JOB" \
                --spk2idx "data/${train_set}/spk2idx" \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --verbose "${verbose}" \
                ${opts}
        echo "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished feature extraction."

    if [ "$(yq ".use_local_condition" "${conf}")" = "true" ]; then
        # calculate statistics for normalization
        echo "Statistics computation start. See the progress via ${dumpdir}/${train_set}/compute_statistics.log."
        ${train_cmd} "${dumpdir}/${train_set}/compute_statistics.log" \
            parallel-wavegan-compute-statistics \
                --config "${conf}" \
                --rootdir "${dumpdir}/${train_set}/raw" \
                --dumpdir "${dumpdir}/${train_set}" \
                --target-feats "local" \
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
                    --target-feats "local" \
                    --verbose "${verbose}"
            echo "Successfully finished normalization of ${name} set."
        ) &
        pids+=($!)
        done
        i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
        [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
        echo "Successfully finished normalization."
    fi
fi

if [ -z "${tag}" ]; then
    expdir="exp/${train_set}_yesno_$(basename "${conf}" .yaml)"
else
    expdir="exp/${train_set}_yesno_${tag}"
fi
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        train="python -m parallel_wavegan.distributed.launch --nproc_per_node ${n_gpus} -c parallel-wavegan-train"
    else
        train="parallel-wavegan-train"
    fi
    if [ "$(yq ".use_local_condition" "${conf}")" = "true" ]; then
        cp "${dumpdir}/${train_set}/stats.${stats_ext}" "${expdir}"
        train_dumpdir="${dumpdir}/${train_set}/norm"
        dev_dumpdir="${dumpdir}/${dev_set}/norm"
    else
        train_dumpdir="${dumpdir}/${train_set}/raw"
        dev_dumpdir="${dumpdir}/${dev_set}/raw"
    fi
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --config "${conf}" \
            --train-dumpdir "${train_dumpdir}" \
            --dev-dumpdir "${dev_dumpdir}" \
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
    for name in "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        # shellcheck disable=SC2030,SC2031
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.log."
        if [ "$(yq ".use_local_condition" "${conf}")" = "true" ]; then
            decode_dumpdir="${dumpdir}/${name}/norm"
        else
            decode_dumpdir="${dumpdir}/${name}/raw"
        fi
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
            parallel-wavegan-decode \
                --dumpdir "${decode_dumpdir}" \
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

#!/bin/bash

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# basic settings
stage=-1       # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=16      # number of parallel jobs in feature extraction

# NOTE(kan-bayashi): renamed to conf to avoid conflict in parse_options.sh
conf=conf/parallel_wavegan_hubert.v1.yaml

# directory path setting
dumpdir=dump           # directory to dump features

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)

# decoding related setting
checkpoint="" # checkpoint path to be used for decoding
              # if not provided, the latest one will be used
              # (e.g. <path>/<to>/checkpoint-400000steps.pkl)

train_set=tr_2mix_trim
dev_set=cv_2mix
eval_set=tt_2mix

hubert_text=downloads/dump_feature_hubert9/label_km100_vctk_16k_tr_2mix_trim/all.txt

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Data preparation"
    echo "Data preparation is not prepared."
    echo "Please manually locate kaldi-style directories as follows:"
    cat << EOF
$ ls data/${train_set}
wav.scp  utt2spk
$ ls data/${dev_set}
wav.scp  utt2spk
$ ls data/${eval_set}
wav.scp  utt2spk
EOF
    # shellcheck disable=SC2016
    echo 'Please make sure "${hubert_text}" is correctly set.'
    echo "Then, please run from stage 1 via --stage 1."
    exit 1;
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Feature extraction"
    if [ ! -e "${hubert_text}" ]; then
        echo "Valid --hubert_text is not provided. Please prepare it by yourself."
        echo "hubert_text should be like kaldi-style text as follows:"
        cat << EOF
utt_id_1 0 0 0 0 1 1 1 1 2 2 2 2
utt_id_2 0 0 0 0 0 0 3 3 3 3 3 3 5 5 5 5
...
EOF
        exit 1
    fi
    # extract raw features
    pids=()
    for name in "${train_set}" "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        utils/make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/raw"
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            local/preprocess_hubert.py \
                --config "${conf}" \
                --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --utt2spk "data/${name}/utt2spk" \
                --spk2idx "data/${train_set}/spk2idx" \
                --text "${hubert_text}" \
                --verbose "${verbose}"
        echo "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished feature extraction."

fi

if [ -z "${tag}" ]; then
    expdir="exp/${train_set}_vctk_$(basename "${conf}" .yaml)"
else
    expdir="exp/${train_set}_vctk_${tag}"
fi
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        train="python -m parallel_wavegan.distributed.launch --nproc_per_node ${n_gpus} -c parallel-wavegan-train"
    else
        train="parallel-wavegan-train"
    fi
    # shellcheck disable=SC2012
    resume="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --config "${conf}" \
            --train-dumpdir "${dumpdir}/${train_set}/raw" \
            --dev-dumpdir "${dumpdir}/${dev_set}/raw" \
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
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
            parallel-wavegan-decode \
                --dumpdir "${dumpdir}/${name}/raw" \
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

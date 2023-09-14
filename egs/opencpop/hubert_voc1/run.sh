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
conf=conf/hifigan_hubert_16k_nodp_f0.v1.yaml

# directory path setting
db_root=/data4/tyx/dataset/opencpop # direcotry including wavfiles (MODIFY BY YOURSELF)
                          # each wav filename in the directory should be unique
                          # e.g.
                          # /path/to/database
                          # ├── utt_1.wav
                          # ├── utt_2.wav
                          # │   ...
                          # └── utt_N.wav
dumpdir=dump           # directory to dump features

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)

# decoding related setting
checkpoint="" # checkpoint path to be used for decoding
              # if not provided, the latest one will be used
              # (e.g. <path>/<to>/checkpoint-400000steps.pkl)

train_set="train"       # name of training data directory
dev_set="dev"           # name of development data direcotry
eval_set="test"         # name of evaluation data direcotry

hubert_text=/data4/tyx/task/discrete_unit/opencpop_wavlm_km1024_noprefix.txt
use_f0=true            # whether add f0 
use_pretrain_feature=true

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Data preparation"
    if [ ! -e "${db_root}" ]; then
        echo "ERROR: Opencpop data does not exist."
    	echo "ERROR: Please download https://wenet.org.cn/opencpop/download/ and locate it at ${download_dir}"
    	exit 1
    fi
    mkdir -p wav_dump
    python local/data_prep.py ${db_root} \
        --wav_dumpdir wav_dump \
        --sr 24000

    sort -o data/train/wav.scp data/train/wav.scp

    dev_num=50
    train_num=$(( $(wc -l < data/train/wav.scp) - dev_num ))

    cp -r data/${train_set} data/${dev_set}
    head -n $train_num data/${train_set}/wav.scp > data/${train_set}/wav.scp.tmp
    tail -n $dev_num data/${train_set}/wav.scp > data/${dev_set}/wav.scp.tmp
    mv data/${dev_set}/wav.scp.tmp data/${dev_set}/wav.scp
    mv data/${train_set}/wav.scp.tmp data/${train_set}/wav.scp
    
    head -n $train_num data/${train_set}/utt2spk > data/${train_set}/utt2spk.tmp
    tail -n $dev_num data/${train_set}/utt2spk > data/${dev_set}/utt2spk.tmp
    mv data/${dev_set}/utt2spk.tmp data/${dev_set}/utt2spk
    mv data/${train_set}/utt2spk.tmp data/${train_set}/utt2spk
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
    # for name in "${dev_set}"; do
    for name in "${train_set}" "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        utils/make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/raw"
        _opts=
        if [ ${use_f0} == true ]; then
            _opts+="--use-f0"
        fi
        if [ ${use_pretrain_feature} == "true" ]; then
            _opts+="--use-pretrain-feature"
        fi
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            local/preprocess_hubert.py \
                --config "${conf}" \
                --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --text "${hubert_text}" \
                --verbose "${verbose}" ${_opts}
        echo "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished feature extraction."

fi

if [ -z "${tag}" ]; then
    expdir="exp/${train_set}_opencpop_$(basename "${conf}" .yaml)"
else
    expdir="exp/${train_set}_opencpop_${tag}"
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        train="python -m parallel_wavegan.distributed.launch --nproc_per_node ${n_gpus} -c parallel-wavegan-train"
    else
        train="parallel-wavegan-train"
    fi
    _opts=
    if [ ${use_f0} == true ]; then
        _opts+="--use-f0"
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
            --verbose "${verbose}" ${_opts}
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
        if [ ${use_f0} == true ]; then
            _opts+="--use-f0"
        fi
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
            parallel-wavegan-decode \
                --dumpdir "${dumpdir}/${name}/raw" \
                --checkpoint "${checkpoint}" \
                --outdir "${outdir}/${name}" \
                --verbose "${verbose}" \
                ${_opts}      
        echo "Successfully finished decoding of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished decoding."
fi
echo "Finished."

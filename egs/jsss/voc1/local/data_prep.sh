#!/bin/bash

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# Prepare kaldi-style data directory for JSSS corpus

fs=24000
num_dev=50
num_eval=50
train_set="train_nodev"
dev_set="dev"
eval_set="eval"
shuffle=false

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db=$1
data_dir_root=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 [Options] <db> <data_dir>"
    echo "e.g.: $0 downloads/jsss_ver1 data"
    echo ""
    echo "Options:"
    echo "    --fs: target sampling rate (default=24000)."
    echo "    --num_dev: number of development uttreances (default=50)."
    echo "    --num_eval: number of evaluation uttreances (default=50)."
    echo "    --train_set: name of train set (default=train_nodev)."
    echo "    --dev_set: name of dev set (default=dev)."
    echo "    --eval_set: name of eval set (default=eval)."
    echo "    --shuffle: whether to perform shuffle in making dev / eval set (default=false)."
    exit 1
fi

set -euo pipefail

######################################
#    process data without segments   #
######################################
dsets_without_segments="
short-form/basic5000
short-form/onomatopee300
short-form/voiceactress100
simplification
"
for dset in ${dsets_without_segments}; do
    # check directory existence
    _data_dir=${data_dir_root}/$(basename "${dset}")
    [ ! -e "${_data_dir}" ] && mkdir -p "${_data_dir}"

    # set filenames
    scp=${_data_dir}/wav.scp
    segments=${_data_dir}/segments

    # check file existence
    [ -e "${scp}" ] && rm "${scp}"
    [ -e "${segments}" ] && rm "${segments}"

    # make wav.scp and segments
    find "${db}/${dset}/wav24kHz16bit" -name "*.wav" | sort | while read -r filename; do
        utt_id=$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")
        lab_filename="${db}/${dset}/lab/$(basename "${filename}" .wav).lab"
        if [ ! -e "${lab_filename}" ]; then
            echo "${lab_filename} does not exist. Skipped."
            continue
        fi
        start_sec=$(head -n 1 "${lab_filename}" | cut -d " " -f 2)
        end_sec=$(tail -n 1 "${lab_filename}" | cut -d " " -f 1)
        echo "${utt_id} ${utt_id} ${start_sec} ${end_sec}" >> "${segments}"
        if [ "${fs}" -eq 24000 ]; then
            # default sampling rate
            echo "${utt_id} ${filename}" >> "${scp}"
        else
            echo "${utt_id} sox ${filename} -t wav -r $fs - |" >> "${scp}"
        fi
    done
    echo "Successfully prepared ${dset}."
done

######################################
#     process data with segments     #
######################################
dsets_with_segments="
long-form/katsura-masakazu
long-form/udon
long-form/washington-dc
summarization
"
for dset in ${dsets_with_segments}; do
    # check directory existence
    _data_dir=${data_dir_root}/$(basename "${dset}")
    [ ! -e "${_data_dir}" ] && mkdir -p "${_data_dir}"

    # set filenames
    scp=${_data_dir}/wav.scp
    segments=${_data_dir}/segments

    # check file existence
    [ -e "${scp}" ] && rm "${scp}"
    [ -e "${segments}" ] && rm "${segments}"

    # make wav.scp
    find "${db}/${dset}/wav24kHz16bit" -name "*.wav" | sort | while read -r filename; do
        wav_id=$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")
        if [ "${fs}" -eq 24000 ]; then
            # default sampling rate
            echo "${wav_id} ${filename}" >> "${scp}"
        else
            echo "${wav_id} sox ${filename} -t wav -r $fs - |" >> "${scp}"
        fi
    done

    # make segments
    find "${db}/${dset}/transcript_utf8" -name "*.txt" | sort | while read -r filename; do
        wav_id=$(basename "${filename}" .txt)
        while read -r line; do
            start_sec=$(echo "${line}" | cut -f 1)
            end_sec=$(echo "${line}" | cut -f 2)
            utt_id=${wav_id}
            utt_id+="_$(printf %010d "$(echo "${start_sec}" | tr -d "." | sed -e "s/^[0]*//g")")"
            utt_id+="_$(printf %010d "$(echo "${end_sec}" | tr -d "." | sed -e "s/^[0]*//g")")"

            # modify segment information with force alignment results
            lab_filename=${db}/${dset}/lab/${utt_id}.lab
            if [ ! -e "${lab_filename}" ]; then
                echo "${lab_filename} does not exist. Skipped."
                continue
            fi
            start_sec_offset=$(head -n 1 "${lab_filename}" | cut -d " " -f 2)
            end_sec_offset=$(tail -n 1 "${lab_filename}" | cut -d " " -f 1)
            start_sec=$(python -c "print(${start_sec} + ${start_sec_offset})")
            end_sec=$(python -c "print(${start_sec} + ${end_sec_offset} - ${start_sec_offset})")
            echo "${utt_id} ${wav_id} ${start_sec} ${end_sec}" >> "${segments}"
        done < "${filename}"
    done

    # fix
    echo "Successfully prepared ${dset}."
done

######################################
#       combine and split data       #
######################################
# combine all data
combined_data_dirs=""
for dset in ${dsets_without_segments} ${dsets_with_segments}; do
    combined_data_dirs+="${data_dir_root}/$(basename "${dset}") "
done
# shellcheck disable=SC2086
utils/combine_data.sh "${data_dir_root}/all" ${combined_data_dirs}
# shellcheck disable=SC2086
rm -rf ${combined_data_dirs}

# split
num_all=$(wc -l < "${data_dir_root}/all/segments")
num_deveval=$((num_dev + num_eval))
num_train=$((num_all - num_deveval))
utils/split_data.sh \
    --num_first "${num_deveval}" \
    --num_second "${num_train}" \
    --shuffle "${shuffle}" \
    "${data_dir_root}/all" \
    "${data_dir_root}/deveval" \
    "${data_dir_root}/${train_set}"
utils/split_data.sh \
    --num_first "${num_eval}" \
    --num_second "${num_dev}" \
    --shuffle "${shuffle}" \
    "${data_dir_root}/deveval" \
    "${data_dir_root}/${eval_set}" \
    "${data_dir_root}/${dev_set}"

# remove tmp directories
rm -rf "${data_dir_root}/all"
rm -rf "${data_dir_root}/deveval"

echo "Successfully prepared data."

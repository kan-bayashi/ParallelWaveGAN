#!/bin/bash

# Split data direcoty into two data direcotries

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

shuffle=false
num_first=0
num_second=0

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
    echo "Usage: $0 <src_dir> <dist_dir_1> <dist_dir_2> ..."
    echo "e.g.: $0 data/all data/train data/deveval"
    echo ""
    echo "Options:"
    echo "    --shuffle: Whether to perform shuffle (default=false)."
    echo "    --num_first: Number of utts in the first dist dir."
    echo "        If set to 0, it will be automatically decided (default=0)."
    echo "    --num_second: Number of utts in the second dist dir."
    echo "        If set to 0, it will be automatically decided (default=0)."
    exit 1
fi

set -eux

src_dir=$1
first_dist_dir=$2
second_dist_dir=$3

src_scp=${src_dir}/wav.scp
if [ -e "${src_dir}/segments" ]; then
    has_segments=true
    src_segments=${src_dir}/segments
    num_src_utts=$(wc -l < "${src_segments}")
else
    has_segments=false
    num_src_utts=$(wc -l < "${src_scp}")
fi
if [ -e "${src_dir}/utt2spk" ]; then
    has_utt2spk=true
    src_utt2spk=${src_dir}/utt2spk
else
    has_utt2spk=false
fi

if ${has_utt2spk}; then
    num_src_utt2spk=$(wc -l < "${src_utt2spk}")
    if [ "${num_src_utt2spk}" -ne "${num_src_utts}" ]; then
        echo "ERROR: wav.scp and utt2spk has different #lines (${num_src_utts} vs ${num_src_utt2spk})." >&2
        exit 1;
    fi
fi

# check number of utts
if [ "${num_first}" -eq 0 ] && [ "${num_second}" -eq 0 ]; then
    num_first=$((num_src_utts / 2 ))
    num_second=$((num_src_utts - num_first))
elif [ "${num_first}" -gt 0 ] && [ "${num_second}" -eq 0 ]; then
    [ "${num_src_utts}" -le "${num_first}" ] && \
        echo "ERROR: num_first must be less than # utts in src. (${num_first} vs ${num_src_utts})" >&2 && \
        exit 1
    num_second=$((num_src_utts - num_first))
elif [ "${num_first}" -eq 0 ] && [ "${num_second}" -gt 0 ]; then
    [ "${num_src_utts}" -le "${num_second}" ] && \
        echo "ERROR: num_second must be less than # utts in src. (${num_second} vs ${num_src_utts})" >&2 && \
        exit 1
    num_first=$((num_src_utts - num_second))
elif [ "${num_first}" -gt 0 ] && [ "${num_second}" -gt 0 ]; then
    [ "${num_src_utts}" -ne "$((num_first + num_second))" ] && \
        echo "ERROR: num_first + num_second must be the same # utts in src. ($((num_first + num_second)) vs ${num_src_utts})" >&2 && \
        exit 1
fi

# check directory existence
[ ! -e "${first_dist_dir}" ] && mkdir -p "${first_dist_dir}"
[ ! -e "${second_dist_dir}" ] && mkdir -p "${second_dist_dir}"

# split
if ! "${has_segments}"; then
    if "${shuffle}"; then
        sort -R "${src_scp}" > "${src_scp}.unsorted"
        head -n "${num_first}" "${src_scp}.unsorted" | sort > "${first_dist_dir}/wav.scp"
        tail -n "${num_second}" "${src_scp}.unsorted" | sort > "${second_dist_dir}/wav.scp"
        rm "${src_scp}.unsorted"
    else
        head -n "${num_first}" "${src_scp}" | sort > "${first_dist_dir}/wav.scp"
        tail -n "${num_second}" "${src_scp}" | sort > "${second_dist_dir}/wav.scp"
    fi
    if "${has_utt2spk}"; then
        grep -f <(awk '{print $1}' "${first_dist_dir}/wav.scp") \
            "${src_utt2spk}" > "${first_dist_dir}/utt2spk"
        grep -f <(awk '{print $1}' "${second_dist_dir}/wav.scp") \
            "${src_utt2spk}" > "${second_dist_dir}/utt2spk"
        diff -q <(awk '{print $1}' "${first_dist_dir}/wav.scp") \
            <(awk '{print $1}' "${first_dist_dir}/utt2spk") > /dev/null
        diff -q <(awk '{print $1}' "${second_dist_dir}/wav.scp") \
            <(awk '{print $1}' "${second_dist_dir}/utt2spk") > /dev/null
    fi
else
    # split segments at first
    if "${shuffle}"; then
        sort -R "${src_segments}" > "${src_segments}.unsorted"
        head -n "${num_first}" "${src_segments}.unsorted" | sort > "${first_dist_dir}/segments"
        tail -n "${num_second}" "${src_segments}.unsorted" | sort > "${second_dist_dir}/segments"
        rm "${src_segments}.unsorted"
    else
        head -n "${num_first}" "${src_segments}" | sort > "${first_dist_dir}/segments"
        tail -n "${num_second}" "${src_segments}" | sort > "${second_dist_dir}/segments"
    fi
    # split wav.scp
    rm -rf "${first_dist_dir}/wav.scp"
    awk '{print $2}' < "${first_dist_dir}/segments" | sort | uniq | while read -r wav_id; do
        grep "^${wav_id} " < "${src_scp}" >> "${first_dist_dir}/wav.scp"
    done
    rm -rf "${second_dist_dir}/wav.scp"
    awk '{print $2}' < "${second_dist_dir}/segments" | sort | uniq | while read -r wav_id; do
        grep "^${wav_id} " < "${src_scp}" >> "${second_dist_dir}/wav.scp"
    done
    if "${has_utt2spk}"; then
        grep -f <(awk '{print $1}' "${first_dist_dir}/segments") \
            "${src_utt2spk}" > "${first_dist_dir}/utt2spk"
        grep -f <(awk '{print $1}' "${second_dist_dir}/segments") \
            "${src_utt2spk}" > "${second_dist_dir}/utt2spk"
        diff -q <(awk '{print $1}' "${first_dist_dir}/segments") \
            <(awk '{print $1}' "${first_dist_dir}/utt2spk") > /dev/null
        diff -q <(awk '{print $1}' "${second_dist_dir}/segments") \
            <(awk '{print $1}' "${second_dist_dir}/utt2spk") > /dev/null
    fi
fi

echo "Successfully split data directory."

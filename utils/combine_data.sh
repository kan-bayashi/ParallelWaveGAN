#!/bin/bash

# Combine data direcotries into a single data direcotry

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

if [ $# -lt 2 ]; then
    echo "Usage: $0 <dist_dir> <src_dir_1> <src_dir_2> ..."
    echo "e.g.: $0 data/all data/spk_1 data/spk_2 data/spk_3"
    exit 1
fi

set -euo pipefail

dist_dir=$1
shift
first_src_dir=$1


[ ! -e "${dist_dir}" ] && mkdir -p "${dist_dir}"

if [ -e "${first_src_dir}/segments" ]; then
    has_segments=true
    segments=${dist_dir}/segments
    segments_tmp=${dist_dir}/segments.unsorted
    [ -e "${segments_tmp}" ] && rm "${segments_tmp}"
else
    has_segments=false
fi
if [ -e "${first_src_dir}/utt2spk" ]; then
    has_utt2spk=true
    utt2spk=${dist_dir}/utt2spk
    utt2spk_tmp=${dist_dir}/utt2spk.unsorted
    [ -e "${utt2spk_tmp}" ] && rm "${utt2spk_tmp}"
else
    has_utt2spk=false
fi
scp=${dist_dir}/wav.scp
scp_tmp=${dist_dir}/wav.scp.unsorted
[ -e "${scp_tmp}" ] && rm "${scp_tmp}"

# concatenate all of wav.scp and segments file
for _ in $(seq 1 ${#}); do
    src_dir=$1

    if "${has_segments}"; then
        [ ! -e "${src_dir}/segments" ] && \
            echo "WARN: Not found segments in ${src_dir}. Skipped." >&2 && shift && continue
        cat "${src_dir}/segments" >> "${segments_tmp}"
    fi

    if "${has_utt2spk}"; then
        [ ! -e "${src_dir}/utt2spk" ] && \
            echo "WARN: Not found utt2spk in ${src_dir}. Skipped." >&2 && shift && continue
        cat "${src_dir}/utt2spk" >> "${utt2spk_tmp}"
    fi

    [ ! -e "${src_dir}/wav.scp" ] && echo "Not found wav.scp in ${src_dir}." >&2 && exit 1;
    cat "${src_dir}/wav.scp" >> "${scp_tmp}"

    shift
done

# sort
sort "${scp_tmp}" > "${scp}"
if "${has_segments}"; then
    sort "${segments_tmp}" > "${segments}"
fi
if "${has_utt2spk}"; then
    sort "${utt2spk_tmp}" > "${utt2spk}"
    diff -q <(awk '{print $1}' "${scp}") <(awk '{print $1}' "${utt2spk}") > /dev/null
fi
rm "${dist_dir}"/*.unsorted

echo "Successfully combined data direcotries."

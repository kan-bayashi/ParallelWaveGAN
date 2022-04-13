#!/bin/bash

# Copyright 2020 Tomoki Hayashi, Adapted in 2022 Gunnar Thor Örnólfsson
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

train_set="train_nodev"
dev_set="dev"
eval_set="eval"

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db_root=$1
data_dir=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 [Options] <db_root> <data_dir>"
    echo ""
    echo "Options:"
    echo "    --train_set: name of train set (default=train_nodev)."
    echo "    --dev_set: name of dev set (default=dev)."
    echo "    --eval_set: name of eval set (default=eval)."
    exit 1
fi

set -euo pipefail

[ ! -e "${data_dir}/${train_set}" ] && mkdir -p "${data_dir}/${train_set}"
[ ! -e "${data_dir}/${eval_set}" ] && mkdir -p "${data_dir}/${eval_set}"
[ ! -e "${data_dir}/${dev_set}" ] && mkdir -p "${data_dir}/${dev_set}"

[ -e "${data_dir}/${train_set}/wav.scp" ] && rm "${data_dir}/${train_set}/wav.scp"
[ -e "${data_dir}/${eval_set}/wav.scp" ] && rm "${data_dir}/${eval_set}/wav.scp"
[ -e "${data_dir}/${dev_set}/wav.scp" ] && rm "${data_dir}/${dev_set}/wav.scp"

# make all scp
for speaker_id in "a" "b" "c" "d" "e" "f" "g" "h"
do
    paste -d " " \
        <(cut -f 1 < "${db_root}/split/${speaker_id}_train.txt") \
        <(cut -f 3 < "${db_root}/split/${speaker_id}_train.txt") \
        >> "${data_dir}/${train_set}/wav.scp"
    paste -d " " \
        <(cut -f 1 < "${db_root}/split/${speaker_id}_test.txt") \
        <(cut -f 3 < "${db_root}/split/${speaker_id}_test.txt") \
        >> "${data_dir}/${eval_set}/wav.scp"
    paste -d " " \
        <(cut -f 1 < "${db_root}/split/${speaker_id}_val.txt") \
        <(cut -f 3 < "${db_root}/split/${speaker_id}_val.txt") \
        >> "${data_dir}/${dev_set}/wav.scp"
done

echo "Successfully prepared data."

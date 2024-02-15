#!/usr/bin/env bash

# Reference from ESPnet's egs2/nit_song070/svs1/local/data.sh
# https://github.com/espnet/espnet/blob/master/egs2/nit_song070/svs1/local/data.sh


set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

db_root=$1
data_dir=$2

SECONDS=0
stage=-1
stop_stage=100
fs=24000
g2p=None

log "$0 $*"

. utils/parse_options.sh || exit 1;

if [ -z "${db_root}" ]; then
    log "Fill the value of 'db_root' of db.sh"
    exit 1
fi

mkdir -p ${db_root}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    if [ -e "${db_root}/todai_child" ] && [ -e "${db_root}/jsut-song_ver1/child_song/wav" ]; then
        echo "The JSUT-song corpus exists. Skip downloading."
    
    elif [ -e "${db_root}/jsut-song_ver1.zip" ] && [ -e "${db_root}/jsut-song_label.zip" ]; then
        echo "Unzipping downloaded zip files for JSUT-song corpus."
        unzip ${db_root}/jsut-song_ver1.zip -d ${db_root}
        unzip ${db_root}/jsut-song_label.zip -d ${db_root}
        rm ${db_root}/jsut-song_ver1.zip
        rm ${db_root}/jsut-song_label.zip

    if [ ! -e "${db_root}/jsut-song_ver1.zip" ] || [ ! -e "${db_root}/jsut-song_label.zip" ]; then
    	echo "ERROR: The JSUT-song corpus does not exist."
    	echo "ERROR: Please download from https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song"
        echo "and locate it at ${db_root}"
        echo "Please ensure that you've downloaded songs (jsut-song_ver1.zip) and labels (jsut-song_label.zip) to ${db_root} before proceeding"
        # Terms from https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song
    	exit 1
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data preparaion "

    mkdir -p score_dump
    mkdir -p wav_dump
    python local/data_prep.py \
        --lab_srcdir ${db_root}/todai_child \
        --wav_srcdir ${db_root}/jsut-song_ver1/child_song/wav \
        --score_dump score_dump \
        --wav_dumpdir wav_dump \
        --sr ${fs}
    for src_data in ${train_set} ${train_dev} ${eval_set}; do
        utils/utt2spk_to_spk2utt.pl < ${data_dir}/${src_data}/utt2spk > ${data_dir}/${src_data}/spk2utt
        utils/fix_data_dir.sh --utt_extra_files "label score.scp" ${data_dir}/${src_data}
    done
fi
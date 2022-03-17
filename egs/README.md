# Kaldi-style all-in-one recipes

This repository provides [Kaldi](https://github.com/kaldi-asr/kaldi)-style recipes, as the same as [ESPnet](https://github.com/espnet/espnet).  
Currently, the following recipes are supported.

- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/): English female speaker
- [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut): Japanese female speaker
- [JSSS](https://sites.google.com/site/shinnosuketakamichi/research-topics/jsss_corpus): Japanese female speaker
- [CSMSC](https://www.data-baker.com/open_source.html): Mandarin female speaker
- [CMU Arctic](http://www.festvox.org/cmu_arctic/): English speakers
- [JNAS](http://research.nii.ac.jp/src/en/JNAS.html): Japanese multi-speaker
- [VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html): English multi-speaker
- [LibriTTS](https://arxiv.org/abs/1904.02882): English multi-speaker
- [YesNo](https://arxiv.org/abs/1904.02882): English speaker (For debugging)
- [KSS](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset): Single Korean female speaker
- [Oniku\_kurumi\_utagoe\_db/](http://onikuru.info/db-download/): Single Japanese female singer (singing voice)
- [Kiritan](https://zunko.jp/kiridev/login.php): Single Japanese male singer (singing voice)
- [Ofuton\_p\_utagoe\_db](https://sites.google.com/view/oftn-utagoedb/%E3%83%9B%E3%83%BC%E3%83%A0): Single Japanese female singer (singing voice)
- [Opencpop](https://wenet.org.cn/opencpop/download/): Single Mandarin female singer (singing voice)
- [CSD](https://zenodo.org/record/4785016/): Single Korean/English female singer (singing voice)
- [KiSing](http://shijt.site/index.php/2021/05/16/kising-the-first-open-source-mandarin-singing-voice-synthesis-corpus/): Single Mandarin female singer (singing voice) 

## How to run the recipe

```bash
# Let us move on the recipe directory
$ cd egs/ljspeech/voc1
# Run the recipe from scratch
$ ./run.sh

# You can change config via command line
$ ./run.sh --conf <your_customized_yaml_config>

# You can select the stage to start and stop
$ ./run.sh --stage 2 --stop_stage 2

# If you want to specify the gpu
$ CUDA_VISIBLE_DEVICES=1 ./run.sh --stage 2

# If you want to resume training from 10000 steps checkpoint
$ ./run.sh --stage 2 --resume <path>/<to>/checkpoint-10000steps.pkl
```

You can check the command line options in `run.sh`.

The integration with job schedulers such as [slurm](https://slurm.schedmd.com/documentation.html) can be done via `cmd.sh` and  `conf/slurm.conf`.  
If you want to use it, please check [this page](https://kaldi-asr.org/doc/queue.html).

All of the hyperparameters are written in a single yaml format configuration file.  
Please check [this example](https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/egs/ljspeech/voc1/conf/parallel_wavegan.v1.yaml) in ljspeech recipe.

You can monitor the training progress via tensorboard.

```bash
$ tensorboard --logdir exp
```

![](https://user-images.githubusercontent.com/22779813/68100080-58bbc500-ff09-11e9-9945-c835186fd7c2.png)

If you want to accelerate the training, you can try distributed multi-gpu training based on apex.  
You need to install apex for distributed training. Please make sure you already installed it.  
Then you can run distributed multi-gpu training via following command:

```bash
# in the case of the number of gpus = 8
$ CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" ./run.sh --stage 2 --n_gpus 8
```

In the case of distributed training, the batch size will be automatically multiplied by the number of gpus.  
Please be careful.

## How to make the recipe for your own dateset

Here, I will show how to make the recipe for your own dataset.

1. Setup your dataset to be the following structure.

    ```bash
    # For single-speaker case
    $ tree /path/to/databse
    /path/to/database
    ├── utt_1.wav
    ├── utt_2.wav
    │   ...
    └── utt_N.wav
    # The directory can be nested, but each filename must be unique

    # For multi-speaker case
    $ tree /path/to/databse
    /path/to/database
    ├── spk_1
    │   ├── utt1.wav
    ├── spk_2
    │   ├── utt1.wav
    │   ...
    └── spk_N
        ├── utt1.wav
        ...
    # The directory under each speaker can be nested, but each filename in each speaker directory must be unique
    ```

2. Copy the template directory.

    ```bash
    cd egs

    # For single speaker case
    cp -r template_single_spk <your_dataset_name>

    # For multi speaker case
    cp -r template_multi_spk <your_dataset_name>

    # Move on your recipe
    cd egs/<your_dataset_name>/voc1
    ```

3. Modify the options in `run.sh`.  
   What you need to change at least in `run.sh` is as follows:
   - `db_root`: Root path of the database.
   - `num_dev`: The number of utterances for development set.
   - `num_eval`: The number of utterances for evaluation set.

4. Modify the hyperpameters in `conf/parallel_wavegan.v1.yaml`.  
   What you need to change at least in config is as follows:
    - `sampling_rate`: If you can specify the lower sampling rate, the audio will be downsampled by sox.

5. (Optional) Change command backend in `cmd.sh`.  
   If you are not familiar with kaldi and run in your local env, you do not need to change.  
   See more info on https://kaldi-asr.org/doc/queue.html.

6. Run your recipe.

    ```bash
    # Run all stages from the first stage
    ./run.sh

    # If you want to specify CUDA device
    CUDA_VISIBLE_DEVICES=0 ./run.sh
    ```

If you want to try the other advanced model, please check the config files in `egs/ljspeech/voc1/conf`.

## Run training using ESPnet2-TTS recipe within 5 minutes

Make sure already you finished the espnet2-tts recipe experiments (at least starting the training).

```bash
cd egs

# Please use single spk template for both single and multi spk case
cp -r template_single_spk <recipe_name>

# Move on your recipe
cd egs/<recipe_name>/voc1

# Make symlink of data directory (Better to use absolute path)
mkdir dump data
ln -s /path/to/espnet/egs2/<recipe_name>/tts1/dump/raw dump/
ln -s /path/to/espnet/egs2/<recipe_name>/tts1/dump/raw/tr_no_dev data/train_nodev
ln -s /path/to/espnet/egs2/<recipe_name>/tts1/dump/raw/dev data/dev
ln -s /path/to/espnet/egs2/<recipe_name>/tts1/dump/raw/eval1 data/eval

# Edit config to match TTS model setting
vim conf/parallel_wavegan.v1.yaml

# Run from stage 1
./run.sh --stage 1 --conf conf/parallel_wavegan.v1.yaml
```

That's it!

## Run finetuning using ESPnet2-TTS GTA outputs

Here, assume that you already finished the training of text2mel model with ESPnet2 recipe and vocoder with this repository.
At first, run teacher-forcing decoding for train and dev sets in ESPnet2 recipe.

```sh
cd espnet/egs2/your_recipe/tts1
./run.sh \
    --ngpu 1 \
    --stage 7 \
    --test_sets "tr_no_dev dev" \
    --inference_args "--use_teacher_forcing true"
```

Then move on vocoder recipe and run fine-tuning:
```sh
cd parallel_wavegan/egs/your_recipe/voc1

# make symlink to text2mel model and dump dirs
ln -s /path/to/espnet/egs2/your_recipe/tts1/dump/raw dump
ln -s /path/to/espnet/egs2/your_recipe/tts1/exp/your_text2mel_mode_dir exp/

# make config for finetune
# e.g.
vim conf/hifigan_melgan.v1.finetune.yaml

# run fine-tuning
. ./path.sh
parallel-wavegan-train \
    --config conf/hifigan_melgan.v1.finetune.yaml \
    --train-wav-scp dump/raw/tr_no_dev/wav.scp \
    --train-feats-scp exp/your_text2mel_mode_dir/decode_use_teacher_forcingtrue_train.loss.ave/tr_no_dev/norm/feats.scp \
    --dev-wav-scp dump/raw/dev/wav.scp \
    --dev-feats-scp exp/your_text2mel_mode_dir/decode_use_teacher_forcingtrue_train.loss.ave/dev/norm/feats.scp \
    --outdir exp/your_finetuned_vocoder_outdir \
    --pretrain /path/to/vocoder_checkpoint.pkl \
    --verbose 1
```

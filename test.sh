
#!/bin/bash
python parallel_wavegan/bin/preprocess.py \
    --config /data3/qt/ParallelWaveGAN/egs/opencpop_f0_exiciation/voc1/conf/hifigan.v1.yaml \
    --rootdir /data3/qt/ParallelWaveGAN/egs/opencpop_f0_exiciation/voc1/wav_dump \
    --dumpdir dump